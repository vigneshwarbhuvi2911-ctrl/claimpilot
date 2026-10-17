"""LangGraph nodes. Each node takes the shared ClaimState and returns the keys it updates.

    intake -> document_reader -> exclusion_screener -> adjudicate
           -> (route) recommendation_writer | request_documents -> finalize

LLM agents: document_reader, exclusion_screener, recommendation_writer.
Tools (plain Python, called by nodes): policy lookup and claims history (SQL), clause search (hybrid
retrieval), and the rules engine. Identity and permissions never come from the model.
"""
from __future__ import annotations

import json
import logging
import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict

from pydantic import ValidationError

from app.agents import heuristics
from app.db import Database
from app.guardrails import Masker, is_grounded
from app.llm.client import LLMClient, LLMError, Usage
from app.llm.prompts import DOCUMENT_READER, EXCLUSION_SCREENER, RECOMMENDATION_WRITER
from app.logging_setup import log_event
from app.models import ExtractedClaim, ScreenFlag
from app.retrieval.clauses import ClauseIndex
from app.rules import assess

log = logging.getLogger(__name__)


class ClaimState(TypedDict, total=False):
    run_id: str
    submission: dict
    docs: dict                 # doc_type -> original text
    masked_docs: dict          # doc_type -> text with personal data masked (what the LLM sees)
    masker: Any                # Masker (in memory only; never persisted)
    usage: Any                 # Usage
    policy: Optional[dict]
    history: list
    product: Optional[dict]
    extracted: dict
    flags: list
    dropped_flags: list
    clauses_used: list
    assessment: dict
    recommendation: dict
    status: str
    requires_senior_review: bool
    trace: Annotated[list, operator.add]


@dataclass
class Deps:
    db: Database
    products: dict
    index: ClauseIndex
    llm: LLMClient
    senior_review_threshold: int


def _step(name: str, t0: float, **info) -> list:
    return [{"node": name, "ms": int((time.time() - t0) * 1000), **info}]


def _docs_block(masked: dict) -> str:
    return "\n\n".join(f"<<<DOCUMENT type={t}\n{txt}\nDOCUMENT>>>" for t, txt in masked.items())


def _number(s) -> float:
    return float(str(s).replace(",", "").replace("Rs", "").strip())


# ------------------------------------------------------------------------------- nodes
def make_nodes(deps: Deps) -> dict:
    def intake(state: ClaimState) -> dict:
        t0 = time.time()
        docs: dict[str, str] = {}
        for d in state["submission"]["documents"]:
            docs[d["doc_type"]] = (docs.get(d["doc_type"], "") + "\n" + d["text"]).strip()
        form = docs.get("claim_form", "")
        m = re.search(r"\b(?:MOT|HLT)-\d{4}\b", form) or re.search(r"\b(?:MOT|HLT)-\d{4}\b", "\n".join(docs.values()))
        policy = deps.db.get_policy(m.group(0)) if m else None          # tool: policy system
        history = deps.db.claims_history(policy["policy_no"]) if policy else []  # tool: claims history
        product = deps.products.get(policy["product_code"]) if policy else None
        masker = Masker()
        if policy:
            masker.add_known("PERSON", policy["full_name"])
            masker.add_known("PHONE", policy["phone"])
            masker.add_known("EMAIL", policy["email"])
            masker.add_known("DOB", policy["dob"])
        masked = {t: masker.mask(x) for t, x in docs.items()}
        return {"docs": docs, "masked_docs": masked, "masker": masker, "usage": Usage(), "policy": policy,
                "history": history, "product": product,
                "trace": _step("intake", t0, policy_found=bool(policy), documents=sorted(docs),
                               prior_claims=len(history), masked_values=len(masker.mapping))}

    def document_reader(state: ClaimState) -> dict:
        t0 = time.time()
        source = "heuristic"
        ex = None
        if not deps.llm.is_mock:
            try:
                out = deps.llm.complete_json(DOCUMENT_READER, _docs_block(state["masked_docs"]), agent="document_reader",
                                             usage=state["usage"], run_id=state["run_id"])
                ex = ExtractedClaim.model_validate(out)
                source = "llm"
            except (LLMError, ValidationError) as exc:
                log_event(log, "agent_fallback", run_id=state["run_id"], agent="document_reader", error=type(exc).__name__)
        if ex is None:
            ex = heuristics.read_documents(state["masked_docs"])
        if state.get("policy"):
            ex.policy_no = state["policy"]["policy_no"]   # the policy system is the source of truth
        return {"extracted": ex.model_dump(), "trace": _step("document_reader", t0, source=source)}

    def exclusion_screener(state: ClaimState) -> dict:
        t0 = time.time()
        product = state.get("product")
        ex = state["extracted"]
        clauses = []
        if product:
            q = " ".join(filter(None, [ex.get("cause"), ex.get("diagnosis"), "exclusions waiting period",
                                       state["masked_docs"].get("claim_form", "")[-300:]]))
            clauses = deps.index.search(q, product["product_code"], k=4)   # tool: clause search
        source, raw = "heuristic", None
        if not deps.llm.is_mock:
            try:
                user = ("POLICY CLAUSES (context):\n" + "\n\n".join(f"[{c.clause_id}] {c.title}\n{c.text}" for c in clauses)
                        + "\n\nCLAIM DOCUMENTS:\n" + _docs_block(state["masked_docs"]))
                out = deps.llm.complete_json(EXCLUSION_SCREENER, user, agent="exclusion_screener",
                                             usage=state["usage"], run_id=state["run_id"])
                raw = out.get("flags", [])
                source = "llm"
            except LLMError as exc:
                log_event(log, "agent_fallback", run_id=state["run_id"], agent="exclusion_screener", error=type(exc).__name__)
        flags, dropped = [], []
        if raw is None:
            flags = heuristics.screen_documents(state["masked_docs"])
        else:
            for item in raw:
                try:
                    f = ScreenFlag.model_validate(item)
                except ValidationError:
                    dropped.append({"item": item, "reason": "invalid flag"})
                    continue
                # Hallucination control: the quote must exist in the documents.
                if is_grounded(f.evidence, list(state["masked_docs"].values())):
                    flags.append(f)
                else:
                    dropped.append({"flag": f.flag, "reason": "evidence not found in documents"})
        return {"flags": [f.model_dump() for f in flags], "dropped_flags": dropped,
                "clauses_used": [c.clause_id for c in clauses],
                "trace": _step("exclusion_screener", t0, source=source, flags=[f.flag for f in flags],
                               dropped=len(dropped), clauses=[c.clause_id for c in clauses])}

    def adjudicate(state: ClaimState) -> dict:
        t0 = time.time()
        product = state.get("product") or next(iter(deps.products.values()))
        a = assess(product, state.get("policy"), state.get("history", []),
                   ExtractedClaim.model_validate(state["extracted"]),
                   [ScreenFlag.model_validate(f) for f in state.get("flags", [])], set(state["docs"]))
        return {"assessment": a.model_dump(),
                "trace": _step("adjudicate", t0, decision=a.decision, payable=a.payable,
                               findings=[f.code for f in a.findings])}

    def request_documents(state: ClaimState) -> dict:
        t0 = time.time()
        rec = heuristics.write_recommendation("REQUEST_DOCUMENTS", state["assessment"])
        return {"recommendation": rec, "trace": _step("request_documents", t0)}

    def recommendation_writer(state: ClaimState) -> dict:
        t0 = time.time()
        a = state["assessment"]
        rec, source = None, "template"
        if not deps.llm.is_mock:
            refs = {f["clause"] for f in a["findings"] if f.get("clause")}
            clause_texts = [c for c in (deps.index.get(r) for r in sorted(refs)) if c]
            payload = {"decision": a["decision"], "claimed": a["claimed"], "payable": a["payable"],
                       "findings": a["findings"], "breakdown": a["breakdown"],
                       "facts": {k: v for k, v in state["extracted"].items() if k not in ("estimate", "bill")},
                       "clauses": [c.as_dict() for c in clause_texts]}
            try:
                out = deps.llm.complete_json(RECOMMENDATION_WRITER, json.dumps(payload, ensure_ascii=False),
                                             agent="recommendation_writer", usage=state["usage"], run_id=state["run_id"])
                problem = _check_recommendation(out, a, state["masked_docs"])
                if problem:
                    log_event(log, "recommendation_rejected", run_id=state["run_id"], reason=problem)
                else:
                    rec, source = out, "llm"
            except LLMError as exc:
                log_event(log, "agent_fallback", run_id=state["run_id"], agent="recommendation_writer", error=type(exc).__name__)
        if rec is None:
            rec = heuristics.write_recommendation(a["decision"], a)
        masker: Masker = state["masker"]
        rec = {"decision": a["decision"],
               "summary": masker.unmask(str(rec.get("summary", ""))),
               "reasons": [masker.unmask(str(x)) for x in rec.get("reasons", [])],
               "next_steps": [masker.unmask(str(x)) for x in rec.get("next_steps", [])]}
        return {"recommendation": rec, "trace": _step("recommendation_writer", t0, source=source)}

    def finalize(state: ClaimState) -> dict:
        t0 = time.time()
        payable = state["assessment"].get("payable") or 0
        return {"status": "awaiting_handler",
                "requires_senior_review": payable > deps.senior_review_threshold,
                "trace": _step("finalize", t0)}

    return {"intake": intake, "document_reader": document_reader, "exclusion_screener": exclusion_screener,
            "adjudicate": adjudicate, "request_documents": request_documents,
            "recommendation_writer": recommendation_writer, "finalize": finalize}


def route_after_adjudication(state: ClaimState) -> str:
    return "request_documents" if state["assessment"]["decision"] == "REQUEST_DOCUMENTS" else "recommendation_writer"


_MONEY = re.compile(r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)")


def _check_recommendation(out: dict, assessment: dict, masked_docs: dict) -> Optional[str]:
    """Output guardrail: the writer may not change the decision or introduce amounts."""
    if str(out.get("decision", "")).upper() != assessment["decision"]:
        return "decision changed"
    allowed = set()
    for v in re.findall(r"\d[\d,]*(?:\.\d+)?", json.dumps(assessment) + " ".join(masked_docs.values())):
        try:
            allowed.add(round(_number(v), 2))
        except ValueError:
            pass
    text = " ".join([str(out.get("summary", ""))] + [str(x) for x in out.get("reasons", []) + out.get("next_steps", [])])
    for m in _MONEY.finditer(text):
        if round(_number(m.group(1)), 2) not in allowed:
            return f"unsupported amount {m.group(0)}"
    return None
