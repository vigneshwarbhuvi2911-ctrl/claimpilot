"""Wires settings, database, products, clause index, LLM client and the graph together."""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.agents.graph import build_graph
from app.agents.nodes import Deps
from app.config import Settings
from app.db import Database
from app.llm.client import make_client
from app.logging_setup import log_event
from app.models import ClaimSubmission
from app.retrieval.clauses import ClauseIndex, HashingEmbedder, OpenAIEmbedder

log = logging.getLogger(__name__)


def load_products(data_dir: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((Path(data_dir) / "products").glob("*.json"))}


class ClaimService:
    def __init__(self, settings: Settings, llm=None, embedder=None):
        self.settings = settings
        self.db = Database(settings.db_path)
        self.db.init(Path(settings.data_dir) / "seed.json")
        self.products = load_products(settings.data_dir)
        self.llm = llm or make_client(settings)
        if embedder is None:
            embedder = (OpenAIEmbedder(settings.openai_embedding_model, settings.llm_timeout_s)
                        if self.llm.provider == "openai" else HashingEmbedder())
        try:
            self.index = ClauseIndex.from_dir(Path(settings.data_dir) / "policies", self.products, embedder)
        except Exception as exc:  # embedding API down: degrade to offline embeddings, keep running
            log_event(log, "embedder_fallback", error=type(exc).__name__)
            self.index = ClauseIndex.from_dir(Path(settings.data_dir) / "policies", self.products, HashingEmbedder())
        self.graph = build_graph(Deps(self.db, self.products, self.index, self.llm, settings.senior_review_threshold))
        log_event(log, "service_ready", provider=self.llm.provider, model=self.llm.model,
                  embedder=self.index.embedder.name, clauses=len(self.index.clauses))

    def process(self, submission: ClaimSubmission) -> dict:
        run_id = str(uuid.uuid4())
        t0 = time.time()
        state = self.graph.invoke({"run_id": run_id, "submission": submission.model_dump(), "trace": []})
        a = state["assessment"]
        result = {
            "run_id": run_id,
            "claim_ref": submission.claim_ref,
            "policy_no": state["extracted"].get("policy_no"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": state["status"],
            "recommendation": a["decision"],
            "payable": a["payable"],
            "claimed": a["claimed"],
            "requires_senior_review": state["requires_senior_review"],
            "note_for_handler": state["recommendation"],
            "findings": a["findings"],
            "breakdown": a["breakdown"],
            "extracted": state["extracted"],
            "flags": state.get("flags", []),
            "dropped_flags": state.get("dropped_flags", []),
            "clauses_used": state.get("clauses_used", []),
            "trace": state["trace"],
            "usage": state["usage"].as_dict(),
            "latency_ms": int((time.time() - t0) * 1000),
        }
        self.db.save_run(result)
        log_event(log, "claim_processed", run_id=run_id, policy_no=result["policy_no"],
                  decision=result["recommendation"], payable=result["payable"], latency_ms=result["latency_ms"],
                  llm_calls=result["usage"]["calls"], cost_usd=result["usage"]["cost_usd"],
                  nodes=[t["node"] for t in result["trace"]])
        return result
