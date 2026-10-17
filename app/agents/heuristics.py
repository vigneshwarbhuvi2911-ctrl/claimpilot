"""Offline fallbacks for the three agents, used when LLM_PROVIDER=mock (tests, CI, demos without an
API key) and as a safety net if a model call fails. They work on the labelled fields of the demo
documents; the LLM agents handle free-form text."""
from __future__ import annotations

import re

from app.models import ExtractedClaim, HospitalBill, MotorEstimate, ScreenFlag

_AMT = r"Rs\.?\s*([\d,]+(?:\.\d+)?)"


def _field(text: str, label: str) -> str | None:
    m = re.search(rf"(?im)^\s*{label}\s*:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m else None


def _amount(text: str, label: str) -> float | None:
    m = re.search(rf"(?im)^\s*{label}\s*:.*?{_AMT}", text)
    return float(m.group(1).replace(",", "")) if m else None


def _date(text: str, label: str) -> str | None:
    v = _field(text, label)
    m = re.search(r"\d{4}-\d{2}-\d{2}", v or "")
    return m.group(0) if m else None


def read_documents(docs: dict[str, str]) -> ExtractedClaim:
    form = docs.get("claim_form", "")
    all_text = "\n".join(docs.values())
    ex = ExtractedClaim()
    m = re.search(r"\b(?:MOT|HLT)-\d{4}\b", form or all_text)
    ex.policy_no = m.group(0) if m else None
    ex.amount_claimed = _amount(form, "Amount claimed")
    if "HEALTH CLAIM" in form.upper() or "discharge_summary" in docs or "final_bill" in docs:
        ex.claim_kind = "health_hospitalisation"
        ex.incident_date = _date(form, "Date of admission")
        ex.discharge_date = _date(form, "Date of discharge")
        ex.intimation_date = _date(form, "Date of claim submission")
        ex.diagnosis = _field(docs.get("discharge_summary", ""), "Diagnosis")
        bill = docs.get("final_bill", "")
        if bill:
            rm = re.search(rf"(?im)^\s*Room rent\s*:\s*{_AMT}\s*per day\s*x\s*(\d+)\s*days?\s*=\s*{_AMT}", bill)
            ex.bill = HospitalBill(
                room_rent_per_day=float(rm.group(1).replace(",", "")) if rm else 0,
                days=int(rm.group(2)) if rm else 0,
                room=float(rm.group(3).replace(",", "")) if rm else (_amount(bill, "Room rent") or 0),
                nursing=_amount(bill, "Nursing charges") or 0, doctor_fees=_amount(bill, "Doctor fees") or 0,
                investigations=_amount(bill, "Investigations") or 0, medicines=_amount(bill, "Medicines") or 0,
                consumables=_amount(bill, "Consumables") or 0)
    else:
        ctype = (_field(form, "Claim type") or "").lower()
        desc = (_field(form, "Description of loss") or "").lower()
        ex.claim_kind = "motor_theft" if "theft" in ctype else "motor_own_damage"
        ex.incident_date = _date(form, "Date of incident")
        ex.intimation_date = _date(form, "Date of intimation")
        if "theft" in ctype or "stolen" in desc:
            ex.cause = "theft"
        elif re.search(r"engine|seized|breakdown|no external impact", desc) and not re.search(r"\bhit\b|collid", desc):
            ex.cause = "mechanical_breakdown"
        elif "fire" in desc:
            ex.cause = "fire"
        elif "flood" in desc:
            ex.cause = "flood"
        else:
            ex.cause = "accident"
        ex.licence_valid_till = _date(docs.get("driving_licence", ""), "Valid till")
        est = docs.get("repair_estimate", "")
        if est:
            ex.estimate = MotorEstimate(metal_parts=_amount(est, "Metal parts") or 0,
                                        plastic_parts=_amount(est, "Plastic parts") or 0,
                                        glass_parts=_amount(est, "Glass parts") or 0, labour=_amount(est, "Labour") or 0)
    return ex


_RULES = [
    ("alcohol", r"[^.\n]*(?:breath analyser[^.\n]*alcohol|under the influence|intoxicated)[^.\n]*"),
    ("no_valid_licence", r"[^.\n]*(?:did not (?:hold|have) (?:a )?(?:valid )?(?:driving )?licen[cs]e|without (?:a )?licen[cs]e)[^.\n]*"),
    ("wear_and_tear", r"[^.\n]*(?:no external impact|wear and tear|mechanical breakdown|seized due to)[^.\n]*"),
    ("commercial_use", r"[^.\n]*(?:ride-hailing|used as a taxi|for hire|carrying paying passengers)[^.\n]*"),
    ("cosmetic", r"[^.\n]*(?:aesthetic reasons|cosmetic)[^.\n]*"),
    ("self_inflicted", r"[^.\n]*(?:self-inflicted|self harm|suicide attempt)[^.\n]*"),
    ("accident", r"[^.\n]*(?:road traffic accident|injured in an accident|fell from)[^.\n]*"),
]
_PED = re.compile(r"[^.\n]*known case of ([a-z0-9 ]+?)(?: mellitus)?(?: for| since|,|\.)[^.\n]*", re.I)
_SPECIFIED = ("cataract", "hernia", "joint replacement")


def screen_documents(docs: dict[str, str]) -> list[ScreenFlag]:
    text = "\n".join(docs.values())
    flags = []
    for flag, rx in _RULES:
        m = re.search(rx, text, re.I)
        if m:
            flags.append(ScreenFlag(flag=flag, evidence=m.group(0).strip()))
    for m in _PED.finditer(text):
        cond = re.sub(r"^type \d\s+", "", m.group(1).strip().lower())
        flags.append(ScreenFlag(flag="pre_existing_condition", evidence=m.group(0).strip(), detail=cond))
    diag = (re.search(r"(?im)^\s*Diagnosis\s*:\s*(.+)$", docs.get("discharge_summary", "")) or [None, ""])[1]
    for s in _SPECIFIED:
        if s in (diag or "").lower():
            flags.append(ScreenFlag(flag="specified_disease", evidence=diag.strip(), detail=s))
    return flags


def write_recommendation(decision: str, assessment: dict) -> dict:
    payable = assessment.get("payable")
    claimed = assessment.get("claimed")
    amt = f" Recommended payable: Rs {payable:,.0f} of Rs {claimed:,.0f} claimed." if payable is not None and claimed else ""
    reasons = [f["message"] + (f" [{f['clause']}]" if f.get("clause") else "") for f in assessment["findings"]
               if f["effect"] != "info"]
    steps = {
        "APPROVE": ["Verify the documents and approve the settlement."],
        "PARTIAL": ["Verify the deductions and approve the partial settlement."],
        "REJECT": ["Confirm the evidence and send a repudiation letter citing the clause."],
        "REFER": ["Review the flagged points before deciding."],
        "REQUEST_DOCUMENTS": ["Ask the customer for the missing documents."],
    }[decision]
    return {"decision": decision, "summary": f"Rules-engine decision: {decision}.{amt}",
            "reasons": reasons or ["All checks passed."], "next_steps": steps}
