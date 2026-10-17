"""Deterministic adjudication. The LLM agents read documents and explain decisions; this module
makes the decision and computes every amount, from the product configuration (data/products/*.json)
and the policy record. Each finding carries the clause it comes from.

Decision precedence: REQUEST_DOCUMENTS > REJECT > REFER > PARTIAL > APPROVE.
- APPROVE: everything claimed is admissible; only standard terms applied (deductible,
  depreciation, co-payment).
- PARTIAL: some amounts are not admissible (non-payable items, room-rent proportion, sum insured cap).
- REFER: needs a human (late notice, claim frequency, fraud history, early claim, undisclosed
  condition, missing facts). An estimate of the payable amount is still computed for the handler.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from typing import Optional

from app.models import Assessment, ExtractedClaim, Finding, ScreenFlag

PRECEDENCE = ["REQUEST_DOCUMENTS", "REJECT", "REFER", "PARTIAL", "APPROVE"]


def d(s: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def add_months(x: date, months: int) -> date:
    y, m = divmod(x.month - 1 + months, 12)
    year, month = x.year + y, m + 1
    return date(year, month, min(x.day, monthrange(year, month)[1]))


def age_on(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def r2(x: float) -> float:
    return round(x + 1e-9, 2)


def assess(product: dict, policy: Optional[dict], history: list[dict], ex: ExtractedClaim,
           flags: list[ScreenFlag], doc_types: set[str]) -> Assessment:
    F: list[Finding] = []
    clauses = product["clauses"]

    # -- documents ------------------------------------------------------------------------
    key = "theft" if ex.claim_kind == "motor_theft" and "theft" in product["required_documents"] else "default"
    missing = [t for t in product["required_documents"][key] if t not in doc_types]
    if missing:
        F.append(Finding(code="missing_documents", clause=clauses["documents"], effect="missing_docs",
                         message="Missing documents: " + ", ".join(m.replace("_", " ") for m in missing)))
        return Assessment(decision="REQUEST_DOCUMENTS", claimed=ex.amount_claimed, payable=None, findings=F)

    if policy is None:
        F.append(Finding(code="policy_not_found", clause=None, effect="refer",
                         message=f"Policy {ex.policy_no or '(not found in documents)'} does not exist in the policy system."))
        return Assessment(decision="REFER", claimed=ex.amount_claimed, payable=None, findings=F)

    loss = d(ex.incident_date)
    if loss is None or ex.amount_claimed is None:
        F.append(Finding(code="insufficient_facts", clause=None, effect="refer",
                         message="Date of loss or amount claimed could not be read from the documents."))
        return Assessment(decision="REFER", claimed=ex.amount_claimed, payable=None, findings=F)

    # -- period ---------------------------------------------------------------------------
    if not (d(policy["start_date"]) <= loss <= d(policy["end_date"])):
        F.append(Finding(code="outside_policy_period", clause=clauses["period"], effect="reject",
                         message=f"Loss on {loss} is outside the period of insurance "
                                 f"{policy['start_date']} to {policy['end_date']}."))

    if product["product_code"] == "MOTOR_PC":
        payable, breakdown, partial = _motor(product, policy, history, ex, flags, loss, F)
    else:
        payable, breakdown, partial = _health(product, policy, history, ex, flags, loss, F)

    effects = {f.effect for f in F}
    if "reject" in effects:
        decision, payable = "REJECT", 0.0
        F = [f for f in F if f.effect != "deduction"]   # deductions are irrelevant once rejected
    elif "refer" in effects:
        decision = "REFER"
    elif partial:
        decision = "PARTIAL"
    else:
        decision = "APPROVE"
    return Assessment(decision=decision, claimed=ex.amount_claimed, payable=None if payable is None else r2(payable),
                      findings=F, breakdown=breakdown)


# ------------------------------------------------------------------------------- motor
def _motor(product, policy, history, ex, flags, loss, F):
    cl = product["clauses"]
    for f in flags:
        if f.flag in product["exclusion_flags"]:
            F.append(Finding(code=f"exclusion_{f.flag}", clause=product["exclusion_flags"][f.flag], effect="reject",
                             message=f"Exclusion applies ({f.flag.replace('_', ' ')}). Evidence: \"{f.evidence}\""))
    if ex.cause and ex.cause not in product["covered_causes"] and not any(f.flag == "wear_and_tear" for f in flags):
        F.append(Finding(code="cause_not_covered", clause=product["exclusion_flags"]["wear_and_tear"]
                         if ex.cause == "mechanical_breakdown" else cl["cover"], effect="reject",
                         message=f"Cause of loss '{ex.cause}' is not an insured peril."))
    if ex.claim_kind != "motor_theft":
        till = d(ex.licence_valid_till)
        if till is not None and till < loss and not any(f.flag == "no_valid_licence" for f in flags):
            F.append(Finding(code="licence_expired", clause=product["exclusion_flags"]["no_valid_licence"], effect="reject",
                             message=f"Driver's licence was valid only till {till}; loss on {loss}."))

    _notice(product, ex, loss, F, from_date=loss)
    first = d(policy["first_start_date"])
    if first and (loss - first).days < product["early_claim_days"]:
        F.append(Finding(code="early_claim", clause=None, effect="refer",
                         message=f"Loss {(loss - first).days} days after the first policy start date "
                                 f"({first}); early-claim review required."))
    _history_checks(history, loss, F)

    idv = float(policy["sum_insured"])
    ded = float(product["compulsory_deductible"])
    if ex.claim_kind == "motor_theft":
        payable = max(idv - ded, 0.0)
        return payable, [{"item": "IDV (theft, not recovered)", "amount": idv, "clause": cl["theft"]},
                         {"item": "Compulsory deductible", "amount": -ded, "clause": cl["deductible"]}], False

    est = ex.estimate
    if est is None:
        F.append(Finding(code="no_estimate", clause=cl["documents"], effect="refer",
                         message="Repair estimate amounts could not be read."))
        return None, [], False
    reg = d(policy["vehicle_reg_date"]) or d(policy["first_start_date"])
    rate = product["depreciation"]["metal_parts_by_age_months"][-1][1]
    for months, r in product["depreciation"]["metal_parts_by_age_months"]:
        if months is None or loss <= add_months(reg, months):
            rate = r
            break
    plastic_rate = product["depreciation"]["plastic_parts"]
    glass_rate = product["depreciation"]["glass_parts"]
    rows = [
        {"item": "Metal parts", "claimed": est.metal_parts, "depreciation": rate},
        {"item": "Plastic/rubber parts", "claimed": est.plastic_parts, "depreciation": plastic_rate},
        {"item": "Glass parts", "claimed": est.glass_parts, "depreciation": glass_rate},
        {"item": "Labour", "claimed": est.labour, "depreciation": 0.0},
    ]
    total = 0.0
    for row in rows:
        row["amount"] = r2(row["claimed"] * (1 - row["depreciation"]))
        row["clause"] = cl["depreciation"]
        total += row["amount"]
    rows.append({"item": "Compulsory deductible", "amount": -ded, "clause": cl["deductible"]})
    payable = max(total - ded, 0.0)
    F.append(Finding(code="depreciation", clause=cl["depreciation"], effect="deduction",
                     message=f"Metal parts depreciated at {rate:.0%} (vehicle registered {reg}); plastic "
                             f"{plastic_rate:.0%}; labour in full.",
                     amount=r2(sum(rw["claimed"] - rw["amount"] for rw in rows[:4]))))
    F.append(Finding(code="deductible", clause=cl["deductible"], effect="deduction",
                     message=f"Compulsory deductible Rs {ded:,.0f}.", amount=ded))
    if payable > idv:
        payable = idv
        F.append(Finding(code="idv_cap", clause=cl["cover"], effect="deduction", message="Limited to the IDV."))
        return payable, rows, True
    return payable, rows, False


# ------------------------------------------------------------------------------ health
def _health(product, policy, history, ex, flags, adm, F):
    cl = product["clauses"]
    first = d(policy["first_start_date"])
    dis = d(ex.discharge_date)
    if dis is not None and (dis - adm).days < 1:
        F.append(Finding(code="under_24h", clause=cl["cover"], effect="reject",
                         message="Hospitalisation shorter than 24 hours."))

    accident = any(f.flag == "accident" for f in flags)
    if first and adm < first + timedelta(days=product["initial_waiting_days"]) and not accident:
        F.append(Finding(code="initial_waiting", clause=cl["initial_waiting"], effect="reject",
                         message=f"Illness within {product['initial_waiting_days']} days of the first policy start "
                                 f"({first})."))

    diag = (ex.diagnosis or "").lower()
    specified = [s for s in product["specified_diseases"] if s in diag] + \
                [f.detail or f.evidence for f in flags if f.flag == "specified_disease"]
    if specified and first and adm < add_months(first, product["specified_disease_waiting_months"]):
        F.append(Finding(code="specified_disease_waiting", clause=cl["specified_waiting"], effect="reject",
                         message=f"'{specified[0]}' is covered only after {product['specified_disease_waiting_months']} "
                                 f"months from {first}."))

    declared = (policy.get("ped_declared") or "").lower()
    for f in flags:
        if f.flag == "pre_existing_condition":
            cond = (f.detail or "").lower()
            if declared and (declared in cond or cond in declared or declared in f.evidence.lower()):
                if first and adm < add_months(first, product["ped_waiting_months"]):
                    F.append(Finding(code="ped_waiting", clause=cl["ped_waiting"], effect="reject",
                                     message=f"Declared pre-existing disease '{declared}' is covered only after "
                                             f"{product['ped_waiting_months']} months from {first}."))
            else:
                F.append(Finding(code="undisclosed_ped", clause=cl["ped_waiting"], effect="refer",
                                 message=f"Pre-existing condition '{f.detail or 'unspecified'}' was not declared at "
                                         f"proposal. Evidence: \"{f.evidence}\""))
        elif f.flag in product["exclusion_flags"]:
            F.append(Finding(code=f"exclusion_{f.flag}", clause=product["exclusion_flags"][f.flag], effect="reject",
                             message=f"Exclusion applies ({f.flag.replace('_', ' ')}). Evidence: \"{f.evidence}\""))

    if dis is not None:
        _notice(product, ex, dis, F, from_date=dis)
    _history_checks(history, adm, F)

    b = ex.bill
    if b is None:
        F.append(Finding(code="no_bill", clause=cl["documents"], effect="refer", message="Hospital bill amounts could not be read."))
        return None, [], False
    si = float(policy["sum_insured"])
    cap = si * product["room_rent_pct_of_si"]
    days = b.days or (max((dis - adm).days, 1) if dis else 1)
    rate = b.room_rent_per_day or (b.room / days if days else 0)
    partial = False
    rows = []
    prop = 1.0
    if rate > cap > 0:
        prop = cap / rate
        room_pay = cap * days
        partial = True
        F.append(Finding(code="room_rent_limit", clause=cl["room_rent"], effect="deduction",
                         message=f"Room rent Rs {rate:,.0f}/day exceeds the limit of Rs {cap:,.0f}/day; room-related "
                                 f"charges paid at {prop:.1%}.", amount=r2(b.room - room_pay)))
    else:
        room_pay = b.room
    rows.append({"item": "Room rent", "claimed": b.room, "amount": r2(room_pay), "clause": cl["room_rent"]})
    for name, val in (("nursing", b.nursing), ("doctor_fees", b.doctor_fees), ("investigations", b.investigations)):
        p = prop if name in product["proportionate_items"] else 1.0
        rows.append({"item": name.replace("_", " ").capitalize(), "claimed": val, "amount": r2(val * p),
                     "clause": cl["room_rent"] if p < 1 else cl["cover"]})
    rows.append({"item": "Medicines", "claimed": b.medicines, "amount": b.medicines, "clause": cl["cover"]})
    if b.consumables:
        partial = True
        F.append(Finding(code="non_payable_items", clause=cl["non_payable"], effect="deduction",
                         message=f"Consumables of Rs {b.consumables:,.0f} are not payable.", amount=b.consumables))
    rows.append({"item": "Consumables", "claimed": b.consumables, "amount": 0.0, "clause": cl["non_payable"]})
    subtotal = sum(r["amount"] for r in rows)

    dob = d(policy.get("dob"))
    if dob and age_on(dob, adm) >= product["copay_age"]:
        copay = subtotal * product["copay_pct"]
        rows.append({"item": f"Co-payment {product['copay_pct']:.0%} (age {age_on(dob, adm)})", "amount": -r2(copay),
                     "clause": cl["copay"]})
        F.append(Finding(code="copay", clause=cl["copay"], effect="deduction",
                         message=f"Insured aged {age_on(dob, adm)} at admission: {product['copay_pct']:.0%} co-payment.",
                         amount=r2(copay)))
        subtotal -= copay

    start, end = d(policy["start_date"]), d(policy["end_date"])
    used = sum(h["amount_paid"] for h in history
               if h["status"] == "settled" and start <= d(h["loss_date"]) <= end)
    available = max(si - used, 0.0)
    if subtotal > available:
        partial = True
        F.append(Finding(code="sum_insured_cap", clause=cl["sum_insured"], effect="deduction",
                         message=f"Only Rs {available:,.0f} of the Rs {si:,.0f} sum insured is available "
                                 f"(Rs {used:,.0f} already paid this policy year).", amount=r2(subtotal - available)))
        rows.append({"item": "Limited to available sum insured", "amount": -r2(subtotal - available), "clause": cl["sum_insured"]})
        subtotal = available
    return subtotal, rows, partial


# ------------------------------------------------------------------------------ shared
def _notice(product, ex, loss_or_discharge, F, from_date):
    notified = d(ex.intimation_date)
    if notified and (notified - from_date).days > product["notice_days"]:
        F.append(Finding(code="late_notice", clause=product["clauses"]["notice"], effect="refer",
                         message=f"Notified {(notified - from_date).days} days after {from_date}; the limit is "
                                 f"{product['notice_days']} days. Reason for delay to be reviewed."))


def _history_checks(history, loss, F):
    if any(h.get("fraud_flag") for h in history):
        F.append(Finding(code="fraud_history", clause=None, effect="refer",
                         message="A previous claim on this policy was flagged for fraud. Investigation review required."))
    recent = [h for h in history if h["status"] == "settled" and d(h["loss_date"])
              and timedelta(0) <= loss - d(h["loss_date"]) <= timedelta(days=365)]
    if len(recent) >= 3:
        F.append(Finding(code="claim_frequency", clause=None, effect="refer",
                         message=f"{len(recent)} settled claims in the 12 months before this loss."))
