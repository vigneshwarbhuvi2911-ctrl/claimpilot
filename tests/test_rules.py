"""Rules engine: amounts and decisions for each policy term, independent of the agents."""
import json
from datetime import date

import pytest

from app.config import ROOT
from app.models import ExtractedClaim, HospitalBill, MotorEstimate, ScreenFlag
from app.rules import add_months, age_on, assess

MOTOR = json.loads((ROOT / "data/products/MOTOR_PC.json").read_text())
HEALTH = json.loads((ROOT / "data/products/HEALTH_IND.json").read_text())
MOTOR_DOCS = {"claim_form", "driving_licence", "repair_estimate"}
HEALTH_DOCS = {"claim_form", "discharge_summary", "final_bill"}


def motor_policy(**kw):
    p = {"policy_no": "MOT-9", "start_date": "2026-01-01", "end_date": "2026-12-31", "first_start_date": "2020-01-01",
         "sum_insured": 500000, "vehicle_reg_date": "2023-01-01", "dob": "1980-01-01", "ped_declared": None}
    p.update(kw)
    return p


def health_policy(**kw):
    p = {"policy_no": "HLT-9", "start_date": "2026-01-01", "end_date": "2026-12-31", "first_start_date": "2020-01-01",
         "sum_insured": 500000, "vehicle_reg_date": None, "dob": "1980-01-01", "ped_declared": None}
    p.update(kw)
    return p


def motor_claim(loss="2026-06-15", **kw):
    c = dict(policy_no="MOT-9", claim_kind="motor_own_damage", incident_date=loss, intimation_date=loss,
             amount_claimed=10000, cause="accident", licence_valid_till="2030-01-01",
             estimate=MotorEstimate(metal_parts=10000))
    c.update(kw)
    return ExtractedClaim(**c)


def health_claim(adm="2026-06-10", dis="2026-06-12", **kw):
    bill = kw.pop("bill", HospitalBill(room_rent_per_day=4000, days=2, room=8000, nursing=2000, doctor_fees=4000,
                                       investigations=4000, medicines=2000))
    c = dict(policy_no="HLT-9", claim_kind="health_hospitalisation", incident_date=adm, discharge_date=dis,
             intimation_date=dis, amount_claimed=20000, diagnosis="Typhoid fever", bill=bill)
    c.update(kw)
    return ExtractedClaim(**c)


def test_add_months_clamps_month_end():
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2023, 3, 10), 36) == date(2026, 3, 10)


def test_age_on_birthday_boundary():
    assert age_on(date(1966, 6, 10), date(2026, 6, 9)) == 59
    assert age_on(date(1966, 6, 10), date(2026, 6, 10)) == 60


@pytest.mark.parametrize("reg,loss,rate", [
    ("2026-01-01", "2026-07-01", 0.0),     # exactly 6 months -> nil
    ("2026-01-01", "2026-07-02", 0.05),    # just over 6 months
    ("2025-01-01", "2026-06-15", 0.10),
    ("2023-06-16", "2026-06-15", 0.15),
    ("2022-06-16", "2026-06-15", 0.25),
    ("2021-06-16", "2026-06-15", 0.35),
    ("2020-01-01", "2026-06-15", 0.40),
])
def test_metal_depreciation_bands(reg, loss, rate):
    a = assess(MOTOR, motor_policy(vehicle_reg_date=reg, start_date="2026-01-01"), [], motor_claim(loss), [], MOTOR_DOCS)
    assert a.payable == 10000 * (1 - rate) - 1000


def test_plastic_50_glass_nil_labour_full():
    est = MotorEstimate(plastic_parts=4000, glass_parts=3000, labour=2000)
    a = assess(MOTOR, motor_policy(), [], motor_claim(estimate=est, amount_claimed=9000), [], MOTOR_DOCS)
    assert a.payable == 2000 + 3000 + 2000 - 1000 and a.decision == "APPROVE"


def test_exclusion_flag_rejects_and_clears_deductions():
    flags = [ScreenFlag(flag="alcohol", evidence="breath analyser positive")]
    a = assess(MOTOR, motor_policy(), [], motor_claim(), flags, MOTOR_DOCS)
    assert a.decision == "REJECT" and a.payable == 0
    assert {f.code for f in a.findings} == {"exclusion_alcohol"}
    assert a.findings[0].clause == "M6(a)"


def test_expired_licence_rejects_but_not_for_theft():
    a = assess(MOTOR, motor_policy(), [], motor_claim(licence_valid_till="2026-01-01"), [], MOTOR_DOCS)
    assert a.decision == "REJECT"
    theft = motor_claim(claim_kind="motor_theft", cause="theft", licence_valid_till="2026-01-01", estimate=None)
    assert assess(MOTOR, motor_policy(), [], theft, [], {"claim_form", "fir"}).decision == "APPROVE"


def test_theft_needs_fir():
    theft = motor_claim(claim_kind="motor_theft", cause="theft", estimate=None)
    a = assess(MOTOR, motor_policy(), [], theft, [], {"claim_form"})
    assert a.decision == "REQUEST_DOCUMENTS" and "fir" in a.findings[0].message


def test_outside_period_rejects():
    a = assess(MOTOR, motor_policy(), [], motor_claim("2027-01-01"), [], MOTOR_DOCS)
    assert a.decision == "REJECT" and a.findings[0].clause == "M2"


def test_late_notice_refers_with_estimate():
    a = assess(MOTOR, motor_policy(), [], motor_claim(intimation_date="2026-06-23"), [], MOTOR_DOCS)
    assert a.decision == "REFER" and a.payable is not None


def test_three_recent_claims_refer_but_old_ones_do_not():
    recent = [{"loss_date": x, "status": "settled", "fraud_flag": 0, "amount_paid": 1}
              for x in ("2025-07-01", "2025-12-01", "2026-03-01")]
    assert assess(MOTOR, motor_policy(), recent, motor_claim(), [], MOTOR_DOCS).decision == "REFER"
    old = [dict(h, loss_date="2024-0%d-01" % i) for i, h in enumerate(recent, 1)]
    assert assess(MOTOR, motor_policy(), old, motor_claim(), [], MOTOR_DOCS).decision == "APPROVE"


def test_reject_beats_refer():
    flags = [ScreenFlag(flag="wear_and_tear", evidence="engine seized")]
    a = assess(MOTOR, motor_policy(), [], motor_claim(intimation_date="2026-07-30"), flags, MOTOR_DOCS)
    assert a.decision == "REJECT"


# ----------------------------------------------------------------------------- health
def test_clean_health_claim():
    a = assess(HEALTH, health_policy(), [], health_claim(), [], HEALTH_DOCS)
    assert a.decision == "APPROVE" and a.payable == 20000


def test_room_rent_proportionate_deduction_spares_medicines():
    bill = HospitalBill(room_rent_per_day=10000, days=2, room=20000, nursing=4000, doctor_fees=8000,
                        investigations=4000, medicines=6000)
    a = assess(HEALTH, health_policy(), [], health_claim(bill=bill), [], HEALTH_DOCS)
    # cap 5,000/day -> proportion 0.5
    assert a.payable == 10000 + 2000 + 4000 + 2000 + 6000 and a.decision == "PARTIAL"


def test_copay_from_age_60():
    a = assess(HEALTH, health_policy(dob="1966-06-10"), [], health_claim(adm="2026-06-10"), [], HEALTH_DOCS)
    assert a.payable == 18000 and a.decision == "APPROVE"    # co-pay is a standard term
    b = assess(HEALTH, health_policy(dob="1966-06-11"), [], health_claim(adm="2026-06-10"), [], HEALTH_DOCS)
    assert b.payable == 20000


def test_sum_insured_counts_only_current_policy_year():
    hist = [{"loss_date": "2026-03-01", "status": "settled", "fraud_flag": 0, "amount_paid": 490000},
            {"loss_date": "2025-12-01", "status": "settled", "fraud_flag": 0, "amount_paid": 500000}]
    a = assess(HEALTH, health_policy(), hist, health_claim(), [], HEALTH_DOCS)
    assert a.payable == 10000 and a.decision == "PARTIAL"


@pytest.mark.parametrize("first,adm,flags,expected", [
    ("2026-05-20", "2026-06-10", [], "REJECT"),                                                     # initial 30 days
    ("2026-05-20", "2026-06-10", [ScreenFlag(flag="accident", evidence="road traffic accident")], "APPROVE"),
    ("2026-05-01", "2026-06-10", [], "APPROVE"),                                                    # day 40
])
def test_initial_waiting_period(first, adm, flags, expected):
    p = health_policy(first_start_date=first, start_date=first)
    assert assess(HEALTH, p, [], health_claim(adm=adm, dis="2026-06-12"), flags, HEALTH_DOCS).decision == expected


def test_specified_disease_waiting():
    p = health_policy(first_start_date="2025-01-01", start_date="2026-01-01")
    a = assess(HEALTH, p, [], health_claim(diagnosis="Inguinal hernia"), [], HEALTH_DOCS)
    assert a.decision == "REJECT" and a.findings[0].clause == "H3(c)"
    p2 = health_policy(first_start_date="2024-01-01")
    assert assess(HEALTH, p2, [], health_claim(diagnosis="Inguinal hernia"), [], HEALTH_DOCS).decision == "APPROVE"


def test_declared_vs_undeclared_pre_existing():
    ped = [ScreenFlag(flag="pre_existing_condition", evidence="known case of diabetes", detail="diabetes")]
    declared = health_policy(first_start_date="2025-01-01", ped_declared="diabetes")
    assert assess(HEALTH, declared, [], health_claim(), ped, HEALTH_DOCS).decision == "REJECT"
    after = health_policy(first_start_date="2022-01-01", ped_declared="diabetes")
    assert assess(HEALTH, after, [], health_claim(), ped, HEALTH_DOCS).decision == "APPROVE"
    undeclared = health_policy(first_start_date="2022-01-01")
    assert assess(HEALTH, undeclared, [], health_claim(), ped, HEALTH_DOCS).decision == "REFER"


def test_under_24_hours_rejected():
    a = assess(HEALTH, health_policy(), [], health_claim(adm="2026-06-10", dis="2026-06-10"), [], HEALTH_DOCS)
    assert a.decision == "REJECT"


def test_unknown_policy_refers():
    assert assess(HEALTH, None, [], health_claim(), [], HEALTH_DOCS).decision == "REFER"
