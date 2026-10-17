"""Typed data passed between the API, the agents and the rules engine."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["claim_form", "driving_licence", "repair_estimate", "fir",
                  "discharge_summary", "final_bill", "other"]
Decision = Literal["APPROVE", "PARTIAL", "REJECT", "REFER", "REQUEST_DOCUMENTS"]


class ClaimDocument(BaseModel):
    doc_type: DocType
    text: str = Field(min_length=1, max_length=50_000)


class ClaimSubmission(BaseModel):
    claim_ref: Optional[str] = None
    submitted_on: Optional[str] = None
    channel: str = "portal"
    documents: list[ClaimDocument] = Field(min_length=1, max_length=20)
    note: Optional[str] = None


class MotorEstimate(BaseModel):
    metal_parts: float = 0
    plastic_parts: float = 0
    glass_parts: float = 0
    labour: float = 0


class HospitalBill(BaseModel):
    room_rent_per_day: float = 0
    days: int = 0
    room: float = 0
    nursing: float = 0
    doctor_fees: float = 0
    investigations: float = 0
    medicines: float = 0
    consumables: float = 0


class ExtractedClaim(BaseModel):
    """Facts read from the documents by the Document Reader agent."""
    policy_no: Optional[str] = None
    claim_kind: Optional[Literal["motor_own_damage", "motor_theft", "health_hospitalisation"]] = None
    incident_date: Optional[str] = None      # motor: date of loss; health: date of admission
    discharge_date: Optional[str] = None
    intimation_date: Optional[str] = None    # motor: notice; health: document submission
    amount_claimed: Optional[float] = None
    cause: Optional[str] = None              # motor: accident | fire | flood | theft | mechanical_breakdown
    diagnosis: Optional[str] = None
    licence_valid_till: Optional[str] = None
    estimate: Optional[MotorEstimate] = None
    bill: Optional[HospitalBill] = None


class ScreenFlag(BaseModel):
    """A risk or exclusion indicator found by the Exclusion Screener agent, with its evidence."""
    flag: Literal["alcohol", "no_valid_licence", "wear_and_tear", "commercial_use", "cosmetic",
                  "self_inflicted", "pre_existing_condition", "specified_disease", "accident"]
    evidence: str
    detail: Optional[str] = None             # e.g. the condition name for pre_existing_condition


class Finding(BaseModel):
    code: str
    clause: Optional[str]
    effect: Literal["info", "deduction", "reject", "refer", "missing_docs"]
    message: str
    amount: Optional[float] = None


class Assessment(BaseModel):
    decision: Decision
    claimed: Optional[float]
    payable: Optional[float]
    findings: list[Finding]
    breakdown: list[dict] = Field(default_factory=list)
