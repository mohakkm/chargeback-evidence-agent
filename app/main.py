"""
FastAPI entrypoint for the Chargeback Evidence Responder.

Offline skeleton: POST /evaluate-dispute uses canned fixture pipeline (no Groq).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.testing.fixtures import SCENARIO_NAMES, evaluate_dispute_from_fixture

app = FastAPI(
    title="Chargeback Evidence Responder",
    description="Defense-only dispute evaluation API (fixture-backed offline skeleton).",
)


class DisputeCaseInput(BaseModel):
    case_id: str = Field(..., description="Dispute case identifier")
    transaction_id: Optional[str] = None
    merchant_category: Optional[str] = "ecommerce"
    dispute_reason_code: str = Field(..., description="Network reason code")
    dispute_amount: float = Field(..., ge=0.0)
    customer_claim_text: str = Field(..., description="Cardholder claim narrative")
    scenario: Optional[str] = Field(
        None,
        description=f"Fixture scenario key. One of: {', '.join(SCENARIO_NAMES)}",
    )


class AuditEntry(BaseModel):
    timestamp_utc: str
    case_id: str
    transaction_id: str
    dispute_reason_code: str
    retrieved_evidence_ids: List[str]
    retrieval_count: int
    scoped_count: int
    low_coverage: bool
    decision: Literal["contest", "no_contest"]
    confidence: float
    reasoning_summary: str
    action: Literal["auto_submit", "flag_for_review"]
    used_fallback: Optional[bool] = None


class EvaluateDisputeResponse(BaseModel):
    case_id: str
    scenario: str
    decision: Literal["contest", "no_contest"]
    confidence: float
    action: Literal["auto_submit", "flag_for_review"]
    low_coverage: bool
    used_fallback: bool
    reasoning_summary: str
    rebuttal_draft: str
    retrieved_evidence: List[Dict[str, Any]]
    audit_entry: AuditEntry
    mode: str = Field("fixture", description="Pipeline mode — always 'fixture' in this skeleton")


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"status": "ok", "mode": "fixture", "evaluate": "POST /evaluate-dispute"}


@app.post("/evaluate-dispute", response_model=EvaluateDisputeResponse)
def evaluate_dispute(case: DisputeCaseInput) -> EvaluateDisputeResponse:
    """
    Run the dispute pipeline using canned fixtures — no live Groq calls.
    """
    try:
        result = evaluate_dispute_from_fixture(
            case=case.model_dump(),
            scenario_name=case.scenario,
            audit_log_path=Path("api_fixture_audit.jsonl"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EvaluateDisputeResponse(
        case_id=result["case_id"],
        scenario=result["scenario"],
        decision=result["decision"],
        confidence=result["confidence"],
        action=result["action"],
        low_coverage=result["low_coverage"],
        used_fallback=bool(result["used_fallback"]),
        reasoning_summary=result["reasoning_summary"],
        rebuttal_draft=result["rebuttal_draft"],
        retrieved_evidence=result["retrieved_evidence"],
        audit_entry=AuditEntry(**result["audit_entry"]),
    )
