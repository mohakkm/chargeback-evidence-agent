"""
Audit Trail Logger - Phase 4 (app/audit/logger.py).

Appends one JSONL record per pipeline decision to audit_logs.jsonl at the
project root (or to any caller-supplied path, e.g. a temp file in tests).

PRIVACY CONTRACT
----------------
This module NEVER records:
  - label_winnable        (evaluation-only ground-truth label)
  - ground_truth_rationale (evaluation-only annotation)
  - quality               (per-evidence evaluation grade)
  - evidence content      (full text is omitted - only evidence IDs are stored)

The AuditRecord model is built exclusively from an explicit field allowlist;
any extra keys in caller-supplied dicts are silently ignored at construction time.

NO external services are called here - no Groq, no Qdrant, no FastAPI, no Streamlit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Default storage path - gitignored at project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_LOG_PATH: Path = _PROJECT_ROOT / "audit_logs.jsonl"

# Fields that must NEVER appear in an audit record under any circumstances.
_BANNED_FIELDS = frozenset({"label_winnable", "ground_truth_rationale", "quality"})


class AuditRecord(BaseModel):
    """
    Validated, privacy-safe audit record for one pipeline decision.

    All fields are derived from an explicit allowlist. Evaluation-only fields
    (label_winnable, ground_truth_rationale, quality) are never accepted.
    """

    # Identity
    timestamp_utc: str = Field(
        ...,
        description="Timezone-aware ISO-8601 UTC timestamp of when the record was written.",
    )
    case_id: str = Field(..., description="Unique identifier for the dispute case.")
    transaction_id: str = Field(..., description="Payment transaction identifier.")
    dispute_reason_code: str = Field(..., description="Reason code from the card network.")

    # Retrieval
    retrieved_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Ordered list of evidence_id values returned by the retriever. Evidence content is never stored here.",
    )
    retrieval_count: int = Field(
        ..., ge=0, description="Total number of evidence docs returned after fallback expansion."
    )
    scoped_count: int = Field(
        ..., ge=0, description="Number of docs scoped strictly to this transaction_id."
    )
    low_coverage: bool = Field(
        ..., description="True when fewer than 2 transaction-scoped docs were retrieved."
    )

    # Decision
    decision: str = Field(..., description="'contest' or 'no_contest'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score.")
    reasoning_summary: str = Field(
        ..., description="2-3 sentence internal rationale written by the decision agent."
    )

    # Action gate
    action: str = Field(
        ..., description="Gate routing outcome: 'auto_submit' or 'flag_for_review'."
    )
    used_fallback: Optional[bool] = Field(
        default=None,
        description="True when the heuristic fallback reasoner was used instead of the LLM.",
    )

    # Validators

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def _require_utc_timestamp(cls, v: Any) -> str:
        """Accept a datetime object or an ISO-8601 string; always produce a UTC string."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                raise ValueError("timestamp_utc datetime must be timezone-aware (UTC required).")
            return v.astimezone(timezone.utc).isoformat()
        if isinstance(v, str):
            try:
                parsed = datetime.fromisoformat(v)
            except ValueError as exc:
                raise ValueError(f"timestamp_utc is not a valid ISO-8601 string: {v!r}") from exc
            if parsed.tzinfo is None:
                raise ValueError(
                    f"timestamp_utc must carry timezone info (got naive datetime string: {v!r})."
                )
            return parsed.astimezone(timezone.utc).isoformat()
        raise TypeError(f"timestamp_utc must be a datetime or str, got {type(v).__name__!r}.")

    @field_validator("decision", mode="before")
    @classmethod
    def _normalise_decision(cls, v: Any) -> str:
        v = str(v).lower().strip()
        if v not in ("contest", "no_contest"):
            raise ValueError(f"decision must be 'contest' or 'no_contest', got {v!r}.")
        return v

    @field_validator("action", mode="before")
    @classmethod
    def _normalise_action(cls, v: Any) -> str:
        v = str(v).lower().strip()
        if v not in ("auto_submit", "flag_for_review"):
            raise ValueError(
                f"action must be 'auto_submit' or 'flag_for_review', got {v!r}."
            )
        return v

    @model_validator(mode="after")
    def _no_banned_fields_in_model(self) -> "AuditRecord":
        """
        Confirm no banned field name leaked into the model schema.
        """
        for field_name in _BANNED_FIELDS:
            if field_name in self.model_fields:
                raise ValueError(
                    f"Banned evaluation field {field_name!r} found in AuditRecord schema."
                )
        return self


def _extract_evidence_ids(evidence_list: List[Any]) -> List[str]:
    """
    Extracts `evidence_id` strings from a list of evidence dicts or objects.
    Never stores content or the `quality` field.
    """
    ids: List[str] = []
    for item in evidence_list:
        if isinstance(item, dict):
            eid = item.get("evidence_id")
        else:
            eid = getattr(item, "evidence_id", None)
        if eid is not None:
            ids.append(str(eid))
    return ids


def _safe_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    return bool(value)


def log_decision(
    *,
    case_id: str,
    transaction_id: str,
    dispute_reason_code: str,
    retrieved_evidence: List[Any],
    retrieval_count: int,
    scoped_count: int,
    low_coverage: bool,
    decision: str,
    confidence: float,
    reasoning_summary: str,
    action: str,
    used_fallback: Optional[bool] = None,
    log_path: Optional[Union[str, Path]] = None,
    timestamp_utc: Optional[Union[datetime, str]] = None,
) -> AuditRecord:
    """
    Build a validated AuditRecord and append it as a single JSON line to the log file.
    """
    target_path = Path(log_path) if log_path is not None else DEFAULT_AUDIT_LOG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if timestamp_utc is None:
        timestamp_utc = datetime.now(tz=timezone.utc).isoformat()

    evidence_ids = _extract_evidence_ids(list(retrieved_evidence))

    record = AuditRecord(
        timestamp_utc=timestamp_utc,
        case_id=case_id,
        transaction_id=transaction_id,
        dispute_reason_code=dispute_reason_code,
        retrieved_evidence_ids=evidence_ids,
        retrieval_count=int(retrieval_count),
        scoped_count=int(scoped_count),
        low_coverage=bool(low_coverage),
        decision=decision,
        confidence=float(confidence),
        reasoning_summary=reasoning_summary,
        action=action,
        used_fallback=_safe_bool(used_fallback),
    )

    json_line = record.model_dump_json() + "\n"
    with target_path.open("a", encoding="utf-8") as fh:
        fh.write(json_line)

    return record


def log_decision_from_dicts(
    *,
    dispute: Dict[str, Any],
    retrieval_output: Dict[str, Any],
    decision_response: Union[Dict[str, Any], Any],
    gate_output: Union[Dict[str, Any], Any],
    case_id: Optional[str] = None,
    log_path: Optional[Union[str, Path]] = None,
) -> AuditRecord:
    """
    Convenience wrapper that accepts raw pipeline dicts / Pydantic objects and
    constructs the AuditRecord from allowlisted fields only.
    """

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    resolved_case_id = case_id or str(_get(dispute, "case_id", ""))
    transaction_id = str(_get(dispute, "transaction_id", ""))
    dispute_reason_code = str(_get(dispute, "dispute_reason_code", ""))

    retrieved_evidence = list(_get(retrieval_output, "retrieved_evidence", []))
    retrieval_count = int(_get(retrieval_output, "retrieval_count", len(retrieved_evidence)))
    scoped_count = int(_get(retrieval_output, "scoped_count", 0))
    low_coverage = bool(_get(retrieval_output, "low_coverage", False))

    decision = str(_get(decision_response, "decision", "no_contest"))
    confidence = float(_get(decision_response, "confidence", 0.0))
    reasoning_summary = str(_get(decision_response, "reasoning_summary", ""))
    used_fallback = _safe_bool(_get(decision_response, "used_fallback", None))

    action = str(_get(gate_output, "action", "flag_for_review"))

    return log_decision(
        case_id=resolved_case_id,
        transaction_id=transaction_id,
        dispute_reason_code=dispute_reason_code,
        retrieved_evidence=retrieved_evidence,
        retrieval_count=retrieval_count,
        scoped_count=scoped_count,
        low_coverage=low_coverage,
        decision=decision,
        confidence=confidence,
        reasoning_summary=reasoning_summary,
        action=action,
        used_fallback=used_fallback,
        log_path=log_path,
    )
