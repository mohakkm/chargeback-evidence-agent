"""
Bounded Action Gate — Phase 3 (app/agent/action_gate.py).

Decides whether to auto-submit a contest response or flag it for human review,
based on the decision agent's output and evidence coverage.

THRESHOLD SELECTION NOTE:
    AUTO_SUBMIT_THRESHOLD is locked from TRAIN calibration_results.jsonl only
    (app/eval/select_threshold.py). Selected value: 0.70 — first threshold where
    the confidence gate excludes eligible cases (coverage 0.762 vs 100% flat
    below 0.70); precision 93.8% at n=16 auto-submit cases. Never tune on holdout.

Action logic (only two possible actions — no automatic rejection):
    "auto_submit"      — only when ALL three conditions hold:
                             decision == "contest"
                         AND confidence >= CONTEST_AUTO_SUBMIT_THRESHOLD
                         AND low_coverage is False
    "flag_for_review"  — every other case (including no_contest, low confidence,
                         sparse evidence, or fallback-produced decisions)

`used_fallback` from the decision agent is preserved in the output for Phase 4
audit purposes but does not influence gate logic in any way.
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.config import AUTO_SUBMIT_CONFIDENCE_THRESHOLD

# TRAIN-locked auto-submit threshold (override via AUTO_SUBMIT_CONFIDENCE_THRESHOLD in .env).
AUTO_SUBMIT_THRESHOLD: float = AUTO_SUBMIT_CONFIDENCE_THRESHOLD
CONTEST_AUTO_SUBMIT_THRESHOLD: float = AUTO_SUBMIT_THRESHOLD

_VALID_DECISIONS = frozenset({"contest", "no_contest"})
_VALID_ACTIONS = frozenset({"auto_submit", "flag_for_review"})


class ActionGateInput(BaseModel):
    """
    Minimal validated slice of DecisionAgentResponse consumed by the gate.
    Accepts either the full Pydantic response object or a plain dict.
    """
    decision: Literal["contest", "no_contest"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    used_fallback: Optional[bool] = None

    @field_validator("decision", mode="before")
    @classmethod
    def normalise_decision(cls, v: str) -> str:
        v = str(v).lower()
        if v not in _VALID_DECISIONS:
            raise ValueError(f"decision must be one of {_VALID_DECISIONS}, got {v!r}")
        return v


class ActionGateOutput(BaseModel):
    """
    Structured gate output returned to downstream pipeline consumers.
    Carries full context needed for Phase 4 audit logging.
    """
    decision: Literal["contest", "no_contest"] = Field(
        ..., description="Decision from the decision agent"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated confidence score"
    )
    action: Literal["auto_submit", "flag_for_review"] = Field(
        ..., description="Gate routing outcome"
    )
    low_coverage: bool = Field(
        ..., description="True when fewer than 2 evidence docs were retrieved"
    )
    used_fallback: Optional[bool] = Field(
        default=None,
        description="Preserved from decision agent; None when not provided. Does not affect gate logic."
    )

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v: str) -> str:
        v = str(v).lower()
        if v not in _VALID_ACTIONS:
            raise ValueError(f"action must be one of {_VALID_ACTIONS}, got {v!r}")
        return v


def apply_action_gate(
    decision_response: Union[ActionGateInput, object, dict],
    low_coverage: bool,
) -> ActionGateOutput:
    """
    Apply the bounded action gate to a decision-agent response.

    Args:
        decision_response:
            A DecisionAgentResponse (or any object/dict with .decision and
            .confidence attributes/keys). Only `decision`, `confidence`, and
            optionally `used_fallback` are read.
        low_coverage:
            True when the retrieval pipeline returned fewer than 2 evidence
            documents for the dispute transaction.

    Returns:
        ActionGateOutput with action set to "auto_submit" or "flag_for_review".

    Action rules:
        auto_submit      iff  decision == "contest"
                         AND  confidence >= CONTEST_AUTO_SUBMIT_THRESHOLD
                         AND  low_coverage is False
        flag_for_review  in every other case
    """
    # --- normalise input to a consistent interface -------------------------
    if isinstance(decision_response, dict):
        gate_input = ActionGateInput(**decision_response)
    elif isinstance(decision_response, ActionGateInput):
        gate_input = decision_response
    else:
        # Accept DecisionAgentResponse or any duck-typed object
        gate_input = ActionGateInput(
            decision=getattr(decision_response, "decision", "no_contest"),
            confidence=getattr(decision_response, "confidence", 0.0),
            used_fallback=getattr(decision_response, "used_fallback", None),
        )

    # --- gate logic (single expression, no hidden branches) ----------------
    all_conditions_met = (
        gate_input.decision == "contest"
        and gate_input.confidence >= CONTEST_AUTO_SUBMIT_THRESHOLD
        and low_coverage is False
    )
    action: Literal["auto_submit", "flag_for_review"] = (
        "auto_submit" if all_conditions_met else "flag_for_review"
    )

    return ActionGateOutput(
        decision=gate_input.decision,
        confidence=gate_input.confidence,
        action=action,
        low_coverage=low_coverage,
        used_fallback=gate_input.used_fallback,
    )
