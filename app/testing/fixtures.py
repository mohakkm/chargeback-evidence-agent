"""
Canned JSON fixtures for offline pipeline, dashboard, and API skeleton tests.

No Groq, Qdrant, or network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent.action_gate import apply_action_gate
from app.agent.decision_agent import (
    DecisionAgentResponse,
    _apply_confidence_calibration,
)
from app.audit.logger import log_decision_from_dicts

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR: Path = _PROJECT_ROOT / "tests" / "fixtures"

SCENARIO_NAMES: tuple[str, ...] = (
    "high_confidence_contest",
    "high_confidence_no_contest",
    "sparse_evidence_capped_confidence",
    "malformed_json_retry",
)


def load_scenario(name: str) -> Dict[str, Any]:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture scenario not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_scenarios() -> List[str]:
    return list(SCENARIO_NAMES)


class FixtureDecisionAgent:
    """
    Returns canned LLM JSON from fixture files. Never contacts Groq.
    """

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.scenario = load_scenario(scenario_name)

    def evaluate_dispute_dict(
        self,
        dispute_dict: Dict[str, Any],
        retrieval_output: Dict[str, Any],
    ) -> DecisionAgentResponse:
        low_coverage = bool(retrieval_output.get("low_coverage", False))
        data = dict(self.scenario["llm_response"])
        confidence = _apply_confidence_calibration(float(data["confidence"]), low_coverage)
        return DecisionAgentResponse(
            decision=data["decision"],
            confidence=confidence,
            rebuttal_draft=str(data.get("rebuttal_draft", "")),
            reasoning_summary=str(data.get("reasoning_summary", "")),
            used_fallback=False,
        )


class FixtureRetriever:
    """Returns retrieval payload from a named fixture scenario."""

    def __init__(self, scenario_name: str):
        self.scenario = load_scenario(scenario_name)

    def retrieve_evidence_for_dispute(self, dispute: Dict[str, Any]) -> Dict[str, Any]:
        retrieval = dict(self.scenario["retrieval"])
        evidence = retrieval.get("retrieved_evidence", [])
        retrieval["retrieved_evidence_ids"] = [
            e.get("evidence_id", "") for e in evidence if e.get("evidence_id")
        ]
        return retrieval


def run_fixture_pipeline(
    scenario_name: str,
    audit_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute retrieval -> fixture agent -> action gate -> audit record for one scenario.
    """
    scenario = load_scenario(scenario_name)
    dispute = dict(scenario["dispute"])
    retrieval = FixtureRetriever(scenario_name).retrieve_evidence_for_dispute(dispute)
    agent = FixtureDecisionAgent(scenario_name)
    decision_response = agent.evaluate_dispute_dict(dispute, retrieval)
    gate_output = apply_action_gate(
        decision_response,
        low_coverage=bool(retrieval.get("low_coverage", False)),
    )

    audit_record = log_decision_from_dicts(
        dispute=dispute,
        retrieval_output=retrieval,
        decision_response=decision_response,
        gate_output=gate_output,
        log_path=audit_log_path,
    )

    return {
        "scenario": scenario_name,
        "decision": gate_output.decision,
        "confidence": gate_output.confidence,
        "action": gate_output.action,
        "low_coverage": gate_output.low_coverage,
        "used_fallback": decision_response.used_fallback,
        "reasoning_summary": decision_response.reasoning_summary,
        "rebuttal_draft": decision_response.rebuttal_draft,
        "retrieved_evidence": retrieval.get("retrieved_evidence", []),
        "audit_entry": audit_record.model_dump(),
    }


def evaluate_dispute_from_fixture(
    case: Dict[str, Any],
    scenario_name: Optional[str] = None,
    audit_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    API entrypoint: resolve scenario from explicit name or case_id match, then run pipeline.
    """
    resolved = scenario_name or str(case.get("scenario", "")).strip()
    if not resolved:
        case_id = str(case.get("case_id", ""))
        for name in SCENARIO_NAMES:
            fixture = load_scenario(name)
            if fixture["dispute"].get("case_id") == case_id:
                resolved = name
                break
    if not resolved or resolved not in SCENARIO_NAMES:
        raise ValueError(
            f"Unknown fixture scenario. Provide scenario= one of {SCENARIO_NAMES}."
        )

    result = run_fixture_pipeline(resolved, audit_log_path=audit_log_path)
    result["case_id"] = case.get("case_id", result["audit_entry"].get("case_id"))
    return result
