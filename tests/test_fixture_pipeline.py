"""
Fixture-based offline tests for decision agent, action gate, and eval pipeline.

Default mode: canned JSON only — zero Groq/Qdrant/network calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.action_gate import apply_action_gate
from app.agent.decision_agent import DecisionAgent, DecisionAgentResponse
from app.eval.run_eval import make_safe_dispute_payload, run_evaluation
from app.testing.fixtures import (
    SCENARIO_NAMES,
    FixtureDecisionAgent,
    FixtureRetriever,
    load_scenario,
    run_fixture_pipeline,
)

PASS = "[PASS]"
FAIL = "[FAIL]"


class MockGroqResponseChoice:
    def __init__(self, content: str):
        self.message = MagicMock(content=content)


class MockGroqResponse:
    def __init__(self, content_dict: dict):
        self.choices = [MockGroqResponseChoice(json.dumps(content_dict))]


def _run_malformed_json_retry_test() -> tuple[bool, str]:
    scenario = load_scenario("malformed_json_retry")
    effects = scenario["groq_side_effects"]
    side_effects = []
    for item in effects:
        if item["type"] == "error":
            side_effects.append(Exception(item["message"]))
        else:
            side_effects.append(MockGroqResponse(item["body"]))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effects

    agent = DecisionAgent(api_key="fake_key")
    agent._client = mock_client

    dispute = scenario["dispute"]
    retrieval = scenario["retrieval"]

    with patch("time.sleep"), patch("time.monotonic", return_value=100.0):
        response = agent.evaluate_dispute_dict(dispute, retrieval)

    expected = scenario["expected"]
    gate = apply_action_gate(response, low_coverage=retrieval["low_coverage"])
    ok = (
        response.used_fallback is False
        and response.decision == expected["decision"]
        and abs(response.confidence - expected["confidence"]) < 0.001
        and gate.action == expected["action"]
        and mock_client.chat.completions.create.call_count == expected["groq_call_count"]
    )
    return ok, (
        f"calls={mock_client.chat.completions.create.call_count}, "
        f"decision={response.decision}, confidence={response.confidence}, action={gate.action}"
    )


def run_fixture_tests() -> bool:
    print("=" * 80)
    print("      FIXTURE PIPELINE TESTS (offline — no live Groq)")
    print("=" * 80)
    print()

    all_passed = True
    results = []

    for name in SCENARIO_NAMES:
        if name == "malformed_json_retry":
            ok, details = _run_malformed_json_retry_test()
            results.append((f"FIXTURE — {name} (mocked Groq retry, no live API)", ok, details))
            continue

        scenario = load_scenario(name)
        expected = scenario["expected"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = run_fixture_pipeline(name, audit_log_path=Path(tmpdir) / "audit.jsonl")

        ok = (
            out["decision"] == expected["decision"]
            and abs(out["confidence"] - expected["confidence"]) < 0.001
            and out["action"] == expected["action"]
            and out["used_fallback"] == expected["used_fallback"]
            and out["low_coverage"] == expected["low_coverage"]
        )
        results.append((
            f"FIXTURE — {name}",
            ok,
            f"decision={out['decision']}, confidence={out['confidence']}, action={out['action']}",
        ))

    # Action gate boundary on fixture contest output
    contest = load_scenario("high_confidence_contest")
    gate_in = DecisionAgentResponse(
        decision=contest["llm_response"]["decision"],
        confidence=contest["llm_response"]["confidence"],
        rebuttal_draft=contest["llm_response"]["rebuttal_draft"],
        reasoning_summary=contest["llm_response"]["reasoning_summary"],
        used_fallback=False,
    )
    gate_out = apply_action_gate(gate_in, low_coverage=False)
    results.append((
        "ACTION GATE — high_confidence_contest routes to auto_submit",
        gate_out.action == "auto_submit",
        f"action={gate_out.action}",
    ))

    # Eval runner isolation: banned fields stripped before agent receives dispute
    polluted = {**contest["dispute"], "label_winnable": True, "quality": "strong"}
    safe = make_safe_dispute_payload(polluted)
    ok_iso = "label_winnable" not in safe and "quality" not in safe
    results.append((
        "EVAL RUNNER — make_safe_dispute_payload strips eval-only fields",
        ok_iso,
        f"safe_keys={sorted(safe.keys())}",
    ))

    # Eval runner end-to-end with fixture doubles
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        train_file = tmp / "train.jsonl"
        train_case = {
            **contest["dispute"],
            "label_winnable": True,
            "dispute_raised_date": "2026-08-01",
            "response_deadline": "2026-08-15",
        }
        train_file.write_text(json.dumps(train_case) + "\n", encoding="utf-8")
        summary = run_evaluation(
            split="train",
            datasets_dir=tmp,
            audit_log_path=tmp / "audit.jsonl",
            retriever=FixtureRetriever("high_confidence_contest"),
            decision_agent=FixtureDecisionAgent("high_confidence_contest"),
            require_live_llm=True,
            allow_fallback=False,
        )
    ok_eval = summary.total_evaluated == 1 and summary.fallback_count == 0
    results.append((
        "EVAL RUNNER — fixture retriever + agent completes one-case train eval",
        ok_eval,
        f"total={summary.total_evaluated}, fallback={summary.fallback_count}",
    ))

    for label, passed, details in results:
        status = PASS if passed else FAIL
        if not passed:
            all_passed = False
        print(f"{status}  {label}")
        print(f"        Details: {details}")
        print()

    print("=" * 80)
    print("      ALL FIXTURE TESTS PASSED" if all_passed else "      ONE OR MORE FIXTURE TESTS FAILED")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_fixture_tests() else 1)
