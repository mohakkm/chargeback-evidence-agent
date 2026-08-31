"""
Deterministic verification test suite for app/agent/decision_agent.py (Phase 3 & 4).

No real Groq API calls, no Qdrant calls, no real sleep calls.
Mocks the Groq client and time functions in-memory.

Tests:
  1. Input field isolation check (sanitize_dispute_input & sanitize_evidence_input).
  2. Sparse evidence confidence calibration check.
  3. Groq HTTP 429 rate limit retry + parsed delay -> success returns live LLM response (used_fallback=False).
  4. Parsed wait time honors "try again in X seconds" (X + 1.0s).
  5. Exhausted 429 retries fall back to heuristic reasoner with used_fallback=True.
  6. Non-429 error does not retry and immediately falls back to heuristic reasoner with used_fallback=True.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.decision_agent import (
    DecisionAgent,
    sanitize_dispute_input,
    sanitize_evidence_input,
    CleanDisputeInput,
    CleanEvidenceInput,
    DecisionAgentResponse,
)

PASS = "[PASS]"
FAIL = "[FAIL]"


class MockGroqResponseChoice:
    def __init__(self, content: str):
        self.message = MagicMock(content=content)


class MockGroqResponse:
    def __init__(self, content_dict: dict):
        self.choices = [MockGroqResponseChoice(json.dumps(content_dict))]


def run_tests():
    print("=" * 80)
    print("      RAZORPAY CHARGEBACK RESPONDER - DECISION AGENT VERIFICATION")
    print("=" * 80)
    print()

    all_passed = True
    results = []

    # -------------------------------------------------------------------------
    # TEST 1: Strict Input Field Isolation
    # -------------------------------------------------------------------------
    raw_polluted_case = {
        "case_id": "CB-99999",
        "dispute_reason_code": "goods_not_received",
        "customer_claim_text": "Merchandise was never delivered.",
        "dispute_amount": 149.99,
        "merchant_category": "ecommerce",
        "label_winnable": True,
        "ground_truth_rationale": "Merchant has signed proof of delivery.",
    }
    raw_polluted_evidence = [
        {
            "evidence_id": "EVD-99999-01",
            "doc_type": "delivery_confirmation",
            "content": "Carrier FedEx Tracking #12345. Status: DELIVERED.",
            "quality": "strong",
        }
    ]

    clean_dispute = sanitize_dispute_input(raw_polluted_case)
    clean_evidence = sanitize_evidence_input(raw_polluted_evidence)

    dispute_fields = set(clean_dispute.model_dump().keys())
    evidence_fields = set(clean_evidence[0].model_dump().keys())

    ok1 = (
        "label_winnable" not in dispute_fields
        and "ground_truth_rationale" not in dispute_fields
        and "quality" not in evidence_fields
    )
    results.append((
        "TEST 1 - Strict input field isolation (ground-truth fields stripped)",
        ok1,
        f"dispute_fields={sorted(dispute_fields)}, evidence_fields={sorted(evidence_fields)}"
    ))

    # -------------------------------------------------------------------------
    # TEST 2: Low coverage confidence calibration
    # -------------------------------------------------------------------------
    agent_heuristic = DecisionAgent(api_key="dummy")
    dispute_sparse = CleanDisputeInput(
        dispute_reason_code="goods_not_received",
        customer_claim_text="Item not delivered",
        dispute_amount=89.50,
    )
    resp_sparse = agent_heuristic.evaluate_dispute(dispute=dispute_sparse, evidence=[], low_coverage=True)

    ok2 = resp_sparse.confidence <= 0.45 and isinstance(resp_sparse.used_fallback, bool)
    results.append((
        "TEST 2 - Sparse low_coverage=True caps confidence at <= 0.45",
        ok2,
        f"decision={resp_sparse.decision}, confidence={resp_sparse.confidence}, fallback={resp_sparse.used_fallback}"
    ))

    # -------------------------------------------------------------------------
    # TEST 3 & 4: Groq 429 retry + parsed delay -> success (used_fallback=False)
    # -------------------------------------------------------------------------
    mock_client = MagicMock()
    error_429 = Exception("Error 429: Rate limit reached. Please try again in 3.4s.")

    valid_llm_json = {
        "decision": "contest",
        "confidence": 0.92,
        "rebuttal_draft": "Formal LLM rebuttal draft.",
        "reasoning_summary": "Strong evidence supports contesting.",
    }
    success_response = MockGroqResponse(valid_llm_json)

    # First call 429, second call succeeds
    mock_client.chat.completions.create.side_effect = [error_429, success_response]

    agent_mocked = DecisionAgent(api_key="fake_key")
    agent_mocked._client = mock_client

    sleep_calls = []
    with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)), \
         patch("time.monotonic", return_value=100.0):

        resp3 = agent_mocked.evaluate_dispute(dispute=dispute_sparse, evidence=[], low_coverage=False)

    ok3 = (
        resp3.used_fallback is False and
        resp3.decision == "contest" and
        mock_client.chat.completions.create.call_count == 2
    )
    # Parsed delay 3.4s -> wait time should be 3.4 + 1.0 = 4.4s
    ok4 = len(sleep_calls) >= 1 and any(abs(s - 4.4) < 0.01 for s in sleep_calls)

    results.append((
        "TEST 3 - 429 rate limit retry followed by success returns live LLM response (used_fallback=False)",
        ok3,
        f"used_fallback={resp3.used_fallback}, decision={resp3.decision}, api_calls={mock_client.chat.completions.create.call_count}"
    ))
    results.append((
        "TEST 4 - Parsed retry delay honors 'try again in 3.4s' (waits 4.4s)",
        ok4,
        f"sleep_calls={sleep_calls}"
    ))

    # -------------------------------------------------------------------------
    # TEST 5: Exhausted 429 retries fall back to heuristic reasoner (used_fallback=True)
    # -------------------------------------------------------------------------
    mock_client_fail = MagicMock()
    mock_client_fail.chat.completions.create.side_effect = Exception("429 Rate limit error")

    agent_fail = DecisionAgent(api_key="fake_key")
    agent_fail._client = mock_client_fail

    sleep_calls_fail = []
    with patch("time.sleep", side_effect=lambda s: sleep_calls_fail.append(s)), \
         patch("time.monotonic", return_value=100.0), \
         patch.dict("os.environ", {"GROQ_MAX_RATE_LIMIT_RETRIES": "2"}):

        resp5 = agent_fail.evaluate_dispute(dispute=dispute_sparse, evidence=[], low_coverage=False)

    ok5 = (
        resp5.used_fallback is True and
        mock_client_fail.chat.completions.create.call_count == 3  # 1 initial + 2 retries
    )
    results.append((
        "TEST 5 - Exhausted 429 retries fall back to heuristic reasoner (used_fallback=True)",
        ok5,
        f"used_fallback={resp5.used_fallback}, calls={mock_client_fail.chat.completions.create.call_count}, sleep_count={len(sleep_calls_fail)}"
    ))

    # -------------------------------------------------------------------------
    # TEST 6: Non-429 error does NOT retry and immediately falls back
    # -------------------------------------------------------------------------
    mock_client_401 = MagicMock()
    mock_client_401.chat.completions.create.side_effect = Exception("401 Invalid API Key")

    agent_401 = DecisionAgent(api_key="fake_key")
    agent_401._client = mock_client_401

    sleep_calls_401 = []
    with patch("time.sleep", side_effect=lambda s: sleep_calls_401.append(s)), \
         patch("time.monotonic", return_value=100.0):

        resp6 = agent_401.evaluate_dispute(dispute=dispute_sparse, evidence=[], low_coverage=False)

    ok6 = (
        resp6.used_fallback is True and
        mock_client_401.chat.completions.create.call_count == 1 and
        len(sleep_calls_401) == 0
    )
    results.append((
        "TEST 6 - Non-429 error (401 Bad Key) does NOT retry and immediately uses fallback",
        ok6,
        f"used_fallback={resp6.used_fallback}, calls={mock_client_401.chat.completions.create.call_count}, sleep_count={len(sleep_calls_401)}"
    ))

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------
    for label, passed, details in results:
        status = PASS if passed else FAIL
        if not passed:
            all_passed = False
        print(f"{status}  {label}")
        print(f"        Details: {details}")
        print()

    print("=" * 80)
    overall = "ALL TESTS PASSED" if all_passed else "ONE OR MORE TESTS FAILED"
    print(f"      {overall}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
