"""
Deterministic verification script for app/agent/action_gate.py.

No Groq calls, no Qdrant calls, no dataset access.
All inputs are constructed inline using fixed values.

Tests:
  1. contest + confidence exactly at threshold + low_coverage=False   => auto_submit
  2. contest + confidence just below threshold + low_coverage=False   => flag_for_review
  3. contest + high confidence   + low_coverage=True                  => flag_for_review
  4. no_contest + high confidence + low_coverage=False                => flag_for_review
  5. used_fallback preserved but does not change a valid gate result
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.action_gate import (
    apply_action_gate,
    ActionGateInput,
    CONTEST_AUTO_SUBMIT_THRESHOLD,
)

PASS = "[PASS]"
FAIL = "[FAIL]"


def _make(decision: str, confidence: float, used_fallback=None) -> ActionGateInput:
    return ActionGateInput(
        decision=decision,
        confidence=confidence,
        used_fallback=used_fallback,
    )


def run_tests():
    print("=" * 80)
    print("      RAZORPAY CHARGEBACK RESPONDER — ACTION GATE VERIFICATION")
    print("=" * 80)
    print(f"\nConfigured threshold : CONTEST_AUTO_SUBMIT_THRESHOLD = {CONTEST_AUTO_SUBMIT_THRESHOLD}\n")

    all_passed = True
    results = []

    # ------------------------------------------------------------------
    # TEST 1: contest, exactly at threshold, low_coverage=False -> auto_submit
    # ------------------------------------------------------------------
    inp1 = _make("contest", CONTEST_AUTO_SUBMIT_THRESHOLD)
    out1 = apply_action_gate(inp1, low_coverage=False)
    ok1 = out1.action == "auto_submit"
    results.append((
        "TEST 1 — contest, confidence==threshold, low_coverage=False",
        ok1, out1, "auto_submit",
    ))

    # ------------------------------------------------------------------
    # TEST 2: contest, just below threshold, low_coverage=False -> flag_for_review
    # ------------------------------------------------------------------
    inp2 = _make("contest", round(CONTEST_AUTO_SUBMIT_THRESHOLD - 0.001, 6))
    out2 = apply_action_gate(inp2, low_coverage=False)
    ok2 = out2.action == "flag_for_review"
    results.append((
        "TEST 2 — contest, confidence just below threshold, low_coverage=False",
        ok2, out2, "flag_for_review",
    ))

    # ------------------------------------------------------------------
    # TEST 3: contest, high confidence, low_coverage=True -> flag_for_review
    # ------------------------------------------------------------------
    inp3 = _make("contest", 0.99)
    out3 = apply_action_gate(inp3, low_coverage=True)
    ok3 = out3.action == "flag_for_review"
    results.append((
        "TEST 3 — contest, high confidence, low_coverage=True",
        ok3, out3, "flag_for_review",
    ))

    # ------------------------------------------------------------------
    # TEST 4: no_contest, high confidence, low_coverage=False -> flag_for_review
    # ------------------------------------------------------------------
    inp4 = _make("no_contest", 0.95)
    out4 = apply_action_gate(inp4, low_coverage=False)
    ok4 = out4.action == "flag_for_review"
    results.append((
        "TEST 4 — no_contest, high confidence, low_coverage=False",
        ok4, out4, "flag_for_review",
    ))

    # ------------------------------------------------------------------
    # TEST 5: used_fallback preserved but does not change a valid gate result
    #         (contest, above threshold, low_coverage=False, used_fallback=True)
    #         Gate must still return auto_submit, used_fallback must survive unchanged.
    # ------------------------------------------------------------------
    inp5 = _make("contest", CONTEST_AUTO_SUBMIT_THRESHOLD, used_fallback=True)
    out5 = apply_action_gate(inp5, low_coverage=False)
    ok5 = out5.action == "auto_submit" and out5.used_fallback is True
    results.append((
        "TEST 5 — contest at threshold, used_fallback=True (must not block auto_submit)",
        ok5, out5, "auto_submit",
    ))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    for label, passed, out, expected_action in results:
        status = PASS if passed else FAIL
        if not passed:
            all_passed = False
        print(f"{status}  {label}")
        print(f"        decision={out.decision!r}  confidence={out.confidence:.4f}"
              f"  low_coverage={out.low_coverage}  action={out.action!r}"
              f"  used_fallback={out.used_fallback}")
        if not passed:
            print(f"        EXPECTED action={expected_action!r}  GOT={out.action!r}")
        print()

    print("=" * 80)
    overall = "ALL TESTS PASSED" if all_passed else "ONE OR MORE TESTS FAILED"
    print(f"      {overall}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
