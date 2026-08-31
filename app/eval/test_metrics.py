"""
Deterministic test suite for app/eval/metrics.py (Phase 4).

No Groq calls, no Qdrant calls, no retriever calls, no dataset/config modifications.
All inputs are constructed in-memory using fixed test fixtures.

Tests:
  1. Fixed small fixture covering TP, FP, TN, FN; assert exact classification metrics.
  2. Fixture covering low coverage, review routing, and auto-submit precision.
  3. False-positive-cost fixture with known amounts and named parameters.
  4. Zero-denominator fixture confirming no crash and default 0.0 outputs.
  5. Confirm metric outputs do not contain raw ground_truth_rationale or evidence quality/content.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.metrics import compute_evaluation_metrics, EvaluationMetricsSummary

PASS = "[PASS]"
FAIL = "[FAIL]"


def run_tests():
    print("=" * 80)
    print("      RAZORPAY CHARGEBACK RESPONDER - EVALUATION METRICS VERIFICATION")
    print("=" * 80)
    print()

    all_passed = True
    results = []

    # ------------------------------------------------------------------
    # TEST 1: TP, FP, TN, FN exact classification metrics
    # ------------------------------------------------------------------
    fixture1 = [
        {"decision": "contest", "label_winnable": True, "dispute_amount": 100.0},     # TP
        {"decision": "contest", "label_winnable": True, "dispute_amount": 150.0},     # TP
        {"decision": "contest", "label_winnable": False, "dispute_amount": 200.0},    # FP
        {"decision": "no_contest", "label_winnable": False, "dispute_amount": 50.0},   # TN
        {"decision": "no_contest", "label_winnable": True, "dispute_amount": 80.0},    # FN
    ]
    m1 = compute_evaluation_metrics(fixture1)
    ok1 = (
        m1.total_cases == 5 and
        m1.tp == 2 and
        m1.fp == 1 and
        m1.tn == 1 and
        m1.fn == 1 and
        m1.precision == round(2 / 3, 6) and
        m1.recall == round(2 / 3, 6) and
        m1.f1 == round(2 / 3, 6) and
        m1.accuracy == round(3 / 5, 6)
    )
    results.append((
        "TEST 1 - Exact classification metrics (TP=2, FP=1, TN=1, FN=1)",
        ok1,
        f"TP={m1.tp}, FP={m1.fp}, TN={m1.tn}, FN={m1.fn}, P={m1.precision}, R={m1.recall}, F1={m1.f1}, Acc={m1.accuracy}"
    ))

    # ------------------------------------------------------------------
    # TEST 2: Low coverage, review routing, and auto-submit precision
    # ------------------------------------------------------------------
    fixture2 = [
        {"decision": "contest", "label_winnable": True, "action": "auto_submit", "low_coverage": False},       # Auto TP
        {"decision": "contest", "label_winnable": True, "action": "auto_submit", "low_coverage": False},       # Auto TP
        {"decision": "contest", "label_winnable": False, "action": "auto_submit", "low_coverage": False},      # Auto FP
        {"decision": "contest", "label_winnable": False, "action": "flag_for_review", "low_coverage": True},   # Review
        {"decision": "no_contest", "label_winnable": False, "action": "flag_for_review", "low_coverage": True}, # Review
    ]
    m2 = compute_evaluation_metrics(fixture2)
    ok2 = (
        m2.total_cases == 5 and
        m2.evidence_coverage_rate == 0.6 and        # 3 sufficient / 5
        m2.auto_submit_count == 3 and
        m2.auto_submit_rate == 0.6 and
        m2.auto_submit_precision == round(2 / 3, 6) and # 2 Auto-TP / 3 Auto-total
        m2.review_count == 2 and
        m2.review_rate == 0.4
    )
    results.append((
        "TEST 2 - Coverage rate, review routing, and auto-submit precision",
        ok2,
        f"coverage_rate={m2.evidence_coverage_rate}, auto_count={m2.auto_submit_count}, auto_prec={m2.auto_submit_precision}, review_count={m2.review_count}"
    ))

    # ------------------------------------------------------------------
    # TEST 3: False-positive cost fixture with known amounts and named parameters
    # ------------------------------------------------------------------
    fixture3 = [
        {"decision": "contest", "label_winnable": False, "dispute_amount": 100.0},  # FP 1
        {"decision": "contest", "label_winnable": False, "dispute_amount": 250.0},  # FP 2
        {"decision": "contest", "label_winnable": True, "dispute_amount": 500.0},   # TP
        {"decision": "no_contest", "label_winnable": False, "dispute_amount": 300.0},# TN
    ]
    # FP exposure = 100 + 250 = 350.0. FP count = 2.
    # cost = 350.0 * 1.5 + 2 * 15.0 = 525.0 + 30.0 = 555.0
    m3 = compute_evaluation_metrics(fixture3, cost_multiplier=1.5, fixed_fee=15.0)
    ok3 = (
        m3.false_positive_count == 2 and
        m3.false_positive_exposure == 350.0 and
        m3.estimated_false_positive_cost == 555.0 and
        m3.cost_multiplier_used == 1.5 and
        m3.fixed_fee_used == 15.0
    )
    results.append((
        "TEST 3 - False positive cost calculation (exposure=$350, cost=$555 with mult=1.5, fee=15)",
        ok3,
        f"FP_count={m3.false_positive_count}, FP_exposure=${m3.false_positive_exposure}, FP_cost=${m3.estimated_false_positive_cost}"
    ))

    # ------------------------------------------------------------------
    # TEST 4: Zero-denominator fixture (empty list)
    # ------------------------------------------------------------------
    m4 = compute_evaluation_metrics([])
    ok4 = (
        m4.total_cases == 0 and
        m4.tp == 0 and m4.fp == 0 and m4.tn == 0 and m4.fn == 0 and
        m4.precision == 0.0 and
        m4.recall == 0.0 and
        m4.f1 == 0.0 and
        m4.accuracy == 0.0 and
        m4.evidence_coverage_rate == 0.0 and
        m4.review_count == 0 and
        m4.review_rate == 0.0 and
        m4.auto_submit_count == 0 and
        m4.auto_submit_rate == 0.0 and
        m4.auto_submit_precision == 0.0 and
        m4.false_positive_count == 0 and
        m4.false_positive_exposure == 0.0 and
        m4.estimated_false_positive_cost == 0.0 and
        m4.fallback_count == 0 and
        m4.fallback_rate == 0.0
    )
    results.append((
        "TEST 4 - Zero-denominator empty fixture safe handling (all metrics default to 0/0.0)",
        ok4,
        f"total={m4.total_cases}, P={m4.precision}, R={m4.recall}, F1={m4.f1}, auto_prec={m4.auto_submit_precision}"
    ))

    # ------------------------------------------------------------------
    # TEST 5: Confirm metric outputs do not leak ground_truth_rationale or quality/content
    # ------------------------------------------------------------------
    polluted_cases = [
        {
            "decision": "contest",
            "label_winnable": True,
            "dispute_amount": 120.0,
            "ground_truth_rationale": "SENSITIVE_GROUND_TRUTH_TEXT_LEAK",
            "quality": "EVAL_QUALITY_HIGH",
            "content": "SECRET_DOC_CONTENT",
            "used_fallback": True,
        }
    ]
    m5 = compute_evaluation_metrics(polluted_cases)
    m5_json = m5.model_dump_json()

    banned_strings = ["SENSITIVE_GROUND_TRUTH_TEXT_LEAK", "EVAL_QUALITY_HIGH", "SECRET_DOC_CONTENT", "ground_truth_rationale", "quality", "content"]
    leaks = [s for s in banned_strings if s in m5_json]

    ok5 = len(leaks) == 0 and m5.fallback_count == 1 and m5.fallback_rate == 1.0
    results.append((
        "TEST 5 - Privacy isolation: Output summary is clean of ground_truth_rationale, quality, content",
        ok5,
        f"leaks_found={leaks}, fallback_count={m5.fallback_count}"
    ))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
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
