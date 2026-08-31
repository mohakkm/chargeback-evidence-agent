"""
Deterministic verification test suite for app/eval/run_eval.py (Phase 4).

No Groq calls, no Qdrant calls, no external model calls.
Uses fake retriever and fake decision agent objects in-memory with temporary files.

Tests:
  1. Verify make_safe_dispute_payload strips all 4 forbidden evaluation fields.
  2. Verify end-to-end pipeline execution with fakes (retrieval -> agent -> gate -> logger -> metrics).
  3. Verify require_live_llm=True rejects fallback response with clear RuntimeError.
  4. Verify allow_fallback=True permits fallback reasoner execution during local testing.
  5. Verify calibrate_auto_submit_threshold and run_train_threshold_calibration reject 'holdout' split and process 'train' split.
  6. Verify normal evaluation run summary does NOT expose per-case labels or records.
  7. Verify CLI argument parsing routes correctly to evaluation and calibration modes using fakes.
"""

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.run_eval import (
    make_safe_dispute_payload,
    run_evaluation,
    run_train_threshold_calibration,
    calibrate_auto_submit_threshold,
    main,
    EvaluationRunSummary,
    ThresholdCalibrationReport,
    BANNED_EVAL_FIELDS,
)
from app.agent.decision_agent import DecisionAgentResponse

PASS = "[PASS]"
FAIL = "[FAIL]"


# ---------------------------------------------------------------------------
# Fakes / Mocks for offline testing
# ---------------------------------------------------------------------------

class FakeRetriever:
    def __init__(self, low_coverage: bool = False):
        self.low_coverage = low_coverage
        self.last_dispute_received = None

    def retrieve_evidence_for_dispute(self, dispute: dict) -> dict:
        self.last_dispute_received = dispute
        return {
            "retrieved_evidence": [
                {"evidence_id": "ev_test_1", "doc_type": "delivery_confirmation", "content": "Proof delivered"}
            ],
            "low_coverage": self.low_coverage,
            "retrieval_count": 1,
            "scoped_count": 1,
        }


class FakeDecisionAgent:
    def __init__(self, decision: str = "contest", confidence: float = 0.90, used_fallback: bool = False):
        self.decision = decision
        self.confidence = confidence
        self.used_fallback = used_fallback
        self.last_dispute_received = None
        self.last_retrieval_received = None

    def evaluate_dispute_dict(self, dispute_dict: dict, retrieval_output: dict) -> DecisionAgentResponse:
        self.last_dispute_received = dispute_dict
        self.last_retrieval_received = retrieval_output
        return DecisionAgentResponse(
            decision=self.decision,
            confidence=self.confidence,
            rebuttal_draft="Formal rebuttal text for testing",
            reasoning_summary="Sufficient evidence provided to contest claim.",
            used_fallback=self.used_fallback,
        )


def run_tests():
    print("=" * 80)
    print("      RAZORPAY CHARGEBACK RESPONDER - EVALUATION RUNNER VERIFICATION")
    print("=" * 80)
    print()

    all_passed = True
    results = []

    # ------------------------------------------------------------------
    # TEST 1: make_safe_dispute_payload strips all banned eval fields
    # ------------------------------------------------------------------
    polluted_raw_case = {
        "case_id": "CB-9999",
        "transaction_id": "txn_9999",
        "merchant_category": "ecommerce",
        "dispute_reason_code": "goods_not_received",
        "dispute_amount": 199.99,
        "dispute_raised_date": "2026-08-01",
        "response_deadline": "2026-08-20",
        "customer_claim_text": "Item never arrived.",
        "label_winnable": True,
        "ground_truth_rationale": "SENSITIVE_GROUND_TRUTH_LEAK",
        "_evidence_docs_obj": [{"quality": "strong"}],
        "quality": "high",
    }

    safe = make_safe_dispute_payload(polluted_raw_case)
    leaks = [b for b in BANNED_EVAL_FIELDS if b in safe]
    ok1 = len(leaks) == 0 and safe.get("case_id") == "CB-9999" and safe.get("dispute_amount") == 199.99
    results.append((
        "TEST 1 - make_safe_dispute_payload strips all 4 forbidden evaluation fields",
        ok1,
        f"leaks_found={leaks}, safe_keys={list(safe.keys())}"
    ))

    # ------------------------------------------------------------------
    # TEST 2: End-to-end pipeline execution with fakes
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        dataset_file = tmp_dir_path / "train.jsonl"
        audit_file = tmp_dir_path / "audit_logs.jsonl"

        case_item = {
            "case_id": "CB-0001",
            "transaction_id": "txn_0001",
            "merchant_category": "ecommerce",
            "dispute_reason_code": "goods_not_received",
            "dispute_amount": 100.0,
            "dispute_raised_date": "2026-08-01",
            "response_deadline": "2026-08-15",
            "customer_claim_text": "Did not receive order",
            "label_winnable": True,
            "ground_truth_rationale": "Secret rationale",
            "_evidence_docs_obj": [],
        }
        dataset_file.write_text(json.dumps(case_item) + "\n", encoding="utf-8")

        fake_retriever = FakeRetriever(low_coverage=False)
        fake_agent = FakeDecisionAgent(decision="contest", confidence=0.90, used_fallback=False)

        summary = run_evaluation(
            split="train",
            retriever=fake_retriever,
            decision_agent=fake_agent,
            datasets_dir=tmp_dir_path,
            audit_log_path=audit_file,
            require_live_llm=True,
            allow_fallback=False,
        )

        received_by_retriever = fake_retriever.last_dispute_received
        received_by_agent = fake_agent.last_dispute_received

        retriever_leaks = [b for b in BANNED_EVAL_FIELDS if b in received_by_retriever]
        agent_leaks = [b for b in BANNED_EVAL_FIELDS if b in received_by_agent]

        audit_lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        audit_ok = len(audit_lines) == 1

        ok2 = (
            summary.total_evaluated == 1 and
            summary.metrics.tp == 1 and
            summary.metrics.auto_submit_count == 1 and
            summary.fallback_count == 0 and
            len(retriever_leaks) == 0 and
            len(agent_leaks) == 0 and
            audit_ok
        )
        results.append((
            "TEST 2 - End-to-end pipeline execution with fakes (retrieval -> agent -> gate -> logger -> metrics)",
            ok2,
            f"eval_count={summary.total_evaluated}, TP={summary.metrics.tp}, auto_count={summary.metrics.auto_submit_count}, audit_lines={len(audit_lines)}"
        ))

    # ------------------------------------------------------------------
    # TEST 3: require_live_llm=True rejects fallback response with RuntimeError
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        dataset_file = tmp_dir_path / "train.jsonl"
        audit_file = tmp_dir_path / "audit_logs.jsonl"

        case_item = {
            "case_id": "CB-0002",
            "transaction_id": "txn_0002",
            "dispute_reason_code": "goods_defective",
            "dispute_amount": 50.0,
            "customer_claim_text": "Item broken",
            "label_winnable": False,
        }
        dataset_file.write_text(json.dumps(case_item) + "\n", encoding="utf-8")

        fake_retriever = FakeRetriever(low_coverage=False)
        fake_agent_fallback = FakeDecisionAgent(decision="contest", confidence=0.85, used_fallback=True)

        rejected = False
        err_msg = ""
        try:
            run_evaluation(
                split="train",
                retriever=fake_retriever,
                decision_agent=fake_agent_fallback,
                datasets_dir=tmp_dir_path,
                audit_log_path=audit_file,
                require_live_llm=True,
                allow_fallback=False,
            )
        except RuntimeError as exc:
            rejected = True
            err_msg = str(exc)

        ok3 = rejected and "Fallback reasoner was used" in err_msg
        results.append((
            "TEST 3 - require_live_llm=True rejects fallback response with clear RuntimeError",
            ok3,
            f"rejected={rejected}, err_msg={err_msg if rejected else None}"
        ))

    # ------------------------------------------------------------------
    # TEST 4: allow_fallback=True permits fallback responses
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        dataset_file = tmp_dir_path / "train.jsonl"
        audit_file = tmp_dir_path / "audit_logs.jsonl"

        case_item = {
            "case_id": "CB-0003",
            "transaction_id": "txn_0003",
            "dispute_reason_code": "duplicate_charge",
            "dispute_amount": 75.0,
            "customer_claim_text": "Charged twice",
            "label_winnable": False,
        }
        dataset_file.write_text(json.dumps(case_item) + "\n", encoding="utf-8")

        fake_retriever = FakeRetriever(low_coverage=False)
        fake_agent_fallback = FakeDecisionAgent(decision="contest", confidence=0.85, used_fallback=True)

        summary4 = run_evaluation(
            split="train",
            retriever=fake_retriever,
            decision_agent=fake_agent_fallback,
            datasets_dir=tmp_dir_path,
            audit_log_path=audit_file,
            require_live_llm=False,
            allow_fallback=True,
        )

        ok4 = (
            summary4.total_evaluated == 1 and
            summary4.fallback_count == 1 and
            summary4.used_fallback_status == "contains_fallback" and
            summary4.allow_fallback is True
        )
        results.append((
            "TEST 4 - allow_fallback=True permits fallback reasoner execution during local testing",
            ok4,
            f"eval_count={summary4.total_evaluated}, fallback_count={summary4.fallback_count}, status={summary4.used_fallback_status}"
        ))

    # ------------------------------------------------------------------
    # TEST 5: Threshold calibration rejects 'holdout' split and runs on 'train'
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        train_file = tmp_dir_path / "train.jsonl"
        audit_file = tmp_dir_path / "audit_logs.jsonl"

        case_item = {
            "case_id": "CB-0005",
            "transaction_id": "txn_0005",
            "dispute_reason_code": "goods_not_received",
            "dispute_amount": 120.0,
            "customer_claim_text": "Not received",
            "label_winnable": True,
        }
        train_file.write_text(json.dumps(case_item) + "\n", encoding="utf-8")

        fake_retriever = FakeRetriever(low_coverage=False)
        fake_agent = FakeDecisionAgent(decision="contest", confidence=0.85, used_fallback=False)

        # 5a. Check calibration rejects holdout
        holdout_rejected = False
        holdout_err = ""
        try:
            run_train_threshold_calibration(
                datasets_dir=tmp_dir_path,
                audit_log_path=audit_file,
                retriever=fake_retriever,
                decision_agent=fake_agent,
                split="holdout",
            )
        except ValueError as exc:
            holdout_rejected = True
            holdout_err = str(exc)

        # 5b. Valid calibration on train split
        report = run_train_threshold_calibration(
            datasets_dir=tmp_dir_path,
            candidate_thresholds=[0.70, 0.80, 0.90],
            audit_log_path=audit_file,
            retriever=fake_retriever,
            decision_agent=fake_agent,
            split="train",
        )

        ok5 = (
            holdout_rejected and
            "permitted ONLY on the 'train' split" in holdout_err and
            isinstance(report, ThresholdCalibrationReport) and
            report.split == "train" and
            report.total_cases == 1 and
            len(report.candidates) == 3
        )
        results.append((
            "TEST 5 - run_train_threshold_calibration rejects 'holdout' split and returns report for 'train'",
            ok5,
            f"holdout_rejected={holdout_rejected}, cand_count={len(report.candidates)}, split={report.split}"
        ))

    # ------------------------------------------------------------------
    # TEST 6: Normal holdout summary does not contain per-case labels/records
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        holdout_file = tmp_dir_path / "holdout.jsonl"
        audit_file = tmp_dir_path / "audit_logs.jsonl"

        case_item = {
            "case_id": "CB-0006",
            "transaction_id": "txn_0006",
            "dispute_reason_code": "goods_defective",
            "dispute_amount": 90.0,
            "customer_claim_text": "Broken item",
            "label_winnable": False,
        }
        holdout_file.write_text(json.dumps(case_item) + "\n", encoding="utf-8")

        fake_retriever = FakeRetriever(low_coverage=False)
        fake_agent = FakeDecisionAgent(decision="no_contest", confidence=0.20, used_fallback=False)

        summary6 = run_evaluation(
            split="holdout",
            retriever=fake_retriever,
            decision_agent=fake_agent,
            datasets_dir=tmp_dir_path,
            audit_log_path=audit_file,
        )

        summary_dict = summary6.model_dump()
        banned_keys = ["label_winnable", "case_records", "eval_records", "ground_truth_rationale"]
        found_banned = [k for k in banned_keys if k in summary_dict]

        ok6 = len(found_banned) == 0 and summary6.split == "holdout"
        results.append((
            "TEST 6 - Normal holdout EvaluationRunSummary does not expose per-case labels/records",
            ok6,
            f"banned_keys_found={found_banned}, summary_keys={list(summary_dict.keys())}"
        ))

    # ------------------------------------------------------------------
    # TEST 7: CLI argument parsing routes correctly to evaluation and calibration modes
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        train_file = tmp_dir_path / "train.jsonl"
        audit_file = tmp_dir_path / "audit_logs.jsonl"

        case_item = {
            "case_id": "CB-0007",
            "transaction_id": "txn_0007",
            "dispute_reason_code": "duplicate_charge",
            "dispute_amount": 40.0,
            "customer_claim_text": "Double charge",
            "label_winnable": True,
        }
        train_file.write_text(json.dumps(case_item) + "\n", encoding="utf-8")

        fake_retriever = FakeRetriever(low_coverage=False)
        fake_agent = FakeDecisionAgent(decision="contest", confidence=0.85, used_fallback=False)

        import io
        captured = io.StringIO()
        sys.stdout = captured

        try:
            main(
                [
                    "--calibrate-train",
                    "--thresholds", "0.75", "0.85",
                    "--datasets-dir", str(tmp_dir_path),
                    "--audit-log-path", str(audit_file),
                    "--allow-fallback",
                ],
                retriever=fake_retriever,
                decision_agent=fake_agent,
            )
        finally:
            sys.stdout = sys.__stdout__

        out_str = captured.getvalue()
        cli_data = json.loads(out_str)

        ok7 = (
            cli_data.get("split") == "train" and
            "candidates" in cli_data and
            len(cli_data.get("candidates", [])) == 2
        )
        results.append((
            "TEST 7 - CLI argument parsing routes correctly to calibration mode without external calls",
            ok7,
            f"split={cli_data.get('split')}, candidates_count={len(cli_data.get('candidates', []))}"
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
