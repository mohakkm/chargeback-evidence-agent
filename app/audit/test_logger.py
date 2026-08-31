import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audit.logger import log_decision, log_decision_from_dicts, AuditRecord

PASS = "[PASS]"
FAIL = "[FAIL]"


def run_tests():
    print("=" * 80)
    print("      RAZORPAY CHARGEBACK RESPONDER - AUDIT LOGGER VERIFICATION")
    print("=" * 80)
    print()

    all_passed = True
    results = []

    # ------------------------------------------------------------------
    # TEST 1: Write one realistic record & verify JSON format, fields, UTC timestamp
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_log = Path(tmpdir) / "test_audit.jsonl"

        rec1 = log_decision(
            case_id="case_1001",
            transaction_id="txn_5501",
            dispute_reason_code="goods_not_received",
            retrieved_evidence=[
                {"evidence_id": "ev_01", "doc_type": "delivery_confirmation", "content": "Delivered to doorstep"},
                {"evidence_id": "ev_02", "doc_type": "order_details", "content": "Order details page"}
            ],
            retrieval_count=2,
            scoped_count=2,
            low_coverage=False,
            decision="contest",
            confidence=0.88,
            reasoning_summary="Sufficient evidence provided showing delivery confirmation.",
            action="auto_submit",
            used_fallback=False,
            log_path=tmp_log,
        )

        lines = tmp_log.read_text(encoding="utf-8").strip().splitlines()
        ok1_line_count = len(lines) == 1

        data1 = json.loads(lines[0]) if ok1_line_count else {}
        expected_fields = {
            "timestamp_utc", "case_id", "transaction_id", "dispute_reason_code",
            "retrieved_evidence_ids", "retrieval_count", "scoped_count", "low_coverage",
            "decision", "confidence", "reasoning_summary", "action", "used_fallback"
        }
        has_all_fields = expected_fields.issubset(data1.keys())

        ts_str = data1.get("timestamp_utc", "")
        try:
            dt = datetime.fromisoformat(ts_str)
            is_utc = dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0
        except Exception:
            is_utc = False

        ev_ids_ok = data1.get("retrieved_evidence_ids") == ["ev_01", "ev_02"]

        ok1 = ok1_line_count and has_all_fields and is_utc and ev_ids_ok
        results.append((
            "TEST 1 - Write one realistic record (valid JSON, expected fields, UTC timestamp)",
            ok1,
            f"lines={len(lines)}, fields_present={has_all_fields}, is_utc={is_utc}, ev_ids={data1.get('retrieved_evidence_ids')}"
        ))

    # ------------------------------------------------------------------
    # TEST 2: Second write appends rather than overwrites
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_log = Path(tmpdir) / "append_test.jsonl"

        log_decision(
            case_id="case_2001",
            transaction_id="txn_6001",
            dispute_reason_code="goods_defective",
            retrieved_evidence=[],
            retrieval_count=0,
            scoped_count=0,
            low_coverage=True,
            decision="no_contest",
            confidence=0.20,
            reasoning_summary="No evidence available.",
            action="flag_for_review",
            log_path=tmp_log,
        )

        log_decision(
            case_id="case_2002",
            transaction_id="txn_6002",
            dispute_reason_code="unauthorized_transaction",
            retrieved_evidence=[{"evidence_id": "ev_99"}],
            retrieval_count=1,
            scoped_count=1,
            low_coverage=True,
            decision="contest",
            confidence=0.45,
            reasoning_summary="Sparse evidence present.",
            action="flag_for_review",
            log_path=tmp_log,
        )

        lines = tmp_log.read_text(encoding="utf-8").strip().splitlines()
        ok2_count = len(lines) == 2
        d1 = json.loads(lines[0]) if len(lines) >= 1 else {}
        d2 = json.loads(lines[1]) if len(lines) >= 2 else {}

        ok2_content = (d1.get("case_id") == "case_2001") and (d2.get("case_id") == "case_2002")
        ok2 = ok2_count and ok2_content
        results.append((
            "TEST 2 - Second write appends rather than overwrites (file has 2 distinct lines)",
            ok2,
            f"line_count={len(lines)}, case1={d1.get('case_id')!r}, case2={d2.get('case_id')!r}"
        ))

    # ------------------------------------------------------------------
    # TEST 3: Pass polluted inputs with label_winnable, ground_truth_rationale, quality
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_log = Path(tmpdir) / "polluted_test.jsonl"

        polluted_dispute = {
            "case_id": "case_3001",
            "transaction_id": "txn_7001",
            "dispute_reason_code": "subscription_canceled_but_charged",
            "label_winnable": True,
            "ground_truth_rationale": "Merchant has clear cancellation policy timestamp.",
        }

        polluted_retrieval = {
            "retrieved_evidence": [
                {
                    "evidence_id": "ev_polluted_1",
                    "doc_type": "refund_policy",
                    "content": "SECRET CONTENT THAT SHOULD NOT BE STORED",
                    "quality": "high",
                }
            ],
            "retrieval_count": 1,
            "scoped_count": 1,
            "low_coverage": True,
        }

        polluted_decision = {
            "decision": "contest",
            "confidence": 0.45,
            "reasoning_summary": "Policy refutes cancellation date.",
            "used_fallback": True,
            "label_winnable": True,
        }

        polluted_gate = {
            "action": "flag_for_review",
            "ground_truth_rationale": "leaked",
        }

        rec3 = log_decision_from_dicts(
            dispute=polluted_dispute,
            retrieval_output=polluted_retrieval,
            decision_response=polluted_decision,
            gate_output=polluted_gate,
            log_path=tmp_log,
        )

        raw_content = tmp_log.read_text(encoding="utf-8")
        banned_terms = ["label_winnable", "ground_truth_rationale", "quality", "SECRET CONTENT"]
        leaks = [term for term in banned_terms if term in raw_content]

        ok3 = len(leaks) == 0
        results.append((
            "TEST 3 - Privacy & Isolation: Polluted inputs produce safe audit log free of banned fields",
            ok3,
            f"leaks_found={leaks}"
        ))

    # ------------------------------------------------------------------
    # TEST 4: Verify used_fallback and action are preserved exactly
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_log = Path(tmpdir) / "preservation_test.jsonl"

        log_decision(
            case_id="case_4001",
            transaction_id="txn_8001",
            dispute_reason_code="goods_not_received",
            retrieved_evidence=[],
            retrieval_count=2,
            scoped_count=2,
            low_coverage=False,
            decision="contest",
            confidence=0.85,
            reasoning_summary="Summary A",
            action="auto_submit",
            used_fallback=True,
            log_path=tmp_log,
        )

        log_decision(
            case_id="case_4002",
            transaction_id="txn_8002",
            dispute_reason_code="goods_not_received",
            retrieved_evidence=[],
            retrieval_count=1,
            scoped_count=1,
            low_coverage=True,
            decision="no_contest",
            confidence=0.30,
            reasoning_summary="Summary B",
            action="flag_for_review",
            used_fallback=False,
            log_path=tmp_log,
        )

        log_decision(
            case_id="case_4003",
            transaction_id="txn_8003",
            dispute_reason_code="goods_not_received",
            retrieved_evidence=[],
            retrieval_count=0,
            scoped_count=0,
            low_coverage=True,
            decision="no_contest",
            confidence=0.10,
            reasoning_summary="Summary C",
            action="flag_for_review",
            used_fallback=None,
            log_path=tmp_log,
        )

        lines = tmp_log.read_text(encoding="utf-8").strip().splitlines()
        rA = json.loads(lines[0])
        rB = json.loads(lines[1])
        rC = json.loads(lines[2])

        okA = rA.get("used_fallback") is True and rA.get("action") == "auto_submit"
        okB = rB.get("used_fallback") is False and rB.get("action") == "flag_for_review"
        okC = rC.get("used_fallback") is None and rC.get("action") == "flag_for_review"

        ok4 = okA and okB and okC
        results.append((
            "TEST 4 - Preserve used_fallback (True/False/None) and action (auto_submit/flag_for_review) exactly",
            ok4,
            f"CaseA=(fallback:{rA.get('used_fallback')}, action:{rA.get('action')!r}), "
            f"CaseB=(fallback:{rB.get('used_fallback')}, action:{rB.get('action')!r}), "
            f"CaseC=(fallback:{rC.get('used_fallback')}, action:{rC.get('action')!r})"
        ))

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
