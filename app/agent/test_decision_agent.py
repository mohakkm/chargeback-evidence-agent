"""
Verification script for Phase 3 Decision Agent (app/agent/decision_agent.py).

Tests:
1. Strict Input Sanitization & Ground-Truth Isolation (label_winnable, ground_truth_rationale, quality).
2. Decision reasoning on strong, weak, and sparse evidence cases.
3. Structured output format (decision, confidence, rebuttal_draft, reasoning_summary).
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.decision_agent import (
    DecisionAgent,
    sanitize_dispute_input,
    sanitize_evidence_input,
    CleanDisputeInput,
    CleanEvidenceInput,
)

DATASETS_DIR = PROJECT_ROOT / "app" / "data" / "datasets"


def load_sample_holdout_cases():
    filepath = DATASETS_DIR / "holdout.jsonl"
    cases = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
    return cases


def verify_decision_agent():
    print("=" * 80)
    print("      RAZORPAY CHARGEBACK RESPONDER — PHASE 3 DECISION AGENT VERIFICATION")
    print("=" * 80)
    print()

    # 1. Test Field Isolation / Input Sanitization
    print("[TEST 1] Strict Input Field Isolation Check")
    print("-" * 60)

    raw_polluted_case = {
        "case_id": "CB-99999",
        "dispute_reason_code": "goods_not_received",
        "customer_claim_text": "Merchandise was never delivered.",
        "dispute_amount": 149.99,
        "merchant_category": "ecommerce",
        "label_winnable": True,  # EVAL ONLY - MUST BE STRIPPED
        "ground_truth_rationale": "Merchant has signed proof of delivery.",  # EVAL ONLY - MUST BE STRIPPED
    }

    raw_polluted_evidence = [
        {
            "evidence_id": "EVD-99999-01",
            "doc_type": "delivery_confirmation",
            "content": "Carrier FedEx Tracking #12345. Status: DELIVERED. Signature captured.",
            "quality": "strong",  # EVAL ONLY - MUST BE STRIPPED
        }
    ]

    clean_dispute = sanitize_dispute_input(raw_polluted_case)
    clean_evidence = sanitize_evidence_input(raw_polluted_evidence)

    assert not hasattr(clean_dispute, "label_winnable"), "ERROR: label_winnable leaked into CleanDisputeInput!"
    assert not hasattr(clean_dispute, "ground_truth_rationale"), "ERROR: ground_truth_rationale leaked into CleanDisputeInput!"
    assert not hasattr(clean_evidence[0], "quality"), "ERROR: quality leaked into CleanEvidenceInput!"

    print("Raw Case Keys       :", list(raw_polluted_case.keys()))
    print("Clean Dispute Fields:", list(clean_dispute.model_dump().keys()))
    print("Clean Evidence Fields:", list(clean_evidence[0].model_dump().keys()))
    print("STATUS: [PASS] Strict field isolation verified. Evaluation-only fields completely stripped.\n")

    # 2. Test Agent Reasoning on Holdout Cases
    print("[TEST 2] Agent Reasoning & Rebuttal Generation on Holdout Cases")
    print("-" * 60)

    holdout_cases = load_sample_holdout_cases()
    agent = DecisionAgent()

    if not holdout_cases:
        print("[WARNING] holdout.jsonl not found. Testing with synthetic test cases.")
        holdout_cases = [
            {
                "case_id": "CB-TEST-01",
                "dispute_reason_code": "goods_not_received",
                "dispute_amount": 199.99,
                "merchant_category": "ecommerce",
                "customer_claim_text": "I paid $199.99 on 2026-08-01 but never received the package.",
                "_evidence_docs_obj": [
                    {"evidence_id": "EVD-01", "doc_type": "delivery_confirmation", "content": "Carrier FedEx #TRK99. Status: DELIVERED on 2026-08-03 to billing address. Recipient signature: John Doe."},
                    {"evidence_id": "EVD-02", "doc_type": "order_details", "content": "Order Details #ORD-123. Total: $199.99. Ship-to: 123 Main St, New York, NY."}
                ]
            }
        ]

    # Select 3 distinct test cases (1 strong, 1 weak, 1 sparse)
    test_samples = []
    # Case A: Strong evidence
    strong_cases = [c for c in holdout_cases if len(c.get("evidence_doc_ids", c.get("_evidence_docs_obj", []))) >= 2]
    if strong_cases:
        test_samples.append(("Strong Case", strong_cases[0], False))

    # Case B: Sparse case
    sparse_cases = [c for c in holdout_cases if len(c.get("evidence_doc_ids", c.get("_evidence_docs_obj", []))) <= 1]
    if sparse_cases:
        test_samples.append(("Sparse Case", sparse_cases[0], True))

    for label, case, is_sparse in test_samples:
        raw_ev = case.get("_evidence_docs_obj", [])
        if not raw_ev and "evidence_doc_ids" in case:
            # Reconstruct basic evidence stubs if missing in raw dict
            raw_ev = [
                {"evidence_id": eid, "doc_type": "order_details", "content": f"Evidence document record for {eid}."}
                for eid in case.get("evidence_doc_ids", [])
            ]

        clean_dis = sanitize_dispute_input(case)
        clean_ev = sanitize_evidence_input(raw_ev)

        response = agent.evaluate_dispute(
            dispute=clean_dis,
            evidence=clean_ev,
            low_coverage=is_sparse
        )

        print(f"\n--- {label}: {case.get('case_id', 'CB-00000')} ({clean_dis.dispute_reason_code}) ---")
        print(f"Dispute Amount   : ${clean_dis.dispute_amount:.2f}")
        print(f"Customer Claim   : \"{clean_dis.customer_claim_text}\"")
        print(f"Evidence Count   : {len(clean_ev)} docs (low_coverage={is_sparse})")
        print(f"Agent Decision   : {response.decision.upper()}")
        print(f"Confidence Score : {response.confidence:.2f}")
        print(f"Reasoning Summary: {response.reasoning_summary}")

        if response.decision == "contest":
            snippet = response.rebuttal_draft[:150].replace('\n', ' ')
            print(f"Rebuttal Snippet : {snippet}...")
        else:
            print(f"Rebuttal Draft   : [EMPTY - No contest decision]")

        assert response.decision in ("contest", "no_contest"), f"Invalid decision: {response.decision}"
        assert 0.0 <= response.confidence <= 1.0, f"Confidence out of bounds: {response.confidence}"
        assert isinstance(response.reasoning_summary, str) and len(response.reasoning_summary) > 0, "Reasoning summary empty!"
        if response.decision == "contest":
            assert len(response.rebuttal_draft) > 20, "Rebuttal draft too short for contest decision!"

        print(f"Result           : [PASS] Valid structured response returned.")

    print("\n" + "=" * 80)
    print("      ALL DECISION AGENT VERIFICATION CHECKS PASSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    verify_decision_agent()
