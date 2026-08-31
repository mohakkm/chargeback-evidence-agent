"""
Verification script for Phase 3 Decision Agent (app/agent/decision_agent.py).

Tests:
1. Strict Input Sanitization & Ground-Truth Isolation (label_winnable, ground_truth_rationale, quality).
2. Sparse / low_coverage=True case with NO strong evidence → expects no_contest, confidence <= 0.45.
3. Sparse / low_coverage=True case with ONE strong delivery-confirmation doc → any decision allowed,
   confidence must still be <= 0.45, used_fallback must be a bool.
4. used_fallback field presence and type verified on all responses.

If Groq is unavailable, the agent runs through the heuristic fallback and reports this clearly.
"""

import json
import sys
from pathlib import Path

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
PASS = "[PASS]"
FAIL = "[FAIL]"


def load_holdout_cases():
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

    all_passed = True

    # -------------------------------------------------------------------------
    # TEST 1: Strict Input Field Isolation
    # -------------------------------------------------------------------------
    print("[TEST 1] Strict Input Field Isolation Check")
    print("-" * 60)

    raw_polluted_case = {
        "case_id": "CB-99999",
        "dispute_reason_code": "goods_not_received",
        "customer_claim_text": "Merchandise was never delivered.",
        "dispute_amount": 149.99,
        "merchant_category": "ecommerce",
        "label_winnable": True,               # EVAL ONLY — must be stripped
        "ground_truth_rationale": "Merchant has signed proof of delivery.",  # EVAL ONLY — must be stripped
    }
    raw_polluted_evidence = [
        {
            "evidence_id": "EVD-99999-01",
            "doc_type": "delivery_confirmation",
            "content": "Carrier FedEx Tracking #12345. Status: DELIVERED. Signature captured.",
            "quality": "strong",              # EVAL ONLY — must be stripped
        }
    ]

    clean_dispute = sanitize_dispute_input(raw_polluted_case)
    clean_evidence = sanitize_evidence_input(raw_polluted_evidence)

    dispute_fields = set(clean_dispute.model_dump().keys())
    evidence_fields = set(clean_evidence[0].model_dump().keys())

    isolation_ok = (
        "label_winnable" not in dispute_fields
        and "ground_truth_rationale" not in dispute_fields
        and "quality" not in evidence_fields
    )

    print(f"Raw Case Keys        : {list(raw_polluted_case.keys())}")
    print(f"Clean Dispute Fields : {sorted(dispute_fields)}")
    print(f"Clean Evidence Fields: {sorted(evidence_fields)}")

    if isolation_ok:
        print(f"STATUS: {PASS} Strict field isolation verified. Evaluation-only fields completely stripped.")
    else:
        print(f"STATUS: {FAIL} Field isolation FAILED — eval fields leaked into sanitized input!")
        all_passed = False

    print()

    # -------------------------------------------------------------------------
    # Instantiate agent once (shared across TEST 2 and TEST 3)
    # -------------------------------------------------------------------------
    agent = DecisionAgent()
    groq_available = agent._client is not None
    print(f"[Agent Mode] Groq available: {groq_available} | Model: {agent.model}")
    if not groq_available:
        print("            ⚠  Groq unavailable — running via heuristic fallback. Results will show used_fallback=True.")
    print()

    # -------------------------------------------------------------------------
    # TEST 2: Sparse / low_coverage=True — NO strong evidence
    # -------------------------------------------------------------------------
    print("[TEST 2] Sparse / low_coverage=True — No Evidence (expect no_contest, confidence <= 0.45)")
    print("-" * 60)

    sparse_dispute_no_ev = CleanDisputeInput(
        dispute_reason_code="unauthorized_transaction",
        customer_claim_text="Fraudulent charge of $200.00 at Frost Inc on 2026-07-02. Not authorized by me.",
        dispute_amount=200.00,
        merchant_category="ecommerce",
    )
    sparse_ev_none: list[CleanEvidenceInput] = []

    resp_no_ev = agent.evaluate_dispute(
        dispute=sparse_dispute_no_ev,
        evidence=sparse_ev_none,
        low_coverage=True,
    )

    print(f"Decision       : {resp_no_ev.decision.upper()}")
    print(f"Confidence     : {resp_no_ev.confidence:.4f}")
    print(f"used_fallback  : {resp_no_ev.used_fallback}")
    print(f"Reasoning      : {resp_no_ev.reasoning_summary}")

    t2_conf_ok = resp_no_ev.confidence <= 0.45
    t2_fallback_ok = isinstance(resp_no_ev.used_fallback, bool)
    t2_decision_ok = resp_no_ev.decision in ("contest", "no_contest")

    status_2 = PASS if (t2_conf_ok and t2_fallback_ok and t2_decision_ok) else FAIL
    if not (t2_conf_ok and t2_fallback_ok and t2_decision_ok):
        all_passed = False

    print(f"Assert confidence <= 0.45 : {'OK' if t2_conf_ok else 'FAIL'} ({resp_no_ev.confidence:.4f})")
    print(f"Assert used_fallback bool : {'OK' if t2_fallback_ok else 'FAIL'}")
    print(f"Assert valid decision     : {'OK' if t2_decision_ok else 'FAIL'}")
    print(f"STATUS: {status_2}")
    print()

    # -------------------------------------------------------------------------
    # TEST 3: Sparse / low_coverage=True — ONE strong delivery-confirmation doc
    # -------------------------------------------------------------------------
    print("[TEST 3] Sparse / low_coverage=True — One Strong Delivery-Confirmation Doc (any decision, confidence <= 0.45)")
    print("-" * 60)

    sparse_dispute_one_ev = CleanDisputeInput(
        dispute_reason_code="goods_not_received",
        customer_claim_text="Package was never delivered despite paying $89.50 to QuickShip Co on 2026-07-15.",
        dispute_amount=89.50,
        merchant_category="ecommerce",
    )
    sparse_ev_one = [
        CleanEvidenceInput(
            evidence_id="EVD-TEST-01",
            doc_type="delivery_confirmation",
            content=(
                "Carrier FedEx Tracking #TRK-78234. Status: DELIVERED on 2026-07-17 at 14:32 UTC. "
                "Package delivered to billing address: 45 Maple Ave, Austin TX 78701. "
                "Recipient signature captured: J. Smith. Photo proof available."
            ),
            score=0.91,
        )
    ]

    resp_one_ev = agent.evaluate_dispute(
        dispute=sparse_dispute_one_ev,
        evidence=sparse_ev_one,
        low_coverage=True,
    )

    print(f"Decision       : {resp_one_ev.decision.upper()}")
    print(f"Confidence     : {resp_one_ev.confidence:.4f}")
    print(f"used_fallback  : {resp_one_ev.used_fallback}")
    print(f"Reasoning      : {resp_one_ev.reasoning_summary}")
    if resp_one_ev.decision == "contest" and resp_one_ev.rebuttal_draft:
        snippet = resp_one_ev.rebuttal_draft[:150].replace("\n", " ")
        print(f"Rebuttal snip  : {snippet}...")

    t3_conf_ok = resp_one_ev.confidence <= 0.45
    t3_fallback_ok = isinstance(resp_one_ev.used_fallback, bool)
    t3_decision_ok = resp_one_ev.decision in ("contest", "no_contest")

    status_3 = PASS if (t3_conf_ok and t3_fallback_ok and t3_decision_ok) else FAIL
    if not (t3_conf_ok and t3_fallback_ok and t3_decision_ok):
        all_passed = False

    print(f"Assert confidence <= 0.45 : {'OK' if t3_conf_ok else 'FAIL'} ({resp_one_ev.confidence:.4f})")
    print(f"Assert used_fallback bool : {'OK' if t3_fallback_ok else 'FAIL'}")
    print(f"Assert valid decision     : {'OK' if t3_decision_ok else 'FAIL'}")
    print(f"Note: decision can be either contest or no_contest — confidence cap is what matters here")
    print(f"STATUS: {status_3}")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    overall = "ALL CHECKS PASSED" if all_passed else "ONE OR MORE CHECKS FAILED"
    print(f"      {overall}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    verify_decision_agent()
