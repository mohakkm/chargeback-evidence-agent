"""
Verification script for Phase 2 Retrieval Pipeline per CHECKLIST.md.

Validates:
1. Qdrant point count vs total unique evidence docs in train.jsonl + holdout.jsonl.
2. Cross-transaction isolation on 3 random holdout cases (checks for leakage).
3. Coverage flag behavior (low_coverage=True for <2 docs, False for >=2 docs).
4. Ground-truth 'quality' field stripping in retriever output vs presence in direct Qdrant payload.
"""

import json
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is first on sys.path and script dir is removed to prevent module name shadowing
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embed import index_evidence_docs
from app.retrieval.qdrant_client import get_qdrant_client, COLLECTION_NAME
from app.retrieval.retriever import EvidenceRetriever, strip_evidence_quality

DATASETS_DIR = Path(__file__).parents[1] / "data" / "datasets"


def stable_point_id(evidence_id: str) -> str:
    """Derives stable UUID v5 from evidence_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, evidence_id))


def load_dataset_cases(filename: str) -> List[Dict[str, Any]]:
    filepath = DATASETS_DIR / filename
    cases = []
    if not filepath.exists():
        print(f"[ERROR] Dataset file not found: {filepath}")
        return cases
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def verify_retrieval_pipeline():
    print("=" * 80)
    print("      RAZORPAY CHARGEBACK RESPONDER — PHASE 2 RETRIEVAL VERIFICATION")
    print("=" * 80)
    print()

    # Reindex Qdrant collection cleanly from scratch using single client instance
    qdrant_client = get_qdrant_client()
    index_evidence_docs(client=qdrant_client, force_reindex=True)
    print()

    # Load datasets
    train_cases = load_dataset_cases("train.jsonl")
    holdout_cases = load_dataset_cases("holdout.jsonl")
    all_cases = train_cases + holdout_cases

    # 1. Point Count Verification
    print("[TEST 1] Qdrant Point Count vs Dataset Evidence Docs")
    print("-" * 60)

    dataset_evidence_map: Dict[str, Dict[str, Any]] = {}
    for case in all_cases:
        for ev in case.get("_evidence_docs_obj", []):
            eid = ev["evidence_id"]
            if eid not in dataset_evidence_map:
                dataset_evidence_map[eid] = ev

    total_dataset_unique_docs = len(dataset_evidence_map)

    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    qdrant_point_count = collection_info.points_count

    print(f"Total Unique Evidence Docs in Datasets : {total_dataset_unique_docs}")
    print(f"Total Points in Qdrant '{COLLECTION_NAME}' : {qdrant_point_count}")

    if total_dataset_unique_docs == qdrant_point_count:
        print(f"STATUS: [PASS] Exact match! {qdrant_point_count} / {total_dataset_unique_docs} docs successfully embedded.\n")
    else:
        print(f"STATUS: [FAIL/WARNING] Mismatch detected! Dataset has {total_dataset_unique_docs} unique docs, but Qdrant has {qdrant_point_count}.\n")

    # Initialize Retriever
    retriever = EvidenceRetriever(client=qdrant_client)

    # 2. Random Holdout Cases Retrieval & Leakage Check (Testing non-sparse cases with >= 2 docs)
    print("[TEST 2] Holdout Cases Retrieval & Cross-Transaction Leakage Check")
    print("-" * 60)

    valid_holdout_cases = [c for c in holdout_cases if len(c.get("evidence_doc_ids", [])) >= 2]
    sampled_cases = random.sample(valid_holdout_cases, min(3, len(valid_holdout_cases)))

    for idx, case in enumerate(sampled_cases, start=1):
        txn_id = case["transaction_id"]
        case_id = case["case_id"]
        reason_code = case["dispute_reason_code"]
        claim = case["customer_claim_text"]
        expected_doc_ids = case.get("evidence_doc_ids", [])

        print(f"\n--- Case #{idx}: {case_id} (Txn: {txn_id}) ---")
        print(f"Reason Code   : {reason_code}")
        print(f"Customer Claim: \"{claim}\"")
        print(f"Expected Evidence IDs ({len(expected_doc_ids)} docs): {expected_doc_ids}")

        retrieval_res = retriever.retrieve_evidence_for_dispute(case, top_k=5)
        retrieved_docs = retrieval_res["retrieved_evidence"]

        print(f"Retrieved {len(retrieved_docs)} docs (low_coverage={retrieval_res['low_coverage']}):")

        leakage_detected = False
        for r_idx, doc in enumerate(retrieved_docs, start=1):
            r_eid = doc.get("evidence_id")
            r_txnid = doc.get("transaction_id")
            r_type = doc.get("doc_type")
            r_score = doc.get("_score")
            snippet = doc.get("content", "")[:85]

            is_txn_match = (r_txnid == txn_id)
            match_status = "MATCH" if is_txn_match else "MISMATCH / LEAKAGE"
            if not is_txn_match:
                leakage_detected = True

            print(f"  [{r_idx}] ID: {r_eid} | Type: {r_type:<24} | Score: {r_score:.4f} | Txn: {r_txnid} [{match_status}]")
            print(f"      Snippet: {snippet}...")

        if not leakage_detected:
            print(f"  Result: [PASS] All retrieved docs match case transaction_id '{txn_id}'. Zero cross-transaction leakage.")
        else:
            print(f"  Result: [FAIL/WARNING] Cross-transaction leakage detected in retrieved results!")

    print()

    # 3. Coverage Flag Checks
    print("[TEST 3] Bounded Action Gate Coverage Flag Checks")
    print("-" * 60)

    # Case A: Low coverage (< 2 docs)
    sparse_case = {
        "transaction_id": "txn_non_existent_99999",
        "dispute_reason_code": "unauthorized_transaction",
        "customer_claim_text": "Unrecognized charge on card statement with missing transaction records.",
    }
    sparse_res = retriever.retrieve_evidence_for_dispute(sparse_case, top_k=5)
    print(f"Sparse Case (non-existent txn_id): scoped_count={sparse_res.get('scoped_count')}, low_coverage={sparse_res['low_coverage']}")
    if sparse_res["low_coverage"] is True:
        print("STATUS: [PASS] Correctly flagged low_coverage=True when transaction-scoped evidence docs < 2.\n")
    else:
        print("STATUS: [FAIL] Expected low_coverage=True for sparse case, got False.\n")

    # Case B: High coverage (3+ strong docs)
    strong_case = None
    for c in holdout_cases:
        strong_docs = [ev for ev in c.get("_evidence_docs_obj", []) if ev.get("quality") == "strong"]
        if len(strong_docs) >= 3:
            strong_case = c
            break

    if not strong_case and train_cases:
        strong_case = train_cases[0]

    if strong_case:
        strong_res = retriever.retrieve_evidence_for_dispute(strong_case, top_k=5)
        print(f"Strong Case ({strong_case['case_id']}, Txn: {strong_case['transaction_id']}): scoped_count={strong_res.get('scoped_count')}, low_coverage={strong_res['low_coverage']}")
        if strong_res["low_coverage"] is False:
            print("STATUS: [PASS] Correctly evaluated low_coverage=False when evidence docs >= 2.\n")
        else:
            print("STATUS: [FAIL] Expected low_coverage=False for strong case, got True.\n")

    # 4. Quality Field Stripping Verification
    print("[TEST 4] Eval Ground-Truth 'quality' Field Isolation")
    print("-" * 60)

    # Check retriever output
    sample_retrieved = retriever.retrieve_for_dispute(all_cases[0], top_k=1)
    has_quality_in_retriever_output = any("quality" in doc for doc in sample_retrieved)

    # Check direct Qdrant point query
    first_ev_id = all_cases[0]["evidence_doc_ids"][0]
    pid = stable_point_id(first_ev_id)
    qdrant_raw_points = qdrant_client.retrieve(collection_name=COLLECTION_NAME, ids=[pid], with_payload=True)

    has_quality_in_qdrant_direct = False
    if qdrant_raw_points and qdrant_raw_points[0].payload:
        has_quality_in_qdrant_direct = "quality" in qdrant_raw_points[0].payload

    print(f"Retriever Output contains 'quality' field : {has_quality_in_retriever_output} (Expected: False)")
    print(f"Direct Qdrant Payload contains 'quality'  : {has_quality_in_qdrant_direct} (Expected: True)")

    if not has_quality_in_retriever_output and has_quality_in_qdrant_direct:
        print("STATUS: [PASS] 'quality' field is safely stripped from decision agent view, but preserved in Qdrant store for evaluation.\n")
    else:
        print("STATUS: [FAIL] Quality field isolation test failed.\n")

    # 5. Summary Metrics & Sanity Checks
    holdout_sparse_ids = [c["case_id"] for c in holdout_cases if len(c.get("evidence_doc_ids", [])) < 2]
    literal_txnid_count = sum(
        1 for c in all_cases
        for ev in c.get("_evidence_docs_obj", [])
        if c.get("transaction_id") and c["transaction_id"] in ev.get("content", "")
    )

    print("=" * 80)
    print("                      VERIFICATION SUMMARY RESULT")
    print("=" * 80)
    print(f"Final Evidence Doc Count               : {total_dataset_unique_docs}")
    print(f"Qdrant Point Count                     : {qdrant_point_count}")
    print(f"Holdout Sparse Case IDs (<2 docs)      : {holdout_sparse_ids}")
    print(f"Literal transaction_id Strings in Content: {literal_txnid_count} (Confirmed Zero: {literal_txnid_count == 0})")
    print("=" * 80)


if __name__ == "__main__":
    verify_retrieval_pipeline()

