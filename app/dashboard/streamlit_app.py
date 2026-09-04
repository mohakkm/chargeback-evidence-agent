"""
Streamlit demo dashboard — fully offline.

Tabs:
  1. Case Walkthrough — sourced from train_audit_consolidated.jsonl + train.jsonl
  2. Metrics — TRAIN calibration results from calibration_results.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLIDATED_AUDIT_PATH = _PROJECT_ROOT / "train_audit_consolidated.jsonl"
CALIBRATION_RESULTS_PATH = _PROJECT_ROOT / "calibration_results.jsonl"
TRAIN_DATASET_PATH = _PROJECT_ROOT / "app" / "data" / "datasets" / "train.jsonl"


@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_train_cases() -> Dict[str, Dict[str, Any]]:
    return {row["case_id"]: row for row in load_jsonl(TRAIN_DATASET_PATH) if row.get("case_id")}


@st.cache_data(show_spinner=False)
def load_audit_by_case() -> Dict[str, Dict[str, Any]]:
    audit_rows = load_jsonl(CONSOLIDATED_AUDIT_PATH)
    by_case: Dict[str, Dict[str, Any]] = {}
    for row in audit_rows:
        cid = row.get("case_id")
        if cid:
            ts = row.get("timestamp_utc", "")
            if cid not in by_case or ts > by_case[cid].get("timestamp_utc", ""):
                by_case[cid] = row
    return by_case


def _evidence_for_case(train_case: Dict[str, Any], evidence_ids: List[str]) -> List[Dict[str, str]]:
    docs = train_case.get("_evidence_docs_obj") or []
    by_id = {d.get("evidence_id"): d for d in docs if d.get("evidence_id")}
    shown = []
    for eid in evidence_ids:
        doc = by_id.get(eid)
        if doc:
            shown.append({
                "evidence_id": eid,
                "doc_type": doc.get("doc_type", ""),
                "content": doc.get("content", ""),
            })
        else:
            shown.append({"evidence_id": eid, "doc_type": "unknown", "content": "(content not in train dataset)"})
    return shown


def compute_train_calibration_metrics(
    calibration_rows: List[Dict[str, Any]],
    train_cases: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    auto_submit = review = 0
    for row in calibration_rows:
        cid = row.get("case_id")
        if not cid or cid not in train_cases:
            continue
        label = bool(train_cases[cid].get("label_winnable", False))
        pred_contest = str(row.get("decision", "")).lower() == "contest"
        if pred_contest and label:
            tp += 1
        elif pred_contest and not label:
            fp += 1
        elif not pred_contest and not label:
            tn += 1
        else:
            fn += 1
        if row.get("action") == "auto_submit":
            auto_submit += 1
        else:
            review += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "total_cases": float(total),
        "precision": precision,
        "recall": recall,
        "auto_submit_rate": auto_submit / total if total else 0.0,
        "review_rate": review / total if total else 0.0,
        "auto_submit_count": float(auto_submit),
        "review_count": float(review),
    }


def render_case_tab(audit_by_case: Dict[str, Dict[str, Any]], train_cases: Dict[str, Dict[str, Any]]) -> None:
    st.subheader("Case Walkthrough")
    if not audit_by_case:
        st.warning(f"No audit records found at `{CONSOLIDATED_AUDIT_PATH.name}`.")
        return

    case_ids = sorted(audit_by_case.keys())
    selected = st.selectbox("Select case", case_ids)
    audit = audit_by_case[selected]
    train_case = train_cases.get(selected, {})

    st.markdown("#### Dispute claim")
    st.write(train_case.get("customer_claim_text", "(claim text unavailable — check train.jsonl)"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Reason code", audit.get("dispute_reason_code", "—"))
    c2.metric("Amount", f"${train_case.get('dispute_amount', 0):.2f}")
    c3.metric("Transaction", audit.get("transaction_id", "—"))

    st.markdown("#### Retrieved evidence")
    evidence_ids = audit.get("retrieved_evidence_ids") or []
    for doc in _evidence_for_case(train_case, evidence_ids):
        with st.expander(f"{doc['evidence_id']} — {doc['doc_type']}"):
            st.write(doc["content"])

    st.markdown("#### Decision")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Decision", audit.get("decision", "—"))
    d2.metric("Confidence", f"{float(audit.get('confidence', 0)):.2f}")
    d3.metric("Action", audit.get("action", "—"))
    d4.metric("Low coverage", str(audit.get("low_coverage", False)))

    st.caption(audit.get("reasoning_summary", ""))
    if audit.get("used_fallback"):
        st.warning("used_fallback=True — heuristic path (excluded from official metrics).")


def render_metrics_tab(
    calibration_rows: List[Dict[str, Any]],
    train_cases: Dict[str, Dict[str, Any]],
) -> None:
    st.subheader("TRAIN calibration results")
    st.info(
        "**TRAIN calibration results** — these metrics are computed from the TRAIN split "
        "calibration run only. They are **not** final holdout evaluation numbers."
    )

    if not calibration_rows:
        st.warning(f"No calibration rows found at `{CALIBRATION_RESULTS_PATH.name}`.")
        return

    metrics = compute_train_calibration_metrics(calibration_rows, train_cases)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{metrics['precision']:.1%}")
    m2.metric("Recall", f"{metrics['recall']:.1%}")
    m3.metric("Auto-submit rate", f"{metrics['auto_submit_rate']:.1%}")
    m4.metric("Review rate", f"{metrics['review_rate']:.1%}")

    st.caption(
        f"Based on {int(metrics['total_cases'])} TRAIN cases with labels from train.jsonl. "
        f"Auto-submit: {int(metrics['auto_submit_count'])} | Review: {int(metrics['review_count'])}"
    )


def main() -> None:
    st.set_page_config(page_title="Chargeback Evidence Responder", layout="wide")
    st.title("Chargeback Evidence Responder")
    st.caption("Offline demo dashboard — no API calls")

    audit_by_case = load_audit_by_case()
    train_cases = load_train_cases()
    calibration_rows = load_jsonl(CALIBRATION_RESULTS_PATH)

    tab_case, tab_metrics = st.tabs(["Case Walkthrough", "Metrics"])

    with tab_case:
        render_case_tab(audit_by_case, train_cases)

    with tab_metrics:
        render_metrics_tab(calibration_rows, train_cases)


if __name__ == "__main__":
    main()
