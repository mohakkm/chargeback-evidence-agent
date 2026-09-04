"""
Post-hoc HOLDOUT official results report: Wilson CI on auto-submit precision + demo case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.eval.metrics import compute_evaluation_metrics
from app.eval.run_eval import load_dataset_split
from app.eval.select_threshold import wilson_score_interval

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT_RESUME = _PROJECT_ROOT / "holdout_results.jsonl"
DEFAULT_HOLDOUT_CONSOLIDATED = _PROJECT_ROOT / "holdout_audit_consolidated.jsonl"
DEFAULT_TRAIN_CALIBRATION = _PROJECT_ROOT / "calibration_results.jsonl"
DEFAULT_REPORT_OUT = _PROJECT_ROOT / "holdout_official_report.json"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _records_from_results(
    results: List[Dict[str, Any]],
    cases_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    eval_records: List[Dict[str, Any]] = []
    for row in results:
        cid = row.get("case_id")
        if cid not in cases_by_id:
            continue
        hc = cases_by_id[cid]
        eval_records.append({
            "decision": row.get("decision", "no_contest"),
            "confidence": float(row.get("confidence", 0.0)),
            "action": row.get("action", "flag_for_review"),
            "low_coverage": bool(row.get("low_coverage", False)),
            "used_fallback": row.get("used_fallback", False),
            "label_winnable": bool(hc.get("label_winnable", False)),
            "dispute_amount": float(hc.get("dispute_amount", 0.0)),
            "case_id": cid,
        })
    return eval_records


def _headline_metrics(metrics: Any) -> Dict[str, Any]:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "tn": metrics.tn,
        "fn": metrics.fn,
    }


def _decision_distribution(eval_records: List[Dict[str, Any]], winnable_only: bool = False) -> Dict[str, int]:
    pool = [r for r in eval_records if r.get("label_winnable")] if winnable_only else eval_records
    contest = sum(1 for r in pool if str(r.get("decision", "")).lower() == "contest")
    no_contest = sum(1 for r in pool if str(r.get("decision", "")).lower() == "no_contest")
    return {
        "contest": contest,
        "no_contest": no_contest,
        "total": len(pool),
    }


def _false_negative_exposure(eval_records: List[Dict[str, Any]]) -> float:
    return round(
        sum(
            r["dispute_amount"]
            for r in eval_records
            if str(r.get("decision", "")).lower() == "no_contest" and r.get("label_winnable")
        ),
        2,
    )


def build_train_headline_comparison(
    calibration_path: Path = DEFAULT_TRAIN_CALIBRATION,
    datasets_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Full TRAIN decision-vs-label metrics (all 90 cases), same formula as HOLDOUT."""
    results = _load_jsonl(calibration_path)
    train_cases = {
        c["case_id"]: c for c in load_dataset_split("train", datasets_dir=datasets_dir)
    }
    eval_records = _records_from_results(results, train_cases)
    metrics = compute_evaluation_metrics(eval_records)
    return {
        "split": "train",
        "source": str(calibration_path.name),
        "total_cases": len(eval_records),
        "headline_metrics": _headline_metrics(metrics),
    }


def build_holdout_report(
    holdout_results_path: Path = DEFAULT_HOLDOUT_RESUME,
    consolidated_audit_path: Path = DEFAULT_HOLDOUT_CONSOLIDATED,
    datasets_dir: Optional[Path] = None,
    cost_multiplier: float = 1.0,
    fixed_fee: float = 0.0,
) -> Dict[str, Any]:
    results = _load_jsonl(holdout_results_path)
    holdout_cases = {
        c["case_id"]: c for c in load_dataset_split("holdout", datasets_dir=datasets_dir)
    }
    eval_records = _records_from_results(results, holdout_cases)

    metrics = compute_evaluation_metrics(
        eval_records,
        cost_multiplier=cost_multiplier,
        fixed_fee=fixed_fee,
    )

    auto_submitted = [r for r in eval_records if r.get("action") == "auto_submit"]
    n_auto = len(auto_submitted)
    successes = sum(1 for r in auto_submitted if r.get("label_winnable"))
    p_hat, ci_low, ci_high = wilson_score_interval(successes, n_auto)
    wilson_summary = (
        f"precision {p_hat * 100:.1f}%, 95% CI [{ci_low * 100:.1f}%, {ci_high * 100:.1f}%], n={n_auto}"
    )

    demo_case_id: Optional[str] = None
    demo_reason: Optional[str] = None
    for row in sorted(eval_records, key=lambda r: r.get("case_id", "")):
        if row.get("action") != "flag_for_review":
            continue
        conf = float(row.get("confidence", 0.0))
        low_cov = bool(row.get("low_coverage", False))
        decision = str(row.get("decision", ""))
        if low_cov:
            demo_case_id = row["case_id"]
            demo_reason = (
                f"low_coverage=True with confidence {conf:.2f} — gate correctly blocked auto-submit "
                f"despite decision={decision!r}."
            )
            break
        if decision == "contest" and conf < 0.70:
            demo_case_id = row["case_id"]
            demo_reason = (
                f"confidence {conf:.2f} below locked threshold 0.70 — gate correctly routed to review."
            )
            break
    if demo_case_id is None:
        for row in eval_records:
            if row.get("action") == "flag_for_review" and row.get("decision") == "no_contest":
                demo_case_id = row["case_id"]
                demo_reason = (
                    f"no_contest at confidence {float(row.get('confidence', 0)):.2f} — "
                    f"correctly flagged for human review, not auto-submit."
                )
                break

    consolidated_count = len(_load_jsonl(consolidated_audit_path)) if consolidated_audit_path.exists() else 0
    fn_exposure = _false_negative_exposure(eval_records)

    return {
        "split": "holdout",
        "total_cases": len(eval_records),
        "consolidated_audit_rows": consolidated_count,
        "train_headline_comparison": build_train_headline_comparison(datasets_dir=datasets_dir),
        "holdout_headline_metrics": _headline_metrics(metrics),
        "metrics": metrics.model_dump(),
        "false_negative_count": metrics.fn,
        "false_negative_exposure": fn_exposure,
        "false_negative_cost_note": (
            "Sum of dispute_amount for FN cases (predicted no_contest, label_winnable=True) — "
            "revenue left uncontested that should have been contested. Mirror of FP exposure proxy."
        ),
        "decision_distribution": {
            "all_holdout_cases": _decision_distribution(eval_records, winnable_only=False),
            "label_winnable_only": _decision_distribution(eval_records, winnable_only=True),
        },
        "auto_submit_precision_wilson": {
            "precision": round(p_hat, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "n": n_auto,
            "successes": successes,
            "summary": wilson_summary,
        },
        "demo_case": {
            "case_id": demo_case_id,
            "reason": demo_reason,
        },
        "fallback_count": metrics.fallback_count,
        "fallback_rate": metrics.fallback_rate,
    }


def main() -> None:
    report = build_holdout_report()
    DEFAULT_REPORT_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
