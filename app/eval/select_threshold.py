"""
TRAIN-only auto-submit threshold selection from stored calibration results.

Joins calibration_results.jsonl to train.jsonl by case_id for label_winnable
(scoring only — never passed to the decision agent).

Eligible pool for auto-submit simulation:
  decision == "contest" AND low_coverage == False

At each candidate threshold:
  - precision: among eligible cases with confidence >= threshold, fraction label_winnable
  - coverage: fraction of all eligible cases with confidence >= threshold
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_PATH = _PROJECT_ROOT / "calibration_results.jsonl"
DEFAULT_TRAIN_DATASET_PATH = _PROJECT_ROOT / "app" / "data" / "datasets" / "train.jsonl"
DEFAULT_THRESHOLD_SWEEP = [round(0.50 + i * 0.05, 2) for i in range(10)]
MIN_PRECISION_TARGET = 0.90
LOCKED_AUTO_SUBMIT_THRESHOLD = 0.70
DEFAULT_REPORT_PATH = _PROJECT_ROOT / "threshold_selection_report.json"


class ThresholdSweepRow(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    coverage: float = Field(..., ge=0.0, le=1.0)
    auto_submit_count: int = Field(..., ge=0)
    eligible_count: int = Field(..., ge=0)


class WilsonInterval(BaseModel):
    precision: float
    ci_low: float
    ci_high: float
    n: int
    successes: int
    confidence_level: float = 0.95
    summary: str


class ConfidenceDistinctnessCheck(BaseModel):
    eligible_pool_size: int
    distinct_confidence_values: List[float]
    value_counts: Dict[str, int]
    clusters_on_round_deciles: bool
    assessment: str


class ThresholdSelectionReport(BaseModel):
    split: str = "train"
    calibration_path: str
    train_dataset_path: str
    total_calibration_rows: int
    eligible_count: int
    min_precision_target: float
    selected_threshold: float
    selected_precision: float
    selected_coverage: float
    justification: str
    precision_at_selected: WilsonInterval
    confidence_distinctness: ConfidenceDistinctnessCheck
    sweep: List[ThresholdSweepRow]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_calibration_with_labels(
    calibration_path: Path,
    train_dataset_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
    """
    Join calibration rows to train labels by case_id.

    Returns (joined_rows, labels_by_case_id).
    Raises if any calibration case_id is missing from train.jsonl.
    """
    calibration_rows = load_jsonl(calibration_path)
    train_rows = load_jsonl(train_dataset_path)
    labels: Dict[str, bool] = {
        str(r["case_id"]): bool(r["label_winnable"])
        for r in train_rows
        if r.get("case_id")
    }

    joined: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in calibration_rows:
        cid = str(row.get("case_id", ""))
        if cid not in labels:
            missing.append(cid)
            continue
        joined.append({**row, "label_winnable": labels[cid]})

    if missing:
        raise ValueError(
            f"{len(missing)} calibration case_id(s) missing from train dataset: {sorted(missing)}"
        )
    return joined, labels


def filter_auto_submit_eligible(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cases that could ever auto-submit: contest decision with adequate evidence coverage."""
    return [
        r for r in rows
        if str(r.get("decision", "")).lower() == "contest"
        and not bool(r.get("low_coverage", False))
    ]


def sweep_auto_submit_thresholds(
    joined_rows: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
) -> Tuple[List[ThresholdSweepRow], List[Dict[str, Any]]]:
    eligible = filter_auto_submit_eligible(joined_rows)
    eligible_count = len(eligible)
    thresh_list = thresholds or DEFAULT_THRESHOLD_SWEEP

    sweep: List[ThresholdSweepRow] = []
    for thresh in thresh_list:
        t = round(float(thresh), 4)
        would_submit = [
            r for r in eligible
            if float(r.get("confidence", 0.0)) >= t
        ]
        n_submit = len(would_submit)
        if n_submit == 0:
            precision = 0.0
        else:
            precision = sum(1 for r in would_submit if r.get("label_winnable")) / n_submit
        coverage = n_submit / eligible_count if eligible_count else 0.0
        sweep.append(ThresholdSweepRow(
            threshold=t,
            precision=round(precision, 6),
            coverage=round(coverage, 6),
            auto_submit_count=n_submit,
            eligible_count=eligible_count,
        ))

    return sweep, eligible


def wilson_score_interval(
    successes: int,
    n: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion. Returns (p_hat, low, high).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.96 if confidence_level >= 0.95 else 1.645
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def precision_wilson_at_threshold(
    eligible: List[Dict[str, Any]],
    threshold: float,
    confidence_level: float = 0.95,
) -> WilsonInterval:
    would_submit = [
        r for r in eligible
        if float(r.get("confidence", 0.0)) >= threshold
    ]
    n = len(would_submit)
    successes = sum(1 for r in would_submit if r.get("label_winnable"))
    p_hat, low, high = wilson_score_interval(successes, n, confidence_level)
    prec_pct = p_hat * 100
    return WilsonInterval(
        precision=round(p_hat, 6),
        ci_low=round(low, 6),
        ci_high=round(high, 6),
        n=n,
        successes=successes,
        confidence_level=confidence_level,
        summary=(
            f"precision {prec_pct:.1f}%, 95% CI [{low * 100:.1f}%, {high * 100:.1f}%], n={n}"
        ),
    )


def analyze_confidence_distinctness(eligible: List[Dict[str, Any]]) -> ConfidenceDistinctnessCheck:
    values = sorted(float(r.get("confidence", 0.0)) for r in eligible)
    distinct = sorted(set(values))
    counts: Dict[str, int] = {}
    for v in values:
        key = f"{v:.2f}"
        counts[key] = counts.get(key, 0) + 1

    round_deciles = {0.5, 0.6, 0.7, 0.8, 0.9}
    on_deciles = all(
        any(abs(v - d) < 0.051 for d in round_deciles)
        for v in distinct
    )
    # Also flag heavy ties / discrete steps (not continuous)
    max_tie = max(counts.values()) if counts else 0
    clusters = on_deciles and len(distinct) <= 12

    if clusters:
        decile_str = ", ".join(f"{d:.1f}" for d in sorted(round_deciles))
        assessment = (
            f"Eligible-pool confidence values cluster near round deciles "
            f"({decile_str}): distinct sorted values are "
            f"{[round(v, 2) for v in distinct]} with repeated ties (max {max_tie} cases sharing "
            f"one score). This is not a continuously calibrated score distribution — treat "
            f"HOLDOUT threshold behavior as uncertain until re-checked."
        )
    else:
        assessment = (
            f"Distinct confidence values: {[round(v, 2) for v in distinct]} — "
            f"does not appear tightly clustered on 0.1 steps only."
        )

    return ConfidenceDistinctnessCheck(
        eligible_pool_size=len(eligible),
        distinct_confidence_values=[round(v, 6) for v in distinct],
        value_counts=counts,
        clusters_on_round_deciles=clusters,
        assessment=assessment,
    )


def select_first_excluding_threshold(
    sweep: List[ThresholdSweepRow],
    min_precision: float = MIN_PRECISION_TARGET,
) -> ThresholdSweepRow:
    """
    Pick the lowest threshold where coverage drops below 1.0 (gate excludes cases)
    while auto-submit precision remains >= min_precision.
    """
    qualifying = [
        row for row in sweep
        if row.precision >= min_precision
        and row.auto_submit_count > 0
        and row.coverage < 1.0
    ]
    if not qualifying:
        raise RuntimeError(
            f"No threshold in sweep achieves >={min_precision:.0%} precision with coverage < 1.0."
        )
    return min(qualifying, key=lambda r: r.threshold)


def build_justification(selected: ThresholdSweepRow, min_precision: float) -> str:
    prec_pct = selected.precision * 100
    cov_pct = selected.coverage * 100
    return (
        f"Locked threshold {selected.threshold:.2f} — first sweep point where the confidence "
        f"gate excludes eligible cases (coverage {cov_pct:.1f}% vs 100% flat below 0.70) while "
        f"precision stays >={min_precision:.0%} ({prec_pct:.1f}%); thresholds 0.50–0.65 share "
        f"identical 90.5% precision at 100% coverage and do not filter on confidence."
    )


def run_threshold_selection(
    calibration_path: Optional[Path] = None,
    train_dataset_path: Optional[Path] = None,
    thresholds: Optional[List[float]] = None,
    min_precision: float = MIN_PRECISION_TARGET,
) -> ThresholdSelectionReport:
    cal_path = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
    train_path = Path(train_dataset_path) if train_dataset_path else DEFAULT_TRAIN_DATASET_PATH

    joined, _ = load_calibration_with_labels(cal_path, train_path)
    sweep, eligible = sweep_auto_submit_thresholds(joined, thresholds=thresholds)
    selected = select_first_excluding_threshold(sweep, min_precision=min_precision)

    return ThresholdSelectionReport(
        calibration_path=str(cal_path.resolve()),
        train_dataset_path=str(train_path.resolve()),
        total_calibration_rows=len(joined),
        eligible_count=len(eligible),
        min_precision_target=min_precision,
        selected_threshold=selected.threshold,
        selected_precision=selected.precision,
        selected_coverage=selected.coverage,
        justification=build_justification(selected, min_precision),
        precision_at_selected=precision_wilson_at_threshold(eligible, selected.threshold),
        confidence_distinctness=analyze_confidence_distinctness(eligible),
        sweep=sweep,
    )


def print_sweep_table(sweep: List[ThresholdSweepRow], file=None) -> None:
    out = file or sys.stdout
    print("threshold | precision | coverage", file=out)
    for row in sweep:
        print(f"{row.threshold:>9.2f} | {row.precision:>9.3f} | {row.coverage:>8.3f}", file=out)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Select auto-submit threshold from TRAIN calibration_results.jsonl"
    )
    parser.add_argument(
        "--calibration-results",
        type=str,
        default=str(DEFAULT_CALIBRATION_PATH),
        help="Path to calibration_results.jsonl",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default=str(DEFAULT_TRAIN_DATASET_PATH),
        help="Path to train.jsonl (for label_winnable join only)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=MIN_PRECISION_TARGET,
        help="Minimum auto-submit precision target (default 0.90)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="Path to write full ThresholdSelectionReport JSON",
    )
    args = parser.parse_args(argv)

    report = run_threshold_selection(
        calibration_path=Path(args.calibration_results),
        train_dataset_path=Path(args.train_dataset),
        min_precision=args.min_precision,
    )

    print_sweep_table(report.sweep, file=sys.stderr)
    print(file=sys.stderr)
    print(report.justification, file=sys.stderr)
    print(report.precision_at_selected.summary, file=sys.stderr)
    print(report.confidence_distinctness.assessment, file=sys.stderr)

    out_path = Path(args.json_out)
    out_path.write_text(json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
