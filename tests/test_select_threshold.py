"""Tests for TRAIN-only threshold selection from calibration_results.jsonl."""

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.select_threshold import (
    LOCKED_AUTO_SUBMIT_THRESHOLD,
    load_calibration_with_labels,
    precision_wilson_at_threshold,
    run_threshold_selection,
    select_first_excluding_threshold,
    sweep_auto_submit_thresholds,
)


def test_threshold_selection_on_real_calibration():
    report = run_threshold_selection()
    assert report.total_calibration_rows == 90
    assert report.eligible_count == 21
    assert report.selected_threshold == LOCKED_AUTO_SUBMIT_THRESHOLD
    assert report.selected_precision >= 0.90
    assert report.selected_coverage < 1.0
    assert report.precision_at_selected.n == 16
    assert report.precision_at_selected.successes == 15
    assert "95% CI" in report.precision_at_selected.summary
    assert len(report.sweep) == 10


def test_synthetic_eligible_precision_sweep():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cal_path = tmp / "cal.jsonl"
        train_path = tmp / "train.jsonl"

        rows = [
            {"case_id": "A", "decision": "contest", "confidence": 0.9, "low_coverage": False},
            {"case_id": "B", "decision": "contest", "confidence": 0.85, "low_coverage": False},
            {"case_id": "C", "decision": "contest", "confidence": 0.6, "low_coverage": False},
            {"case_id": "D", "decision": "no_contest", "confidence": 0.95, "low_coverage": False},
        ]
        cal_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        train_path.write_text(
            "\n".join(
                json.dumps({"case_id": cid, "label_winnable": win})
                for cid, win in [("A", True), ("B", True), ("C", False), ("D", True)]
            )
            + "\n",
            encoding="utf-8",
        )

        joined, _ = load_calibration_with_labels(cal_path, train_path)
        sweep, eligible = sweep_auto_submit_thresholds(joined, thresholds=[0.50, 0.80, 0.90])
        assert len(eligible) == 3

        row_50 = next(r for r in sweep if r.threshold == 0.50)
        assert row_50.auto_submit_count == 3
        assert abs(row_50.precision - (2 / 3)) < 0.001

        row_80 = next(r for r in sweep if r.threshold == 0.80)
        assert row_80.auto_submit_count == 2
        assert row_80.precision == 1.0

        selected = select_first_excluding_threshold(sweep, min_precision=0.90)
        assert selected.threshold == 0.80


if __name__ == "__main__":
    test_threshold_selection_on_real_calibration()
    test_synthetic_eligible_precision_sweep()
    print("select_threshold tests passed")
