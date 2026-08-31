"""
Evaluation Runner - Phase 4 (app/eval/run_eval.py).

Composes the end-to-end evaluation pipeline over dataset splits:
  dataset -> retrieval -> decision agent -> action gate -> audit logger -> metrics

PRIVACY & EVALUATION ISOLATION CONTRACT:
- Ground-truth evaluation fields (label_winnable, ground_truth_rationale,
  _evidence_docs_obj, quality) are stripped before sending case data to retrieval
  or decision-agent components.
- In-memory evaluation records (containing predictions + label_winnable) are passed
  only to compute_evaluation_metrics after execution.

HONEST EVALUATION RULE:
- By default (require_live_llm=True), if any case uses the heuristic fallback reasoner,
  evaluation halts with a RuntimeError.
- Local smoke testing with fallback requires explicit opt-in (allow_fallback=True).

THRESHOLD CALIBRATION RULE:
- Threshold calibration is permitted ONLY on the 'train' split.
- It calculates gate metrics across candidate thresholds without mutating config.py or .env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal, Tuple

from pydantic import BaseModel, Field

from app.agent.action_gate import apply_action_gate, ActionGateOutput
from app.audit.logger import log_decision_from_dicts, DEFAULT_AUDIT_LOG_PATH
from app.eval.metrics import compute_evaluation_metrics, EvaluationMetricsSummary

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR: Path = _PROJECT_ROOT / "app" / "data" / "datasets"

SAFE_DISPUTE_FIELDS = (
    "case_id",
    "transaction_id",
    "merchant_category",
    "dispute_reason_code",
    "dispute_amount",
    "dispute_raised_date",
    "response_deadline",
    "customer_claim_text",
)

BANNED_EVAL_FIELDS = (
    "label_winnable",
    "ground_truth_rationale",
    "_evidence_docs_obj",
    "quality",
)


class EvaluationRunSummary(BaseModel):
    """
    Structured summary of an evaluation run over a dataset split.
    """
    split: str = Field(..., description="Dataset split evaluated: 'train' or 'holdout'")
    total_evaluated: int = Field(..., ge=0, description="Total number of cases evaluated")
    metrics: EvaluationMetricsSummary = Field(..., description="Computed classification and cost metrics")
    audit_log_path: str = Field(..., description="Path to the JSONL audit log produced during the run")
    fallback_count: int = Field(..., ge=0, description="Number of cases that used heuristic fallback")
    fallback_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of cases that used heuristic fallback")
    require_live_llm: bool = Field(..., description="Whether live LLM execution was required for this run")
    allow_fallback: bool = Field(..., description="Whether fallback reasoner execution was explicitly allowed")
    used_fallback_status: str = Field(
        ..., description="'all_llm' if 0 fallbacks used; 'contains_fallback' if fallback allowed and used"
    )


class ThresholdCandidateResult(BaseModel):
    """
    Metrics breakdown for one candidate confidence threshold during calibration.
    """
    threshold: float = Field(..., ge=0.0, le=1.0, description="Candidate auto-submit confidence threshold")
    auto_submit_count: int = Field(..., ge=0, description="Cases auto-submitted at this threshold")
    auto_submit_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction auto-submitted")
    auto_submit_precision: float = Field(..., ge=0.0, le=1.0, description="Precision of auto-submitted cases")
    review_count: int = Field(..., ge=0, description="Cases flagged for human review")
    review_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction flagged for review")
    false_positive_count: int = Field(..., ge=0, description="False positives under this threshold")
    false_positive_exposure: float = Field(..., ge=0.0, description="FP disputed amount sum")
    precision: float = Field(..., ge=0.0, le=1.0, description="Overall precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Overall recall")
    f1: float = Field(..., ge=0.0, le=1.0, description="Overall F1 score")


class ThresholdCalibrationReport(BaseModel):
    """
    Summary report comparing candidate auto-submit confidence thresholds on TRAIN split.
    """
    split: str = Field("train", description="Split used for calibration; MUST be 'train'")
    total_cases: int = Field(..., ge=0, description="Total TRAIN cases evaluated")
    candidates: List[ThresholdCandidateResult] = Field(..., description="Metrics per candidate threshold")


def make_safe_dispute_payload(raw_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs a clean dispute dictionary containing ONLY non-ground-truth fields.

    Strips label_winnable, ground_truth_rationale, _evidence_docs_obj, and quality.
    """
    safe_dict = {}
    for field in SAFE_DISPUTE_FIELDS:
        if field in raw_case:
            safe_dict[field] = raw_case[field]

    for banned in BANNED_EVAL_FIELDS:
        if banned in safe_dict:
            safe_dict.pop(banned, None)

    return safe_dict


def load_dataset_split(
    split: str = "train",
    datasets_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Loads dataset cases from a JSONL file for the given split.

    Args:
        split: 'train' or 'holdout'
        datasets_dir: Directory containing split JSONL files (defaults to app/data/datasets)

    Returns:
        List of raw dispute case dicts from JSONL.
    """
    split_name = str(split).lower().strip()
    if split_name not in ("train", "holdout"):
        raise ValueError(f"Invalid split '{split}'. Split must be 'train' or 'holdout'.")

    base_dir = Path(datasets_dir) if datasets_dir is not None else DEFAULT_DATASETS_DIR
    split_file = base_dir / f"{split_name}.jsonl"

    if not split_file.exists():
        raise FileNotFoundError(f"Dataset split file not found: {split_file}")

    cases = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                cases.append(json.loads(line_str))

    return cases


def _execute_evaluation_pipeline(
    split: str,
    retriever: Optional[Any] = None,
    decision_agent: Optional[Any] = None,
    datasets_dir: Optional[Union[str, Path]] = None,
    audit_log_path: Optional[Union[str, Path]] = None,
    require_live_llm: bool = True,
    allow_fallback: bool = False,
    cost_multiplier: float = 1.0,
    fixed_fee: float = 0.0,
) -> Tuple[List[Dict[str, Any]], EvaluationMetricsSummary, int, Path, int]:
    """
    Internal helper that executes the full safe pipeline over a dataset split
    and retains in-memory evaluation records for internal callers.
    """
    split_name = str(split).lower().strip()
    if split_name not in ("train", "holdout"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'holdout'.")

    if retriever is None:
        from app.retrieval.retriever import EvidenceRetriever
        retriever = EvidenceRetriever()

    if decision_agent is None:
        from app.agent.decision_agent import DecisionAgent
        decision_agent = DecisionAgent()

    raw_cases = load_dataset_split(split=split_name, datasets_dir=datasets_dir)
    target_audit_path = Path(audit_log_path) if audit_log_path is not None else DEFAULT_AUDIT_LOG_PATH

    eval_records: List[Dict[str, Any]] = []
    fallback_count = 0

    for idx, raw_case in enumerate(raw_cases, start=1):
        # 1. Build safe dispute payload (zero eval fields)
        safe_dispute = make_safe_dispute_payload(raw_case)

        # 2. Retrieve evidence
        retrieval_output = retriever.retrieve_evidence_for_dispute(safe_dispute)

        # 3. Decision agent evaluation
        decision_response = decision_agent.evaluate_dispute_dict(safe_dispute, retrieval_output)

        # Extract fallback status
        used_fallback = getattr(decision_response, "used_fallback", None)
        if used_fallback is True:
            fallback_count += 1
            if require_live_llm and not allow_fallback:
                case_id_str = safe_dispute.get("case_id", f"case_{idx}")
                raise RuntimeError(
                    f"Evaluation halted on case '{case_id_str}': Fallback reasoner was used. "
                    f"Official evaluation requires a live LLM (require_live_llm=True). "
                    f"To run local smoke tests with fallback enabled, set allow_fallback=True."
                )

        # 4. Action gate routing
        low_coverage = retrieval_output.get("low_coverage", False)
        gate_output: ActionGateOutput = apply_action_gate(decision_response, low_coverage=low_coverage)

        # 5. Write privacy-safe audit record
        log_decision_from_dicts(
            dispute=safe_dispute,
            retrieval_output=retrieval_output,
            decision_response=decision_response,
            gate_output=gate_output,
            log_path=target_audit_path,
        )

        # 6. Create in-memory evaluation record (for metrics computation only)
        eval_records.append({
            "decision": gate_output.decision,
            "confidence": gate_output.confidence,
            "action": gate_output.action,
            "low_coverage": gate_output.low_coverage,
            "used_fallback": used_fallback,
            "label_winnable": bool(raw_case.get("label_winnable", False)),
            "dispute_amount": float(raw_case.get("dispute_amount", 0.0)),
        })

    metrics_summary = compute_evaluation_metrics(
        case_results=eval_records,
        cost_multiplier=cost_multiplier,
        fixed_fee=fixed_fee,
    )

    total_evaluated = len(raw_cases)
    return eval_records, metrics_summary, fallback_count, target_audit_path, total_evaluated


def run_evaluation(
    split: Literal["train", "holdout"] = "train",
    retriever: Optional[Any] = None,
    decision_agent: Optional[Any] = None,
    datasets_dir: Optional[Union[str, Path]] = None,
    audit_log_path: Optional[Union[str, Path]] = None,
    require_live_llm: bool = True,
    allow_fallback: bool = False,
    cost_multiplier: float = 1.0,
    fixed_fee: float = 0.0,
) -> EvaluationRunSummary:
    """
    Executes end-to-end evaluation over a dataset split and returns an aggregate summary.

    Does not expose raw per-case records or labels in the return value.
    """
    eval_records, metrics_summary, fallback_count, target_audit_path, total_evaluated = _execute_evaluation_pipeline(
        split=split,
        retriever=retriever,
        decision_agent=decision_agent,
        datasets_dir=datasets_dir,
        audit_log_path=audit_log_path,
        require_live_llm=require_live_llm,
        allow_fallback=allow_fallback,
        cost_multiplier=cost_multiplier,
        fixed_fee=fixed_fee,
    )

    fallback_rate = (fallback_count / total_evaluated) if total_evaluated > 0 else 0.0
    status_str = "all_llm" if fallback_count == 0 else "contains_fallback"

    return EvaluationRunSummary(
        split=str(split).lower().strip(),
        total_evaluated=total_evaluated,
        metrics=metrics_summary,
        audit_log_path=str(target_audit_path.resolve()),
        fallback_count=fallback_count,
        fallback_rate=round(fallback_rate, 6),
        require_live_llm=require_live_llm,
        allow_fallback=allow_fallback,
        used_fallback_status=status_str,
    )


def calibrate_auto_submit_threshold(
    train_eval_records: List[Dict[str, Any]],
    candidate_thresholds: Optional[List[float]] = None,
    split: str = "train",
    cost_multiplier: float = 1.0,
    fixed_fee: float = 0.0,
) -> ThresholdCalibrationReport:
    """
    Pure threshold-calibration helper over TRAIN split evaluation records.

    REJECTS any split other than 'train' to prevent data snooping / overfitting on holdout.
    """
    split_name = str(split).lower().strip()
    if split_name != "train":
        raise ValueError(
            f"Threshold calibration is permitted ONLY on the 'train' split (got split='{split}'). "
            f"Tuning thresholds on holdout data is strictly prohibited to prevent data snooping."
        )

    if candidate_thresholds is None:
        candidate_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    candidate_results: List[ThresholdCandidateResult] = []
    total_cases = len(train_eval_records)

    for thresh in candidate_thresholds:
        thresh_float = round(float(thresh), 4)

        candidate_eval_records = []
        for rec in train_eval_records:
            decision = str(rec.get("decision", "no_contest")).lower().strip()
            confidence = float(rec.get("confidence", 0.0))
            low_coverage = bool(rec.get("low_coverage", False))

            is_auto_submit = (
                decision == "contest"
                and confidence >= thresh_float
                and not low_coverage
            )
            candidate_action = "auto_submit" if is_auto_submit else "flag_for_review"

            candidate_eval_records.append({
                "decision": decision,
                "confidence": confidence,
                "action": candidate_action,
                "low_coverage": low_coverage,
                "label_winnable": rec.get("label_winnable", False),
                "dispute_amount": rec.get("dispute_amount", 0.0),
                "used_fallback": rec.get("used_fallback", None),
            })

        m = compute_evaluation_metrics(
            candidate_eval_records,
            cost_multiplier=cost_multiplier,
            fixed_fee=fixed_fee,
        )

        candidate_results.append(ThresholdCandidateResult(
            threshold=thresh_float,
            auto_submit_count=m.auto_submit_count,
            auto_submit_rate=m.auto_submit_rate,
            auto_submit_precision=m.auto_submit_precision,
            review_count=m.review_count,
            review_rate=m.review_rate,
            false_positive_count=m.false_positive_count,
            false_positive_exposure=m.false_positive_exposure,
            precision=m.precision,
            recall=m.recall,
            f1=m.f1,
        ))

    return ThresholdCalibrationReport(
        split="train",
        total_cases=total_cases,
        candidates=candidate_results,
    )


def run_train_threshold_calibration(
    datasets_dir: Optional[Union[str, Path]] = None,
    candidate_thresholds: Optional[List[float]] = None,
    audit_log_path: Optional[Union[str, Path]] = None,
    require_live_llm: bool = True,
    allow_fallback: bool = False,
    cost_multiplier: float = 1.0,
    fixed_fee: float = 0.0,
    retriever: Optional[Any] = None,
    decision_agent: Optional[Any] = None,
    split: str = "train",
) -> ThresholdCalibrationReport:
    """
    Runs the full safe evaluation pipeline over the TRAIN split only and performs
    threshold calibration across candidate confidence thresholds.

    Rejects any attempt to specify split != 'train'.
    """
    split_name = str(split).lower().strip()
    if split_name != "train":
        raise ValueError(
            f"run_train_threshold_calibration is permitted ONLY on the 'train' split (got split='{split}'). "
            f"Tuning thresholds on holdout data is strictly prohibited to prevent data snooping."
        )

    eval_records, _, _, _, _ = _execute_evaluation_pipeline(
        split="train",
        retriever=retriever,
        decision_agent=decision_agent,
        datasets_dir=datasets_dir,
        audit_log_path=audit_log_path,
        require_live_llm=require_live_llm,
        allow_fallback=allow_fallback,
        cost_multiplier=cost_multiplier,
        fixed_fee=fixed_fee,
    )

    return calibrate_auto_submit_threshold(
        train_eval_records=eval_records,
        candidate_thresholds=candidate_thresholds,
        split="train",
        cost_multiplier=cost_multiplier,
        fixed_fee=fixed_fee,
    )


def main(
    args_list: Optional[List[str]] = None,
    retriever: Optional[Any] = None,
    decision_agent: Optional[Any] = None,
) -> None:
    """
    CLI entrypoint for evaluation runner.
    """
    parser = argparse.ArgumentParser(
        description="Phase 4 Evaluation Runner - Chargeback Evidence Responder"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--split",
        choices=["train", "holdout"],
        help="Dataset split to evaluate ('train' or 'holdout')",
    )
    group.add_argument(
        "--calibrate-train",
        action="store_true",
        help="Run auto-submit confidence threshold calibration on TRAIN split only",
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Candidate thresholds for calibration (e.g. --thresholds 0.60 0.70 0.80)",
    )
    parser.add_argument(
        "--audit-log-path",
        type=str,
        help="Path for writing JSONL audit log",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        help="Directory containing train.jsonl and holdout.jsonl",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow heuristic fallback reasoner (opt-in for local smoke testing only)",
    )
    parser.add_argument(
        "--cost-multiplier",
        type=float,
        default=1.0,
        help="Cost multiplier for false positive exposure (default 1.0)",
    )
    parser.add_argument(
        "--fixed-fee",
        type=float,
        default=0.0,
        help="Fixed fee per false positive case (default 0.0)",
    )

    args = parser.parse_args(args_list)

    require_live_llm = not args.allow_fallback

    if args.calibrate_train:
        report = run_train_threshold_calibration(
            datasets_dir=args.datasets_dir,
            candidate_thresholds=args.thresholds,
            audit_log_path=args.audit_log_path,
            require_live_llm=require_live_llm,
            allow_fallback=args.allow_fallback,
            cost_multiplier=args.cost_multiplier,
            fixed_fee=args.fixed_fee,
            retriever=retriever,
            decision_agent=decision_agent,
            split="train",
        )
        print(json.dumps(report.model_dump(), indent=2))
    else:
        summary = run_evaluation(
            split=args.split,
            datasets_dir=args.datasets_dir,
            audit_log_path=args.audit_log_path,
            require_live_llm=require_live_llm,
            allow_fallback=args.allow_fallback,
            cost_multiplier=args.cost_multiplier,
            fixed_fee=args.fixed_fee,
            retriever=retriever,
            decision_agent=decision_agent,
        )
        print(json.dumps(summary.model_dump(), indent=2))


if __name__ == "__main__":
    main()
