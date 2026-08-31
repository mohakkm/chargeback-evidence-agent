"""
Evaluation Metrics Module - Phase 4 (app/eval/metrics.py).

Provides pure, deterministic evaluation metric calculations over agent case results.

PRIVACY & ISOLATION CONTRACT:
- Evaluation ground-truth labels (label_winnable) are used strictly inside this
  evaluation module to calculate precision, recall, F1, accuracy, and FP cost.
- Evaluation labels and annotations MUST NEVER be passed upstream to the decision agent,
  retrieval pipeline, action gate, or audit logger.
- Output metrics contain aggregate numerical summaries only and never expose raw
  ground_truth_rationale or evidence content/quality.

ZERO-DENOMINATOR CONVENTIONS:
- All rate and ratio metrics (precision, recall, f1, accuracy, auto_submit_precision,
  coverage_rate, review_rate, auto_submit_rate, fallback_rate) default safely to 0.0
  when the corresponding denominator is zero.

FALSE-POSITIVE COST ESTIMATE:
- Reported via false_positive_exposure (sum of disputed amounts for FP cases) and
  estimated_false_positive_cost (exposure * cost_multiplier + FP_count * fixed_fee).
- SYNTHETIC PROXY NOTE: This metric is a synthetic cost proxy for evaluation
  tradeoff analysis only and does not represent an official Razorpay fee schedule.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class EvalCaseInput(BaseModel):
    """
    Validated representation of one case evaluation result.
    Does not mutate input records.
    """
    decision: str = Field(..., description="'contest' or 'no_contest'")
    label_winnable: bool = Field(..., description="Ground truth: True if winnable, False otherwise")
    dispute_amount: float = Field(default=0.0, ge=0.0, description="Disputed amount in currency units")
    action: str = Field(default="flag_for_review", description="'auto_submit' or 'flag_for_review'")
    low_coverage: bool = Field(default=False, description="True if evidence docs < 2")
    used_fallback: Optional[bool] = Field(default=None, description="True if fallback reasoner was used")


class EvaluationMetricsSummary(BaseModel):
    """
    Comprehensive, validated metrics summary for Phase 4 evaluation reporting.
    """
    total_cases: int = Field(..., ge=0, description="Total number of evaluated cases")

    # Confusion matrix
    tp: int = Field(..., ge=0, description="True Positives: predicted contest & label_winnable=True")
    fp: int = Field(..., ge=0, description="False Positives: predicted contest & label_winnable=False")
    tn: int = Field(..., ge=0, description="True Negatives: predicted no_contest & label_winnable=False")
    fn: int = Field(..., ge=0, description="False Negatives: predicted no_contest & label_winnable=True")

    # Standard classification metrics
    precision: float = Field(..., ge=0.0, le=1.0, description="TP / (TP + FP), 0.0 if denominator is 0")
    recall: float = Field(..., ge=0.0, le=1.0, description="TP / (TP + FN), 0.0 if denominator is 0")
    f1: float = Field(..., ge=0.0, le=1.0, description="2 * P * R / (P + R), 0.0 if denominator is 0")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="(TP + TN) / total_cases, 0.0 if denominator is 0")

    # Retrieval & gate routing metrics
    evidence_coverage_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of cases with low_coverage=False"
    )
    review_count: int = Field(..., ge=0, description="Count of cases routed to flag_for_review")
    review_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of cases routed to flag_for_review")
    auto_submit_count: int = Field(..., ge=0, description="Count of cases routed to auto_submit")
    auto_submit_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of cases routed to auto_submit")
    auto_submit_precision: float = Field(
        ..., ge=0.0, le=1.0,
        description="Precision restricted strictly to auto_submit decisions. 0.0 if auto_submit_count is 0"
    )

    # Cost & financial exposure metrics
    false_positive_count: int = Field(..., ge=0, description="Count of false positives (equals FP)")
    false_positive_exposure: float = Field(
        ..., ge=0.0, description="Sum of dispute_amount for all false positive cases"
    )
    estimated_false_positive_cost: float = Field(
        ..., ge=0.0,
        description="Estimated cost = (exposure * cost_multiplier) + (fp_count * fixed_fee)"
    )
    cost_multiplier_used: float = Field(..., ge=0.0, description="Cost multiplier parameter used in estimation")
    fixed_fee_used: float = Field(..., ge=0.0, description="Fixed fee parameter used in estimation")

    # Fallback metrics
    fallback_count: int = Field(..., ge=0, description="Count of decisions produced by fallback reasoner")
    fallback_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of cases using fallback reasoner")


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to access dict key or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def compute_evaluation_metrics(
    case_results: List[Union[Dict[str, Any], Any]],
    cost_multiplier: float = 1.0,
    fixed_fee: float = 0.0,
) -> EvaluationMetricsSummary:
    """
    Computes deterministic metrics over a list of case evaluation results.

    Args:
        case_results: List of dicts or objects containing at minimum:
                      `decision` ("contest"/"no_contest"),
                      `label_winnable` (bool),
                      `dispute_amount` (float),
                      `action` ("auto_submit"/"flag_for_review"),
                      `low_coverage` (bool),
                      `used_fallback` (Optional[bool]).
        cost_multiplier: Multiplier applied to FP dispute amount exposure (default 1.0).
        fixed_fee: Fixed monetary fee penalty per FP case (default 0.0).

    Returns:
        Validated EvaluationMetricsSummary object.

    Note:
        Does not mutate the input case_results list or items.
        The false positive cost calculation is a synthetic proxy estimation for evaluation
        tradeoff analysis only and does not represent an official Razorpay fee schedule.
    """
    total_cases = len(case_results)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    sufficient_coverage_count = 0
    review_count = 0
    auto_submit_count = 0

    auto_submit_tp = 0
    auto_submit_fp = 0

    fp_exposure = 0.0
    fallback_count = 0

    for item in case_results:
        # Extract fields without mutating input item
        raw_decision = str(_get_val(item, "decision", "no_contest")).lower().strip()
        label_winnable = bool(_get_val(item, "label_winnable", False))
        dispute_amount = float(_get_val(item, "dispute_amount", 0.0))
        raw_action = str(_get_val(item, "action", "flag_for_review")).lower().strip()
        low_coverage = bool(_get_val(item, "low_coverage", False))
        used_fallback = _get_val(item, "used_fallback", None)

        is_predicted_contest = raw_decision == "contest"

        # Confusion matrix
        if is_predicted_contest and label_winnable:
            tp += 1
        elif is_predicted_contest and not label_winnable:
            fp += 1
            fp_exposure += dispute_amount
        elif not is_predicted_contest and not label_winnable:
            tn += 1
        else:  # not predicted contest and label_winnable
            fn += 1

        # Coverage
        if not low_coverage:
            sufficient_coverage_count += 1

        # Action routing
        if raw_action == "auto_submit":
            auto_submit_count += 1
            if label_winnable:
                auto_submit_tp += 1
            else:
                auto_submit_fp += 1
        else:
            review_count += 1

        # Fallback tracking
        if used_fallback is True:
            fallback_count += 1

    # Standard classification metrics with zero-denominator safety
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / total_cases) if total_cases > 0 else 0.0

    # Coverage, routing, and auto-submit metrics
    coverage_rate = (sufficient_coverage_count / total_cases) if total_cases > 0 else 0.0
    review_rate = (review_count / total_cases) if total_cases > 0 else 0.0
    auto_submit_rate = (auto_submit_count / total_cases) if total_cases > 0 else 0.0
    auto_submit_precision = (auto_submit_tp / auto_submit_count) if auto_submit_count > 0 else 0.0

    # FP cost calculation (synthetic proxy)
    estimated_fp_cost = (fp_exposure * cost_multiplier) + (fp * fixed_fee)

    # Fallback rate
    fallback_rate = (fallback_count / total_cases) if total_cases > 0 else 0.0

    return EvaluationMetricsSummary(
        total_cases=total_cases,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        accuracy=round(accuracy, 6),
        evidence_coverage_rate=round(coverage_rate, 6),
        review_count=review_count,
        review_rate=round(review_rate, 6),
        auto_submit_count=auto_submit_count,
        auto_submit_rate=round(auto_submit_rate, 6),
        auto_submit_precision=round(auto_submit_precision, 6),
        false_positive_count=fp,
        false_positive_exposure=round(fp_exposure, 2),
        estimated_false_positive_cost=round(estimated_fp_cost, 2),
        cost_multiplier_used=float(cost_multiplier),
        fixed_fee_used=float(fixed_fee),
        fallback_count=fallback_count,
        fallback_rate=round(fallback_rate, 6),
    )
