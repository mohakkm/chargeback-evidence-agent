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
from typing import Any, Dict, List, Optional, Set, Union, Literal, Tuple

from pydantic import BaseModel, Field

from app.agent.action_gate import apply_action_gate, ActionGateOutput
from app.audit.logger import log_decision_from_dicts, DEFAULT_AUDIT_LOG_PATH
from app.eval.metrics import compute_evaluation_metrics, EvaluationMetricsSummary

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR: Path = _PROJECT_ROOT / "app" / "data" / "datasets"
DEFAULT_CONSOLIDATED_AUDIT_PATH: Path = _PROJECT_ROOT / "train_audit_consolidated.jsonl"
DEFAULT_HOLDOUT_CONSOLIDATED_AUDIT_PATH: Path = _PROJECT_ROOT / "holdout_audit_consolidated.jsonl"
DEFAULT_HOLDOUT_RESUME_PATH: Path = _PROJECT_ROOT / "holdout_results.jsonl"
DEFAULT_HOLDOUT_AUDIT_LOG_PATH: Path = _PROJECT_ROOT / "audit_logs_holdout.jsonl"
# Interrupted-run audit logs salvaged before resume/checkpoint workflow existed.
SALVAGED_AUDIT_LOG_PATHS: Tuple[Path, ...] = (
    _PROJECT_ROOT / "audit_logs_train_calibration_clean.jsonl",
    _PROJECT_ROOT / "audit_logs_train_calibration_run3.jsonl",
    _PROJECT_ROOT / "audit_logs_train_calibration_final.jsonl",
)

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


def _load_resume_records(
    resume_from_path: Optional[Union[str, Path]],
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """
    Parse a resume JSONL file into case_id -> record (latest timestamp_utc wins).

    Malformed JSON lines are counted, never silently dropped. Prints a stderr summary:
      "resume index: N loaded, M malformed lines skipped, F missing files."

    Returns (index, malformed_line_count, missing_files_count).
    """
    if resume_from_path is None:
        return {}, 0, 0

    p = Path(resume_from_path)
    missing_files = 0
    malformed = 0

    if not p.exists():
        missing_files = 1
        print(
            f"resume index: 0 loaded, 0 malformed lines skipped, {missing_files} missing files.",
            file=sys.stderr,
        )
        return {}, malformed, missing_files

    index: Dict[str, Dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("used_fallback") is False:
                cid = rec.get("case_id")
                if cid:
                    ts = rec.get("timestamp_utc", "")
                    if cid not in index or ts > index[cid].get("timestamp_utc", ""):
                        index[cid] = rec
        except json.JSONDecodeError:
            malformed += 1

    print(
        f"resume index: {len(index)} loaded, {malformed} malformed lines skipped, {missing_files} missing files.",
        file=sys.stderr,
    )
    return index, malformed, missing_files


def load_resume_index(
    resume_from_path: Optional[Union[str, Path]]
) -> Set[str]:
    """
    Loads a JSONL file of previously completed results and returns the set of
    case_ids that were already successfully evaluated by the live LLM
    (used_fallback=False).

    Malformed lines are counted and reported to stderr — they are never silently dropped.
    Missing files are also reported. Summary format:
      "resume index: N loaded, M malformed lines skipped, F missing files."
    """
    index, _, _ = _load_resume_records(resume_from_path)
    return set(index.keys())


def default_consolidation_source_paths(
    resume_from_path: Optional[Union[str, Path]] = None,
    run_audit_log_path: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Build the ordered list of JSONL sources for train audit consolidation:
    salvaged interrupted-run logs, then resume checkpoint, then this run's audit log.
    """
    sources: List[Path] = list(SALVAGED_AUDIT_LOG_PATHS)
    if resume_from_path is not None:
        sources.append(Path(resume_from_path))
    if run_audit_log_path is not None:
        sources.append(Path(run_audit_log_path))
    return sources


def consolidate_audit_trail(
    source_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    expected_case_ids: Optional[Set[str]] = None,
) -> int:
    """
    Merges multiple JSONL audit log files into one canonical file at output_path.

    Deduplication rule: one row per case_id, latest timestamp_utc wins.
    Malformed lines and entries without a case_id are skipped with a stderr warning.

    If expected_case_ids is provided, raises RuntimeError after merging if any
    expected case_id is absent from the output — never emits a partial file silently.

    Returns the number of unique case_id rows written.
    """
    best: Dict[str, Dict[str, Any]] = {}
    total_lines = 0
    malformed = 0

    for src in source_paths:
        src_path = Path(src)
        if not src_path.exists():
            print(
                f"[consolidate_audit_trail] WARNING: source file not found, skipping: {src_path}",
                file=sys.stderr,
            )
            continue
        for line in src_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                print(
                    f"[consolidate_audit_trail] WARNING: malformed line in {src_path.name}, skipping.",
                    file=sys.stderr,
                )
                continue
            cid = rec.get("case_id")
            if not cid:
                print(
                    f"[consolidate_audit_trail] WARNING: line without case_id in {src_path.name}, skipping.",
                    file=sys.stderr,
                )
                continue
            ts = rec.get("timestamp_utc", "")
            if cid not in best or ts > best[cid].get("timestamp_utc", ""):
                best[cid] = rec

    out = Path(output_path)
    # Validate completeness BEFORE writing
    if expected_case_ids is not None:
        missing = sorted(expected_case_ids - set(best.keys()))
        if missing:
            raise RuntimeError(
                f"consolidate_audit_trail: {len(missing)} expected case_id(s) missing from merged output "
                f"— refusing to write a partial file. Missing: {missing}"
            )
        rows = sorted(
            (best[cid] for cid in expected_case_ids),
            key=lambda r: r.get("case_id", ""),
        )
    else:
        rows = sorted(best.values(), key=lambda r: r.get("case_id", ""))
    content = "".join(json.dumps(row) + "\n" for row in rows)

    out.write_text(content, encoding="utf-8")
    print(
        f"[consolidate_audit_trail] Wrote {len(rows)} rows to {out} "
        f"(from {total_lines} total input lines; {malformed} malformed skipped).",
        file=sys.stderr,
    )
    return len(rows)


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
    resume_from_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Dict[str, Any]], EvaluationMetricsSummary, int, Path, int]:
    """
    Internal helper that executes the full safe pipeline over a dataset split
    and retains in-memory evaluation records for internal callers.

    If resume_from_path is provided, case_ids already present in that file with
    used_fallback=False are skipped; their records are replayed into eval_records
    from the resume file so that metrics are computed over the full dataset.
    Each new result is appended to resume_from_path immediately on receipt.
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

    # --- Resume index: map case_id -> completed audit record ---
    resume_index, _, _ = _load_resume_records(resume_from_path)
    if resume_index:
        print(
            f"[run_eval] Resume: {len(resume_index)} case(s) already completed "
            f"(used_fallback=False) — skipping them.",
            file=sys.stderr,
        )

    # Open resume file for unbuffered appending if requested
    resume_append_fh = None
    if resume_from_path is not None:
        resume_append_fh = open(Path(resume_from_path), "a", encoding="utf-8", buffering=1)

    eval_records: List[Dict[str, Any]] = []
    fallback_count = 0

    for idx, raw_case in enumerate(raw_cases, start=1):
        # 1. Build safe dispute payload (zero eval fields)
        safe_dispute = make_safe_dispute_payload(raw_case)
        case_id = safe_dispute.get("case_id", f"case_{idx}")

        # --- Resume skip: replay completed record without calling LLM ---
        if case_id in resume_index:
            completed_rec = resume_index[case_id]
            eval_records.append({
                "decision": completed_rec.get("decision", "no_contest"),
                "confidence": float(completed_rec.get("confidence", 0.0)),
                "action": completed_rec.get("action", "flag_for_review"),
                "low_coverage": bool(completed_rec.get("low_coverage", False)),
                "used_fallback": False,
                "label_winnable": bool(raw_case.get("label_winnable", False)),
                "dispute_amount": float(raw_case.get("dispute_amount", 0.0)),
            })
            continue

        # 2. Retrieve evidence
        retrieval_output = retriever.retrieve_evidence_for_dispute(safe_dispute)

        # 3. Decision agent evaluation
        decision_response = decision_agent.evaluate_dispute_dict(safe_dispute, retrieval_output)

        # Extract fallback status
        used_fallback = getattr(decision_response, "used_fallback", None)
        if used_fallback is True:
            fallback_count += 1
            if require_live_llm and not allow_fallback:
                if resume_append_fh is not None:
                    resume_append_fh.close()
                raise RuntimeError(
                    f"Evaluation halted on case '{case_id}': Fallback reasoner was used. "
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

        # 5b. Append new result immediately to resume file (unbuffered, line=1)
        if resume_append_fh is not None and used_fallback is False:
            from datetime import datetime, timezone
            resume_record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": case_id,
                "transaction_id": safe_dispute.get("transaction_id", ""),
                "dispute_reason_code": safe_dispute.get("dispute_reason_code", ""),
                "retrieved_evidence_ids": retrieval_output.get("retrieved_evidence_ids", []),
                "retrieval_count": retrieval_output.get("retrieval_count", 0),
                "scoped_count": retrieval_output.get("scoped_count", 0),
                "low_coverage": low_coverage,
                "decision": gate_output.decision,
                "confidence": gate_output.confidence,
                "reasoning_summary": getattr(decision_response, "reasoning_summary", ""),
                "action": gate_output.action,
                "used_fallback": False,
            }
            resume_append_fh.write(json.dumps(resume_record) + "\n")

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

    if resume_append_fh is not None:
        resume_append_fh.close()

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
    resume_from_path: Optional[Union[str, Path]] = None,
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
        resume_from_path=resume_from_path,
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


def run_holdout_evaluation(
    datasets_dir: Optional[Union[str, Path]] = None,
    audit_log_path: Optional[Union[str, Path]] = None,
    require_live_llm: bool = True,
    allow_fallback: bool = False,
    cost_multiplier: float = 1.0,
    fixed_fee: float = 0.0,
    retriever: Optional[Any] = None,
    decision_agent: Optional[Any] = None,
    resume_from_path: Optional[Union[str, Path]] = None,
    consolidated_audit_path: Optional[Union[str, Path]] = None,
) -> Tuple[EvaluationRunSummary, List[Dict[str, Any]]]:
    """
    Official HOLDOUT evaluation: full pipeline, resume-aware, consolidates audit log.

    Threshold is NOT tuned here — uses locked AUTO_SUBMIT_THRESHOLD from config.
    """
    eval_records, metrics_summary, fallback_count, target_audit_path, total_evaluated = (
        _execute_evaluation_pipeline(
            split="holdout",
            retriever=retriever,
            decision_agent=decision_agent,
            datasets_dir=datasets_dir,
            audit_log_path=audit_log_path,
            require_live_llm=require_live_llm,
            allow_fallback=allow_fallback,
            cost_multiplier=cost_multiplier,
            fixed_fee=fixed_fee,
            resume_from_path=resume_from_path,
        )
    )

    out_path = (
        Path(consolidated_audit_path)
        if consolidated_audit_path is not None
        else DEFAULT_HOLDOUT_CONSOLIDATED_AUDIT_PATH
    )
    sources: List[Path] = []
    if resume_from_path is not None:
        sources.append(Path(resume_from_path))
    if target_audit_path.exists():
        sources.append(target_audit_path)
    raw_cases = load_dataset_split(split="holdout", datasets_dir=datasets_dir)
    expected_ids: Set[str] = {
        str(c.get("case_id", "")) for c in raw_cases if c.get("case_id")
    }
    consolidate_audit_trail(
        source_paths=sources,
        output_path=out_path,
        expected_case_ids=expected_ids,
    )

    fallback_rate = (fallback_count / total_evaluated) if total_evaluated > 0 else 0.0
    status_str = "all_llm" if fallback_count == 0 else "contains_fallback"

    summary = EvaluationRunSummary(
        split="holdout",
        total_evaluated=total_evaluated,
        metrics=metrics_summary,
        audit_log_path=str(target_audit_path.resolve()),
        fallback_count=fallback_count,
        fallback_rate=round(fallback_rate, 6),
        require_live_llm=require_live_llm,
        allow_fallback=allow_fallback,
        used_fallback_status=status_str,
    )
    return summary, eval_records


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
    resume_from_path: Optional[Union[str, Path]] = None,
    consolidated_audit_path: Optional[Union[str, Path]] = None,
) -> ThresholdCalibrationReport:
    """
    Runs the full safe evaluation pipeline over the TRAIN split only and performs
    threshold calibration across candidate confidence thresholds.

    Rejects any attempt to specify split != 'train'.

    After all train cases complete, consolidate_audit_trail() always runs: it merges
    salvaged interrupted-run logs + resume checkpoint + this run's audit log into
    train_audit_consolidated.jsonl (or consolidated_audit_path if overridden).
    Fails loudly if any expected case_id is absent.
    """
    split_name = str(split).lower().strip()
    if split_name != "train":
        raise ValueError(
            f"run_train_threshold_calibration is permitted ONLY on the 'train' split (got split='{split}'). "
            f"Tuning thresholds on holdout data is strictly prohibited to prevent data snooping."
        )

    eval_records, _, _, target_audit_path, _ = _execute_evaluation_pipeline(
        split="train",
        retriever=retriever,
        decision_agent=decision_agent,
        datasets_dir=datasets_dir,
        audit_log_path=audit_log_path,
        require_live_llm=require_live_llm,
        allow_fallback=allow_fallback,
        cost_multiplier=cost_multiplier,
        fixed_fee=fixed_fee,
        resume_from_path=resume_from_path,
    )

    # --- Consolidation step: one canonical row per train case_id ---
    out_path = (
        Path(consolidated_audit_path)
        if consolidated_audit_path is not None
        else DEFAULT_CONSOLIDATED_AUDIT_PATH
    )
    sources = default_consolidation_source_paths(
        resume_from_path=resume_from_path,
        run_audit_log_path=target_audit_path,
    )
    raw_cases = load_dataset_split(split="train", datasets_dir=datasets_dir)
    expected_ids: Set[str] = {
        str(c.get("case_id", "")) for c in raw_cases if c.get("case_id")
    }
    consolidate_audit_trail(
        source_paths=sources,
        output_path=out_path,
        expected_case_ids=expected_ids,
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
        "--resume-from",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSONL file of previously completed results (e.g. calibration_results.jsonl). "
            "On startup, case_ids already present with used_fallback=False are skipped. "
            "Each new result is appended immediately to this file."
        ),
    )
    parser.add_argument(
        "--consolidated-audit-log",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path for merged canonical audit log. "
            "holdout: holdout_audit_consolidated.jsonl (default). "
            "calibrate-train: train_audit_consolidated.jsonl (default)."
        ),
    )
    parser.add_argument(
        "--official-holdout",
        action="store_true",
        help=(
            "Run the one-shot official HOLDOUT evaluation with resume checkpoint "
            f"(default resume: {DEFAULT_HOLDOUT_RESUME_PATH.name}) and consolidated audit."
        ),
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
    resume_from_path = getattr(args, "resume_from", None)
    consolidated_audit_path = getattr(args, "consolidated_audit_log", None)

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
            resume_from_path=resume_from_path,
            consolidated_audit_path=consolidated_audit_path,
        )
        print(json.dumps(report.model_dump(), indent=2))
    elif getattr(args, "official_holdout", False) or args.split == "holdout":
        holdout_resume = resume_from_path or str(DEFAULT_HOLDOUT_RESUME_PATH)
        holdout_audit = args.audit_log_path or str(DEFAULT_HOLDOUT_AUDIT_LOG_PATH)
        holdout_consolidated = consolidated_audit_path or str(DEFAULT_HOLDOUT_CONSOLIDATED_AUDIT_PATH)
        summary, _ = run_holdout_evaluation(
            datasets_dir=args.datasets_dir,
            audit_log_path=holdout_audit,
            require_live_llm=require_live_llm,
            allow_fallback=args.allow_fallback,
            cost_multiplier=args.cost_multiplier,
            fixed_fee=args.fixed_fee,
            retriever=retriever,
            decision_agent=decision_agent,
            resume_from_path=holdout_resume,
            consolidated_audit_path=holdout_consolidated,
        )
        print(json.dumps(summary.model_dump(), indent=2))
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
            resume_from_path=resume_from_path,
        )
        print(json.dumps(summary.model_dump(), indent=2))


if __name__ == "__main__":
    main()
