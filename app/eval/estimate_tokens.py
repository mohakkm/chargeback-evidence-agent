"""
Rough token-cost estimate for TRAIN calibration runs (no live API calls).

Reconstructs decision-agent prompt sizes from train.jsonl + calibration_results.jsonl
and estimates completion tokens from stored reasoning summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent.decision_agent import DecisionAgent, sanitize_dispute_input, sanitize_evidence_input
from app.eval.select_threshold import DEFAULT_CALIBRATION_PATH, DEFAULT_TRAIN_DATASET_PATH, load_jsonl

HOLDOUT_CASE_COUNT = 30
_CHARS_PER_TOKEN = 4.0  # conservative heuristic when tiktoken unavailable


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_case_tokens(
    train_case: Dict[str, Any],
    calibration_row: Dict[str, Any],
) -> Dict[str, int]:
    agent = DecisionAgent(api_key="dummy")
    dispute = sanitize_dispute_input(train_case)
    evidence_ids = set(calibration_row.get("retrieved_evidence_ids") or [])
    raw_evidence = [
        doc for doc in (train_case.get("_evidence_docs_obj") or [])
        if doc.get("evidence_id") in evidence_ids
    ]
    evidence = sanitize_evidence_input(raw_evidence)
    low_coverage = bool(calibration_row.get("low_coverage", False))

    system_prompt = agent._build_system_prompt(compact=False)
    user_prompt = agent._build_user_prompt(dispute, evidence, low_coverage)

    # Completion: reasoning stored; rebuttal not in calibration log — estimate from decision
    reasoning = str(calibration_row.get("reasoning_summary", ""))
    decision = str(calibration_row.get("decision", "no_contest"))
    # rebuttal_draft not stored in calibration_results — typical contest JSON output ~120-200 tokens
    rebuttal_placeholder = (
        "REBUTTAL STATEMENT. Merchant records demonstrate authorization and fulfillment. "
        "Evidence cited per card network representment guidelines. "
        * 8
    ) if decision == "contest" else ""
    completion_text = reasoning + ("\n" + rebuttal_placeholder if rebuttal_placeholder else "")

    prompt_tokens = _count_tokens(system_prompt) + _count_tokens(user_prompt)
    completion_tokens = _count_tokens(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def estimate_train_token_usage(
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    train_dataset_path: Path = DEFAULT_TRAIN_DATASET_PATH,
    api_calls_per_case: float = 1.0,
) -> Dict[str, Any]:
    """
    Average tokens per TRAIN case from stored calibration artifacts.

    api_calls_per_case: multiplier for retries (1.0 = one successful call per case on TRAIN).
    """
    calibration_rows = load_jsonl(calibration_path)
    train_by_id = {
        str(r["case_id"]): r for r in load_jsonl(train_dataset_path) if r.get("case_id")
    }

    per_case: List[Dict[str, int]] = []
    for row in calibration_rows:
        cid = str(row.get("case_id", ""))
        if cid not in train_by_id:
            continue
        per_case.append(estimate_case_tokens(train_by_id[cid], row))

    n = len(per_case)
    if n == 0:
        raise ValueError("No overlapping calibration/train cases for token estimate.")

    avg_prompt = sum(p["prompt_tokens"] for p in per_case) / n
    avg_completion = sum(p["completion_tokens"] for p in per_case) / n
    avg_per_call = (avg_prompt + avg_completion)
    avg_per_case_with_retries = avg_per_call * api_calls_per_case

    holdout_estimate = avg_per_case_with_retries * HOLDOUT_CASE_COUNT

    return {
        "train_cases": n,
        "avg_prompt_tokens_per_call": round(avg_prompt, 1),
        "avg_completion_tokens_per_call": round(avg_completion, 1),
        "avg_total_tokens_per_call": round(avg_per_call, 1),
        "api_calls_per_case_assumed": api_calls_per_case,
        "avg_total_tokens_per_case": round(avg_per_case_with_retries, 1),
        "holdout_cases": HOLDOUT_CASE_COUNT,
        "estimated_holdout_total_tokens": round(holdout_estimate, 0),
        "note": (
            "Completion tokens use stored reasoning_summary plus a fixed contest rebuttal "
            "estimate (rebuttal not stored in calibration_results.jsonl). "
            "TRAIN run had used_fallback=false for all 90 cases — api_calls_per_case=1.0."
        ),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate token usage from TRAIN calibration logs")
    parser.add_argument("--calibration-results", type=str, default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--train-dataset", type=str, default=str(DEFAULT_TRAIN_DATASET_PATH))
    parser.add_argument(
        "--api-calls-per-case",
        type=float,
        default=1.0,
        help="Average Groq calls per case including retries (TRAIN: 1.0)",
    )
    args = parser.parse_args(argv)

    report = estimate_train_token_usage(
        calibration_path=Path(args.calibration_results),
        train_dataset_path=Path(args.train_dataset),
        api_calls_per_case=args.api_calls_per_case,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
