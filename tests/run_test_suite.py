#!/usr/bin/env python3
"""
Unified test runner for the chargeback responder.

Default (no flags): fixture-based offline tests only — no live Groq API.
--live: additionally runs module unit tests that mock Groq internally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_script(rel_path: str) -> int:
    path = PROJECT_ROOT / rel_path
    print(f"\n>>> Running {rel_path}\n")
    return subprocess.call([sys.executable, str(path)], cwd=str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chargeback responder test suite")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run Groq-mocked unit tests (decision_agent, action_gate, eval). "
        "Does not call the live Groq API unless GROQ_API_KEY is set and tests opt in.",
    )
    args = parser.parse_args(argv)

    exit_codes = [_run_script("tests/test_fixture_pipeline.py")]

    if args.live:
        exit_codes.extend([
            _run_script("app/agent/test_decision_agent.py"),
            _run_script("app/agent/test_action_gate.py"),
            _run_script("app/eval/test_run_eval.py"),
        ])

    if any(code != 0 for code in exit_codes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
