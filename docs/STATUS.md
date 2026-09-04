# STATUS — Razorpay Buildathon (Chargeback Evidence Responder)

Last updated: 2026-09-04

## Decided
- Track: 02 — AI Risk Manager
- Sub-idea: Chargeback evidence responder
- Stack: Python/FastAPI backend, Groq API for reasoning/drafting, Qdrant for
  evidence retrieval (local embedded mode), Streamlit for demo UI
- Auto-submit threshold: **0.70** (locked from TRAIN calibration; not tuned on HOLDOUT)

## Completed

### Phase 1 — Data & Setup
- Folder structure scaffolding matching ARCHITECTURE.md
- Empty `__init__.py` package initializers across all subdirectories
- `requirements.txt` with fastapi, uvicorn, qdrant-client, groq, anthropic,
  sentence-transformers, streamlit, pydantic, python-dotenv
- `README.md` stub with repository layout
- Synthetic data generation (`app/data/generate_synthetic.py`): 120 cases
  (90 train / 30 holdout), 2–4 evidence docs per case, reason-code taxonomy,
  evaluation field stripping

### Phase 2 — Retrieval
- `app/retrieval/qdrant_client.py`: local embedded Qdrant at `./qdrant_data`
- `app/retrieval/embed.py`: `multilingual-e5-large`, 342 evidence docs indexed
- `app/retrieval/retriever.py`: `EvidenceRetriever` with reason-code hints and
  transaction-scoped fallback

### Phase 3 — Decision Agent
- `app/agent/decision_agent.py`: Groq LLM reasoning, confidence scoring,
  low-coverage cap, JSON retry path, heuristic fallback
- `app/agent/action_gate.py`: bounded gate (`auto_submit` vs `flag_for_review`)

### Phase 4 — Audit Trail & Eval
- `app/audit/logger.py`: privacy-safe JSONL audit trail
- `app/eval/run_eval.py`: eval harness, resume/checkpoint, train calibration,
  holdout consolidation
- `app/eval/metrics.py`, `app/eval/select_threshold.py`, `app/eval/holdout_report.py`
- Official TRAIN calibration: 90/90 (`calibration_results.jsonl`,
  `train_audit_consolidated.jsonl`)
- Threshold selection report (`threshold_selection_report.json`); locked at 0.70
- **Official one-time HOLDOUT run:** 30/30, 0 fallbacks
  (`holdout_results.jsonl`, `holdout_audit_consolidated.jsonl`,
  `holdout_official_report.json`)
- Failure-handled demo case: `CB-00091` (low coverage, gate blocked auto-submit)
- Fixture test suite, FastAPI skeleton (`POST /evaluate-dispute`), mocked pipeline

### Phase 5 — Demo & Polish
- Streamlit dashboard (`app/dashboard/streamlit_app.py`): case walkthrough +
  TRAIN-labeled metrics tab (offline)
- `docs/demo_narrative.md`: problem → architecture → CB-00106 walkthrough →
  honest metrics → CB-00091 gate demo
- `docs/limitations.md`: honest limitations (no pitch-deck framing)

## Not started yet
- Phase 6: deadline confirmation, final submission, panel rehearsal
- Demo narrative verbal rehearsal (draft written; not yet rehearsed end-to-end)

## Open questions
- Exact submission deadline not confirmed on Razorpay's own page — third-party
  sources say Sept 5, 2026. Verify directly before final submission.
