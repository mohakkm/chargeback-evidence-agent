# STATUS — Razorpay Buildathon (Chargeback Evidence Responder)

Last updated: 2026-08-29

## Decided
- Track: 02 — AI Risk Manager
- Sub-idea: Chargeback evidence responder
- Stack: Python/FastAPI backend, Claude / Groq API for reasoning/drafting, Qdrant for
  evidence retrieval, Streamlit for demo UI

## Completed
- Folder structure scaffolding matching ARCHITECTURE.md
- Empty `__init__.py` package initializers across all subdirectories
- `requirements.txt` with fastapi, uvicorn, qdrant-client, groq, anthropic, streamlit, pydantic, python-dotenv
- `README.md` stub with repository layout
- Module stub files for app and test components
- Phase 1 Synthetic Data Generation (`app/data/generate_synthetic.py`): 120 total dispute cases (90 train / 30 holdout split) generated with 2-4 evidence docs per case, enforcing reason code taxonomy, auth_log for unauthorized_transaction, winnable pre-assignment (~55/45), 5 holdout ambiguous cases, and evaluation field stripping utility.

## Not started yet
- Evidence store setup (Qdrant)
- Retrieval pipeline
- Decision agent logic
- Bounded action gate + audit trail
- Eval harness
- Demo dashboard

## Open questions
- Exact submission deadline not confirmed on Razorpay's own page — third-party
  sources say Sept 5, 2026. Verify directly before planning the final days.
- Stack confirmation (Python-first vs. default Next.js/Supabase/Vercel)
