# STATUS — Razorpay Buildathon (Chargeback Evidence Responder)

Last updated: 2026-08-29

## Decided
- Track: 02 — AI Risk Manager
- Sub-idea: Chargeback evidence responder
- Stack: Python/FastAPI backend, Groq / Claude API for reasoning/drafting, Qdrant for
  evidence retrieval (local embedded mode), Streamlit for demo UI

## Completed
- Folder structure scaffolding matching ARCHITECTURE.md
- Empty `__init__.py` package initializers across all subdirectories
- `requirements.txt` with fastapi, uvicorn, qdrant-client, groq, anthropic, sentence-transformers, streamlit, pydantic, python-dotenv
- `README.md` stub with repository layout
- Phase 1 Synthetic Data Generation (`app/data/generate_synthetic.py`): 120 total dispute cases (90 train / 30 holdout split) generated with 2-4 evidence docs per case, enforcing reason code taxonomy, auth_log for unauthorized_transaction, winnable pre-assignment (~55/45), 5 holdout ambiguous cases, and evaluation field stripping utility.
- Phase 2 Retrieval Pipeline (`app/retrieval/`):
  - `qdrant_client.py`: Local embedded Qdrant instance storing data at `./qdrant_data`.
  - `embed.py`: Loaded `multilingual-e5-large` embedding model, indexed all 342 unique evidence documents across train and holdout datasets.
  - `retriever.py`: Built `EvidenceRetriever` with reason-code hint augmentation and transaction-scoped fallback logic.

## Not started yet
- Decision agent logic (Phase 3)
- Bounded action gate + audit trail (Phase 3 & 4)
- Eval harness (Phase 4)
- Demo dashboard (Phase 5)

## Open questions
- Exact submission deadline not confirmed on Razorpay's own page — third-party
  sources say Sept 5, 2026. Verify directly before planning the final days.
- Stack confirmation (Python-first vs. default Next.js/Supabase/Vercel)
