# Razorpay Buildathon — Chargeback Evidence Responder — ARCHITECTURE

## Track & Idea
Track 02: AI Risk Manager. Sub-idea: Chargeback evidence responder (defense-only).

## Problem
A chargeback/dispute lands on a merchant. The agent decides whether it's contestable,
retrieves supporting evidence, drafts a rebuttal packet, and auto-submits — or flags
for human review if confidence is low.

## Stack (recommended — Python-first, override if needed)
- Backend: Python, FastAPI
- LLM (reasoning + drafting): Claude API
- Vector DB (evidence retrieval): Qdrant
- Embeddings: multilingual-e5-large (reused from prior RAG project) — swap to a
  lighter English-only model if the synthetic data ends up English-only
- Demo UI: Streamlit (not Next.js — avoids frontend/TypeScript time sink)
- Data: synthetic dispute + evidence generator (Python, LLM-assisted)
- Deployment: local for build/eval; Streamlit Cloud or Render only if a live link
  is needed for submission
- Version control: trunk-based on main

## Components / Data Flow
1. **Synthetic Data Generator** — produces labeled dispute cases (winnable /
   not winnable) plus associated evidence docs (delivery confirmation, comms log,
   order/refund policy text). Held-out split for eval, ~20-30%.
2. **Evidence Store** — evidence docs embedded and indexed in Qdrant per
   transaction/case.
3. **Retrieval Layer** — given a dispute + reason code, retrieve the relevant
   evidence chunks for that case.
4. **Decision Agent** — Claude reasons over the dispute + retrieved evidence,
   outputs: contest / don't-contest decision, confidence score, drafted evidence
   packet (rebuttal text).
5. **Bounded Action Gate** — auto-submit only if confidence clears a set
   threshold; below threshold → flagged for human review, never silently
   dropped or auto-rejected without a paper trail.
6. **Audit Trail Logger** — every case logs: dispute id, evidence retrieved,
   reasoning, decision, action taken, timestamp.
7. **Eval Harness** — runs the agent over the held-out set, computes precision,
   recall, false-positive cost.
8. **Demo Dashboard (Streamlit)** — walks through a live case end-to-end, shows
   the metrics panel, and shows one injected failure case handled gracefully.

## Safety / Bounded-Action Rules (map directly to the track's bar)
- **Defense-only**: the agent never takes adverse action against a customer —
  it only assembles/submits evidence in response to disputes already raised
  against the merchant. No offense-capable behavior anywhere in the loop.
- **Confidence-gated auto-action**: nothing auto-submits below the threshold.
- **Full audit trail**: every decision is reconstructable after the fact.

## Metrics (the "honest metrics" bar)
- Precision / recall on the held-out labeled dispute set
- False-positive cost estimate (cost of contesting a non-winnable case)
- Evidence coverage rate (% of cases with enough retrieved evidence to decide
  confidently vs. flagged for review)

## Folder Structure
```
chargeback-responder/
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── config.py
│   ├── data/
│   │   ├── generate_synthetic.py  # dispute + evidence generator
│   │   └── datasets/
│   │       ├── train.jsonl
│   │       └── holdout.jsonl
│   ├── retrieval/
│   │   ├── embed.py
│   │   ├── qdrant_client.py
│   │   └── retriever.py
│   ├── agent/
│   │   ├── decision_agent.py      # LLM reasoning + drafting
│   │   ├── confidence.py
│   │   └── action_gate.py         # bounded action gate
│   ├── audit/
│   │   └── logger.py
│   ├── eval/
│   │   ├── run_eval.py
│   │   └── metrics.py
│   └── dashboard/
│       └── streamlit_app.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── STATUS.md
│   └── CHECKLIST.md
├── tests/
├── requirements.txt
└── README.md
```
