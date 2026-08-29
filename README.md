# Chargeback Evidence Responder — Razorpay Buildathon

> **Track 02: AI Risk Manager** | Defense-Only Chargeback Evidence Responder

An intelligent, bounded defense agent that automates chargeback response for merchants. It evaluates inbound disputes, retrieves transaction evidence using vector RAG, synthesizes rebuttal packages using LLMs, and safely submits or flags cases for human review based on confidence scores.

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
