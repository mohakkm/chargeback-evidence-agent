# Chargeback Evidence Responder

**Razorpay AI Buildathon 2026 · Track 02 (AI Risk Manager)**

A defense-only chargeback evidence responder for merchants: when a dispute arrives, the system retrieves transaction-scoped evidence, reasons over the claim with an LLM, drafts a rebuttal packet, and either auto-submits or flags the case for human review behind a confidence-and-coverage gate. It never takes adverse action against a customer — it only assembles merchant defense evidence in response to disputes already raised. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/STATUS.md](docs/STATUS.md) for design and build status.

---

## Quickstart

### Environment

Copy [`.env.example`](.env.example) to `.env` and set:

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Live LLM calls (decision agent, official eval runs) |
| `AUTO_SUBMIT_CONFIDENCE_THRESHOLD` | Locked at `0.70` (TRAIN-calibrated; see `threshold_selection_report.json`) |
| `QDRANT_HOST` / `QDRANT_PORT` | Local Qdrant (embedded mode uses `./qdrant_data` by default) |

**Dataset generation only** (`app/data/generate_synthetic.py`): downloads IEEE-CIS and CFPB grounding data via `kagglehub`. Configure [Kaggle API credentials](https://github.com/Kaggle/kagglehub#authentication) (`KAGGLE_USERNAME` / `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`) before running the generator. Pre-built `train.jsonl` and `holdout.jsonl` are already in `app/data/datasets/` if you skip this step.

### Setup

```bash
pip install -r requirements.txt
```

### Run order

1. **Synthetic data** (optional if using committed datasets):
   ```bash
   python app/data/generate_synthetic.py
   ```
2. **Embed + index evidence** (Qdrant local store at `./qdrant_data`):
   ```bash
   python app/retrieval/embed.py
   ```
3. **Pipeline / eval** (retrieval → decision agent → action gate → audit logger):
   ```bash
   # Example: official-style holdout eval (live Groq, resume-aware)
   python -m app.eval.run_eval --split holdout --resume-from holdout_results.jsonl
   ```
   See `app/eval/run_eval.py` for TRAIN calibration (`--calibrate-train`) and other flags.
4. **Dashboard** (offline — reads local JSONL artifacts, no API calls):
   ```bash
   streamlit run app/dashboard/streamlit_app.py
   ```

FastAPI skeleton (fixture-backed, no live Groq): `uvicorn app.main:app --reload`

---

## Repo layout

See the [folder structure in docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#folder-structure).

---

## Results

**Official HOLDOUT (n=30):** precision 54.5%, auto-submit precision 80.0% (95% CI [37.6%, 96.4%], n=5)

- Full demo walkthrough and metric context: [docs/demo_narrative.md](docs/demo_narrative.md)
- Raw numbers and Wilson CI: [holdout_official_report.json](holdout_official_report.json)
- TRAIN vs HOLDOUT comparison and decision-distribution analysis: same report file

The official HOLDOUT run is complete (30/30, threshold locked at 0.70 before the run). Do not re-run for tuning. See [docs/limitations.md](docs/limitations.md).

---

## Known limitations

See [docs/limitations.md](docs/limitations.md) — stated plainly, not summarized here.

---

## Testing

**Default — no API calls, no Groq quota burned:**

```bash
python tests/run_test_suite.py
```

Runs fixture-based tests only (`tests/test_fixture_pipeline.py`): canned JSON scenarios, fake retriever/agent, zero network calls to Groq.

**`--live` — still does not call the real Groq API:**

```bash
python tests/run_test_suite.py --live
```

Additionally runs `app/agent/test_decision_agent.py`, `app/agent/test_action_gate.py`, and `app/eval/test_run_eval.py`. These tests **mock** the Groq client internally (429 retries, JSON validation, etc.). The `--live` flag means “run the fuller mocked unit suite,” not “hit the live API.”

**What actually calls Groq (costs quota):**

- `python -m app.eval.run_eval --split holdout` (or `--split train`, `--calibrate-train`)
- Any direct use of `DecisionAgent` with a valid `GROQ_API_KEY` and no mock

Official eval requires live LLM unless you pass `--allow-fallback` (smoke testing only; excluded from official metrics).

---

## Safety & scope

This is a **defense-only** agent: it responds to chargebacks already filed against the merchant by retrieving evidence and drafting/submitting a contest packet. It does not initiate disputes, block customers, or take any offense-capable risk action. A **bounded action gate** enforces auto-submit only when `decision == contest`, confidence clears the locked threshold (0.70), and evidence coverage is adequate (`low_coverage == false`); everything else routes to `flag_for_review` with a full audit trail. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#safety--bounded-action-rules-map-directly-to-the-tracks-bar).

---

## Submission

Exact Razorpay submission deadline is **not confirmed** in this repo — verify on Razorpay's portal before final submission. Phase 6 (submission, panel prep) is open; see [docs/CHECKLIST.md](docs/CHECKLIST.md).
