# Limitations — Honest Metrics Submission

This document states what the build does **not** prove. It is not a pitch deck.

---

## Generalization is not proven at this sample size

TRAIN precision: **82.6%** (n=90). HOLDOUT precision: **54.5%** (n=30).

That gap may reflect distribution shift, small-sample noise, or overfitting to TRAIN patterns. Thirty holdout cases cannot establish that the system generalizes. The TRAIN number is not a reliable forecast of production performance.

---

## Auto-submit precision is not a settled number

Official HOLDOUT auto-submit precision: **80.0%** (4/5 correct).

95% Wilson CI: **[37.6%, 96.4%]**, **n=5**.

Five auto-submitted cases produce a confidence interval that spans from unacceptable to excellent. Any claim about auto-submit reliability beyond "promising but unproven on tiny n" is unsupported by the data.

---

## Confidence is not a calibrated continuous score

TRAIN eligible-pool analysis shows confidence values clustering near round deciles (0.65, 0.66, 0.68, 0.72, 0.75, 0.78, 0.85, 0.88, 0.90, 0.92) with heavy ties — not a smooth distribution.

Using an LLM's self-reported confidence directly as a gate input is a limitation. The threshold (0.70) was selected on TRAIN under this constraint; HOLDOUT gate behavior may not track intuitive probability.

---

## Systematic conservative bias — revenue under-recovery

HOLDOUT decision distribution:

- All 30 cases: **63% `no_contest`** (19/30)
- Winnable cases only (n=15): **60% `no_contest`** (9/15)

The agent under-contests even when ground truth marks the case winnable. FN cost proxy: **$1,185.79** across 9 false-negative cases. The system currently trades recall for precision and **under-recovers revenue** as an explicit behavioral pattern, not an accident.

---

## No real Razorpay integration — synthetic data only

There is no Razorpay test-mode API integration, no live transaction feed, and no production dispute webhook.

The dataset is synthetic: grounded in IEEE-CIS and CFPB dispute patterns, but not real merchant transactions. Evidence documents, reason codes, and win/loss labels are generated and labeled offline. Results do not validate behavior on actual Razorpay chargeback flows.

---

## Scope boundaries

- **English only** — prompts, evidence, and evaluation are English-language.
- **Single agent** — one decision agent; no multi-agent orchestration, no specialist sub-agents per reason code.
- **Single use case** — chargeback evidence response only; not fraud prevention, not customer-facing dispute initiation, not offense-capable risk action.

---

## Fallback path is untested in the run that counts

A heuristic fallback reasoner exists for Groq API outages (`used_fallback=true`).

Official HOLDOUT run: **0/30 fallbacks**. The fallback path was never exercised in the one evaluation that matters. Its behavior on real cases is unknown.

---

## What this build does demonstrate

Despite the above: defense-only boundary is enforced; evaluation labels are stripped before inference; threshold was locked on TRAIN before HOLDOUT was run once; audit trail is complete for all 30 holdout cases; the action gate blocked auto-submit on sparse-evidence cases (e.g. `CB-00091`). Those are architectural properties the eval supports. They do not override the metric and sample-size limitations above.
