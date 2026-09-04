# Chargeback Evidence Responder — Demo Narrative

## The Problem

When a cardholder disputes a charge, the merchant has a narrow window to assemble evidence and respond. Most merchants lack dedicated dispute-ops staff; wrong calls waste representment fees on unwinnable cases, while missed contests leave recoverable revenue on the table. This system is a **defense-only** responder: it never takes adverse action against a customer. It retrieves merchant evidence, reasons over the claim, drafts a rebuttal packet, and either auto-submits or routes to human review.

## Architecture (One Paragraph)

A dispute case enters the pipeline without any ground-truth labels. The retrieval layer (Qdrant + `multilingual-e5-large` embeddings) fetches transaction-scoped evidence keyed on reason code. The decision agent (Groq LLM) outputs `contest` or `no_contest`, a confidence score, and a reasoning summary — evaluation labels never reach this step. A bounded action gate applies a **locked** auto-submit threshold (0.70, calibrated on TRAIN only) plus a hard `low_coverage` block: auto-submit requires `decision == contest`, `confidence >= 0.70`, and at least two scoped evidence documents. Everything else is `flag_for_review`. Each case appends a privacy-safe JSONL audit record (evidence IDs only, no label leakage).

---

## Live Case Walkthrough — System Working as Intended

**Case:** `CB-00106` (official HOLDOUT run; auto-submitted true positive)

| Field | Value |
|-------|-------|
| Reason code | `unauthorized_transaction` |
| Amount | $26.95 |
| Merchant | Williamson and Sons Cloud (subscription) |
| Decision | **contest** |
| Confidence | **0.88** |
| Action | **auto_submit** |
| `low_coverage` | false (4 scoped evidence docs) |
| `used_fallback` | false |

**Customer claim:**  
*"Fraudulent charge of $26.95 at Williamson and Sons Cloud on 2026-07-02. Transaction was not authorized by me. Requesting immediate dispute resolution."*

**Retrieved evidence (summary):**

1. **auth_log** — Authentication log for order ORD-597107: client IP passed biometric and SMS challenge for session SESS-72684; no account-takeover indicators during authorization.
2. **order_details** — Invoice INV-57002 issued 2026-07-02; payment settled with zero authorization exceptions.
3. **communication_log** — Support ticket TCK-8402 closed resolved; customer correspondence confirmed satisfaction.
4. **refund_policy** — Standard terms of sale on file (context for merchant compliance).

**Agent reasoning (audit log):**  
*"Strong auth log and settled invoice prove legitimate, authorized transaction; contest with high confidence."*

**Why this case auto-submitted:** The agent recommended contest at 0.88 — above the locked 0.70 threshold — with adequate evidence coverage. Ground truth (`label_winnable=true`) confirms this was a correct auto-submit: winnable unauthorized-transaction dispute with strong authorization proof.

---

## Honest Metrics (Official Eval — No Rounding Up)

### Classification (decision vs. `label_winnable`, full set)

| Split | n | Precision | Recall | Accuracy |
|-------|---|-----------|--------|----------|
| TRAIN | 90 | 82.6% | 39.6% | 63.3% |
| HOLDOUT | 30 | 54.5% | 40.0% | 53.3% |

TRAIN and HOLDOUT use the same formula. The precision gap is real; generalization is not proven at this sample size (see `limitations.md`).

### Auto-submit gate (HOLDOUT, n=5 auto-submitted)

- **Auto-submit precision:** 80.0%
- **95% Wilson CI:** [37.6%, 96.4%]
- **n = 5** (4 correct, 1 incorrect)

The CI is wide because the auto-submit pool is tiny. Do not treat 80.0% as a settled production number.

### Cost proxies (HOLDOUT, synthetic dollar amounts)

| Proxy | Exposure | Cases |
|-------|----------|-------|
| False-positive (contested, not winnable) | **$1,329.42** | 5 |
| False-negative (not contested, winnable) | **$1,185.79** | 9 |

FP proxy = money spent contesting cases that should not have been contested. FN proxy = winnable revenue left uncontested.

### Named finding: Systematic conservative bias

Across all 30 HOLDOUT cases: **63% `no_contest`** (19 of 30).

Among the **15 genuinely winnable** cases only: **60% `no_contest`** (9 of 15).

This is not random error. The agent under-contests even when ground truth says the merchant could win. The system trades recall for precision — it contests selectively and leaves recoverable cases on the table.

---

## Failure Handled Gracefully — `CB-00091`

Not every case auto-submits. **`CB-00091`** demonstrates the gate working as a safety layer:

| Field | Value |
|-------|-------|
| Reason code | `unauthorized_transaction` |
| Decision | `no_contest` |
| Confidence | **0.35** |
| `low_coverage` | **true** (0 transaction-scoped docs despite retrieval expansion) |
| Action | **flag_for_review** |

The agent leaned `no_contest` with low confidence on sparse, poorly scoped evidence. Even if the decision had been `contest`, `low_coverage=true` and confidence 0.35 are both below gate requirements — **no auto-submit**. A human reviewer sees the audit record and decides next steps. The system did not silently drop the case or auto-act on thin evidence.

---

## Why Bounded Action Beats Raw Model Trust

HOLDOUT **accuracy is 53.3%** — barely better than a coin flip on the full decision task. That is not a hidden flaw; it is the reason the confidence-and-coverage gate exists. Raw model output is not trusted for irreversible merchant action. Only 5 of 30 HOLDOUT cases (16.7%) cleared auto-submit; 25 went to review (83.3%). The design assumption is: **when the model is wrong or uncertain, default to human review** — not to automated representment. Weak aggregate accuracy makes the gate more important, not less.

---

*Demo case source: `holdout_results.jsonl` / `holdout_audit_consolidated.jsonl`, official one-time HOLDOUT run. Metrics source: `holdout_official_report.json`, `calibration_results.jsonl`.*
