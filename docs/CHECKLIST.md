# CHECKLIST - Chargeback Evidence Responder

## Phase 1 - Data & Setup
- [ ] Confirm exact submission deadline directly on Razorpay's page/portal
- [x] Set up repo, FastAPI skeleton, trunk-based on main
- [x] Write synthetic dispute + evidence generator script
- [x] Generate labeled dataset (winnable / not winnable) - target 100+ cases,
      hold out 20-30%

## Phase 2 - Retrieval
- [x] Stand up Qdrant instance (embedded local mode at ./qdrant_data)
- [x] Embed evidence documents (multilingual-e5-large model)
- [x] Build retrieval pipeline keyed on dispute reason code

## Phase 3 - Decision Agent
- [x] LLM reasoning step: dispute + evidence -> contest/don't-contest + drafted
      rebuttal packet
- [x] Confidence scoring
- [x] Bounded action gate (auto-submit vs. flag-for-review threshold)

## Phase 4 - Audit Trail & Eval
- [x] Audit trail logger (decision, evidence, reasoning, timestamp per case)
- [ ] Run agent over held-out set
- [ ] Compute precision, recall, false-positive cost
- [ ] Deliberately inject one failure case - document how it's handled

## Phase 5 - Demo & Polish
- [ ] Streamlit dashboard: live case walkthrough + metrics view
- [ ] Rehearse demo narrative: problem -> architecture -> live case -> metrics ->
      failure case handled
- [ ] Write an honest limitations section - what this doesn't cover

## Phase 6 - Buffer / Submission
- [ ] Slippage buffer
- [ ] Final submission per Razorpay's process
- [ ] Panel prep: know precision/recall cold, know why FP cost matters, know
      exactly where the "defense-only" boundary sits in your build
