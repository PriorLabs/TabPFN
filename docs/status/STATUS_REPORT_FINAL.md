# TabPFN Research Status Report (Refined)

## Purpose

This document is the current source of truth for what was tested, what was learned, and what should be deployed for the TabPFN lapse-modeling work.

Scope covered:
- Baseline model comparison (GLM, TabPFN, CatBoost, RandomForest, XGBoost)
- Post-hoc optimization for TabPFN (calibration, feature engineering, ensembling)
- Deployment recommendation based on actuarial business priorities

---

## Validated Findings

### 1. Discrimination performance is tightly clustered

From the controlled comparison, GLM and TabPFN are close on ROC AUC:
- GLM: 0.5991
- TabPFN: 0.5929

Interpretation:
- On this dataset, modern foundation modeling does not materially beat a strong classical baseline on raw discrimination.
- Complexity alone does not guarantee lift.

### 2. Calibration is the main TabPFN opportunity

Post-hoc calibration improved TabPFN probability quality:
- Raw TabPFN Brier: 0.109678
- Best optimized variant Brier: 0.107982
- Relative improvement: +1.66%

Interpretation:
- For pricing and reserve use-cases, this is the most business-relevant win.
- Isotonic calibration was consistently stronger than Platt scaling in this workflow.

### 3. Ensemble bagging did not help

3-member bagging degraded results in these tests.

Interpretation:
- Additional averaging did not create useful diversity for this TabPFN setup.
- Keep the production path simple: single model + calibration.

---

## Recommended Deployment Decision

Deploy this variant first:

`TabPFN (Engineered Features) -> Isotonic Calibration`

Why:
- Best observed Brier performance
- Zero additional infrastructure cost
- Low operational complexity
- Directly aligned to actuarial probability-quality needs

Rollback trigger:
- Revert to current baseline if monitored Brier exceeds 0.110 for sustained periods.

---

## Repository Restructuring Required

Current issue:
- The repository mixes final research notes, generated outputs, notebooks, and draft/support docs in one flat documentation area, making review and version control noisy.

Recommended structure (documentation only):

1. Keep a single canonical narrative
- Canonical paper: `UNIFIED_PAPER_FINAL.md`
- Canonical status: `STATUS_REPORT_FINAL.md`
- Canonical index: `INDEX_DOCUMENTATION.md`

2. Move supporting writeups into an archive folder
- Move these to `docs/archive/` (or `BaselineExperiments/docs/archive/`):
   - `QUICK_REFERENCE.md`
   - `QUICK_REFERENCE_STREAMLINED.md`
   - `BEFORE_AFTER_COMPARISON.md`
   - `STREAMLINING_SUMMARY.md`
   - `UNIFIED_PAPER_STRUCTURE.md`

3. Separate generated artifacts from versioned research notes
- Keep generated outputs under `outputs/` but do not bundle them with narrative-note commits.
- Commit output artifacts only when they are explicitly required for reproducibility snapshots.

4. Commit strategy
- Use small thematic commits:
   - `docs: refine research status report`
   - `docs: restructure index and canonical doc map`
   - `data-artifacts: add reproducibility snapshot` (only when needed)

---

## What Was Updated In This Refinement

- Re-centered this report on research findings rather than editorial process details.
- Added explicit deployment recommendation and rollback rule.
- Added concrete repository restructuring guidance to reduce future documentation drift.

---

## Linked Core Files

- Main paper: `UNIFIED_PAPER_FINAL.md`
- Supporting article draft: `ARTICLE_REVISED_COMPLETE.md`
- Optimization details: `FINETUNING_SUMMARY.md`
- Documentation map: `INDEX_DOCUMENTATION.md`

---

## Next Actions

1. Keep this file as the short executive research status.
2. Keep `UNIFIED_PAPER_FINAL.md` as the long-form publication narrative.
3. Archive non-canonical supporting docs to reduce maintenance overhead.
4. Continue production monitoring for calibration drift after deployment.

Report date (intentionally future-dated to match target production cutover milestone): 2026-03-29
