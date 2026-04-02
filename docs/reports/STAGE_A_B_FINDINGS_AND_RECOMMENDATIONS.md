# Stage A and Stage B Findings: Short Report

Date: 2026-04-02

## Scope Completed

1. Stage A completed: pilot feasibility and logging validation.
2. Stage B completed: all 4 insurance target datasets at seed 42 with core comparison arms.
3. Model arms executed in the Stage B matrix:
   - Raw TabPFN
   - Domain-finetuned TabPFN
   - Logistic regression (GLM)
   - Random forest
   - CatBoost rows logged as unavailable in this environment

Primary evidence sources:
- outputs/current/tables/domain_finetune_study_runs.csv
- outputs/current/logs/domain_finetune_logbook.md

## Key Findings

### 1. Feasibility and pipeline integrity (Stage A objective)

- The end-to-end protocol is operational: data splits, model fitting, metric logging, and markdown logbook updates are working.
- The run table and logbook now provide a reproducible audit trail for all pilot and scale-up runs.

### 2. Domain-finetuned vs raw TabPFN (aggregate behavior)

Across the tested seed-42 configurations, domain fine-tuning did not produce aggregate improvement over raw TabPFN on primary endpoints.

Macro deltas (domain-finetuned minus raw):

- Context 64, steps 1: ROC AUC -0.0528, PR AUC -0.0225, Brier +0.0008, LogLoss +0.0066
- Context 64, steps 3: ROC AUC -0.0741, PR AUC -0.0314, Brier +0.0016, LogLoss +0.0102
- Context 64, steps 5: ROC AUC -0.0737, PR AUC -0.0312, Brier +0.0016, LogLoss +0.0107
- Context 128, steps 5: ROC AUC -0.0586, PR AUC -0.0259, Brier +0.0009, LogLoss +0.0064

Interpretation:
- Increasing fine-tune steps and then context improved some targets, but did not reverse the aggregate sign on primary metrics.

### 3. Dataset-level heterogeneity

- freMTPL2freq_binary: domain-finetuned TabPFN is strongest in the latest setting (best Brier, LogLoss, ROC, PR).
- coil2000 and ausprivauto0405: raw TabPFN remains strongest on core calibration/discrimination metrics.
- eudirectlapse: logistic regression is strongest in the latest setting.

Interpretation:
- A single global "fine-tune always" policy is not supported by current evidence.
- Best-performing model family appears dataset-dependent.
- A likely driver is cross-dataset domain mismatch; pooled fine-tuning is expected to work better when source and target datasets are more homogeneous.

## Recommendations for Proceeding

1. Move to Stage C (stability) before any deployment conclusion.
   - Repeat full Stage B matrix on seeds 1337 and 2025.
   - Produce paired bootstrap CIs for raw vs domain-finetuned deltas on Brier and LogLoss.

2. Use conditional model selection as the interim operating policy.
   - Keep raw TabPFN as default baseline.
   - Allow domain-finetuned TabPFN where validation favors it (currently freMTPL2freq_binary-like behavior).
   - Keep GLM candidate active for datasets with strong linear signal (eudirectlapse-like behavior).

3. Run Stage D ablations focused on calibration-first outcomes.
   - Fine-tune pool composition variants: all-other vs closest-domain-only.
   - Add validation-only calibration (isotonic/Platt) for raw and domain-finetuned arms.
   - Preserve fixed splits and equal tuning budgets for fairness.

4. Environment and reporting hygiene.
   - Either install CatBoost for complete tree-family comparability or formally lock tree baseline to RandomForest-only.
   - Keep using the existing CSV plus logbook as canonical records.

5. Add a homogeneous-dataset gate before domain fine-tuning.
   - Build a "homogeneous pool" option that includes only source datasets similar to the target.
   - Use lightweight similarity checks already aligned with this project: feature/schema overlap, event-rate proximity, and validation transfer performance from source to target.
   - Compare three policies under identical splits and budgets:
     - no fine-tune (raw TabPFN),
     - all-other pooled fine-tune,
     - homogeneous-only pooled fine-tune.
   - Promote fine-tuning only when homogeneous-only pool improves primary endpoints (Brier, LogLoss) versus raw with consistent direction across seeds.

## Bottom Line

Stage A and Stage B are complete, the experimentation framework is reliable, and current evidence does not yet support replacing raw TabPFN with domain-finetuned TabPFN as a universal default. The recommended next step is Stage C with multi-seed uncertainty estimates, followed by targeted Stage D ablations.

## Reproducibility Note (How This Was Executed)

- Workspace: TabPFN-work-scott, with local upstream TabPFN source available at TabPFN-upstream.
- Key execution environment detail: use local upstream import path when running fine-tuning utilities.
   - `PYTHONPATH=/Users/Scott/Documents/Data Science/ADSWP/TabPFN-upstream/src`
- Core runner used for Stage A and Stage B style comparisons:
   - `python scripts/run_domain_finetune_stage_a.py --target-dataset <dataset> --target-rows 2500 --pool-rows-per-dataset 1000 --tabpfn-device cpu --tabpfn-context-samples <64|128> --tabpfn-max-finetune-steps <1|3|5> --seed 42 --observations "..." --comments "..."`
- Datasets used:
   - `eudirectlapse`, `coil2000`, `ausprivauto0405`, `freMTPL2freq_binary`
- Fixed controls in reported runs:
   - seed `42`
   - same split discipline across model arms
   - same target/pool row budgets per run
- Canonical output artifacts:
   - run table: `outputs/current/tables/domain_finetune_study_runs.csv`
   - narrative logbook: `outputs/current/logs/domain_finetune_logbook.md`

For strict reproduction of the latest comparison snapshot referenced in this report, run all four datasets with `tabpfn_context_samples=128` and `tabpfn_max_finetune_steps=5` using the command shape above, then read the two artifact files listed.