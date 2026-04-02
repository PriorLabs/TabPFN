# TabPFN Fine-Tuning Limits on Apple Silicon: An Empirical Study

## Abstract

This report documents an empirical evaluation of local TabPFN classifier fine-tuning on an Apple Silicon (`macOS`, M1-class) machine. We designed and executed staged stress tests to determine practical operating limits, compare `cpu` and `mps` performance, and validate model persistence through save/reload workflows. Across 300 to 3000 row trials, we observed a device crossover: `cpu` was faster in smaller runs, while `mps` became faster and materially more memory-efficient at larger workloads (3000 rows). We also validated persistent model artifact generation and reload checks, while identifying an important caveat that reload-evaluation metrics typically align with pre-fine-tune clone-evaluation behavior rather than always reproducing post-step in-memory evaluation values.

## 1. Introduction

TabPFN fine-tuning introduces practical systems questions for local experimentation:

1. Which device is best at a given workload size?
2. What row/context/step settings remain reasonable for iterative work?
3. Is save/reload robust enough for reproducible experiments?

This study answers those questions with a reproducible harness and logged trial history.

## 2. Experimental Setup

### 2.1 Environment

- Platform: `macOS` Apple Silicon (M1-class)
- Repositories:
  - `TabPFN-work-scott` (experiment harness and logs)
  - `TabPFN-upstream` (source APIs for fine-tuning and model loading)
- Import policy: local upstream source preferred (`PYTHONPATH=.../TabPFN-upstream/src`) to avoid installed-package API mismatch.

### 2.2 Dataset and Task

- Dataset: `data/raw/coil2000.csv`
- Target: `CARAVAN`
- Task: binary classification
- Split: `70/30` train/test with deterministic sampling (`seed=42`)

### 2.3 Fixed Model/Run Defaults

- Estimator: `TabPFNClassifier`
- `n_estimators=2`
- Fine-tuning style: batched one-step/short-step trials
- Context sizes tested: `64`, `128`
- Steps tested: `1`, `3`

### 2.4 Logging and Artifacts

- Trial metrics: `outputs/current/tables/tabpfn_finetune_trial_results.csv`
- Reload checks: `outputs/current/tables/tabpfn_finetune_reload_checks.csv`
- Saved artifacts: `outputs/current/models/*.tabpfn_fit`

## 3. Methodology

We used a staged limit-finding strategy:

1. Small readiness trials (300 rows)
2. First batch scaling (500/1000 rows, CPU-first with MPS spot checks)
3. Stress batch at 2000 rows
4. Crossover confirmation at 3000 rows (matched CPU vs MPS pairs)

Each trial included:

- Initial evaluation
- Fine-tuning steps
- Post-step evaluation
- Artifact save (`.tabpfn_fit`)
- Reload check validation

## 4. Results

### 4.1 Representative Runtime and Memory Trends

| Rows | Context | Steps | Device | Wall Time (s) | Max RSS (bytes) |
| --- | ---: | ---: | --- | ---: | ---: |
| 1000 | 64 | 1 | cpu | 14.51 | 662,945,792 |
| 1000 | 64 | 1 | mps | 19.18 | 495,763,456 |
| 2000 | 64 | 1 | cpu | 23.12 | 696,762,368 |
| 2000 | 64 | 1 | mps | 21.67 | 544,276,480 |
| 3000 | 64 | 1 | cpu | 27.87 | 710,426,624 |
| 3000 | 64 | 1 | mps | 26.50 | 529,022,976 |
| 3000 | 128 | 1 | cpu | 36.29 | 1,187,856,384 |
| 3000 | 128 | 1 | mps | 34.92 | 638,271,488 |

### 4.2 Key Observations

1. **Early-stage advantage for CPU**: At smaller runs (for example 1000/64/1), CPU had lower wall time than MPS.
2. **Crossover at larger scale**: At 2000 and especially 3000 row workloads, MPS became faster for matched settings.
3. **Strong memory advantage for MPS at higher context**: At 3000 rows, context 128, MPS used roughly half the max RSS of CPU.
4. **Context and steps increase cost predictably**: Increasing context from 64 to 128 and steps from 1 to 3 increased wall time and memory pressure.

### 4.3 Metric Behavior

- Post-step metrics varied modestly per run.
- In many trials, post-step changes were small, consistent with smoke/stress testing objectives focused on systems behavior over model quality optimization.

## 5. Persistence and Reload Validation

### 5.1 What Worked

- Fine-tuned artifacts were saved successfully to `.tabpfn_fit` files.
- Reload validation workflow executed successfully for all batched test runs.
- Reload checks were logged systematically.

### 5.2 Important Caveat

Reload-evaluation values commonly aligned with the logged pre-fine-tune clone-evaluation level rather than always matching post-step in-memory evaluation values exactly. This indicates:

- persistence is operational and reproducible as a workflow,
- but interpretation of post-step equivalence across save/reload boundaries should be conservative until deeper API-level equivalence testing is completed.

## 6. Practical Limit Guidance (Current)

Based on observed behavior in this study:

1. Use `cpu` as a default for smaller local exploratory runs.
2. Prefer `mps` once workload size approaches the higher tested range (observed advantage at 3000 rows).
3. For higher contexts (128+) and larger rows, monitor memory closely; MPS currently provides better memory headroom.
4. Keep save/reload checks in-loop for every stress batch.

## 7. Limitations

1. Single dataset (`coil2000`) and one target; crossover point may shift with different data geometry.
2. Single-machine study; results may differ on other Apple Silicon generations.
3. Smoke/stress methodology prioritizes systems behavior, not final predictive optimization.
4. Reload equivalence needs dedicated deep-dive testing for strict post-fine-tune parity claims.

## 8. Conclusion

The experiments establish a usable local fine-tuning regime for TabPFN on Apple Silicon and identify a practical device crossover:

- `cpu` is competitive or faster at smaller workloads,
- `mps` becomes better as workload scale increases,
- and at 3000-row matched tests MPS is both faster and much more memory efficient.

The study also delivers a reproducible persistence pipeline with automatic logging and reload checks, providing a solid operational foundation for further scaling and optimization.

## Appendix A: Reproducibility Artifacts

- Trial runner scripts:
  - `scripts/run_finetune_first_batch.sh`
  - `scripts/run_finetune_stress_batch_2000.sh`
  - `scripts/run_finetune_crossover_batch_3000.sh`
- Trial harness:
  - `scripts/run_small_finetune_classifier_trial.py`
- Reload validator:
  - `scripts/check_saved_finetune_classifier_model.py`
- Logs:
  - `outputs/current/tables/tabpfn_finetune_trial_results.csv`
  - `outputs/current/tables/tabpfn_finetune_reload_checks.csv`
