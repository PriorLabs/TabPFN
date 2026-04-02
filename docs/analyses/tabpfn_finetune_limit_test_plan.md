# TabPFN Fine-Tuning Limit Test Plan

Scope note: this limit plan covers binary classification fine-tuning with `TabPFNClassifier` only.

## Goal

Measure the largest TabPFN fine-tuning workload that remains reasonable on this Apple Silicon machine.

For this plan, a run is considered reasonable if it:

1. completes without import, preprocessing, save/load, or device errors
2. keeps wall time short enough to support iteration
3. stays within acceptable memory pressure
4. preserves a functioning save/load path

## Working Assumptions

- Machine: macOS Apple Silicon (M1-class)
- Current evidence: `cpu` performed better than `mps` on the small local smoke workload
- Dataset for limit finding: `data/raw/coil2000.csv`
- Target: `CARAVAN`
- Baseline harness: `scripts/run_small_finetune_classifier_trial.py`
- Save/load validator: `scripts/check_saved_finetune_classifier_model.py`

## What We Are Testing

We want to isolate which dimension becomes the bottleneck first:

1. sample count (`rows`)
2. inference context size (`context_samples`)
3. fine-tuning length (`max_finetune_steps`)
4. device choice (`cpu` vs `mps`)

## Reasonable Limit Criteria

Treat a configuration as still reasonable if all of the following are true:

- wall time is under `120` seconds for a single trial
- no memory-related failure occurs
- save succeeds to `.tabpfn_fit`
- reload check succeeds
- the run can be repeated without obvious instability

Treat a configuration as approaching the limit if any of the following happen:

- wall time exceeds `120` seconds
- repeated runs become unreliable
- memory usage jumps sharply relative to the previous step
- `mps` becomes slower than `cpu` without improving stability
- save/load stops reproducing a usable inference path

## Test Execution Rules

1. Run tests in order.
2. Do not change multiple variables at once unless the phase explicitly says so.
3. Record every run in `outputs/current/tables/tabpfn_finetune_trial_results.csv`.
4. For every saved model, run `scripts/check_saved_finetune_classifier_model.py`.
5. If a run fails, stop that branch and do not scale it further.

## Stage A: CPU Row Scaling Baseline

Purpose: find the row-count range where CPU remains comfortable while keeping context fixed.

| Test ID | Device | Rows | Context | Max Steps | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| A1 | `cpu` | 300 | 64 | 1 | Known-good baseline |
| A2 | `cpu` | 500 | 64 | 1 | Small scale-up |
| A3 | `cpu` | 1000 | 64 | 1 | First medium pilot |
| A4 | `cpu` | 2000 | 64 | 1 | Stress sample count |
| A5 | `cpu` | 4000 | 64 | 1 | Upper exploratory bound |

Decision after Stage A:

- If A4 is still fast and stable, continue to Stage B.
- If A4 is slow or unstable, treat the A3 range as the current practical limit.

## Stage B: CPU Context Scaling

Purpose: determine whether context size becomes the bottleneck before row count does.

Use the largest stable row count from Stage A.

| Test ID | Device | Rows | Context | Max Steps | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| B1 | `cpu` | stable_rows | 64 | 1 | Reference point |
| B2 | `cpu` | stable_rows | 128 | 1 | Standard context increase |
| B3 | `cpu` | stable_rows | 256 | 1 | Aggressive context test |

Decision after Stage B:

- If `256` is unstable or slow, keep the best result from `64` or `128` as the default.

## Stage C: CPU Fine-Tuning Length

Purpose: test whether longer fine-tuning runs remain reasonable once rows and context are fixed.

Use the best stable `(rows, context)` pair from Stages A and B.

| Test ID | Device | Rows | Context | Max Steps | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| C1 | `cpu` | best_rows | best_context | 1 | Reference |
| C2 | `cpu` | best_rows | best_context | 3 | Moderate fine-tune length |
| C3 | `cpu` | best_rows | best_context | 5 | Heavier local trial |

Decision after Stage C:

- If `5` steps is still fast and stable, CPU remains viable for small iterative fine-tuning.
- If `3` or `5` steps grows too slow, keep one-step local smoke tests and reserve longer runs for remote hardware.

## Stage D: MPS Spot Checks

Purpose: re-test whether `mps` becomes attractive only after workload scale increases.

Do not benchmark all MPS combinations. Only compare against the best CPU settings from earlier stages.

| Test ID | Device | Rows | Context | Max Steps | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| D1 | `mps` | A3_rows | 64 | 1 | Medium-row comparison |
| D2 | `mps` | best_rows | best_context | 1 | Best CPU setting, MPS comparison |
| D3 | `mps` | best_rows | best_context | 3 | Longer-run MPS check |

Decision after Stage D:

- If `mps` is still slower or less stable, keep `cpu` as the local default.
- If `mps` becomes competitive at larger scale, record the crossover point clearly.

## Stage E: Persistence Checks

Purpose: verify that save/reload remains reliable as workloads grow.

Run on the best stable configurations from Stages A to D.

| Test ID | Based On | Check |
| --- | --- | --- |
| E1 | Best small CPU run | save and reload |
| E2 | Best medium CPU run | save and reload |
| E3 | Best MPS run, if any | save and reload |

Important note:

Current evidence shows the reload-check path reproduces a usable evaluation flow, but not the post-step metric exactly. That means persistence is operational, but still needs interpretation before we treat saved fine-tuned artifacts as fully equivalent to the in-memory post-step object.

## Recommended First Batch

Start with these five tests only:

1. A2: `cpu`, `500` rows, `64` context, `1` step
2. A3: `cpu`, `1000` rows, `64` context, `1` step
3. B2: `cpu`, `1000` rows, `128` context, `1` step
4. C2: `cpu`, `1000` rows, `128` context, `3` steps
5. D1: `mps`, `1000` rows, `64` context, `1` step

This batch should tell us whether the practical limit is driven more by rows, context, or device choice.

## Command Pattern

Use this pattern for each test:

```bash
cd /Users/Scott/Documents/Data\ Science/ADSWP/TabPFN-work-scott
python scripts/run_small_finetune_classifier_trial.py --device cpu --rows 1000 --context-samples 128 --max-finetune-steps 1
python scripts/check_saved_finetune_classifier_model.py --device cpu
```

Swap `cpu` for `mps` only on the explicit MPS spot checks.

## Output Artifacts

- Trial log: `outputs/current/tables/tabpfn_finetune_trial_results.csv`
- Reload check log: `outputs/current/tables/tabpfn_finetune_reload_checks.csv`
- Saved models: `outputs/current/models/*.tabpfn_fit`

## Decision We Want at the End

At the end of this plan we should be able to say:

1. the largest row count that remains practical locally
2. the largest context size that remains practical locally
3. whether `cpu` or `mps` is the better local default
4. whether local save/load is good enough for iterative fine-tuning experiments
5. when it becomes more sensible to move to larger remote hardware
