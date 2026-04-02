---
name: tabpfn-regress
description: 'Run TabPFN regression experiments. Use for continuous targets, pilot regression baselines, target sanity checks, save/load validation, and reproducible regression result reporting.'
argument-hint: 'Dataset path, target column, metric, and run size'
---

# TabPFN Regress

## When to Use
- Continuous-target regression with TabPFN
- Small pilot regression baselines before scaling up
- RMSE, MAE, or similar regression evaluation
- Checking whether target transformation or scaling is worth testing

## Procedure
1. Confirm dataset path, target column, and metric.
2. Validate that the target is continuous enough for regression.
3. Start with a small subset if runtime or memory is uncertain.
4. Run a baseline regressor with explicit seed and logged settings.
5. Capture time, memory, and regression metrics.
6. Recommend one next change only: more rows, target transformation, preprocessing adjustment, or a benchmark comparison.

## Reporting
1. What was run
2. Metric table
3. Whether the baseline is ready to scale
4. One concrete next step
