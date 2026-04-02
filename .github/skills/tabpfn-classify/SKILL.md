---
name: tabpfn-classify
description: 'Run TabPFN classification experiments. Use for binary or multiclass tabular classification, pilot baselines, preprocessing checks, probability evaluation, and reproducible classification result reporting.'
argument-hint: 'Dataset path, target column, metric, and run size'
---

# TabPFN Classify

## When to Use
- Binary or multiclass classification with TabPFN
- Small pilot baselines before scaling up
- Probability-based evaluation such as ROC AUC or log loss
- Comparing preprocessing or environment choices on the same classification task

## Procedure
1. Confirm dataset path, target column, and evaluation metric.
2. Validate that the target is suitable for classification.
3. Start with a pilot subset if runtime or memory is uncertain.
4. Run a baseline classifier with explicit seed and logged settings.
5. Capture time, memory, and classification metrics.
6. Recommend one next change only: more rows, more estimators, preprocessing adjustment, or a device comparison.

## Reporting
1. What was run
2. Metric table
3. Whether the baseline is stable enough to scale
4. One concrete next step
