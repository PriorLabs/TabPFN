---
name: tabpfn-explore
description: 'Inspect and validate tabular datasets for TabPFN. Use for schema checks, target validation, missingness review, pilot subset creation, environment-path checks, and deciding whether to run classifier, regressor, fine-tune, or benchmark workflows.'
argument-hint: 'Dataset path, target column, and what you want to validate'
---

# TabPFN Explore

## When to Use
- Quick dataset readiness checks before a TabPFN run
- Deciding whether the task is classification or regression
- Inspecting row counts, feature types, target balance, and missingness
- Creating a tiny pilot subset before a larger experiment
- Verifying the active `tabpfn` import path and avoiding environment mismatch

## Procedure
1. Confirm the dataset path and target column.
2. Inspect row count, column count, dtypes, missingness, and target distribution.
3. Decide task type: classification if target is categorical or low-cardinality labels, regression if target is continuous.
4. Create a tiny stratified subset for smoke tests when appropriate.
5. Verify whether the run should use the local upstream TabPFN source tree instead of an installed package.
6. Recommend the next TabPFN path: classification, regression, fine-tuning, benchmarking, or save/load validation.

## Local Workspace Guidance
- In this workspace, repository-specific APIs may exist only in the local upstream source tree.
- If fine-tuning utilities are missing from the installed package, prefer `PYTHONPATH=/Users/Scott/Documents/Data Science/ADSWP/TabPFN-upstream/src`.
- For Apple Silicon smoke tests, keep first runs small.
