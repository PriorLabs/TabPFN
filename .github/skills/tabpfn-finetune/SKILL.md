---
name: tabpfn-finetune
description: 'Run TabPFN fine-tuning workflows. Use for fine-tuning smoke tests, local hardware readiness, one-epoch pilot runs, context-size trials, save/load checks, and environment fixes when local upstream APIs differ from an installed package.'
argument-hint: 'Dataset path, target column, rows, device, epochs, and context samples'
---

# TabPFN Finetune

## When to Use
- Testing whether local hardware can execute TabPFN fine-tuning
- Running one-epoch smoke tests before larger fine-tuning jobs
- Comparing `cpu`, `mps`, and `cuda` using identical settings
- Validating save/load paths for fine-tuned models
- Fixing environment-path issues where installed `tabpfn` differs from the repo version

## Procedure
1. Confirm dataset path, target column, device candidates, and target metric.
2. Verify the import path and prefer the local upstream source tree when fine-tuning APIs are repo-specific.
3. Create a tiny stratified subset for the first run.
4. Run a one-epoch or one-step fine-tuning smoke test with timing and memory capture.
5. Re-run the same config across devices for a fair comparison.
6. Recommend the safest next scale-up step.

## Apple Silicon Guidance
- This workspace runs on macOS Apple Silicon.
- For the tested local smoke workload on this machine, `cpu` outperformed `mps`.
- Start with `cpu` for small local fine-tuning trials, then re-check empirically as row count or context size increases.

## Practical Defaults
- Rows: 128 to 500 for first pass
- Epochs: 1
- Context samples: 64 to 128
- Always record wall time and memory before scaling
