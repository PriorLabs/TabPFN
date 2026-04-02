---
name: tabpfn-benchmark
description: 'Benchmark TabPFN runs across devices or configurations. Use for CPU vs MPS vs CUDA comparisons, timing and memory capture, pilot benchmark tables, and evidence-based decisions about which setup to scale.'
argument-hint: 'Dataset path, target column, devices or configs to compare, and trial size'
---

# TabPFN Benchmark

## When to Use
- Comparing `cpu`, `mps`, and `cuda` on the same TabPFN workload
- Comparing context sizes, sample counts, or epochs
- Producing benchmark tables for planning larger runs
- Verifying whether a faster-looking device is actually better on current workload size

## Procedure
1. Fix one workload definition: dataset, target, rows, seed, epochs, and context size.
2. Run each candidate device or config with identical settings.
3. Capture wall time, memory footprint, and task metric.
4. Report results in a compact comparison table.
5. Recommend the single best next configuration to scale.

## Local Workspace Guidance
- On this Apple Silicon machine, small fine-tuning smoke tests favored `cpu` over `mps`.
- Do not assume that result holds for larger runs; benchmark again when scale changes materially.
