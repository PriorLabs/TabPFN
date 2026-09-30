# Aggregation compilation experiment

This is the selected inference prototype from RES-2159, refreshed on main at
`09b4188e` for the released `tabpfn-v3.5-fast-20260909.safetensors` checkpoint.
It is opt-in and lives alongside the architecture. No model source is changed.

`aggregation.install()` patches v3.5 classes for the lifetime of its Python
process. It compiles the distribution cross-attention blocks, feature aggregation
attention and MLP tensor regions, and final CLS readout. Python retains chunking,
residual accumulation, cache dispatch, and memory recovery. Preprocessing and ICL
remain eager. Default compiler tuning and estimator precision are preserved;
`fullgraph=True` exposes graph breaks as errors. This is an inference experiment,
not a public estimator option or a validated training implementation.

The main opportunity is fusing normalization, casts, and surrounding tensor
operations to reduce temporary buffers and memory traffic. Attention backends
and matrix multiplications largely retain their existing kernels. Extensive
autotuning and ICL compilation are deliberately outside this experiment.

From an environment with this checkout's dependencies installed and a CUDA GPU:

```bash
python examples/compilation/run.py \
  --checkpoint ~/.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors \
  --out examples/compilation/results/rtx-run-1
```

The output directory must not exist. Defaults are 50,000 numeric training rows,
200 features, 1,024 test rows, three classes, and exactly four ensemble members
(`auto_scale_n_estimators=False`). Shape arguments can override these defaults.
Current main's ensemble batching policy is retained.

The harness runs three separate processes: eager, compiled with empty dedicated
Inductor/Triton caches, and compiled reusing those disk caches. Each measures the
first `predict_proba` and two subsequent calls. First-call timings exclude Python
imports, data generation, and `fit`; `fit_s` and total process wall time are also
recorded. This is compiler-cache cold, not an OS/filesystem-cache cold boot.
Warm time is the median of the two subsequent calls.

JSON reports include versions, GPU, checkpoint hash, revision, peak allocated
memory, Dynamo counters, repeatability, and probability/label differences versus
eager. The runner requires finite normalized probabilities, exact repeatability,
no graph breaks, cache hits without misses in the cached process, and identical
fresh-compiled/cached outputs. Eager/compiled differences are reported, not
treated as a general model-quality guarantee from one synthetic dataset.

Keep logs, predictions, and compiler caches under the ignored `results/`
directory. Caches from the previous A100 experiments must not be reused for the
RTX run; toolchain and device compatibility matter.
