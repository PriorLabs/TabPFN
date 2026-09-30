# Aggregation compilation experiment

The selected regions from RES-2159 are integrated into the v3.5 architecture
through the existing `PerformanceOptions.enable_torch_compile` flag. The former
`aggregation.install()` runtime patch helper has been removed. v3 keeps its
existing compilation behavior.

```python
from dataclasses import replace

options = replace(model.get_default_performance_options(), enable_torch_compile=True)
output = model(x, y, task_type="multiclass", performance_options=options)
```

The marked regions are distribution cross-attention blocks, feature aggregation
attention and MLP tensor regions, and final CLS readout. Python retains chunking,
residual accumulation, cache dispatch, and memory recovery. Preprocessing and ICL
remain eager. The flag is per forward call and can be turned off again without
rebuilding the model. Compiler initialization is lazy and adds no compiler state
to model serialization.

Compilation uses default tuning, `dynamic=True`, and `fullgraph=True`. Leading
batch, row, and sequence dimensions are explicitly marked independently dynamic;
embedding width and model parameter shapes stay specialized. This avoids
accidental equality guards when unrelated data dimensions happen to match in the
first input. Singleton dimensions and changes in control flow, dtype, strides,
or attention backend can still require additional variants.

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

Pass `--repo /path/to/another/TabPFN/checkout` to run the same harness against
another revision, including unmodified main. The runner records the imported
checkout's revision and source path.

The sklearn API does not directly expose `PerformanceOptions`. The benchmark
uses a forward pre-hook to set that option on its own estimator instances; it
does not replace any architecture methods. Region selection is implemented in
the architecture source.

To check graph reuse while train/test lengths, feature counts, and batch sizes
change in one process:

```bash
python examples/compilation/shapes.py \
  --checkpoint ~/.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors \
  --out examples/compilation/results/dynamic-shapes.json
```

This exercises full, singleton-batch, and chunked paths. It records new and
cumulative graph counts as shapes change, eager/compiled timings, and numerical
differences versus eager execution. Recompilation is an observation rather than
a failing assertion. A failed path is recorded and stopped without retries.

To additionally test model KV-cache creation and reuse, plus compiler disk-cache
reuse in a fresh process, against a specific checkout:

```bash
python examples/compilation/run_shapes.py \
  --repo /path/to/TabPFN-checkout \
  --checkpoint ~/.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors \
  --out examples/compilation/results/shape-cache-run-1
```

The runner uses separate compiler caches for the cases with and without model
KV caching. Each starts empty and is reused in a second process. Model KV caches
are rebuilt for each training dataset, then reused for 256, 512, and 768 test
rows with that context fixed. Eager references run before each compiled call.
Failed paths are excluded from the disk-cache replay; each process has a
10-minute timeout. These direct model probes use FP16 autocast, numeric random
inputs, and the same row/column/batch shapes as the original shape check.

The harness runs three separate processes: eager, compiled with empty dedicated
Inductor/Triton caches, and compiled reusing those disk caches. Each measures the
first `predict_proba` and two subsequent calls. First-call timings exclude Python
imports, data generation, and `fit`; `fit_s` and total process wall time are also
recorded. This is compiler-cache cold, not an OS/filesystem-cache cold boot.
Warm time is the median of the two subsequent calls.

JSON reports include versions, GPU, checkpoint hash, revision, peak allocated
memory, Dynamo counters, repeatability, and probability/label differences versus
eager. The runner requires finite normalized probabilities, exact repeatability,
no graph breaks, FX graph cache hits without misses in the cached process, and identical
fresh-compiled/cached outputs. Eager/compiled differences are reported, not
treated as a general model-quality guarantee from one synthetic dataset.

Keep logs, predictions, and compiler caches under the ignored `results/`
directory. Caches from the previous A100 experiments must not be reused for the
RTX run; toolchain and device compatibility matter.

## Native flag and dynamic shapes, 2026-09-30

The native flag implementation was validated on the same RTX and checkpoint,
using PyTorch 2.10.0+cu128 and the default benchmark shape:

| Mode | First prediction | Warm prediction |
| --- | ---: | ---: |
| Eager | 9.97 s | 9.60 s |
| Native flag, empty caches | 23.35 s | 5.05 s |
| Native flag, disk caches reused in a new process | 11.88 s | 5.05 s |

The larger benchmark retained the warm speedup and memory reduction of the
prototype. All labels matched eager on this synthetic classifier dataset;
probability differences had mean absolute error 0.0000487 and maximum 0.01365.
The cached process had seven FX graph cache hits and no misses. All calls were
finite and repeatable, with no graph breaks and identical cold/cached outputs.

The separate shape probe changed row counts, train lengths, feature counts, and
batch sizes in one process. Cumulative graph counts were `[5, 5, 5]` for full
inference, `[6, 6]` after switching to singleton batches, and `[7, 7, 7]` after
switching to chunked inference. Each new path needed a variant; subsequent
dataset shape changes reused it. On these random-noise inputs, probability
differences stayed below 0.000486, while label agreement varied from 98.3% to
99.7%. This is a compilation/shape check, not a predictive-quality benchmark.

CPU validation passed the selective-compilation, changing-shape, KV-cache,
serialization, flag-toggle, and gradient checks, plus the existing architecture
tests. Full measurements are in [native_flag_20260930.json](native_flag_20260930.json).

## Unmodified main comparison, 2026-09-30

The same harness also ran against unmodified main at `09b4188e`, the base of
this branch, in a separate detached worktree. Hardware, PyTorch, checkpoint,
numeric dataset, precision, and ensemble size matched the native flag run above:
50,000 training rows, 1,024 test rows, 200 features, and four estimators.
Eager predictions were bit-identical across the two revisions.

| Implementation | First prediction, empty caches | First prediction, disk caches reused | Warm prediction | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Main, eager | 10.04 s | N/A | 9.60 s | 7.12 GB |
| Main, existing compile flag | 95.38 s | 45.32 s | 4.53 s | 4.04 GB |
| This branch, smaller compiled regions | 23.35 s | 11.88 s | 5.05 s | 4.84 GB |

Main's larger regions reduced warm prediction time by another 10.3% versus this
branch, and used less GPU memory. The smaller regions reduced the first-call
time by 4.1x with empty caches and 3.8x with disk caches. First-call measurements
include prediction execution, but exclude imports, data generation, and fit.
Warm timings use the empty-cache process's two subsequent predictions; the
cached main process was similar at 4.52 s.

Main captured 4,524 operations across four FX graphs, versus 424 operations
across seven graphs on this branch. Disk cache reuse still involves tracing
and graph processing; the cached first call is not equivalent to a warm call.
The smaller regions favor short jobs and frequent process starts, while main's
larger regions retain an advantage for many repeated predictions of this shape.
Changing-shape reuse on main was not tested in this run.

Main completed without intervention despite symbolic-shape warnings. All
benchmark checks passed: finite normalized probabilities, exact repeatability,
no graph breaks, four FX graph cache hits and no misses in the cached process,
and identical cold/cached outputs. All predicted labels matched eager on this
dataset; mean absolute probability difference was 0.0000429 and maximum 0.00778.
Full measurements are in [main_flag_20260930.json](main_flag_20260930.json).

## Original runtime-patch RTX check, 2026-09-30

These historical measurements used the former runtime patch helper. On the
existing SkyPilot RTX PRO 6000 Blackwell (96 GB), using PyTorch
2.10.0+cu128 and the default shape above:

| Mode | First prediction | Warm prediction | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| Eager | 10.04 s | 9.60 s | 7.12 GB |
| Aggregation compiled, empty caches | 22.79 s | 5.06 s | 4.84 GB |
| Aggregation compiled, disk caches reused in a new process | 11.54 s | 5.06 s | 4.84 GB |

This quick check gives 1.90x warm speedup and 32% lower peak allocated memory.
Seven FX graphs were compiled; the cached process had seven FX cache hits and
zero FX cache misses. All outputs were finite and repeatable, with no graph
breaks and identical cold/cached outputs. All 1,024 predicted labels matched
eager. Mean absolute probability difference was 0.0000422; the maximum was
0.01243, so probability outputs are not bit-identical to eager.

Full measurements and provenance are in [rtx_20260930.json](rtx_20260930.json).
Each warm measurement summarizes two calls on one synthetic dataset. These
numbers are not directly comparable to the earlier A100 run: device, PyTorch,
checkpoint, and main's ensemble batching implementation differ.
