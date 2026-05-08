# FlashAttention-3 (Hopper) backend

TabPFN v3 can dispatch attention to FlashAttention-3 instead of PyTorch's
SDPA. FA3 is expected to be meaningfully faster on Hopper-class GPUs
(H100, H200): an externally contributed micro-benchmark on v2.5 attention
shapes reported ~1.5–1.7× over SDPA (we plan to re-run this internally
before citing in the v3 model report). v3 numbers will land alongside the
v3 report itself.

This is **opt-in**. Default is `attention_backend="auto"`, which uses FA3
when eligible and SDPA otherwise.

## When FA3 is used

The dispatcher routes a call to FA3 only when **all** of the following hold:

- The `flash_attn_interface` Python package is importable.
- The attention is on a CUDA tensor whose device is Hopper-class
  (compute capability 9.0+).
- The dtype is `torch.float16` or `torch.bfloat16`.
- The head dimension is one of `{64, 96, 128, 192, 256}`.

Calls that don't satisfy these (e.g. v3's `dist_embedder` with head_dim=16)
fall back to SDPA in `auto` mode, or raise in `fa3` mode.

### `auto` also applies a sequence-length threshold

Beyond the capability gates above, `auto` mode additionally requires
`max(seq_q, seq_kv) >= _FA3_MIN_SEQLEN_FOR_SPEEDUP` (currently `10_000`,
defined in `fa3_backend.py`). Below that threshold, FA3's per-call
dispatch overhead exceeds its kernel-throughput win and SDPA is faster
end-to-end — so `auto` falls back to SDPA. The threshold is a pure
performance tuning, not a capability gate, and is therefore **bypassed
when `attention_backend="fa3"` is forced** (force still requires the
capability gates, but always picks FA3 once those pass, even on small
shapes).

The 10 000 crossover comes from a v3 forward-pass H100 benchmark: SDPA wins
by 10–15 % at `n_train=1k`; FA3 starts to win at `n_train=10k` (decisively
at `n_features=10`, near parity at `n_features=100/500`); FA3 wins uniformly
from `n_train=100k` upward, reaching 1.49–1.73× at `n_train=10⁶`. Update the
constant if later kernels move the crossover.

`max(seq_q, seq_kv)` rather than `seq_q` alone so cross-attention with
small queries against a large support set (e.g. test rows querying a
million-row train cache) still routes through FA3 — the per-call work
there is dominated by the K/V side and amortises FA3's overhead.

## Building the FA3 wheel

FA3 is not on PyPI — it's built from source against your CUDA toolkit.

### On a Hopper machine (in-place install)

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

This installs `flash_attn_interface`. Verify with:

```python
from flash_attn_interface import flash_attn_func  # noqa: F401
```

### Cross-compile on a non-Hopper / no-GPU node

You can build the wheel on a CPU-only build node (a VM or login node with
the CUDA toolkit but no GPU attached) and run it on an H100 elsewhere. Tell
`nvcc` the target arch explicitly so it doesn't introspect the build host's
GPU:

```bash
export TORCH_CUDA_ARCH_LIST="9.0a"   # FA3 uses sm_90a (WGMMA + TMA)
export MAX_JOBS=4                     # tune to login-node RAM (~32 GB needed)

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
pip wheel . --no-build-isolation -w /tmp/fa3-wheel/
```

`pip wheel` produces a transferable `.whl` (preferable to
`python setup.py install`, which installs in place on the build node).
On the H100 node:

```bash
pip install /tmp/fa3-wheel/flash_attn_3-*.whl
python -c "from flash_attn_interface import flash_attn_func; print('ok')"
```

The build node and runtime node must agree on:

- **CUDA toolkit major version** — ≥ 12.3 for FA3 hopper (12.4+ recommended).
  The login node needs the toolkit installed; the H100 node only needs a
  compatible driver.
- **PyTorch version + Python minor version** — build against the same
  `torch` wheel that the runtime env will use.
- **glibc** — usually fine when build and runtime nodes share an image; can
  bite when they diverge.

Build takes ~30–60 min and ~32 GB RAM; cache the wheel on shared storage so
later H100 nodes can `pip install` directly without rebuilding.

## Selecting the backend

Pass `attention_backend` via `PerformanceOptions`:

```python
from tabpfn.architectures.interface import PerformanceOptions

# Auto: FA3 if eligible, SDPA otherwise (recommended)
options = PerformanceOptions(attention_backend="auto")

# Force SDPA — useful for A/B comparison
options = PerformanceOptions(attention_backend="sdpa")

# Force FA3 — raises RuntimeError if ineligible. Bypasses the
# auto-mode seqlen threshold, so FA3 will run even at small shapes
# where SDPA would win — useful for A/B comparison and for users
# who know their workload sits above the crossover anyway.
options = PerformanceOptions(attention_backend="fa3")

model(x, y, performance_options=options)
```

The setting applies for the duration of one `forward()` call only.

## Numerical equivalence

`tests/test_architectures/test_attention_backends.py` carries Hopper-marked
tests (`@pytest.mark.hopper`) that assert FA3 matches SDPA within
`atol=rtol=5e-3` on fp16/bf16 over v3's attention shapes. They skip on
non-Hopper hosts; run them manually on an H100 until a Hopper CI runner is
in place.
