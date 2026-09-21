# FlashAttention-4 (Hopper + Blackwell) backend

TabPFN v3 can dispatch attention to FlashAttention-4 instead of PyTorch's
SDPA. FA4 is the CuTeDSL rewrite of FlashAttention and ships kernels for
Hopper (sm_90), Blackwell datacenter (sm_100/sm_110) and Blackwell consumer /
DGX Spark (sm_120/sm_121), so one backend covers the GPUs FA3 serves and the
ones it cannot. On Hopper, FA4 replaces the FA3 backend: measured on H100 it
matches or beats FA3 at every sequence length and, unlike FA3, has no
short-sequence penalty against SDPA (see [measured speedups](#measured-speedups)).

## When FA4 is used

The dispatcher routes a call to FA4 only when **all** of the following hold:

- The `flash-attn-4` package is importable as `flash_attn.cute`.
- The attention is on a CUDA tensor whose device has compute capability
  9.x, 10.x, 11.x or 12.x. ROCm is rejected explicitly. Ampere (8.x) is left
  to SDPA, which already dispatches FA2 there.
- The dtype is `torch.float16`, or `torch.bfloat16` on a pre-Blackwell GPU.
  On compute capability 10.x+ bf16 stays on SDPA, with a one-time warning
  (see [bf16 on Blackwell](#bf16-on-blackwell)).
- The head dimension is a multiple of 8 within the architecture's range:
  8–256 on sm_90, 8–128 on sm_100 and above (`_FA4_MAX_HEAD_DIM` in
  `fa4_backend.py`). This is wider than FA3's `{64, 96, 128, 192, 256}`, and in
  particular includes the head_dim-16 feature-attention stages, which FA3
  could not serve. On Blackwell those stages are where all of FA4's gain is
  (see below).

There is **no sequence-length threshold**, unlike FA3's
`_FA3_MIN_SEQLEN_FOR_SPEEDUP = 10_000`: FA4 has no measured short-sequence
penalty on either architecture.

## Installing

FA4 is on PyPI, beta releases only:

```bash
pip install "tabpfn[fa4]"        # CUDA 12 torch builds
pip install "tabpfn[fa4-cu13]"   # CUDA 13 torch builds
```

or directly, `pip install --pre "flash-attn-4[cu13]"`. The `[cu13]` extra
selects the CuTeDSL runtime libraries for CUDA 13; the base package pulls the
CUDA 12 ones. Match it to `torch.version.cuda`. No source build is needed —
kernels are JIT-compiled by CuTeDSL on first use for each new shape
(a few seconds per shape, cached for the process).

Verify with:

```python
from flash_attn.cute import flash_attn_func  # noqa: F401
from tabpfn.architectures.shared.fa4_backend import FA4_BACKEND
assert FA4_BACKEND.is_available()
```

**Pinning.** FA4 has weekly betas and no stable release yet. The backend was
written and measured against `4.0.0b30`; the optional extra floors there
rather than pinning, since betas fix things weekly. If a later beta regresses,
`pip install "flash-attn-4==4.0.0b30"` is the known-good.

## Measured speedups

TabPFN v3 `predict()` wall-clock, `n_estimators=1`, fp16 autocast, synthetic
data, `n_test = n_train/10`; ratio = SDPA time / FA4 time (>1 is faster than
SDPA). Full data and method in
[TabPFN#1235](https://github.com/PriorLabs/TabPFN/issues/1235).

**H100 (sm_90)**, torch 2.14.0+cu130, flash-attn-4 4.0.0b30:

| n_train | n_features 10 | 100 | 500 |
|---:|---:|---:|---:|
| 100–300 | 1.01–1.04 | 1.01–1.02 | – |
| 1k | 1.02 | 0.97 | 1.04 |
| 3k | 1.05 | 1.07 | 1.07 |
| 10k | 1.03 | 1.07 | 1.10 |
| 30k | 1.19 | 1.17 | 1.13 |
| 100k | 1.25 | 1.19 | 1.17 |
| 300k | 1.30 | 1.27 | – |

FA4 also matched or beat the FA3 backend it replaces at every point from 10k
up (by 4–10%), and tied below.

**GB200 (sm_100)**, same software:

| n_train | n_features 10 | 100 | 500 |
|---:|---:|---:|---:|
| 300–3k | 0.95–1.04 | 0.98–1.02 | 0.98–1.03 |
| 10k | 1.03 | 0.97 | 1.03 |
| 30k | 1.05 | 1.13 | 1.12 |
| 100k | 1.03 | 1.21 | 1.26 |
| 300k | 1.02 | 1.13 | – |

On Blackwell, torch's SDPA selects cuDNN attention, which already runs the
head_dim-64 ICL calls at FA4 speed; the gain there comes entirely from the
head_dim-16 feature-attention stages, hence the dependence on `n_features`.

Below ~3k rows FA4 is within noise of SDPA on both GPUs, which is why there is
no sequence-length gate.

**Cached prediction, few test rows** (H100, `fit_mode="fit_with_cache"`,
`predict_proba` on n_test rows, n_estimators=1, median of 7):

| n_train | n_test | SDPA | FA4 | FA3 (removed backend) |
|---:|---:|---:|---:|---:|
| 100k | 16 | 36.7 ms | 28.4 ms | 23.1 ms |
| 100k | 256 | 38.8 ms | 27.1 ms | 23.0 ms |
| 300k | 16 | 89.3 ms | 27.7 ms | 23.0 ms |
| 300k | 256 | 92.2 ms | 28.2 ms | 25.1 ms |

The per-predict attention here is 24 calls of `(1, n_test, 8, 64)` against
`(1, n_train, 1, 64)`; the out-of-kernel split-KV path is what keeps FA4 near
FA3 (unsplit, FA4 equals SDPA on these).

## bf16 on Blackwell

On sm_100, FA4's bf16 forward is 18–23% slower than SDPA's from ~10k rows up
(and ~20% slower than FA4's own fp16); fp16 shows no such dip, and Hopper
has none in either dtype (flash-attn-4 4.0.0b30). The backend therefore
declines bf16 calls on compute capability 10.x and above and emits one
`UserWarning` per process saying so. TabPFN's default on CUDA is fp16
autocast, so this only affects users who pass
`inference_precision=torch.bfloat16`. Re-measure with each FA4 beta; the gate
is `_FA4_BF16_SLOW_COMPUTE_CAPABILITY_MAJOR` in `fa4_backend.py`.

## Differences from FA3

FA4 replaces the FA3 (`flash_attn_interface`) backend TabPFN shipped for
Hopper. Things in `fa4_backend.py` that exist because FA4 4.0.0b30 differs
from it:

- `flash_attn.cute.flash_attn_func` returns `(out, lse)` unconditionally.
- Split-KV is not implemented on sm_90 and sm_12x only accepts `num_splits=1`.
  On sm_100/110, where it exists, `num_splits=0` asks FA4's own heuristic.
  Elsewhere the backend splits outside the kernel for short-Q / long-KV calls
  at small batch (`_split_kv_plan`, `_fa4_split_kv`): KV chunks are folded
  into the batch dimension for one launch and the partial outputs combined
  with the log-sum-exps FA4 returns. This is the cached-prediction shape,
  a few test rows against a large training cache; unsplit, such a call
  fills only a handful of SMs.
- The kernel launches one grid entry per batch element, so `batch > 65535`
  fails with `cudaErrorInvalidValue`. `fa4_attn_func` chunks the batch.

## Numerical equivalence

`tests/test_architectures/test_attention_backends.py` carries
`@pytest.mark.hopper` / `@pytest.mark.blackwell` tests asserting FA4 matches
SDPA within `atol=rtol=5e-3` on fp16/bf16 over v3's attention shapes, plus a
100k-key cross-attention and a `batch=70_000` chunking case. They skip
automatically where the GPU or the package is missing.
