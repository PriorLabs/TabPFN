# FlashAttention-4 (Hopper + Blackwell) backend

TabPFN v3 can dispatch attention to FlashAttention-4 instead of PyTorch's
SDPA. FA4 is the CuTeDSL rewrite of FlashAttention and ships kernels for
Hopper (sm_90), Blackwell datacenter (sm_100/sm_110) and Blackwell consumer /
DGX Spark (sm_120/sm_121), so one backend covers the GPUs FA3 serves and the
ones it cannot. On Hopper, FA4 replaces the FA3 backend: measured on H100 it
matches or beats FA3 at every sequence length and, unlike FA3, has no
short-sequence penalty against SDPA (see [thresholds](#sequence-length-threshold)).

## When FA4 is used

The dispatcher routes a call to FA4 only when **all** of the following hold:

- The `flash-attn-4` package is importable as `flash_attn.cute`.
- The attention is on a CUDA tensor whose device has compute capability
  9.x, 10.x, 11.x or 12.x. ROCm is rejected explicitly. Ampere (8.x) is left
  to SDPA, which already dispatches FA2 there.
- The dtype is `torch.float16` or `torch.bfloat16`.
- The head dimension is a multiple of 8 within the architecture's range:
  8–256 on sm_90, 8–128 on sm_100 and above (`_FA4_MAX_HEAD_DIM` in
  `fa4_backend.py`). This is wider than FA3's `{64, 96, 128, 192, 256}`, and in
  particular includes the head_dim-16 feature-attention stages, which FA3
  cannot serve and which are worth 5–10% of a forward pass at
  `n_features >= 100`.
- `max(seq_q, seq_kv) >= _FA4_MIN_SEQLEN_FOR_SPEEDUP`.

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

## Sequence-length threshold

`_FA4_MIN_SEQLEN_FOR_SPEEDUP` is currently a **placeholder** inherited from
FA3's 10 000 so the two backends dispatch at the same call sites while
benchmarks are in progress. It is not FA4's measured crossover.

Measured on H100 (v3 `predict()`, `n_estimators=1`, fp16, `n_test = n_train/10`,
ratio = SDPA time / FA4 time):

| n_train | n_features 10 | 100 | 500 |
|---:|---:|---:|---:|
| 100–300 | 1.01–1.04 | 1.01–1.02 | – |
| 1k | 1.02 | 0.97 | 1.04 |
| 3k | 1.05 | 1.07 | 1.07 |
| 10k | 1.03 | 1.07 | 1.10 |
| 30k | 1.19 | 1.17 | 1.13 |
| 100k | 1.25 | 1.19 | 1.17 |
| 300k | 1.30 | 1.27 | – |

FA4 is within noise of SDPA from `n_train=100` and ahead from 3k, so on Hopper
the threshold can be 0. Whether Blackwell wants the same number, and whether
the constant should become per-architecture, is decided by the Blackwell
measurement (tracked in TabPFN#1235). Update the constant and this table when
that lands.

## Differences from the FA3 backend

Things in `fa4_backend.py` that exist because FA4 4.0.0b30 differs from
`flash_attn_interface`:

- `flash_attn.cute.flash_attn_func` returns `(out, lse)` unconditionally.
- Split-KV is not implemented on sm_90 and sm_12x only accepts `num_splits=1`,
  so FA3's manual short-Q split rule is not carried over. On sm_100/110, where
  split-KV exists, `num_splits=0` asks FA4's own heuristic.
- The kernel launches one grid entry per batch element, so `batch > 65535`
  fails with `cudaErrorInvalidValue`. `fa4_attn_func` chunks the batch.

## Numerical equivalence

`tests/test_architectures/test_attention_backends.py` carries
`@pytest.mark.hopper` / `@pytest.mark.blackwell` tests asserting FA4 matches
SDPA within `atol=rtol=5e-3` on fp16/bf16 over v3's attention shapes, plus a
100k-key cross-attention and a `batch=70_000` chunking case. They skip
automatically where the GPU or the package is missing.
