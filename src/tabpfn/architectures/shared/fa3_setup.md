# FlashAttention-3 (Hopper) backend

TabPFN v3 can dispatch attention to FlashAttention-3 instead of PyTorch's
SDPA. On Hopper-class GPUs (H100, H200), FA3 outperforms SDPA above a
sequence-length crossover empirically measured on v3 ICL self-attention —
see the comment on `_FA3_MIN_SEQLEN_FOR_SPEEDUP` in `fa3_backend.py` (next
to this file) for the measured numbers and the [auto-mode threshold
section](#auto-also-applies-a-sequence-length-threshold) below for how the
constant is applied at dispatch time.

## When FA3 is used

The dispatcher routes a call to FA3 only when **all** of the following hold:

- The `flash_attn_interface` Python package is importable.
- The attention is on a CUDA tensor whose device is Hopper-class
  (compute capability 9.0+).
- The dtype is `torch.float16` or `torch.bfloat16`.
- The head dimension is one of `{64, 96, 128, 192, 256}`.
- `max(seq_q, seq_kv) >= _FA3_MIN_SEQLEN_FOR_SPEEDUP` (currently `10_000`,
  defined in `fa3_backend.py`) The 10 000 crossover comes from a v3 forward-pass
  H100 benchmark: SDPA wins by 10–15 % at `n_train=1k`; FA3 starts to win
  at `n_train=10k` (decisively at `n_features=10`, near parity at `n_features=100/500`);
  FA3 wins uniformly from `n_train=100k` upward, reaching 1.49–1.73× at `n_train=10⁶`.
  Update the constant if later kernels move the crossover.

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


## Numerical equivalence

`tests/test_architectures/test_attention_backends.py` carries Hopper-marked
tests (`@pytest.mark.hopper`) that assert FA3 matches SDPA within
`atol=rtol=5e-3` on fp16/bf16 over v3's attention shapes. They skip on
non-Hopper hosts; run them manually on an H100 until a Hopper CI runner is
in place.
