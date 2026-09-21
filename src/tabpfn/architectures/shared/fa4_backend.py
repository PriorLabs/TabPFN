#  Copyright (c) Prior Labs GmbH 2026.

"""FlashAttention-4 (CuTeDSL) backend availability and dispatch.

FA4 ships on PyPI as the ``flash-attn-4`` package (``pip install
"tabpfn[fa4]"`` or ``"tabpfn[fa4-cu13]"``; beta releases only) and is imported
as ``flash_attn.cute``. See ``fa4_setup.md`` next to this file. Its kernels
are written in CuTeDSL and cover Hopper (sm_90) and Blackwell (sm_100), the
two architectures this backend dispatches on. FA4 requires fp16/bf16 inputs;
the supported head dims depend on the architecture (see
``_fa4_max_head_dim``), and bf16 is left to SDPA on Blackwell (see
``_is_bf16_slow``).

Properties of ``flash_attn.cute`` (as of ``flash-attn-4==4.0.0b30``) that
this module accounts for:

- ``flash_attn_func`` returns a ``(out, lse)`` tuple; ``lse`` is ``None``
  unless ``return_lse=True``.
- Split-KV exists on sm_100 only, with a working ``num_splits=0``
  heuristic; sm_90 has none. There, short-Q / long-KV inference calls are
  split outside the kernel using the returned LSE (``_fa4_split_kv``).
- The kernel launches one grid entry per batch element, so ``batch > 65535``
  fails with ``cudaErrorInvalidValue``; ``fa4_attn_func`` chunks the batch.
- ``flash_attn_func`` is not Dynamo-traceable, so under ``torch.compile``
  each FA4 call is a graph break. Measured on v3 ``predict()`` (H100): within
  noise at 30k rows, ~4% at 100k, with FA4 still well ahead of SDPA.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from tabpfn.architectures.shared.attention_backends import AttentionBackend

if TYPE_CHECKING:
    from tabpfn.architectures.shared.attention_backends import AttentionSpec

# Largest head dim FA4 accepts per compute-capability major; head dims must
# also be a multiple of 8. sm_90 takes up to 256, sm_100 up to 128 (plus
# DeepSeek-specific shapes TabPFN does not use). Only architectures this
# backend has been measured on are listed: FA4 also has kernels for Ampere
# (sm_80, where SDPA already dispatches FA2), sm_110 and consumer Blackwell
# (sm_12x, an SM80-style kernel that FA4 does not validate head dims for);
# those stay on SDPA until measured.
_FA4_MAX_HEAD_DIM: dict[int, int] = {9: 256, 10: 128}
_FA4_HEAD_DIM_ALIGNMENT = 8

# No sequence-length gate. Measured on H100 and GB200 (TabPFN#1235), FA4 is
# within noise of SDPA from n_train=100 up and ahead from ~3k, so there is no
# short-sequence regime where SDPA should be preferred.

# On sm_100 FA4's bf16 kernels run ~20% slower than SDPA's from ~10k rows
# up, while fp16 does not (measured on GB200, flash-attn-4 4.0.0b30). bf16 is
# therefore left to SDPA there, with a one-time warning so the user knows why
# the backend is not being used. This is a measurement, not a property of the
# architecture: re-check it when moving to a newer FA4 beta.
_FA4_BF16_SLOW_COMPUTE_CAPABILITY_MAJORS = frozenset({10})

# ``num_splits`` passed to FA4 where it implements split-KV (sm_100): ``0``
# asks its own heuristic.
_FA4_NUM_SPLITS_SPLIT_KV_ARCHS = 0

# Split-KV done outside the kernel, for architectures where FA4 has none
# (sm_90 in particular). A short-Q call against a long KV at small batch
# runs on a handful of SMs: (1, 16, 8, 64) x (1, 1M, 1, 64) takes 9.9 ms
# unsplit on H100 and 0.4 ms split 32 ways. ``_split_kv_plan`` folds KV
# chunks into the batch dimension so one FA4 launch fills the GPU; the
# partial outputs are combined with the per-chunk log-sum-exps FA4 returns.
# The plan is tuned by measurement on H100, not derived from occupancy:
# FA4 packs GQA query heads into its m-tiles, so by tile count an 8:1 call
# with seq_q=256 already fills the GPU, yet splitting it 32 ways is 20%
# faster than 8 ways and 4x faster than unsplit, because each CTA's serial
# loop over KV is the long pole. Treat ``_FA4_SPLIT_KV_Q_TILE`` as a
# work-item size, not the kernel's tile.
_FA4_SPLIT_KV_Q_TILE = 128
_FA4_SPLIT_KV_TARGET_TILES = 128  # ~one CTA per SM on H100 (132)
_FA4_SPLIT_KV_MAX_SPLITS = 32
_FA4_SPLIT_KV_MIN_CHUNK = 2048  # keys per chunk below which splitting costs more

# FA4 launches one grid entry per batch element; CUDA caps that dimension.
_FA4_MAX_BATCH_PER_CALL = 65_535


@functools.cache
def _load_fa4_func() -> Callable | None:
    """Lazily import ``flash_attn.cute.flash_attn_func``; ``None`` if missing."""
    try:
        from flash_attn.cute import (  # type: ignore[import-not-found,import-untyped]  # noqa: PLC0415
            flash_attn_func,
        )
    except ImportError:
        return None
    return flash_attn_func


@functools.cache
def _compute_capability_major(device: torch.device) -> int | None:
    """Compute-capability major of an Nvidia CUDA ``device``, else ``None``."""
    # Reject ROCm: torch.cuda.is_available() is True on ROCm and gfx90a reports
    # capability (9, 0) — same as Hopper sm_90 — but FA4 is Nvidia/CUDA only.
    if torch.version.hip is not None:
        return None
    if not torch.cuda.is_available() or device.type != "cuda":
        return None
    return torch.cuda.get_device_capability(device)[0]


def _fa4_max_head_dim(device: torch.device) -> int | None:
    """Largest head dim FA4 serves on ``device``; ``None`` if unsupported."""
    major = _compute_capability_major(device)
    return None if major is None else _FA4_MAX_HEAD_DIM.get(major)


def is_fa4_eligible(device: torch.device, dtype: torch.dtype, head_dim: int) -> bool:
    """True iff FA4 can serve such an attention call (capability gate, not perf).

    Assumes the package is installed — see :meth:`FA4Backend.is_available`.
    """
    max_head_dim = _fa4_max_head_dim(device)
    if (
        max_head_dim is None
        or dtype not in (torch.float16, torch.bfloat16)
        or not _FA4_HEAD_DIM_ALIGNMENT <= head_dim <= max_head_dim
        or head_dim % _FA4_HEAD_DIM_ALIGNMENT != 0
    ):
        return False
    if dtype is torch.bfloat16 and _is_bf16_slow(device):
        # dtype is the only reason this call is declined, so the advice to
        # use fp16 holds. Not while tracing: ``warnings.warn`` is not
        # traceable (it breaks the graph), and the registry consults backends
        # inside compiled regions.
        if not torch.compiler.is_compiling():
            _warn_bf16_blackwell_once()
        return False
    return True


def _is_bf16_slow(device: torch.device) -> bool:
    """True on architectures where FA4 bf16 loses to SDPA (sm_100)."""
    return _compute_capability_major(device) in _FA4_BF16_SLOW_COMPUTE_CAPABILITY_MAJORS


@functools.cache
def _warn_bf16_blackwell_once() -> None:
    """Tell the user, once per process, why bf16 calls stay on SDPA."""
    warnings.warn(
        "FlashAttention-4 is installed but is not used for bfloat16 attention "
        "on Blackwell GPUs, where its bf16 kernels are slower than PyTorch "
        "SDPA; falling back to SDPA. Use float16 (TabPFN's default autocast "
        "dtype on CUDA) to enable FA4.",
        stacklevel=4,
    )


def _num_splits_for(device: torch.device) -> int:
    """``num_splits`` FA4 accepts on ``device``; split-KV is sm_100 only."""
    if _compute_capability_major(device) == 10:
        return _FA4_NUM_SPLITS_SPLIT_KV_ARCHS
    return 1


def _split_kv_plan(batch: int, seq_q: int, seq_kv: int) -> int:
    """Chunks to split KV into outside the kernel; 1 means do not split.

    Splits only when the unsplit launch would under-fill the GPU (few
    ``batch x q-tile`` work items) and the chunks stay long enough to be
    worth a launch each.
    """
    tiles = batch * -(-seq_q // _FA4_SPLIT_KV_Q_TILE)
    if tiles == 0:  # empty query: nothing to split
        return 1
    splits = min(
        _FA4_SPLIT_KV_MAX_SPLITS,
        _FA4_SPLIT_KV_TARGET_TILES // tiles,
        seq_kv // _FA4_SPLIT_KV_MIN_CHUNK,
    )
    if batch > 1:
        # A slice along the sequence of a (B, S, ...) tensor is only
        # contiguous for B == 1, so the batch > 1 fold must divide KV
        # exactly to stay copy-free: take the largest divisor.
        while splits > 1 and seq_kv % splits:
            splits -= 1
    return max(splits, 1)


def _fa4_split_kv(
    fn: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, splits: int
) -> torch.Tensor:
    """Attention over ``splits`` KV chunks in one launch, combined by LSE."""
    batch, seq_q, n_heads, head_dim = q.shape
    seq_kv, n_kv_heads = k.shape[1], k.shape[2]
    chunk = seq_kv // splits
    main = chunk * splits

    def fold(t: torch.Tensor) -> torch.Tensor:
        return t[:, :main].reshape(batch * splits, chunk, n_kv_heads, head_dim)

    q_rep = (
        q.unsqueeze(1)
        .expand(batch, splits, seq_q, n_heads, head_dim)
        .reshape(batch * splits, seq_q, n_heads, head_dim)
    )
    out, lse = fn(q_rep, fold(k), fold(v), return_lse=True)
    outs = [out.view(batch, splits, seq_q, n_heads, head_dim)]
    lses = [lse.view(batch, splits, n_heads, seq_q)]
    if main < seq_kv:  # the remainder that did not divide evenly
        out_t, lse_t = fn(q, k[:, main:], v[:, main:], return_lse=True)
        outs.append(out_t.unsqueeze(1))
        lses.append(lse_t.unsqueeze(1))
    # FA4 ships a fused combine for its own split-KV path, but it is not part
    # of the public interface. This eager combine reweights fp16-rounded
    # partial outputs in fp32; measured against SDPA it stays within 6e-5.
    out_all = torch.cat(outs, dim=1).float()  # (B, S, Q, H, D)
    lse_all = (
        torch.cat(lses, dim=1).permute(0, 1, 3, 2).unsqueeze(-1)
    )  # (B, S, Q, H, 1)
    weight = torch.exp(lse_all - lse_all.max(dim=1, keepdim=True).values)
    combined = (out_all * weight).sum(dim=1) / weight.sum(dim=1)
    return combined.to(q.dtype)


def fa4_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Call ``flash_attn.cute.flash_attn_func`` with the v3 layout (B, S, H, D).

    GQA is handled natively by FA4 when ``nheads_q % nheads_k == 0``.
    ``num_splits=None`` picks the per-architecture default: FA4's own
    split-KV heuristic where it has one, otherwise split-KV outside the
    kernel for short-Q / long-KV calls (see ``_split_kv_plan``); pass a
    value to override (benchmarks). Batches above ``_FA4_MAX_BATCH_PER_CALL``
    are run in chunks.
    """
    fn = _load_fa4_func()
    if fn is None:
        raise RuntimeError(
            "FA4 path requested but flash_attn.cute is not importable; "
            "install it with `pip install 'tabpfn[fa4]'` "
            "(see fa4_setup.md next to this file)."
        )
    batch = q.shape[0]
    if num_splits is None:
        num_splits = _num_splits_for(q.device)
        needs_grad = torch.is_grad_enabled() and (
            q.requires_grad or k.requires_grad or v.requires_grad
        )
        # No kernel split-KV here: do it outside. Inference only; a backward
        # through the eager combine is not something this path is tested for.
        if num_splits == 1 and not needs_grad:
            splits = _split_kv_plan(batch, q.shape[1], k.shape[1])
            if splits > 1:
                return _fa4_split_kv(fn, q, k, v, splits)

    if batch <= _FA4_MAX_BATCH_PER_CALL:
        out, _lse = fn(q, k, v, num_splits=num_splits)
        return out
    outputs = []
    for start in range(0, batch, _FA4_MAX_BATCH_PER_CALL):
        stop = start + _FA4_MAX_BATCH_PER_CALL
        out, _lse = fn(
            q[start:stop], k[start:stop], v[start:stop], num_splits=num_splits
        )
        outputs.append(out)
    return torch.cat(outputs)


class FA4Backend(AttentionBackend):
    """FA4 as an :class:`~.attention_backends.AttentionBackend`.

    An eligibility gate and a ``run`` that hands dense ``k``/``v`` to the
    kernel.
    """

    name = "fa4"

    @staticmethod
    def is_available() -> bool:
        """Whether ``flash-attn-4`` is installed (checked once, at registration)."""
        return _load_fa4_func() is not None

    def is_preferred(self, spec: AttentionSpec) -> bool:
        """Eligibility alone: FA4 has no measured short-sequence penalty, so
        every call it can serve, it takes (see the module comment on the
        absent sequence-length gate and the bf16 exception on Blackwell).
        """
        return is_fa4_eligible(spec.device, spec.dtype, spec.head_dim)

    def run(
        self,
        q_BSHD: torch.Tensor,
        k_BSJD: torch.Tensor | None,
        v_BSJD: torch.Tensor | None,
        **_informational: Any,  # forward-compat context; safe to ignore
    ) -> torch.Tensor:
        """Run FA4 (k/v arrive dense; the SDPA wrapper dequantizes)."""
        assert k_BSJD is not None
        assert v_BSJD is not None
        return fa4_attn_func(
            q_BSHD.contiguous(), k_BSJD.contiguous(), v_BSJD.contiguous()
        )


FA4_BACKEND = FA4Backend()
