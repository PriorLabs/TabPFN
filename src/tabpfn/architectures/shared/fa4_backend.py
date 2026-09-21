#  Copyright (c) Prior Labs GmbH 2026.

"""FlashAttention-4 (CuTeDSL) backend availability and dispatch.

FA4 is the ``flash-attn-4`` package (``pip install "tabpfn[fa4]"``), imported
as ``flash_attn.cute``; it serves Hopper (sm_90) and Blackwell (sm_100) with
fp16/bf16 inputs. Install, dispatch conditions and measurements are in
``fa4_setup.md`` next to this file. Verified against ``flash-attn-4==4.0.0b30``;
``flash_attn_func`` is not Dynamo-traceable, so each call is a graph break
under ``torch.compile``.
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

# Largest head dim per compute-capability major (head dims must also be a
# multiple of 8). Only measured architectures are listed; FA4 also has
# kernels for sm_80, sm_110 and sm_12x, which stay on SDPA until measured.
_FA4_MAX_HEAD_DIM: dict[int, int] = {9: 256, 10: 128}
_FA4_HEAD_DIM_ALIGNMENT = 8

# FA4's bf16 kernels on sm_100 are ~20% slower than SDPA's from ~10k rows
# (fp16 is not; measured on GB200 with 4.0.0b30). Re-check per FA4 beta.
_FA4_BF16_SLOW_COMPUTE_CAPABILITY_MAJORS = frozenset({10})

# Where FA4 implements split-KV (sm_100), ``0`` asks its own heuristic.
_FA4_NUM_SPLITS_SPLIT_KV_ARCHS = 0

# Split-KV outside the kernel where FA4 has none (sm_90): a short-Q call
# against a long KV at small batch otherwise runs on a handful of SMs
# (9.9 ms unsplit vs 0.4 ms split 32 ways for (1, 16) x (1, 1M) on H100).
# The plan is tuned by measurement, not tile occupancy: splitting keeps
# paying past nominal occupancy because each CTA's KV loop is the long pole.
_FA4_SPLIT_KV_Q_TILE = 128  # query rows per work item
_FA4_SPLIT_KV_TARGET_TILES = 128  # ~one per SM on H100
_FA4_SPLIT_KV_MAX_SPLITS = 32
_FA4_SPLIT_KV_MIN_CHUNK = 2048  # keys per chunk below which splitting costs more

# One grid entry per batch element; CUDA caps that dimension.
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
        # Not while tracing: ``warnings.warn`` breaks the graph.
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
    """Chunks to split KV into outside the kernel; 1 means do not split."""
    tiles = batch * -(-seq_q // _FA4_SPLIT_KV_Q_TILE)
    if tiles == 0:
        return 1
    splits = min(
        _FA4_SPLIT_KV_MAX_SPLITS,
        _FA4_SPLIT_KV_TARGET_TILES // tiles,
        seq_kv // _FA4_SPLIT_KV_MIN_CHUNK,
    )
    if batch > 1:
        # The fold is copy-free only if KV divides exactly (a sequence slice
        # of a (B, S, ...) tensor is contiguous only for B == 1).
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
    if main < seq_kv:  # remainder (batch == 1 only, see _split_kv_plan)
        out_t, lse_t = fn(q, k[:, main:], v[:, main:], return_lse=True)
        outs.append(out_t.unsqueeze(1))
        lses.append(lse_t.unsqueeze(1))
    # FA4's own fused combine is not public API; this eager one is within
    # 6e-5 of SDPA.
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

    ``num_splits=None`` picks the per-architecture default: FA4's own
    split-KV where it has one, else split-KV outside the kernel for short-Q
    / long-KV inference calls. Batches above ``_FA4_MAX_BATCH_PER_CALL`` run
    in chunks.
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
        if num_splits == 1 and not needs_grad:  # split outside, inference only
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
    """FA4 as an :class:`~.attention_backends.AttentionBackend`."""

    name = "fa4"

    @staticmethod
    def is_available() -> bool:
        """Whether ``flash-attn-4`` is installed (checked once, at registration)."""
        return _load_fa4_func() is not None

    def is_preferred(self, spec: AttentionSpec) -> bool:
        """Eligibility alone: FA4 has no measured short-sequence penalty."""
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
