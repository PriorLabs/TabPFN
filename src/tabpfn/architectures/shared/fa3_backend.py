#  Copyright (c) Prior Labs GmbH 2026.

"""FlashAttention-3 (Hopper) backend availability and dispatch.

FA3 ships as the ``flash_attn_interface`` package built from the ``hopper/``
directory of Dao-AILab/flash-attention. It is Hopper-only (sm_90 WGMMA/TMA);
Blackwell will need FA4. See ``fa3_setup.md`` next to this file for build
instructions. FA3 supports head dims ``{64, 96, 128, 192, 256}`` and requires
fp16/bf16 inputs.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch

_HOPPER_COMPUTE_CAPABILITY_MAJOR = 9
_FA3_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 192, 256})

# Below this Q/K sequence length, FA3's per-call dispatch overhead exceeds
# its kernel-throughput win, so SDPA is faster end-to-end. Used by
# ``is_fa3_preferred`` to gate the auto dispatch.
# Crossover measured on H100 + v3 ICL self-attention: SDPA wins at n_train=1k
# by 10-15%; FA3 wins at n_train=10k (decisively at n_features=10, ~parity at
# n_features=100/500); FA3 wins uniformly from n_train=100k upward.
_FA3_MIN_SEQLEN_FOR_SPEEDUP = 10_000


@functools.cache
def _load_fa3_func() -> Callable | None:
    """Lazily import ``flash_attn_interface.flash_attn_func``; ``None`` if missing."""
    try:
        from flash_attn_interface import (  # type: ignore[import-not-found]  # noqa: PLC0415
            flash_attn_func,
        )
    except ImportError:
        return None
    return flash_attn_func


def is_fa3_importable() -> bool:
    """True iff the FA3 Hopper Python package (``flash_attn_interface``) imports."""
    return _load_fa3_func() is not None


@functools.cache
def _is_hopper(device: torch.device) -> bool:
    """True iff ``device`` is an Nvidia Hopper GPU (compute capability 9.x)."""
    # Reject ROCm: torch.cuda.is_available() is True on ROCm and gfx90a reports
    # capability (9, 0) — same as Hopper sm_90 — but FA3 is Nvidia/CUDA only.
    if torch.version.hip is not None:
        return False
    if not torch.cuda.is_available() or device.type != "cuda":
        return False
    cap = torch.cuda.get_device_capability(device)
    return cap[0] == _HOPPER_COMPUTE_CAPABILITY_MAJOR


def is_fa3_eligible(q: torch.Tensor) -> bool:
    """True iff FA3 can serve this attention call (capability gate, not perf)."""
    head_dim = q.shape[-1]
    return (
        is_fa3_importable()
        and q.is_cuda
        and _is_hopper(q.device)
        and q.dtype in (torch.float16, torch.bfloat16)
        and head_dim in _FA3_SUPPORTED_HEAD_DIMS
    )


def is_fa3_preferred(q: torch.Tensor, k: torch.Tensor) -> bool:
    """True iff FA3 is eligible and past the speedup threshold.

    Compares ``max(seq_q, seq_kv)`` against :data:`_FA3_MIN_SEQLEN_FOR_SPEEDUP`.
    """
    # max(seq_q, seq_kv) so cross-attention with small Q against a large K
    # (test queries vs train cache) still routes through FA3.
    max_seq_len = max(q.shape[1], k.shape[1])
    return is_fa3_eligible(q) and max_seq_len >= _FA3_MIN_SEQLEN_FOR_SPEEDUP


def fa3_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Call ``flash_attn_func`` with the v3 attention layout (B, S, H, D).

    GQA is handled natively by FA3 when ``nheads_q % nheads_k == 0``.
    """
    fn = _load_fa3_func()
    if fn is None:
        raise RuntimeError(
            "FA3 path requested but flash_attn_interface is not importable; "
            "see fa3_setup.md (next to this file)."
        )
    return fn(q, k, v)
