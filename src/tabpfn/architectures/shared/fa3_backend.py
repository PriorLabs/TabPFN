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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from tabpfn.architectures.shared.attention_backends import AttentionBackend

if TYPE_CHECKING:
    from tabpfn.architectures.shared.attention_backends import AttentionSpec

_HOPPER_COMPUTE_CAPABILITY_MAJOR = 9
_FA3_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 192, 256})

# Below this Q/K sequence length, FA3's per-call dispatch overhead exceeds
# its kernel-throughput win, so SDPA is faster end-to-end. Used by
# ``FA3Backend.is_preferred`` to gate the auto dispatch.
# Crossover measured on H100 + v3 ICL self-attention: SDPA wins at n_train=1k
# by 10-15%; FA3 wins at n_train=10k (decisively at n_features=10, ~parity at
# n_features=100/500); FA3 wins uniformly from n_train=100k upward.
_FA3_MIN_SEQLEN_FOR_SPEEDUP = 10_000

_FA3_LONG_KV_SEQLEN = 50_000
_FA3_SPLIT_KV_MAX_SEQLEN_Q = 1024
_FA3_SPLIT_KV_MAX_SEQLEN_Q_LONG_KV = 4096
_FA3_SPLIT_KV_NUM_SPLITS = 32


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


def is_fa3_eligible(device: torch.device, dtype: torch.dtype, head_dim: int) -> bool:
    """True iff FA3 can serve such an attention call (capability gate, not perf).

    Assumes the wheel is installed — see :meth:`FA3Backend.is_available`.
    """
    return (
        device.type == "cuda"
        and _is_hopper(device)
        and dtype in (torch.float16, torch.bfloat16)
        and head_dim in _FA3_SUPPORTED_HEAD_DIMS
    )


def fa3_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Call ``flash_attn_func`` with the v3 attention layout (B, S, H, D).

    GQA is handled natively by FA3 when ``nheads_q % nheads_k == 0``. Short
    query sequences run split over the KV sequence; see
    ``_FA3_SPLIT_KV_NUM_SPLITS``.
    """
    fn = _load_fa3_func()
    if fn is None:
        raise RuntimeError(
            "FA3 path requested but flash_attn_interface is not importable; "
            "see fa3_setup.md (next to this file)."
        )
    # determine num_splits manually, because FA3's heuristic num_splits=0
    # does not compile
    max_seqlen_q = (
        _FA3_SPLIT_KV_MAX_SEQLEN_Q_LONG_KV
        if k.shape[1] > _FA3_LONG_KV_SEQLEN
        else _FA3_SPLIT_KV_MAX_SEQLEN_Q
    )
    splits = _FA3_SPLIT_KV_NUM_SPLITS if q.shape[1] <= max_seqlen_q else 1
    return fn(q, k, v, num_splits=splits)


class FA3Backend(AttentionBackend):
    """FA3 as an :class:`~.attention_backends.AttentionBackend`.

    Wraps the eligibility/crossover logic above unchanged; registered at
    import (below) so FA3 keeps its historical always-on auto-dispatch on
    Hopper.
    """

    name = "fa3"

    @staticmethod
    def is_available() -> bool:
        """Whether the FA3 wheel is installed (checked once, at registration)."""
        return _load_fa3_func() is not None

    def is_preferred(self, spec: AttentionSpec) -> bool:
        """Eligibility plus the measured speedup crossover.

        The threshold compares ``max(seq_q, seq_kv)`` so cross-attention with
        small Q against a large K (test queries vs train cache) still routes
        through FA3; unknown (None) lengths count as 0, i.e. never argue for
        FA3. A quantized KV cache entry is fine: the SDPA wrapper dequantizes
        to q's dtype, which is what the eligibility check validates.
        """
        max_seq_len = max(spec.seq_len_q or 0, spec.seq_len_kv or 0)
        return (
            is_fa3_eligible(spec.device, spec.dtype, spec.head_dim)
            and max_seq_len >= _FA3_MIN_SEQLEN_FOR_SPEEDUP
        )

    def run(
        self,
        q_BSHD: torch.Tensor,
        k_BSJD: torch.Tensor | None,
        v_BSJD: torch.Tensor | None,
        **_informational: Any,  # forward-compat context; safe to ignore
    ) -> torch.Tensor:
        """Run FA3 (k/v arrive dense; the SDPA wrapper dequantizes)."""
        assert k_BSJD is not None
        assert v_BSJD is not None
        return fa3_attn_func(
            q_BSHD.contiguous(), k_BSJD.contiguous(), v_BSJD.contiguous()
        )


# Registered by the shared SDPA module, which owns the consult order.
FA3_BACKEND = FA3Backend()
