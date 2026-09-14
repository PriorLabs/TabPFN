#  Copyright (c) Prior Labs GmbH 2026.

"""FlashAttention-4 (CuTeDSL) backend availability and dispatch.

FA4 ships on PyPI as the ``flash-attn-4`` package (``pip install
"tabpfn[fa4]"`` or ``"tabpfn[fa4-cu13]"``; beta releases only) and is imported
as ``flash_attn.cute``. See ``fa4_setup.md`` next to this file. Its kernels
are written in CuTeDSL and cover Hopper (sm_90), Blackwell datacenter
(sm_100/sm_110) and Blackwell consumer / DGX Spark (sm_120/sm_121), so one
backend serves both the architecture FA3 already covers and the one it cannot.
FA4 requires fp16/bf16 inputs; the supported head dims depend on the
architecture (see ``_fa4_max_head_dim``).

FA4 replaces the FA3 backend this module descends from. Differences from
``flash_attn_interface`` (FA3) that shaped it:

- ``flash_attn.cute.flash_attn_func`` returns ``(out, lse)`` unconditionally.
- Split-KV is not implemented for sm_90 (``"SplitKV not supported on SM 9.0"``)
  so FA3's manual ``num_splits`` rule cannot be carried over on Hopper, and
  sm_12x only accepts ``num_splits=1``. FA4 does have a working
  ``num_splits=0`` heuristic where split-KV exists.
- The kernel launches one grid entry per batch element, so ``batch > 65535``
  fails with ``cudaErrorInvalidValue``; ``fa4_attn_func`` chunks the batch.

Verified against ``flash-attn-4==4.0.0b30``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from tabpfn.architectures.shared.attention_backends import AttentionBackend

if TYPE_CHECKING:
    from tabpfn.architectures.shared.attention_backends import AttentionSpec

# Largest head dim FA4 accepts per compute-capability major; head dims must
# also be a multiple of 8. sm_90 takes up to 256, sm_100/110 up to 128 (plus
# DeepSeek-specific shapes TabPFN does not use). sm_12x is not validated by
# FA4 itself; keep it at the sm_100 range until measured.
_FA4_MAX_HEAD_DIM: dict[int, int] = {9: 256, 10: 128, 11: 128, 12: 128}
_FA4_HEAD_DIM_ALIGNMENT = 8

# FA4 also has kernels for Ampere (sm_80), but SDPA already dispatches FA2
# there, so Ampere is deliberately absent from the table above.

# PLACEHOLDER, NOT MEASURED FOR FA4. Inherited from the FA3 backend's
# H100 crossover. On Hopper, FA4 shows no low-end penalty (see fa4_setup.md);
# the value is set by the benchmark, per architecture if Hopper and Blackwell
# cross at different lengths (TabPFN#1235).
_FA4_MIN_SEQLEN_FOR_SPEEDUP = 10_000

# ``num_splits`` passed to FA4 where split-KV exists (sm_100/110). ``0``
# asks FA4's own heuristic. Whether that beats ``1`` on TabPFN's short-Q /
# long-KV cross-attention is a benchmark question, not a settled one.
_FA4_NUM_SPLITS_SPLIT_KV_ARCHS = 0

# FA4 launches one grid entry per batch element; CUDA caps that dimension.
_FA4_MAX_BATCH_PER_CALL = 65_535


@functools.cache
def _load_fa4_func() -> Callable | None:
    """Lazily import ``flash_attn.cute.flash_attn_func``; ``None`` if missing."""
    try:
        from flash_attn.cute import (  # type: ignore[import-not-found]  # noqa: PLC0415
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
    return (
        max_head_dim is not None
        and dtype in (torch.float16, torch.bfloat16)
        and _FA4_HEAD_DIM_ALIGNMENT <= head_dim <= max_head_dim
        and head_dim % _FA4_HEAD_DIM_ALIGNMENT == 0
    )


def _num_splits_for(device: torch.device) -> int:
    """``num_splits`` FA4 accepts on ``device``; split-KV is sm_100/110 only."""
    major = _compute_capability_major(device)
    if major in (10, 11):
        return _FA4_NUM_SPLITS_SPLIT_KV_ARCHS
    return 1


def fa4_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Call ``flash_attn.cute.flash_attn_func`` with the v3 layout (B, S, H, D).

    GQA is handled natively by FA4 when ``nheads_q % nheads_k == 0``.
    ``num_splits=None`` picks the per-architecture default; pass a value to
    override it (benchmarks). Batches above ``_FA4_MAX_BATCH_PER_CALL`` are
    run in chunks.
    """
    fn = _load_fa4_func()
    if fn is None:
        raise RuntimeError(
            "FA4 path requested but flash_attn.cute is not importable; "
            "install it with `pip install 'tabpfn[fa4]'` "
            "(see fa4_setup.md next to this file)."
        )
    if num_splits is None:
        num_splits = _num_splits_for(q.device)

    batch = q.shape[0]
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

    An eligibility gate, a sequence-length crossover, and a ``run`` that
    hands dense ``k``/``v`` to the kernel. Registered by the shared SDPA
    module, which owns the consult order.
    """

    name = "fa4"

    @staticmethod
    def is_available() -> bool:
        """Whether ``flash-attn-4`` is installed (checked once, at registration)."""
        return _load_fa4_func() is not None

    def is_preferred(self, spec: AttentionSpec) -> bool:
        """Eligibility plus the speedup crossover.

        Compares ``max(seq_q, seq_kv)``
        so cross-attention with small Q against a large K still routes here;
        unknown (None) lengths count as 0 and never argue for FA4.
        """
        max_seq_len = max(spec.seq_len_q or 0, spec.seq_len_kv or 0)
        return (
            is_fa4_eligible(spec.device, spec.dtype, spec.head_dim)
            and max_seq_len >= _FA4_MIN_SEQLEN_FOR_SPEEDUP
        )

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


# Registered by the shared SDPA module, which owns the consult order.
FA4_BACKEND = FA4Backend()
