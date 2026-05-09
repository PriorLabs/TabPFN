#  Copyright (c) Prior Labs GmbH 2026.

"""Numerical-equivalence tests for the v3 attention backend selector.

The non-Hopper tests (sdpa-only, eligibility checks, error paths) run on
any GPU — or CPU — and exercise the dispatch logic with FA3 unavailable.

The ``hopper``-marked tests require a Hopper-class GPU (compute
capability 9.0+) AND the ``flash_attn_interface`` package built from
Dao-AILab/flash-attention's ``hopper/`` directory. They ``skip``
automatically on any other host; run them manually on an H100 until a
Hopper CI runner is in place.
"""

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.interface import AttentionBackend
from tabpfn.architectures.shared import fa3_backend
from tabpfn.architectures.shared.fa3_backend import (
    is_fa3_eligible_for,
    is_fa3_importable,
    is_fa3_preferred_for,
)
from tabpfn.architectures.tabpfn_v3 import _batched_scaled_dot_product_attention


def _has_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 9


_FA3_RUNNABLE = _has_hopper() and is_fa3_importable()
_skip_unless_fa3 = pytest.mark.skipif(
    not _FA3_RUNNABLE, reason="requires Hopper GPU and flash_attn_interface"
)


def _make_qkv(
    *,
    batch: int,
    seq_q: int,
    seq_kv: int,
    n_heads_q: int,
    n_heads_kv: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(0)
    kw = {"device": device, "dtype": dtype, "generator": g}
    q = torch.randn(batch, seq_q, n_heads_q, head_dim, **kw)
    k = torch.randn(batch, seq_kv, n_heads_kv, head_dim, **kw)
    v = torch.randn(batch, seq_kv, n_heads_kv, head_dim, **kw)
    return q, k, v


# ---------------------------------------------------------------------
# Eligibility & dispatch logic — runnable anywhere
# ---------------------------------------------------------------------


def test__sdpa_backend_default_path_unchanged_when_fa3_unavailable() -> None:
    """Auto on CPU/non-Hopper falls back silently to SDPA; matches explicit sdpa."""
    q, k, v = _make_qkv(
        batch=1,
        seq_q=8,
        seq_kv=8,
        n_heads_q=2,
        n_heads_kv=2,
        head_dim=16,
        device="cpu",
        dtype=torch.float32,
    )

    out_auto = _batched_scaled_dot_product_attention(
        q, k, v, attention_backend=AttentionBackend.AUTO
    )
    out_sdpa = _batched_scaled_dot_product_attention(
        q, k, v, attention_backend=AttentionBackend.SDPA
    )
    # The enum is a ``str``-mixin, so passing the bare string should also work.
    out_sdpa_str = _batched_scaled_dot_product_attention(
        q, k, v, attention_backend="sdpa"
    )

    torch.testing.assert_close(out_auto, out_sdpa)
    torch.testing.assert_close(out_sdpa, out_sdpa_str)


def test__forced_fa3_raises_with_actionable_message_when_ineligible() -> None:
    """Forced FA3 on CPU/non-Hopper raises a clear error, not a silent fallback."""
    q, k, v = _make_qkv(
        batch=1,
        seq_q=4,
        seq_kv=4,
        n_heads_q=2,
        n_heads_kv=2,
        head_dim=16,
        device="cpu",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="FA3"):
        _batched_scaled_dot_product_attention(
            q, k, v, attention_backend=AttentionBackend.FA3
        )


def test__eligibility_rejects_unsupported_head_dim() -> None:
    """Eligibility gate rules out head_dim=16 (v3 dist-embedder shape)."""
    if not torch.cuda.is_available():
        pytest.skip("eligibility check needs CUDA tensor")
    q = torch.zeros(1, 4, 2, 16, device="cuda", dtype=torch.float16)
    assert not is_fa3_eligible_for(q)


def test__preferred_falls_back_to_sdpa_below_seqlen_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto dispatch must skip FA3 when both seq_q and seq_kv are too small.

    Capability is mocked True so we exercise just the perf threshold; this
    keeps the test runnable on any host (no Hopper required).
    """
    monkeypatch.setattr(fa3_backend, "is_fa3_eligible_for", lambda _q: True)

    seq_below = fa3_backend._FA3_MIN_SEQLEN_FOR_SPEEDUP - 1
    seq_at = fa3_backend._FA3_MIN_SEQLEN_FOR_SPEEDUP
    q_small = torch.zeros(1, seq_below, 8, 64)
    k_small = torch.zeros(1, seq_below, 8, 64)
    assert not is_fa3_preferred_for(q_small, k_small)

    q_at = torch.zeros(1, seq_at, 8, 64)
    k_at = torch.zeros(1, seq_at, 8, 64)
    assert is_fa3_preferred_for(q_at, k_at)

    # Cross-attention with small Q but large K (e.g. test queries against
    # a large support set) should still route through FA3 — the per-call
    # work is dominated by K and amortises FA3's overhead.
    q_small_q = torch.zeros(1, 256, 8, 64)
    k_large_k = torch.zeros(1, 100_000, 8, 64)
    assert is_fa3_preferred_for(q_small_q, k_large_k)


# ---------------------------------------------------------------------
# Numerical equivalence on Hopper — needs the FA3 wheel
# ---------------------------------------------------------------------


@pytest.mark.hopper
@_skip_unless_fa3
def test__fa3_batch_heads_above_cuda_max_grid() -> None:
    """FA3 must handle B*H > CUDA_MAX_GRID (65536) without silent failure.

    The SDPA path explicitly chunks at 65536 to work around pytorch
    issue #142228; we want to know whether FA3's kernels have the same
    constraint. The current ``_fa3_attention`` doesn't chunk, so this
    test settles whether that's safe. If FA3 has the same grid limit,
    this test will fail (with crash or numerical mismatch) and we
    should add chunking to ``_fa3_attention``.

    Designed to put B*H comfortably past 65536 with a tiny tensor
    footprint so it runs in seconds on a single H100.
    """
    batch = 70_000  # > 65_536
    seq, head_dim = 16, 64
    n_heads = 1
    q = torch.randn(batch, seq, n_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq, n_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, seq, n_heads, head_dim, device="cuda", dtype=torch.float16)

    out_sdpa = _batched_scaled_dot_product_attention(
        q, k, v, attention_backend=AttentionBackend.SDPA
    )
    out_fa3 = _batched_scaled_dot_product_attention(
        q, k, v, attention_backend=AttentionBackend.FA3
    )

    torch.testing.assert_close(out_fa3, out_sdpa, atol=5e-3, rtol=5e-3)


@pytest.mark.hopper
@_skip_unless_fa3
@pytest.mark.parametrize(
    ("seq_q", "seq_kv", "n_heads_q", "n_heads_kv"),
    [
        # MHA self-attn over training rows (icl_emsize=512, 8 heads, head_dim=64)
        (1024, 1024, 8, 8),
        # MQA cross-attn for test rows (test queries vs train keys)
        (256, 1024, 8, 1),
        # GQA mid-point (e.g. icl_num_kv_heads=2)
        (512, 512, 8, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test__fa3_matches_sdpa_within_tolerance(
    seq_q: int,
    seq_kv: int,
    n_heads_q: int,
    n_heads_kv: int,
    dtype: torch.dtype,
) -> None:
    q, k, v = _make_qkv(
        batch=2,
        seq_q=seq_q,
        seq_kv=seq_kv,
        n_heads_q=n_heads_q,
        n_heads_kv=n_heads_kv,
        head_dim=64,
        device="cuda",
        dtype=dtype,
    )

    out_sdpa = _batched_scaled_dot_product_attention(
        q, k, v, attention_backend=AttentionBackend.SDPA
    )
    out_fa3 = _batched_scaled_dot_product_attention(
        q, k, v, attention_backend=AttentionBackend.FA3
    )

    # 5e-3 abs matches the contributor's test_fa3.py for the same shapes.
    torch.testing.assert_close(out_fa3, out_sdpa, atol=5e-3, rtol=5e-3)
