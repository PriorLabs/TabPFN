#  Copyright (c) Prior Labs GmbH 2026.

"""Numerical-equivalence tests for the v3 attention backend selector.

The non-Hopper tests (sdpa-only, eligibility checks, error paths) run on
any GPU — or CPU — and exercise the dispatch logic with FA3 unavailable.

The ``hopper``-marked tests require a Hopper-class GPU (compute
capability 9.0+) AND the ``flash_attn_interface`` package built from
Dao-AILab/flash-attention's ``hopper/`` directory. They ``skip``
automatically on any other host; run them manually on an H100 until a
Hopper CI runner is in place.

The FA4 tests carry both ``hopper`` and ``blackwell`` markers: FA4 runs on
either, and needs the ``flash-attn-4`` package (``pip install tabpfn[fa4]``).
"""

from __future__ import annotations

import pytest
import torch

import tabpfn.architectures.shared.scaled_dot_product_attention as _sdpa_mod
from tabpfn.architectures.shared import (
    fa3_backend,
    fa4_backend,
    torch_mps_backend as _torch_mps_mod,
)
from tabpfn.architectures.shared.attention_backends import AttentionSpec
from tabpfn.architectures.shared.fa3_backend import FA3_BACKEND, is_fa3_eligible
from tabpfn.architectures.shared.fa4_backend import FA4_BACKEND, is_fa4_eligible
from tabpfn.architectures.shared.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)


def _has_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 9


_FA3_RUNNABLE = _has_hopper() and FA3_BACKEND.is_available()
_skip_unless_fa3 = pytest.mark.skipif(
    not _FA3_RUNNABLE, reason="requires Hopper GPU and flash_attn_interface"
)


def _has_fa4_gpu() -> bool:
    """Hopper or Blackwell: the architectures ``fa4_backend`` dispatches on."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] in fa4_backend._FA4_MAX_HEAD_DIM


_FA4_RUNNABLE = _has_fa4_gpu() and FA4_BACKEND.is_available()


def _skip_unless_fa4(test):  # noqa: ANN202
    """Mark an FA4 GPU test: ``hopper`` and ``blackwell`` (it runs on either),
    skipped unless such a GPU and ``flash-attn-4`` are present.
    """
    skip = pytest.mark.skipif(
        not _FA4_RUNNABLE, reason="requires Hopper/Blackwell GPU and flash-attn-4"
    )
    return pytest.mark.blackwell(pytest.mark.hopper(skip(test)))


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
    """Auto on CPU/non-Hopper falls back silently to SDPA; output is correct."""
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

    # On CPU no backend can be selected, so auto must equal forced SDPA.
    out_forced_sdpa = scaled_dot_product_attention(q, k, v, backend=None)
    out_auto = scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(out_forced_sdpa, out_auto)


def test__eligibility_rejects_unsupported_head_dim() -> None:
    """Eligibility gate rules out head_dim=16 (v3 dist-embedder shape)."""
    if not torch.cuda.is_available():
        pytest.skip("eligibility check needs CUDA")
    assert not is_fa3_eligible(torch.device("cuda"), torch.float16, head_dim=16)


def test__preferred_falls_back_to_sdpa_below_seqlen_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto dispatch must skip FA3 when both seq_q and seq_kv are too small.

    Capability is mocked True so we exercise just the perf threshold; this
    keeps the test runnable on any host (no Hopper required).
    """
    monkeypatch.setattr(fa3_backend, "is_fa3_eligible", lambda *_a, **_k: True)

    def spec(seq_len_q: int | None, seq_len_kv: int | None) -> AttentionSpec:
        return AttentionSpec(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device=torch.device("cpu"),
            batch_size=1,
        )

    seq_below = fa3_backend._FA3_MIN_SEQLEN_FOR_SPEEDUP - 1
    seq_at = fa3_backend._FA3_MIN_SEQLEN_FOR_SPEEDUP
    assert not FA3_BACKEND.is_preferred(spec(seq_below, seq_below))
    assert FA3_BACKEND.is_preferred(spec(seq_at, seq_at))

    # Cross-attention with small Q but large K (e.g. test queries against
    # a large support set) should still route through FA3 — the per-call
    # work is dominated by K and amortises FA3's overhead.
    assert FA3_BACKEND.is_preferred(spec(256, 100_000))

    # Unknown (chunk-dependent) lengths never argue for FA3.
    assert not FA3_BACKEND.is_preferred(spec(None, None))


def test__fa4_preferred_falls_back_to_sdpa_below_seqlen_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract as the FA3 test above, for the FA4 backend.

    ``_FA4_MIN_SEQLEN_FOR_SPEEDUP`` is a placeholder until the FA4 benchmark
    lands; the test pins the *shape* of the rule, not the number.
    """
    monkeypatch.setattr(fa4_backend, "is_fa4_eligible", lambda *_a, **_k: True)

    def spec(seq_len_q: int | None, seq_len_kv: int | None) -> AttentionSpec:
        return AttentionSpec(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device=torch.device("cpu"),
            batch_size=1,
        )

    seq_below = fa4_backend._FA4_MIN_SEQLEN_FOR_SPEEDUP - 1
    seq_at = fa4_backend._FA4_MIN_SEQLEN_FOR_SPEEDUP
    assert not FA4_BACKEND.is_preferred(spec(seq_below, seq_below))
    assert FA4_BACKEND.is_preferred(spec(seq_at, seq_at))
    assert FA4_BACKEND.is_preferred(spec(256, 100_000))
    assert not FA4_BACKEND.is_preferred(spec(None, None))


def test__fa4_eligibility_head_dim_range_per_arch() -> None:
    """FA4's head-dim range differs by architecture: 256 on sm_90, 128 on sm_100+."""
    if not torch.cuda.is_available():
        pytest.skip("eligibility check needs CUDA")
    device = torch.device("cuda")
    major = torch.cuda.get_device_capability(device)[0]
    if major not in fa4_backend._FA4_MAX_HEAD_DIM:
        pytest.skip(f"FA4 has no kernels for compute capability {major}.x")
    max_hd = fa4_backend._FA4_MAX_HEAD_DIM[major]
    assert is_fa4_eligible(device, torch.float16, head_dim=64)
    assert is_fa4_eligible(device, torch.bfloat16, head_dim=max_hd)
    assert not is_fa4_eligible(device, torch.float16, head_dim=max_hd + 8)
    assert not is_fa4_eligible(device, torch.float32, head_dim=64)
    assert not is_fa4_eligible(device, torch.float16, head_dim=60)  # not %8
    # Unlike FA3, FA4 takes any multiple of 8 from 8 up, so the v3
    # dist-embedder shape (head_dim=16) is eligible; the seqlen gate still applies.
    assert is_fa4_eligible(device, torch.float16, head_dim=16)


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

    out_sdpa = scaled_dot_product_attention(q, k, v, backend=None)
    # FA3 regardless of the seqlen threshold (seq=16 is below it).
    out_fa3 = scaled_dot_product_attention(q, k, v, backend=FA3_BACKEND)

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

    out_sdpa = scaled_dot_product_attention(q, k, v, backend=None)
    # FA3 regardless of the seqlen threshold.
    out_fa3 = scaled_dot_product_attention(q, k, v, backend=FA3_BACKEND)

    # 5e-3 abs matches the contributor's test_fa3.py for the same shapes.
    torch.testing.assert_close(out_fa3, out_sdpa, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------
# Numerical equivalence for FA4 — needs flash-attn-4 and Hopper/Blackwell
# ---------------------------------------------------------------------


@_skip_unless_fa4
def test__fa4_batch_above_cuda_max_grid() -> None:
    """FA4 launches one grid entry per batch element, so ``batch > 65535``
    fails with ``cudaErrorInvalidValue``; ``fa4_attn_func`` must chunk.

    ``fa3_backend`` gets this for free from FA3's kernel; FA4 does not.
    """
    batch = 70_000  # > 65_535
    seq, head_dim = 16, 64
    n_heads = 1
    q = torch.randn(batch, seq, n_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq, n_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, seq, n_heads, head_dim, device="cuda", dtype=torch.float16)

    out_sdpa = scaled_dot_product_attention(q, k, v, backend=None)
    out_fa4 = scaled_dot_product_attention(q, k, v, backend=FA4_BACKEND)

    torch.testing.assert_close(out_fa4, out_sdpa, atol=5e-3, rtol=5e-3)


@_skip_unless_fa4
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
def test__fa4_matches_sdpa_within_tolerance(
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

    out_sdpa = scaled_dot_product_attention(q, k, v, backend=None)
    # FA4 regardless of the seqlen threshold.
    out_fa4 = scaled_dot_product_attention(q, k, v, backend=FA4_BACKEND)

    # Same tolerance as the FA3 test above.
    torch.testing.assert_close(out_fa4, out_sdpa, atol=5e-3, rtol=5e-3)


@_skip_unless_fa4
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test__fa4_long_kv_cross_attention_matches_sdpa(dtype: torch.dtype) -> None:
    """Short Q against a long KV: the shape FA3 split-KV was written for.

    FA4 has no split-KV on sm_90, so this runs unsplit there; it must still
    be numerically right whichever ``num_splits`` the architecture allows.
    """
    q, k, v = _make_qkv(
        batch=1,
        seq_q=256,
        seq_kv=100_000,
        n_heads_q=8,
        n_heads_kv=1,
        head_dim=64,
        device="cuda",
        dtype=dtype,
    )
    out_sdpa = scaled_dot_product_attention(q, k, v, backend=None)
    out_fa4 = scaled_dot_product_attention(q, k, v, backend=FA4_BACKEND)
    torch.testing.assert_close(out_fa4, out_sdpa, atol=5e-3, rtol=5e-3)


def _gqa_inputs(
    num_q_heads: int, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    batch, seq, head_dim = 2, 5, 8
    q = torch.randn(batch, seq, num_q_heads, head_dim)
    k = torch.randn(batch, seq, num_kv_heads, head_dim)
    v = torch.randn(batch, seq, num_kv_heads, head_dim)
    return q, k, v


@pytest.mark.skipif(
    torch.__version__ < "2.5", reason="enable_gqa requires torch >= 2.5"
)
def test__torch_mps_sdpa__gqa_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that GQA works for torch mps branch.

    Force the torch-MPS branch (on CPU) with mismatched head counts: the
    real torch_mps_sdpa must not crash and must match the default path's
    repeat_interleave GQA reference.
    """
    q, k, v = _gqa_inputs(num_q_heads=8, num_kv_heads=2)
    reference = _sdpa_mod.scaled_dot_product_attention(q, k, v)

    monkeypatch.setattr(_torch_mps_mod, "is_torch_mps_preferred", lambda *_: True)
    out = _sdpa_mod.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
