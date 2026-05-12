#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the MLX attention backend.

Eligibility and dispatch-logic tests run on any host (no MPS required).
Conversion and numerical-equivalence tests require both ``mlx`` and an
Apple MPS device; they skip automatically everywhere else.
"""

from __future__ import annotations

import pytest
import torch

import tabpfn.architectures.shared.scaled_dot_product_attention as _sdpa_mod
from tabpfn.architectures.shared import mlx_backend
from tabpfn.architectures.shared.mlx_backend import (
    _mlx_to_torch,
    _torch_to_mlx,
    flash_attention_mlx,
    is_eligible_for_mlx,
    is_mlx_preferred,
)

try:
    import mlx.core as mx
except ImportError:
    mx = None

_MPS_AVAILABLE = torch.backends.mps.is_available()
_MLX_AVAILABLE = mlx_backend.mx is not None

_skip_unless_mps = pytest.mark.skipif(not _MPS_AVAILABLE, reason="requires MPS device")
_skip_unless_mlx = pytest.mark.skipif(not _MLX_AVAILABLE, reason="requires mlx package")
_skip_unless_mlx_and_mps = pytest.mark.skipif(
    not (_MPS_AVAILABLE and _MLX_AVAILABLE),
    reason="requires MPS device and mlx package",
)


def _mps_qkv(
    batch: int = 1,
    seq_q: int = 16,
    seq_kv: int = 16,
    n_heads: int = 2,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (B, H, S, D) tensors on MPS with a fixed seed."""
    g = torch.Generator(device="mps").manual_seed(42)
    kw = {"device": "mps", "dtype": dtype, "generator": g}
    q = torch.randn(batch, n_heads, seq_q, head_dim, **kw)
    k = torch.randn(batch, n_heads, seq_kv, head_dim, **kw)
    v = torch.randn(batch, n_heads, seq_kv, head_dim, **kw)
    return q, k, v


def _reference_attn_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Exact float32 attention on CPU — used as numerical reference."""
    scale = head_dim**-0.5
    q32 = q.cpu().float()
    k32 = k.cpu().float()
    v32 = v.cpu().float()
    scores = torch.einsum("bhsd,bhjd->bhsj", q32 * scale, k32)
    return torch.einsum("bhsj,bhjd->bhsd", scores.softmax(dim=-1), v32).to(q.dtype)


@_skip_unless_mps
def test__eligible_false_when_mx_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """mx=None (import failure) makes every tensor ineligible."""
    monkeypatch.setattr(mlx_backend, "mx", None)
    q = torch.zeros(1, 2, 8, 64, dtype=torch.float16, device="mps")
    assert not is_eligible_for_mlx(q, q, q)


def test__eligible_false_for_cpu_tensors() -> None:
    """CPU tensors are never eligible (not MPS)."""
    q = torch.zeros(1, 2, 8, 64, dtype=torch.float16)
    assert not is_eligible_for_mlx(q, q, q)


@_skip_unless_mps
def test__eligible_false_for_head_dim_over_128() -> None:
    """head_dim=129 is rejected regardless of other properties."""
    q = torch.zeros(1, 2, 8, 129, dtype=torch.float16, device="mps")
    assert not is_eligible_for_mlx(q, q, q)


@_skip_unless_mps
def test__eligible_false_when_requires_grad() -> None:
    q = torch.zeros(1, 2, 8, 64, device="mps", dtype=torch.float32, requires_grad=True)
    k = torch.zeros(1, 2, 8, 64, device="mps", dtype=torch.float32)
    v = torch.zeros(1, 2, 8, 64, device="mps", dtype=torch.float32)
    assert not is_eligible_for_mlx(q, k, v)


@_skip_unless_mps
def test__eligible_false_for_integer_dtype() -> None:
    """Integer tensors are not a supported dtype (MPS supports int32)."""
    if not _MLX_AVAILABLE:
        pytest.skip("mlx not installed")
    q = torch.zeros(1, 2, 8, 64, device="mps", dtype=torch.int32)
    assert not is_eligible_for_mlx(q, q, q)


@_skip_unless_mlx_and_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test__eligible_true_for_valid_mps_tensor(dtype: torch.dtype) -> None:
    q = torch.zeros(1, 2, 8, 64, device="mps", dtype=dtype)
    assert is_eligible_for_mlx(q, q, q)


@_skip_unless_mlx_and_mps
def test__eligible_true_at_head_dim_boundary_128() -> None:
    q = torch.zeros(1, 2, 8, 128, device="mps", dtype=torch.float16)
    assert is_eligible_for_mlx(q, q, q)


@_skip_unless_mps
def test__eligible_false_at_head_dim_129() -> None:
    q = torch.zeros(1, 2, 8, 129, device="mps", dtype=torch.float16)
    assert not is_eligible_for_mlx(q, q, q)


def test__preferred_false_when_mx_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlx_backend, "mx", None)
    q = torch.zeros(1, 2, 8, 64)
    k = torch.zeros(1, 2, 2048, 64)
    assert not is_mlx_preferred(q, k, k)


@_skip_unless_mlx_and_mps
def test__preferred_requires_seq_kv_ge_129() -> None:
    """Threshold is 128; one below must not prefer MLX, at-threshold must."""
    q = torch.zeros(1, 2, 8, 64, device="mps", dtype=torch.float16)
    k_below = torch.zeros(1, 2, 127, 64, device="mps", dtype=torch.float16)
    k_at = torch.zeros(1, 2, 128, 64, device="mps", dtype=torch.float16)
    assert not is_mlx_preferred(q, k_below, k_below)
    assert is_mlx_preferred(q, k_at, k_at)


@_skip_unless_mlx
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test__conversion_roundtrip(dtype: torch.dtype) -> None:
    """float16 and float32 tensors survive a to-MLX-and-back trip exactly."""
    t = torch.randn(4, 8, dtype=dtype)
    arr = _torch_to_mlx(t)
    mx.eval(arr)
    result = _mlx_to_torch(arr, t.device, t.dtype)
    torch.testing.assert_close(result, t)


@_skip_unless_mlx
def test__conversion_roundtrip_bfloat16() -> None:
    """bfloat16 uses the int16 view trick; bit-pattern must survive intact."""
    t = torch.randn(4, 8, dtype=torch.bfloat16)
    arr = _torch_to_mlx(t)
    mx.eval(arr)
    result = _mlx_to_torch(arr, t.device, t.dtype)
    torch.testing.assert_close(result, t)


@_skip_unless_mlx_and_mps
def test__flash_attention_mlx_rejects_head_dim_over_128() -> None:
    q = torch.zeros(1, 2, 8, 129, device="mps", dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attention_mlx(q, q, q)


# ---------------------------------------------------------------------------
# flash_attention_mlx — numerical equivalence
# ---------------------------------------------------------------------------


@_skip_unless_mlx_and_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("seq_q", "seq_kv", "head_dim", "description"),
    [
        (16, 16, 32, "D<64 padded to 64"),
        (16, 16, 64, "D=64 no padding"),
        (16, 16, 96, "64<D<128 padded to 128"),
        (16, 16, 128, "D=128 no padding"),
    ],
)
def test__flash_attention_mlx_matches_reference(
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    description: str,
    dtype: torch.dtype,
) -> None:
    """MLX output is numerically close to an exact float32 SDPA reference."""
    q, k, v = _mps_qkv(
        batch=2, seq_q=seq_q, seq_kv=seq_kv, n_heads=4, head_dim=head_dim, dtype=dtype
    )
    ref = _reference_attn_cpu(q, k, v, head_dim)

    out = flash_attention_mlx(q, k, v)

    assert out.shape == q.shape, f"shape mismatch for {description}"
    assert out.dtype == q.dtype
    assert out.is_mps

    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (5e-4, 5e-4)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=atol, rtol=rtol)


@_skip_unless_mlx_and_mps
def test__flash_attention_mlx_bfloat16_matches_reference() -> None:
    """bfloat16 path (int16 view trick) matches float32 reference."""
    head_dim = 64
    q, k, v = _mps_qkv(
        batch=1, seq_q=32, seq_kv=32, n_heads=4, head_dim=head_dim, dtype=torch.bfloat16
    )
    ref = _reference_attn_cpu(q, k, v, head_dim)
    out = flash_attention_mlx(q, k, v)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-2, rtol=1e-2)


@_skip_unless_mlx_and_mps
def test__flash_attention_mlx_gqa() -> None:
    """MLX supports GQA (n_heads_q != n_heads_kv) without repeat-interleave."""
    head_dim = 64
    g = torch.Generator(device="mps").manual_seed(7)
    kw = {"device": "mps", "dtype": torch.float16, "generator": g}
    q = torch.randn(1, 8, 16, head_dim, **kw)  # 8 query heads
    k = torch.randn(1, 2, 16, head_dim, **kw)  # 2 kv heads
    v = torch.randn(1, 2, 16, head_dim, **kw)

    # Reference: expand kv heads to match q heads then run SDPA on CPU.
    repeat = 8 // 2
    ref_k = k.repeat_interleave(repeat, dim=1)
    ref_v = v.repeat_interleave(repeat, dim=1)
    ref = _reference_attn_cpu(q, ref_k, ref_v, head_dim)

    out = flash_attention_mlx(q, k, v)

    assert out.shape == q.shape
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Dispatch integration — MLX branch exercised through scaled_dot_product_attention
# ---------------------------------------------------------------------------


@_skip_unless_mlx_and_mps
def test__dispatch_routes_through_mlx_when_preferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When is_mlx_preferred returns True, scaled_dot_product_attention uses MLX."""
    monkeypatch.setattr(_sdpa_mod, "is_mlx_preferred", lambda *_: True)

    # Use (B, S, H, D) tensors as expected by scaled_dot_product_attention.
    q, k, v = _mps_qkv(batch=1, seq_q=16, seq_kv=16, n_heads=2, head_dim=64)
    q_bshd = q.permute(0, 2, 1, 3)
    k_bshd = k.permute(0, 2, 1, 3)
    v_bshd = v.permute(0, 2, 1, 3)

    out = _sdpa_mod.scaled_dot_product_attention(q_bshd, k_bshd, v_bshd)
    assert out.shape == q_bshd.shape
    assert out.is_mps


@_skip_unless_mlx_and_mps
@pytest.mark.skipif(
    torch.__version__ >= "2.13.dev20260510", reason="torch 2.13 uses SDPA"
)
@pytest.mark.parametrize(
    ("n_q_heads", "n_kv_heads"),
    [(4, 4), (8, 2)],
    ids=["mha", "gqa"],
)
def test__dispatch_mlx_matches_sdpa_reference(
    monkeypatch: pytest.MonkeyPatch,
    n_q_heads: int,
    n_kv_heads: int,
) -> None:
    """Output matches SDPA reference and flash_attention_mlx is called."""
    monkeypatch.setattr(_sdpa_mod, "is_mlx_preferred", lambda *_: True)

    called: list[bool] = []
    original_fn = _sdpa_mod.flash_attention_mlx

    def _spy(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
        called.append(True)
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(_sdpa_mod, "flash_attention_mlx", _spy)

    head_dim = 64
    g = torch.Generator(device="mps").manual_seed(42)
    kw = {"device": "mps", "dtype": torch.float16, "generator": g}
    q_bshd = torch.randn(1, 32, n_q_heads, head_dim, **kw)
    k_bshd = torch.randn(1, 32, n_kv_heads, head_dim, **kw)
    v_bshd = torch.randn(1, 32, n_kv_heads, head_dim, **kw)

    out_mlx = _sdpa_mod.scaled_dot_product_attention(q_bshd, k_bshd, v_bshd)

    assert called, "flash_attention_mlx was not called"

    q_cpu = q_bshd.cpu().float().permute(0, 2, 1, 3)  # BHSD
    k_cpu = k_bshd.cpu().float().permute(0, 2, 1, 3)
    v_cpu = v_bshd.cpu().float().permute(0, 2, 1, 3)
    if n_q_heads != n_kv_heads:
        repeat = n_q_heads // n_kv_heads
        k_cpu = k_cpu.repeat_interleave(repeat, dim=1)
        v_cpu = v_cpu.repeat_interleave(repeat, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
    ref = ref.permute(0, 2, 1, 3).half()  # back to BSHD

    torch.testing.assert_close(out_mlx.cpu(), ref.cpu(), atol=1e-2, rtol=1e-2)
