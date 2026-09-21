#  Copyright (c) Prior Labs GmbH 2026.

"""Numerical-equivalence tests for the v3 attention backend selector.

The non-Hopper tests (sdpa-only, eligibility checks, error paths) run on
any GPU — or CPU — and exercise the dispatch logic with FA4 unavailable.

The ``hopper``/``blackwell``-marked tests require a Hopper- or Blackwell-class
GPU AND the ``flash-attn-4`` package (``pip install "tabpfn[fa4]"``). They
``skip`` automatically on any other host; run them manually on such a GPU
until a CI runner is in place.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import tabpfn.architectures.shared.scaled_dot_product_attention as _sdpa_mod
from tabpfn.architectures.shared import (
    fa4_backend,
    torch_mps_backend as _torch_mps_mod,
)
from tabpfn.architectures.shared.attention_backends import AttentionSpec
from tabpfn.architectures.shared.fa4_backend import FA4_BACKEND, is_fa4_eligible
from tabpfn.architectures.shared.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)


def _has_fa4_gpu() -> bool:
    """Hopper or Blackwell: the architectures ``fa4_backend`` dispatches on."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] in fa4_backend._FA4_MAX_HEAD_DIM


_FA4_RUNNABLE = _has_fa4_gpu() and FA4_BACKEND.is_available()


@pytest.fixture(autouse=True)
def _reset_fa4_once_warning():  # noqa: ANN202
    """The bf16 warning fires once per process; give every test a fresh one."""
    fa4_backend._warn_bf16_blackwell_once.cache_clear()
    try:
        yield
    finally:
        fa4_backend._warn_bf16_blackwell_once.cache_clear()


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


def test__sdpa_backend_default_path_unchanged_when_fa4_unavailable() -> None:
    """Auto on CPU/unsupported GPU falls back silently to SDPA; output is correct."""
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


def _spec(
    seq_len_q: int | None,
    seq_len_kv: int | None,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> AttentionSpec:
    return AttentionSpec(
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        dtype=dtype,
        device=torch.device(device),
        batch_size=1,
    )


def test__fa4_preferred_is_eligibility_with_no_seqlen_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FA4 has no short-sequence penalty (TabPFN#1235), so ``is_preferred`` is
    eligibility alone: tiny, huge, and unknown lengths all route to FA4 once
    the call is eligible, and none do when it is not.
    """
    monkeypatch.setattr(fa4_backend, "is_fa4_eligible", lambda *_a, **_k: True)
    assert FA4_BACKEND.is_preferred(_spec(8, 8))
    assert FA4_BACKEND.is_preferred(_spec(256, 100_000))
    assert FA4_BACKEND.is_preferred(_spec(None, None))

    monkeypatch.setattr(fa4_backend, "is_fa4_eligible", lambda *_a, **_k: False)
    assert not FA4_BACKEND.is_preferred(_spec(100_000, 100_000))


def test__fa4_bf16_declined_on_blackwell_with_one_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On compute capability >= 10, bf16 stays on SDPA and warns once; fp16
    is unaffected, and pre-Blackwell parts take bf16 without a warning.

    The capability lookup is mocked so this runs on any host.
    """
    monkeypatch.setattr(fa4_backend, "_fa4_max_head_dim", lambda _d: 128)
    monkeypatch.setattr(fa4_backend, "_compute_capability_major", lambda _d: 10)
    device = torch.device("cpu")
    with pytest.warns(UserWarning, match="not used for bfloat16 .* Blackwell"):
        assert not fa4_backend.is_fa4_eligible(device, torch.bfloat16, head_dim=64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second warning would fail the test
        assert not fa4_backend.is_fa4_eligible(device, torch.bfloat16, head_dim=64)
        assert fa4_backend.is_fa4_eligible(device, torch.float16, head_dim=64)
        # A call FA4 could not serve in any dtype does not get the fp16 advice.
        assert not fa4_backend.is_fa4_eligible(device, torch.bfloat16, head_dim=192)

    monkeypatch.setattr(fa4_backend, "_compute_capability_major", lambda _d: 9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fa4_backend.is_fa4_eligible(device, torch.bfloat16, head_dim=64)


def test__fa4_bf16_gate_does_not_break_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry consults ``is_preferred`` inside compiled regions, so the
    Blackwell bf16 decline must be traceable: no ``warnings.warn`` while
    Dynamo is tracing (it cannot trace the builtin and would break the graph).
    """
    monkeypatch.setattr(fa4_backend, "_fa4_max_head_dim", lambda _d: 128)
    monkeypatch.setattr(fa4_backend, "_compute_capability_major", lambda _d: 10)
    spec = _spec(64, 64, dtype=torch.bfloat16)

    torch._dynamo.reset()
    compiled = torch.compile(FA4_BACKEND.is_preferred, fullgraph=True, backend="eager")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="FlashAttention-4")
        preferred = compiled(spec)
    assert preferred is False


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
    assert is_fa4_eligible(device, torch.float16, head_dim=max_hd)
    # bf16 is served on Hopper but left to SDPA on Blackwell.
    bf16_expected = major not in fa4_backend._FA4_BF16_SLOW_COMPUTE_CAPABILITY_MAJORS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert is_fa4_eligible(device, torch.bfloat16, head_dim=64) is bf16_expected
    assert not is_fa4_eligible(device, torch.float16, head_dim=max_hd + 8)
    assert not is_fa4_eligible(device, torch.float32, head_dim=64)
    assert not is_fa4_eligible(device, torch.float16, head_dim=60)  # not %8
    # Unlike FA3, FA4 takes any multiple of 8 from 8 up, so the v3
    # dist-embedder shape (head_dim=16) is eligible.
    assert is_fa4_eligible(device, torch.float16, head_dim=16)


def test__fa4_split_kv_plan() -> None:
    """Split-KV outside the kernel triggers only for short Q against long KV at
    small batch, never below the chunk minimum, and never past 32 chunks.
    """
    plan = fa4_backend._split_kv_plan
    # The cached-prediction shape: a few test rows, one batch, long cache.
    assert plan(batch=1, seq_q=16, seq_kv=1_000_000) == 32
    assert plan(batch=1, seq_q=256, seq_kv=100_000) == 32
    assert plan(batch=1, seq_q=1024, seq_kv=100_000) == 16
    # Enough q-tiles already: no split.
    assert plan(batch=1, seq_q=100_000, seq_kv=100_000) == 1
    assert plan(batch=64, seq_q=256, seq_kv=100_000) == 1
    # KV too short to be worth chunking.
    assert plan(batch=1, seq_q=16, seq_kv=1_000) == 1
    assert plan(batch=1, seq_q=16, seq_kv=4_095) == 1
    assert plan(batch=1, seq_q=16, seq_kv=4_096) == 2
    # Empty query: nothing to split, and no division by zero.
    assert plan(batch=1, seq_q=0, seq_kv=1_000_000) == 1
    # batch > 1 must divide KV exactly (copy-free fold): largest divisor.
    assert plan(batch=2, seq_q=16, seq_kv=100_000) == 32
    assert plan(batch=2, seq_q=16, seq_kv=100_001) == 11  # 100_001 = 11 * 9_091
    assert plan(batch=2, seq_q=16, seq_kv=99_990) == 30


# ---------------------------------------------------------------------
# Numerical equivalence for FA4 — needs flash-attn-4 and Hopper/Blackwell
# ---------------------------------------------------------------------


@_skip_unless_fa4
def test__fa4_backward_matches_sdpa() -> None:
    """Under autograd FA4 runs its own kernel unsplit (the out-of-kernel
    split-KV path is inference-only); its gradients must match SDPA's.
    """
    torch.manual_seed(0)
    make = lambda *shape: torch.randn(
        *shape, device="cuda", dtype=torch.float16, requires_grad=True
    )
    q, k, v = make(1, 256, 8, 64), make(1, 8192, 1, 64), make(1, 8192, 1, 64)
    grad_out = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.float16)

    with torch.enable_grad():
        scaled_dot_product_attention(q, k, v, backend=None).backward(grad_out)
        grads_sdpa = (q.grad, k.grad, v.grad)
        q.grad = k.grad = v.grad = None
        scaled_dot_product_attention(q, k, v, backend=FA4_BACKEND).backward(grad_out)
        grads_fa4 = (q.grad, k.grad, v.grad)

    for g_fa4, g_sdpa in zip(grads_fa4, grads_sdpa, strict=False):
        torch.testing.assert_close(g_fa4, g_sdpa, atol=1e-2, rtol=1e-2)


@_skip_unless_fa4
def test__fa4_batch_above_cuda_max_grid() -> None:
    """FA4 launches one grid entry per batch element, so ``batch > 65535``
    fails with ``cudaErrorInvalidValue``; ``fa4_attn_func`` must chunk.

    FA3's kernel had no such limit; FA4 does.
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

    # 5e-3 abs matches the tolerance the FA3 backend was tested at.
    torch.testing.assert_close(out_fa4, out_sdpa, atol=5e-3, rtol=5e-3)


@_skip_unless_fa4
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("batch", "seq_q", "seq_kv"),
    [
        (1, 256, 100_000),  # test rows vs train cache
        (1, 16, 100_001),  # tiny Q; KV not a multiple of the chunk count
        (2, 16, 50_003),  # batch > 1 with a remainder
    ],
)
def test__fa4_long_kv_cross_attention_matches_sdpa(
    batch: int, seq_q: int, seq_kv: int, dtype: torch.dtype
) -> None:
    """Short Q against a long KV: the cached-prediction shape.

    On sm_90 this goes through the out-of-kernel split-KV path
    (``_fa4_split_kv``), including its remainder handling; on sm_100 through
    FA4's own split-KV heuristic. Either way it must match SDPA.
    """
    q, k, v = _make_qkv(
        batch=batch,
        seq_q=seq_q,
        seq_kv=seq_kv,
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
