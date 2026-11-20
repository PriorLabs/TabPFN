#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.architectures.shared.scaled_dot_product_attention."""

from __future__ import annotations

import io
import os
from typing import Any, TypedDict

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.torch_version import TorchVersion

from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.scaled_dot_product_attention import (
    _FLASH_BACKWARD_MAX_ELEMENTS,
    _FLASH_MIN_HEAD_DIM,
    _FLASH_QUERY_BLOCK,
    _torch_sdpa,
)


class _Shape(TypedDict):
    """The query shape of an SDPA call, unpacked into the helpers below."""

    batch: int
    seq_q: int
    heads: int
    head_dim: int


# batch * heads * round_up(seq_q, 128) * max(head_dim, 32) = 2,149,580,800, just
# over the 2**31 element range of FlashAttention's backward index. Without
# chunking this faults in backward with an illegal memory access.
_OVER_RANGE_SHAPE: _Shape = {
    "batch": 400,
    "seq_q": 10_496,
    "heads": 4,
    "head_dim": 128,
}
# Same count, reached with a head dim small enough to be padded up to 32. The raw
# element count is only half the range, so a check that ignored the padding would
# let this through.
_OVER_RANGE_SMALL_HEAD_DIM: _Shape = {
    "batch": 800,
    "seq_q": 10_496,
    "heads": 8,
    "head_dim": 16,
}
_SEQ_KV = 128


def _padded_elements(batch: int, heads: int, seq_q: int, head_dim: int) -> int:
    block = _FLASH_QUERY_BLOCK
    padded_seq = ((seq_q + block - 1) // block) * block
    return batch * heads * padded_seq * max(head_dim, _FLASH_MIN_HEAD_DIM)


def _record_sdpa_calls(monkeypatch: pytest.MonkeyPatch) -> list[torch.Size]:
    """Replace SDPA with a shape-preserving stub and record the query shapes."""
    shapes: list[torch.Size] = []

    def fake_sdpa(
        query: torch.Tensor,
        _key: torch.Tensor,
        value: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        shapes.append(query.shape)
        return torch.empty(
            (*query.shape[:-1], value.shape[-1]),
            dtype=query.dtype,
            device=query.device,
        )

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    return shapes


def _run_on_meta(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch: int,
    seq_q: int,
    heads: int,
    head_dim: int,
    backends: list[SDPBackend] | None,
    requires_grad: bool = True,
) -> list[torch.Size]:
    """Drive `_torch_sdpa` on meta tensors, which carry shape but no storage.

    Only the backward exceeds the index range, so the inputs require grad
    unless a test is about the inference path.
    """
    shapes = _record_sdpa_calls(monkeypatch)

    def meta_input(*shape: int) -> torch.Tensor:
        return torch.empty(
            shape, device="meta", dtype=torch.bfloat16, requires_grad=requires_grad
        )

    q = meta_input(batch, seq_q, heads, head_dim)
    k = meta_input(batch, _SEQ_KV, heads, head_dim)
    v = meta_input(batch, _SEQ_KV, heads, head_dim)
    _torch_sdpa(q, k, v, backends)
    return shapes


def test__torch_sdpa__query_over_flash_index_range__chunks_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each FlashAttention call must stay inside the backward's 32-bit range."""
    shapes = _run_on_meta(
        monkeypatch, **_OVER_RANGE_SHAPE, backends=[SDPBackend.FLASH_ATTENTION]
    )

    assert len(shapes) > 1, "expected the batch to be chunked"
    for shape in shapes:
        sub_batch, heads, seq_q, head_dim = shape
        assert (
            _padded_elements(sub_batch, heads, seq_q, head_dim)
            <= _FLASH_BACKWARD_MAX_ELEMENTS
        )
    assert sum(shape[0] for shape in shapes) == _OVER_RANGE_SHAPE["batch"]


def test__torch_sdpa__small_head_dim_over_range__chunks_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head dim below 32 is counted as 32, so it hits the range sooner."""
    shape = _OVER_RANGE_SMALL_HEAD_DIM
    raw = shape["batch"] * shape["heads"] * shape["seq_q"] * shape["head_dim"]
    assert raw < _FLASH_BACKWARD_MAX_ELEMENTS, "raw count should look safe"
    assert (
        _padded_elements(
            shape["batch"], shape["heads"], shape["seq_q"], shape["head_dim"]
        )
        > _FLASH_BACKWARD_MAX_ELEMENTS
    )

    shapes = _run_on_meta(monkeypatch, **shape, backends=[SDPBackend.FLASH_ATTENTION])

    assert len(shapes) > 1, "expected the batch to be chunked"
    for chunk in shapes:
        sub_batch, heads, seq_q, head_dim = chunk
        assert (
            _padded_elements(sub_batch, heads, seq_q, head_dim)
            <= _FLASH_BACKWARD_MAX_ELEMENTS
        )


def test__torch_sdpa__query_within_flash_index_range__single_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shape inside the range must not be chunked, so nothing slows down."""
    # 8192 needs no padding, giving exactly 2**31 elements, which is in range.
    shapes = _run_on_meta(
        monkeypatch,
        batch=512,
        seq_q=8_192,
        heads=4,
        head_dim=128,
        backends=[SDPBackend.FLASH_ATTENTION],
    )

    assert len(shapes) == 1


def test__torch_sdpa__flash_not_selectable__does_not_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit is FlashAttention's, so other backends should not be chunked."""
    shapes = _run_on_meta(
        monkeypatch, **_OVER_RANGE_SHAPE, backends=[SDPBackend.EFFICIENT_ATTENTION]
    )

    assert len(shapes) == 1


def test__torch_sdpa__inputs_without_grad__does_not_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the backward faults, so inference must not pay for chunking."""
    shapes = _run_on_meta(
        monkeypatch,
        **_OVER_RANGE_SHAPE,
        backends=[SDPBackend.FLASH_ATTENTION],
        requires_grad=False,
    )

    assert len(shapes) == 1


def test__torch_sdpa__grad_disabled__does_not_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs requiring grad under `no_grad` still record no backward."""
    with torch.no_grad():
        shapes = _run_on_meta(
            monkeypatch, **_OVER_RANGE_SHAPE, backends=[SDPBackend.FLASH_ATTENTION]
        )

    assert len(shapes) == 1


def _sdpa_input_storages(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> tuple[list[int], list[int]]:
    """Storages SDPA receives, and the storages of the inputs to `_torch_sdpa`."""
    received: list[int] = []

    def fake_sdpa(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **_kwargs: Any
    ) -> torch.Tensor:
        received.extend(t.untyped_storage().data_ptr() for t in (query, key, value))
        return torch.empty_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    q = torch.randn(2, 5, 4, 8, device=device)
    k = torch.randn(2, 7, 4, 8, device=device)
    v = torch.randn(2, 7, 4, 8, device=device)

    _torch_sdpa(q, k, v)

    return received, [t.untyped_storage().data_ptr() for t in (q, k, v)]


def test__torch_sdpa__passes_views_of_the_inputs_without_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    received, inputs = _sdpa_input_storages(monkeypatch, "cpu")
    assert received == inputs


def test__torch_sdpa__on_rocm__copies_the_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.4")
    received, inputs = _sdpa_input_storages(monkeypatch, "cpu")
    assert not set(received) & set(inputs)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test__torch_sdpa__on_mps__copies_the_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received, inputs = _sdpa_input_storages(monkeypatch, "mps")
    assert not set(received) & set(inputs)


def _cuda_bytes_available() -> int:
    if not torch.cuda.is_available():
        return 0
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes


# The query alone is 4 GiB; the backward additionally holds its gradient and an
# fp32 dQ accumulator, so leave room for roughly 24 GiB.
_NEEDED_BYTES = 24 * 1024**3


@pytest.mark.slow
@pytest.mark.skipif(
    _cuda_bytes_available() < _NEEDED_BYTES,
    reason="needs a CUDA device with ~24 GiB free",
)
def test__torch_sdpa__query_over_flash_index_range__backward_runs_on_cuda() -> None:
    """The shape that faults in an unchunked FlashAttention backward now runs."""
    shape = _OVER_RANGE_SHAPE
    assert (
        _padded_elements(
            shape["batch"], shape["heads"], shape["seq_q"], shape["head_dim"]
        )
        > _FLASH_BACKWARD_MAX_ELEMENTS
    ), "shape no longer exceeds the range this test is about"

    def cuda_input(*size: int) -> torch.Tensor:
        return torch.randn(
            size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

    q = cuda_input(shape["batch"], shape["seq_q"], shape["heads"], shape["head_dim"])
    k = cuda_input(shape["batch"], _SEQ_KV, shape["heads"], shape["head_dim"])
    v = cuda_input(shape["batch"], _SEQ_KV, shape["heads"], shape["head_dim"])

    out = _torch_sdpa(q, k, v, [SDPBackend.FLASH_ATTENTION])
    out.sum().backward()
    torch.cuda.synchronize()

    assert q.grad is not None


@pytest.mark.skipif(
    not gqa_is_supported(torch.device("cpu"), torch.float32),
    reason="needs CPU GQA on Torch >=2.9",
)
def test__torch_sdpa__cpu_mqa__shares_kv_and_matches_copied_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU MQA passes the single K/V head to SDPA instead of copying it per head."""
    q = torch.randn(2, 7, 4, 8)
    k = torch.randn(2, 37, 1, 8)
    v = torch.randn_like(k)
    with torch.inference_mode():
        # K/V expanded to all 4 query heads takes the path that copies K/V.
        expected = _torch_sdpa(q, k.expand(-1, -1, 4, -1), v.expand(-1, -1, 4, -1))

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls: list[tuple[torch.Tensor, dict[str, Any]]] = []

    def record(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        calls.append((key, kwargs))
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    with torch.inference_mode():
        actual = _torch_sdpa(q, k, v)

    [(key, kwargs)] = calls
    assert kwargs.get("enable_gqa") is True
    assert key.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not gqa_is_supported(torch.device("cpu"), torch.float32),
    reason="needs CPU GQA on Torch >=2.9",
)
@pytest.mark.skipif(os.name == "nt", reason="onnx export is not tested on windows")
def test__torch_sdpa__cpu_mqa_onnx_export__exports_without_enable_gqa() -> None:
    """The TorchScript ONNX exporter cannot convert SDPA with enable_gqa."""
    pytest.importorskip("onnx")

    class Attention(torch.nn.Module):
        def forward(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            return _torch_sdpa(q, k, v)

    q = torch.randn(2, 7, 4, 8)
    k = torch.randn(2, 37, 1, 8)
    v = torch.randn_like(k)

    torch.onnx.export(Attention(), (q, k, v), io.BytesIO(), dynamo=False)


@pytest.mark.parametrize(
    ("device", "dtype", "version", "expected"),
    [
        ("cpu", torch.float32, "2.8.0", False),
        ("cpu", torch.float32, "2.9.0", True),
        ("cpu", torch.float32, "2.9.0a0", True),
        ("cpu", torch.float32, "2.9.0.dev20250801+cpu", True),
        ("cpu", torch.float64, "2.9.0", True),
        ("cuda:0", torch.float16, "2.9.0", False),
        ("cuda:1", torch.float16, "2.4.0", False),
        ("cuda:1", torch.float16, "2.5.0rc1", True),
        ("cuda:1", torch.float16, "2.9.0", True),
        ("cuda:1", torch.bfloat16, "2.9.0", True),
        ("cuda:1", torch.float32, "2.9.0", False),
        ("meta", torch.float16, "2.9.0", False),
    ],
)
def test__gqa_is_supported__per_device__follows_version_capability_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    dtype: torch.dtype,
    version: str,
    *,
    expected: bool,
) -> None:
    """CPU depends on the Torch version, CUDA also on the GPU and the dtype."""
    capabilities = {0: (7, 5), 1: (8, 0)}
    monkeypatch.setattr(torch, "__version__", TorchVersion(version))
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda gpu: capabilities[gpu.index]
    )
    gqa_is_supported.cache_clear()
    try:
        assert gqa_is_supported(torch.device(device), dtype) is expected
    finally:
        gqa_is_supported.cache_clear()


# Packing never runs on ROCm, so tests that expect it need NVIDIA CUDA. Tests that
# expect no packing set `torch.version.hip` themselves and run on any CUDA device.
_CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_NVIDIA_CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="needs NVIDIA CUDA",
)


@_NVIDIA_CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("head_dim", "value_dim"), [(64, 64), (16, 32)])
def test__torch_sdpa__mqa__packs_heads_into_query_rows(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    head_dim: int,
    value_dim: int,
) -> None:
    """With one shared K/V head, SDPA gets a single query head with all rows."""
    seq_q = 10
    shapes = _record_sdpa_calls(monkeypatch)
    q = torch.empty(2, seq_q, 16, head_dim, device="cuda", dtype=dtype)
    k = torch.empty(2, 1_024, 1, head_dim, device="cuda", dtype=dtype)
    v = torch.empty(2, 1_024, 1, value_dim, device="cuda", dtype=dtype)

    with torch.inference_mode():
        out = _torch_sdpa(q, k, v)

    assert shapes == [torch.Size([2, 1, seq_q * 16, head_dim])]
    assert out.shape == (2, seq_q, 16, value_dim)


@_NVIDIA_CUDA_ONLY
def test__torch_sdpa__packed_mqa__empty_query_preserves_value_width() -> None:
    """Empty queries still return the value head width."""
    q = torch.empty(2, 0, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.empty(2, 8, 1, 64, device="cuda", dtype=torch.float16)
    v = torch.empty(2, 8, 1, 32, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        out = _torch_sdpa(q, k, v, [SDPBackend.MATH])

    assert out.shape == (2, 0, 16, 32)


@pytest.mark.parametrize(
    "case",
    ["cpu"]
    + [
        pytest.param(case, marks=_CUDA_ONLY)
        for case in (
            "fp32",
            "grouped_kv",
            "grad",
            "compiling",
            "rocm",
            "flash_override",
        )
    ],
)
def test__torch_sdpa__packing_not_applicable__keeps_query_heads(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Keep the query heads when packing is disabled."""
    device = "cpu" if case == "cpu" else "cuda"
    dtype = torch.float32 if case == "fp32" else torch.float16
    seq_q = 10
    kv_heads = 2 if case == "grouped_kv" else 1
    q = torch.empty(2, seq_q, 16, 64, device=device, dtype=dtype)
    k = torch.empty(2, 1_024, kv_heads, 64, device=device, dtype=dtype)
    v = torch.empty_like(k)
    shapes = _record_sdpa_calls(monkeypatch)
    if case == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4" if case == "rocm" else None)
    if case == "flash_override":
        monkeypatch.setattr(
            torch.nn.attention,
            "current_flash_attention_impl",
            lambda: "FA4",
            raising=False,
        )

    with torch.set_grad_enabled(case == "grad"):
        out = _torch_sdpa(q, k, v)

    assert shapes == [torch.Size([2, 16, seq_q, 64])]
    assert out.shape == q.shape


@_NVIDIA_CUDA_ONLY
@pytest.mark.parametrize("seq_q", [1, 10, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test__torch_sdpa__packed_mqa__matches_fp32_attention(
    seq_q: int, dtype: torch.dtype, *, strided: bool
) -> None:
    """Each original row/head attends to the same KV, including strided inputs."""
    factor = 2 if strided else 1
    q = torch.randn(2, seq_q * factor, 16, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 1_024 * factor, 1, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    if strided:
        q, k, v = q[:, ::2], k[:, ::2], v[:, ::2]

    with torch.inference_mode():
        out = _torch_sdpa(q, k, v)
        with sdpa_kernel([SDPBackend.MATH]):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.float().permute(0, 2, 1, 3),
                k.float().permute(0, 2, 1, 3),
                v.float().permute(0, 2, 1, 3),
                enable_gqa=True,
            ).permute(0, 2, 1, 3)

    assert out.shape == q.shape
    assert out.dtype == dtype
    atol, rtol = (2e-4, 2e-3) if dtype == torch.float16 else (1e-3, 1e-2)
    torch.testing.assert_close(out.float(), expected, atol=atol, rtol=rtol)
