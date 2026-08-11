#  Copyright (c) Prior Labs GmbH 2026.
"""Contract tests for the attention backend registry, as plugins rely on it."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from tabpfn.architectures.kv_cache import FP8_KV_DTYPE, KVCacheEntry
from tabpfn.architectures.shared import attention_backends
from tabpfn.architectures.shared.fa3_backend import FA3_BACKEND
from tabpfn.architectures.shared.mlx_backend import MLX_BACKEND
from tabpfn.architectures.shared.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)
from tabpfn.architectures.shared.torch_mps_backend import TORCH_MPS_BACKEND


class _RecordingBackend:
    """Backend double: returns a recognizable constant when preferred."""

    consumes_quantized_kv = False

    def __init__(self, name: str = "test-backend", *, preferred: bool = True):
        self.name = name
        self.preferred = preferred
        self.specs: list[attention_backends.AttentionSpec] = []
        self.calls: list[dict] = []

    def is_preferred(self, spec: attention_backends.AttentionSpec) -> bool:
        self.specs.append(spec)
        return self.preferred

    def run(  # noqa: ANN202
        self,
        q,
        k,
        v,
        *,
        quantized_kv=None,
        **_informational,
    ):
        self.calls.append({"q": q, "k": k, "v": v, "quantized_kv": quantized_kv})
        return torch.full_like(q, 42.0)


@pytest.fixture
def registry_sandbox() -> Iterator[dict]:
    """Snapshot and restore the registry (and consult order) around each test."""
    saved_registry = dict(attention_backends._registry)
    saved_order = attention_backends._consult_order
    attention_backends._registry.clear()
    attention_backends._consult_order = ()
    try:
        yield attention_backends._registry
    finally:
        attention_backends._registry.clear()
        attention_backends._registry.update(saved_registry)
        attention_backends._consult_order = saved_order


def _qkv(
    b: int = 1, s: int = 8, h: int = 2, j: int = 2, d: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, j, d)
    v = torch.randn(b, s, j, d)
    return q, k, v


@pytest.mark.usefixtures("registry_sandbox")
def test_registered_backend_output_is_used() -> None:
    backend = _RecordingBackend()
    attention_backends.register_attention_backend(backend)
    q, k, v = _qkv()
    out = scaled_dot_product_attention(q, k, v)
    assert torch.all(out == 42.0)
    assert len(backend.calls) == 1


@pytest.mark.usefixtures("registry_sandbox")
def test_declining_backend_falls_through_to_sdpa() -> None:
    backend = _RecordingBackend(preferred=False)
    attention_backends.register_attention_backend(backend)
    q, k, v = _qkv()
    out = scaled_dot_product_attention(q, k, v)
    expected = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected)
    assert backend.calls == []


@pytest.mark.usefixtures("registry_sandbox")
def test_consult_order_is_newest_first_in_the_order_given() -> None:
    earlier = _RecordingBackend("earlier")
    a = _RecordingBackend("a")
    b = _RecordingBackend("b")
    attention_backends.register_attention_backend(earlier)
    attention_backends.register_attention_backend(a, b)
    assert attention_backends.registered_attention_backends() == (a, b, earlier)
    attention_backends.unregister_attention_backend("a")
    assert attention_backends.registered_attention_backends() == (b, earlier)


@pytest.mark.usefixtures("registry_sandbox")
def test_reregistration_is_reentrant_but_conflicts_raise() -> None:
    backend = _RecordingBackend()
    attention_backends.register_attention_backend(backend)
    attention_backends.register_attention_backend(backend)  # no-op
    with pytest.raises(ValueError, match="already registered"):
        attention_backends.register_attention_backend(_RecordingBackend(backend.name))


@pytest.mark.usefixtures("registry_sandbox")
@pytest.mark.parametrize("consumes_quantized_kv", [True, False])
def test_quantized_kv_is_passed_on_or_dequantized_once(
    consumes_quantized_kv: bool,
) -> None:
    """The entry is passed on only to a backend that declares it consumes
    one; otherwise the SDPA wrapper dequantizes, once.
    """
    backend = _RecordingBackend()
    backend.consumes_quantized_kv = consumes_quantized_kv
    attention_backends.register_attention_backend(backend)
    q, k, v = _qkv()
    entry = KVCacheEntry(key=k, value=v).quantize(FP8_KV_DTYPE)

    scaled_dot_product_attention(q, None, None, quantized_kv=entry)

    call = backend.calls[0]
    if consumes_quantized_kv:
        assert call["quantized_kv"] is entry
        assert call["k"] is None
    else:
        assert call["quantized_kv"] is None
        dequant = entry.dequantize(q.dtype)
        torch.testing.assert_close(call["k"], dequant.key)
        torch.testing.assert_close(call["v"], dequant.value)


@pytest.mark.usefixtures("registry_sandbox")
def test_explicit_backend_argument_bypasses_the_registry() -> None:
    """``None`` forces SDPA, a backend runs it — neither consults anyone."""
    registered = _RecordingBackend("registered")
    forced = _RecordingBackend("forced", preferred=False)
    attention_backends.register_attention_backend(registered)
    q, k, v = _qkv()

    assert not torch.all(scaled_dot_product_attention(q, k, v, backend=None) == 42.0)
    assert torch.all(scaled_dot_product_attention(q, k, v, backend=forced) == 42.0)
    assert registered.specs == []  # never consulted


@pytest.mark.usefixtures("registry_sandbox")
def test_lazy_spec_describes_the_live_call() -> None:
    """The spec describes the call the tensors represent."""
    backend = _RecordingBackend(preferred=False)
    attention_backends.register_attention_backend(backend)
    q, k, v = _qkv(b=3, s=8, h=4, j=2, d=16)
    scaled_dot_product_attention(q, k, v)
    (spec,) = backend.specs
    assert spec.seq_len_q == 8
    assert spec.seq_len_kv == 8
    assert spec.num_heads == 4
    assert spec.num_kv_heads == 2
    assert spec.head_dim == 16
    assert spec.batch_size == 3
    assert spec.dtype == q.dtype
    assert spec.device == q.device
    assert spec.quantized_kv_dtype is None

    backend.specs.clear()
    entry = KVCacheEntry(key=k, value=v).quantize(FP8_KV_DTYPE)
    with torch.no_grad():
        scaled_dot_product_attention(q, None, None, quantized_kv=entry)
    (spec,) = backend.specs
    assert spec.quantized_kv_dtype == FP8_KV_DTYPE
    assert spec.seq_len_kv == 8
    assert spec.num_kv_heads == 2
    assert spec.is_grad_enabled is False


def test_in_tree_backends_are_registered_when_available() -> None:
    """Only backends whose dependency is installed get registered."""
    names = [b.name for b in attention_backends.registered_attention_backends()]
    for backend in (FA3_BACKEND, TORCH_MPS_BACKEND, MLX_BACKEND):
        assert (backend.name in names) is backend.is_available(), backend.name
