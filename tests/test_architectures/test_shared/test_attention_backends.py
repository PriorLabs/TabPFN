#  Copyright (c) Prior Labs GmbH 2026.
"""Contract tests for the external attention backend registry.

These pin the behaviour plugins rely on: a registered backend that prefers a
call receives it and its output is used; a backend that declines (or none
being registered) leaves the standard SDPA path intact, including quantized
KV cache entries being dequantized at the chokepoint.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from tabpfn.architectures.kv_cache import FP8_KV_DTYPE, KVCacheEntry
from tabpfn.architectures.shared import attention_backends
from tabpfn.architectures.shared.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)


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
def test_most_recently_registered_backend_wins() -> None:
    first = _RecordingBackend("first")
    second = _RecordingBackend("second")
    attention_backends.register_attention_backend(first)
    attention_backends.register_attention_backend(second)
    q, k, v = _qkv()
    scaled_dot_product_attention(q, k, v)
    assert len(second.calls) == 1
    assert first.calls == []


@pytest.mark.usefixtures("registry_sandbox")
def test_group_registration_keeps_given_order_ahead_of_earlier() -> None:
    """One call registers several backends: consulted in the order given,
    before anything registered earlier.
    """
    earlier = _RecordingBackend("earlier")
    a = _RecordingBackend("a")
    b = _RecordingBackend("b")
    attention_backends.register_attention_backend(earlier)
    attention_backends.register_attention_backend(a, b)
    assert attention_backends.registered_attention_backends() == (a, b, earlier)


@pytest.mark.usefixtures("registry_sandbox")
def test_reregistration_is_reentrant_but_conflicts_raise() -> None:
    backend = _RecordingBackend()
    attention_backends.register_attention_backend(backend)
    attention_backends.register_attention_backend(backend)  # no-op
    with pytest.raises(ValueError, match="already registered"):
        attention_backends.register_attention_backend(_RecordingBackend(backend.name))


@pytest.mark.usefixtures("registry_sandbox")
def test_quantized_kv_reaches_declaring_backend_undequantized() -> None:
    backend = _RecordingBackend()
    backend.consumes_quantized_kv = True
    attention_backends.register_attention_backend(backend)
    q, k, v = _qkv()
    entry = KVCacheEntry(key=k, value=v).quantize(FP8_KV_DTYPE)
    scaled_dot_product_attention(q, None, None, quantized_kv=entry)
    assert backend.calls[0]["quantized_kv"] is entry
    assert backend.calls[0]["k"] is None


@pytest.mark.usefixtures("registry_sandbox")
def test_quantized_kv_dequantized_once_for_ordinary_backends() -> None:
    """A backend without consumes_quantized_kv receives dense k/v."""
    backend = _RecordingBackend()
    attention_backends.register_attention_backend(backend)
    q, k, v = _qkv()
    entry = KVCacheEntry(key=k, value=v).quantize(FP8_KV_DTYPE)
    scaled_dot_product_attention(q, None, None, quantized_kv=entry)
    call = backend.calls[0]
    assert call["quantized_kv"] is None
    dequant = entry.dequantize(q.dtype)
    torch.testing.assert_close(call["k"], dequant.key)
    torch.testing.assert_close(call["v"], dequant.value)


@pytest.mark.usefixtures("registry_sandbox")
def test_quantized_kv_dequantized_when_no_backend_takes_it() -> None:
    q, k, v = _qkv()
    entry = KVCacheEntry(key=k, value=v).quantize(FP8_KV_DTYPE)
    out = scaled_dot_product_attention(q, None, None, quantized_kv=entry)
    dequant = entry.dequantize(q.dtype)
    expected = scaled_dot_product_attention(q, dequant.key, dequant.value)
    torch.testing.assert_close(out, expected)


@pytest.mark.usefixtures("registry_sandbox")
def test_planned_none_skips_registry_consult() -> None:
    """An explicitly planned ``None`` runs SDPA without consulting anyone."""
    backend = _RecordingBackend()
    attention_backends.register_attention_backend(backend)
    q, k, v = _qkv()
    out = scaled_dot_product_attention(q, k, v, backend=None)
    assert not torch.all(out == 42.0)
    assert backend.specs == []
    assert backend.calls == []


@pytest.mark.usefixtures("registry_sandbox")
def test_planned_backend_runs_without_consult() -> None:
    """A planned backend object takes the call; is_preferred is not re-asked."""
    backend = _RecordingBackend(preferred=False)  # would decline if consulted
    out = scaled_dot_product_attention(*_qkv(), backend=backend)
    assert torch.all(out == 42.0)
    assert backend.specs == []
    assert len(backend.calls) == 1


@pytest.mark.usefixtures("registry_sandbox")
def test_lazy_spec_describes_the_live_call() -> None:
    """The per-call ("auto") path builds a faithful spec from the tensors."""
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


def test_fa3_backend_is_registered_by_default() -> None:
    """Importing tabpfn (via the module imports above) registers FA3."""
    names = [b.name for b in attention_backends.registered_attention_backends()]
    assert "fa3" in names


@pytest.mark.usefixtures("registry_sandbox")
def test_consult_order_updates_on_register_and_unregister() -> None:
    assert attention_backends.registered_attention_backends() == ()
    first = _RecordingBackend("first")
    second = _RecordingBackend("second")
    attention_backends.register_attention_backend(first)
    attention_backends.register_attention_backend(second)
    assert attention_backends.registered_attention_backends() == (second, first)
    attention_backends.unregister_attention_backend("second")
    assert attention_backends.registered_attention_backends() == (first,)
