#  Copyright (c) Prior Labs GmbH 2026.
"""TabPFN v3 dispatches its attention calls through the registry.

The architectures own no selection logic: every attention call reaches the
shared chokepoint, which describes it as an :class:`AttentionSpec` and asks
the registry. These tests check that from the outside — a registered backend
sees the calls it should, described as they really are.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from tabpfn.architectures import tabpfn_v3
from tabpfn.architectures.kv_cache import FP8_KV_DTYPE
from tabpfn.architectures.shared import attention_backends
from tabpfn.architectures.shared.attention_backends import AttentionSpec


def _is_icl_spec(spec: AttentionSpec) -> bool:
    """Identify the ICL calls of the tiny test model by their geometry.

    ICL is the only stage there with head_dim 64 and 3 heads (the embedder
    and aggregator run head_dim 16, the decoder 6 heads).
    """
    return spec.head_dim == 64 and spec.num_heads == 3


class _RecordingBackend:
    """Records every spec it is asked about; takes the calls it matches."""

    name = "recorder"

    def __init__(self, take=None):
        self.take = take
        self.specs: list[AttentionSpec] = []
        self.runs = 0

    def is_preferred(self, spec: AttentionSpec) -> bool:
        self.specs.append(spec)
        return self.take is not None and self.take(spec)

    def run(self, q, _k, _v, **_kwargs):  # noqa: ANN202
        self.runs += 1
        return torch.zeros_like(q)


@pytest.fixture
def registry_sandbox() -> Iterator[None]:
    """Snapshot and restore the registry around each test."""
    saved_registry = dict(attention_backends._registry)
    saved_order = attention_backends._consult_order
    attention_backends._registry.clear()
    attention_backends._consult_order = ()
    try:
        yield
    finally:
        attention_backends._registry.clear()
        attention_backends._registry.update(saved_registry)
        attention_backends._consult_order = saved_order


def _model(nlayers: int = 2) -> tabpfn_v3.TabPFNV3:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=10,
        num_buckets=5,
        embed_dim=48,
        nlayers=nlayers,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float32)
    return model


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(20, 2, 5, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [10, 2], dtype=torch.float32)
    return x, y


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_backend_receives_the_calls_it_prefers() -> None:
    """A backend preferring the ICL shape runs for every ICL layer, and the
    specs describe the call it actually gets.
    """
    backend = _RecordingBackend(take=_is_icl_spec)
    attention_backends.register_attention_backend(backend)
    model = _model(nlayers=2)
    x, y = _inputs()

    model(x, y)

    assert backend.runs == 2  # one per ICL layer
    icl_specs = [spec for spec in backend.specs if _is_icl_spec(spec)]
    assert {spec.seq_len_q for spec in icl_specs} == {20}  # train + test rows
    assert {spec.seq_len_kv for spec in icl_specs} == {10}  # train rows only
    assert {spec.batch_size for spec in icl_specs} == {2}
    assert all(spec.is_grad_enabled is False for spec in icl_specs)
    # Other stages were offered too, and declined.
    assert any(not _is_icl_spec(spec) for spec in backend.specs)


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_cached_predict_specs_report_the_quantized_cache() -> None:
    """On the cache path the ICL spec carries the stored KV dtype, so a
    backend can decide whether to consume the entry as it is stored.
    """
    backend = _RecordingBackend()  # observe only
    attention_backends.register_attention_backend(backend)
    model = _model()
    x, y = _inputs()

    _, cache = model(x, y, return_kv_cache=True)
    cache = cache.quantize(FP8_KV_DTYPE)
    backend.specs.clear()

    model(x[10:], y, kv_cache=cache, x_is_test_only=True)

    quantized = [s for s in backend.specs if s.quantized_kv_dtype is not None]
    assert quantized, "no spec reported the quantized cache"
    for spec in quantized:
        assert spec.quantized_kv_dtype == FP8_KV_DTYPE
        assert spec.seq_len_q == 10  # test rows query the cache
        assert spec.seq_len_kv == 10  # cached train rows


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_specs_report_the_autocast_dtype() -> None:
    """Under autocast the specs describe the dtype attention really runs in,
    not the input's — the spec is built from the projected tensors.
    """
    backend = _RecordingBackend()
    attention_backends.register_attention_backend(backend)
    model = _model()
    x, y = _inputs()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(x, y)

    assert backend.specs
    assert {spec.dtype for spec in backend.specs} == {torch.bfloat16}
