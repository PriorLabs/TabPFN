#  Copyright (c) Prior Labs GmbH 2026.
"""Tests for per-forward upfront attention-backend planning in TabPFN v3.

The planner (``TabPFNV3._plan_attention_backends``) resolves the backend for
every attention stage once per forward pass — instead of per attention call.
These tests pin that contract.
"""

from __future__ import annotations

import copy
import pickle
import threading
from collections.abc import Iterator

import pytest
import torch

from tabpfn.architectures import tabpfn_v3
from tabpfn.architectures.kv_cache import FP8_KV_DTYPE
from tabpfn.architectures.shared import attention_backends
from tabpfn.architectures.shared.attention_backends import (
    AttentionSpec,
)


def _is_icl_spec(spec: AttentionSpec) -> bool:
    """Identify the ICL stage of the tiny test model by its geometry.

    In `_get_model` the ICL attention is the only stage with head_dim 64 and
    3 heads (dist-embedder/aggregator run head_dim 16, the decoder 6 heads).
    """
    return spec.head_dim == 64 and spec.num_heads == 3


class _SpecRecordingBackend:
    """Records every consulted spec; takes the calls matching a predicate."""

    def __init__(
        self,
        name: str = "recorder",
        *,
        take=None,
    ):
        self.name = name
        self.take = take
        self.specs: list[AttentionSpec] = []
        self.run_calls: list[str] = []

    def is_preferred(self, spec: AttentionSpec) -> bool:
        self.specs.append(spec)
        return self.take is not None and self.take(spec)

    def run(  # noqa: ANN202
        self,
        q,
        k,  # noqa: ARG002
        v,  # noqa: ARG002
        *,
        quantized_kv=None,  # noqa: ARG002
        **_informational,
    ):
        self.run_calls.append("run")
        return torch.full_like(q, 0.5)


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


def _get_model(nlayers: int = 2) -> tabpfn_v3.TabPFNV3:
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


def _fit_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(20, 2, 5, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [10, 2], dtype=torch.float32)
    return x, y


def _all_planned_slots(model: tabpfn_v3.TabPFNV3) -> list:
    slots = []
    for block in model.feature_distribution_embedder.layers:
        slots.append(block.cross_attn_block1.attn.planned_backend)
        slots.append(block.cross_attn_block2.attn.planned_backend)
    for block in model.column_aggregator.blocks:
        slots.append(block.attention.planned_backend)
    for block in model.icl_blocks:
        slots.append(block.icl_attention.planned_backend)
        slots.append(block.icl_attention.planned_backend_test)
    if model.task_type == "multiclass":
        slots.append(model.many_class_decoder.planned_backend)
    return slots


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_planner_resolves_once_per_stage_not_per_call() -> None:
    """One registry consult per stage per forward, not per attention call."""
    backend = _SpecRecordingBackend(take=_is_icl_spec)
    attention_backends.register_attention_backend(backend)
    model = _get_model(nlayers=2)
    x, y = _fit_inputs()

    model(x, y)

    # Fit-style forward: six stages (dist-embed inducing + rows, feature
    # attention + readout, ICL, decoder), each consulted exactly once,
    # despite two ICL layers, three dist-embedder blocks and three
    # aggregator blocks.
    assert len(backend.specs) == 6
    # ...while the planned backend ran once per ICL layer.
    assert len(backend.run_calls) == 2

    (icl_spec,) = [s for s in backend.specs if _is_icl_spec(s)]
    assert icl_spec.seq_len_q == 20  # train + test rows
    assert icl_spec.seq_len_kv == 10  # train rows only
    assert icl_spec.batch_size == 2
    assert icl_spec.num_heads == 3
    assert icl_spec.head_dim == model.icl_emsize // 3
    assert icl_spec.is_grad_enabled is False
    (decoder_spec,) = [s for s in backend.specs if s.num_heads == 6]
    assert decoder_spec.seq_len_q == 10  # test rows
    assert decoder_spec.seq_len_kv == 10  # train rows


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_planner_cached_predict_spec_carries_quantized_dtype() -> None:
    """On the cached path the ICL spec reflects the cache: lengths + dtype."""
    backend = _SpecRecordingBackend()  # observe only, never take a call
    attention_backends.register_attention_backend(backend)
    model = _get_model()
    x, y = _fit_inputs()

    _, cache = model(x, y, return_kv_cache=True)
    cache = cache.quantize(FP8_KV_DTYPE)
    backend.specs.clear()

    model(x, y, kv_cache=cache)

    (icl_spec,) = [s for s in backend.specs if s.quantized_kv_dtype is not None]
    assert icl_spec.seq_len_q == 20  # every input row queries the cache
    assert icl_spec.seq_len_kv == 10  # cached train rows
    assert icl_spec.quantized_kv_dtype == FP8_KV_DTYPE


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_planner_uses_effective_autocast_dtype() -> None:
    """Under autocast the plan describes the autocast dtype, not the input's."""
    backend = _SpecRecordingBackend()
    attention_backends.register_attention_backend(backend)
    model = _get_model()
    x, y = _fit_inputs()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(x, y)

    assert backend.specs
    assert all(spec.dtype == torch.bfloat16 for spec in backend.specs)


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_plan_is_reset_after_forward_even_on_error() -> None:
    class _ExplodingBackend(_SpecRecordingBackend):
        def run(self, q, k, v, **kwargs):  # noqa: ANN202, ARG002
            raise RuntimeError("boom")

    backend = _ExplodingBackend(take=_is_icl_spec)
    attention_backends.register_attention_backend(backend)
    model = _get_model()
    x, y = _fit_inputs()

    with pytest.raises(RuntimeError, match="boom"):
        model(x, y)

    assert all(slot == "auto" for slot in _all_planned_slots(model))

    attention_backends.unregister_attention_backend(backend.name)
    model(x, y)  # normal forward: also resets
    assert all(slot == "auto" for slot in _all_planned_slots(model))


@pytest.mark.usefixtures("registry_sandbox")
@torch.no_grad()
def test_model_pickles_and_deepcopies_with_unpicklable_backend() -> None:
    """The plan never outlives a forward, so a lock-holding backend that was
    planned into the modules cannot poison pickling or deepcopy at rest.
    """

    class _LockedBackend(_SpecRecordingBackend):
        def __init__(self) -> None:
            super().__init__("locked", take=_is_icl_spec)
            self.lock = threading.Lock()  # not picklable

    attention_backends.register_attention_backend(_LockedBackend())
    model = _get_model()
    x, y = _fit_inputs()
    model(x, y)

    pickle.dumps(model)
    copy.deepcopy(model)
