#  Copyright (c) Prior Labs GmbH 2026.
"""TabPFN v3 and v3.5 route their attention calls through the registry.

Checked from the outside: a registered backend sees the calls it should,
described as they really are.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest
import torch

from tabpfn.architectures import tabpfn_v2_6, tabpfn_v3, tabpfn_v3_5
from tabpfn.architectures.interface import Architecture, ArchitectureModule
from tabpfn.architectures.kv_cache import FP8_KV_DTYPE
from tabpfn.architectures.shared import attention_backends
from tabpfn.architectures.shared.attention_backends import (
    AttentionRole,
    AttentionSpec,
)


def _is_icl_spec(spec: AttentionSpec) -> bool:
    """The test model's ICL calls: its only ones with head_dim 64 and 3 heads
    (embedder and aggregator run head_dim 16, the decoder 6 heads).
    """
    return spec.head_dim == 64 and spec.num_heads == 3


class _RecordingBackend:
    """Records every spec it is asked about; takes the calls it matches."""

    name = "recorder"

    def __init__(self, take=None):
        self.take = take
        self.specs: list[AttentionSpec] = []
        self.runs = 0
        self.run_roles: list[AttentionRole | None] = []

    def is_preferred(self, spec: AttentionSpec) -> bool:
        self.specs.append(spec)
        return self.take is not None and self.take(spec)

    def run(self, q, _k, _v, **kwargs):  # noqa: ANN202
        self.runs += 1
        self.run_roles.append(kwargs.get("role"))
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


# Every architecture that dispatches through the registry.
_ARCHITECTURES = [tabpfn_v3, tabpfn_v3_5]


def _model(
    architecture: ArchitectureModule,
    nlayers: int = 2,
    icl_num_kv_heads_test: int | None = None,
) -> Architecture:
    config, _ = architecture.parse_config(
        {
            "max_num_classes": 10,
            "num_buckets": 5,
            "embed_dim": 48,
            "nlayers": nlayers,
            "icl_num_heads": 3,
            "dist_embed_num_heads": 3,
            "feat_agg_num_heads": 3,
            # Both architectures then run their ICL attention at head_dim 64, in
            # one call per layer over train and test rows together.
            "feat_agg_num_cls_tokens": 4,
            "icl_num_kv_heads_test": icl_num_kv_heads_test,
        }
    )
    model = architecture.get_architecture(config)
    model.to(torch.float32)
    return model


def _forward_kwargs(model: Architecture) -> dict[str, str]:
    """v3.5 takes the task per call; v3 is built for one task."""
    if "task_type" in inspect.signature(model.forward).parameters:
        return {"task_type": "multiclass"}
    return {}


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(20, 2, 5, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [10, 2], dtype=torch.float32)
    return x, y


@pytest.mark.usefixtures("registry_sandbox")
@pytest.mark.parametrize("architecture", _ARCHITECTURES, ids=lambda m: m.__name__)
@torch.no_grad()
def test_backend_receives_the_calls_it_prefers(
    architecture: ArchitectureModule,
) -> None:
    """A backend preferring the ICL shape runs once per ICL layer."""
    backend = _RecordingBackend(take=_is_icl_spec)
    attention_backends.register_attention_backend(backend)
    model = _model(architecture, nlayers=2)
    x, y = _inputs()

    model(x, y, **_forward_kwargs(model))

    assert backend.runs == 2  # one per ICL layer
    icl_specs = [spec for spec in backend.specs if _is_icl_spec(spec)]
    assert {spec.seq_len_q for spec in icl_specs} == {20}  # train + test rows
    assert {spec.seq_len_kv for spec in icl_specs} == {10}  # train rows only
    assert {spec.batch_size for spec in icl_specs} == {2}
    assert all(spec.is_grad_enabled is False for spec in icl_specs)
    # Other stages were offered too, and declined.
    assert any(not _is_icl_spec(spec) for spec in backend.specs)


@pytest.mark.usefixtures("registry_sandbox")
@pytest.mark.parametrize("architecture", _ARCHITECTURES, ids=lambda m: m.__name__)
@torch.no_grad()
def test_cached_predict_specs_report_the_quantized_cache(
    architecture: ArchitectureModule,
) -> None:
    """On the cache path the specs carry the stored KV dtype."""
    backend = _RecordingBackend()  # observe only
    attention_backends.register_attention_backend(backend)
    model = _model(architecture)
    x, y = _inputs()

    _, cache = model(x, y, return_kv_cache=True, **_forward_kwargs(model))
    cache = cache.quantize(FP8_KV_DTYPE)
    backend.specs.clear()

    model(x[10:], y, kv_cache=cache, x_is_test_only=True, **_forward_kwargs(model))

    quantized = [s for s in backend.specs if s.quantized_kv_dtype is not None]
    assert quantized, "no spec reported the quantized cache"
    for spec in quantized:
        assert spec.quantized_kv_dtype == FP8_KV_DTYPE
        assert spec.seq_len_q == 10  # test rows query the cache
        assert spec.seq_len_kv == 10  # cached train rows


@pytest.mark.usefixtures("registry_sandbox")
@pytest.mark.parametrize("architecture", _ARCHITECTURES, ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    "icl_num_kv_heads_test", [None, 1], ids=["fused-icl", "split-icl"]
)
@torch.no_grad()
def test_only_the_icl_train_calls_carry_a_role(
    architecture: ArchitectureModule, icl_num_kv_heads_test: int | None
) -> None:
    """The ICL train call is labelled; the test call and the decoder are not.

    With as many test rows as training rows, the many-class decoder's calls have
    the ICL train call's geometry (S_q == S_kv, no GQA); only the role separates
    them, which is what a backend writing to its k/v in place relies on.
    """
    backend = _RecordingBackend(take=lambda _spec: True)
    attention_backends.register_attention_backend(backend)
    model = _model(architecture, icl_num_kv_heads_test=icl_num_kv_heads_test)
    x, y = _inputs()  # 10 train rows, 10 test rows

    model(x, y, **_forward_kwargs(model))

    labelled = [s for s in backend.specs if s.role is not None]
    assert len(labelled) == 2  # one per ICL layer
    for spec in labelled:
        assert spec.role is AttentionRole.ICL_TRAIN
        assert _is_icl_spec(spec)
        assert spec.seq_len_kv == 10
        assert spec.num_kv_heads == spec.num_heads
        # Train rows alone when split, train and test rows together when fused.
        assert spec.seq_len_q == (20 if icl_num_kv_heads_test is None else 10)
    # The role reaches run() only on the labelled calls.
    assert backend.run_roles.count(AttentionRole.ICL_TRAIN) == 2
    assert backend.runs == len(backend.specs)

    lookalikes = [
        s
        for s in backend.specs
        if not _is_icl_spec(s)
        and s.seq_len_q == s.seq_len_kv == 10
        and s.num_kv_heads == s.num_heads
    ]
    assert lookalikes, "expected the decoder to share the ICL train geometry"
    assert all(s.role is None for s in lookalikes)
    if icl_num_kv_heads_test is not None:
        test_calls = [
            s for s in backend.specs if _is_icl_spec(s) and s.num_kv_heads == 1
        ]
        assert len(test_calls) == 2
        assert all(s.role is None for s in test_calls)


@pytest.mark.usefixtures("registry_sandbox")
@pytest.mark.parametrize("all_train", [False, True], ids=["train-test", "all-train"])
@torch.no_grad()
def test_v2_6_icl_train_calls_carry_a_role(all_train: bool) -> None:
    """v2.x labels its per-column train call, split or alone."""
    backend = _RecordingBackend()  # observe only
    attention_backends.register_attention_backend(backend)
    nlayers = 2
    model = tabpfn_v2_6.get_architecture(
        tabpfn_v2_6.TabPFNV2p6Config(
            max_num_classes=10,
            num_buckets=5,
            emsize=48,
            nlayers=nlayers,
            nhead=3,
            features_per_group=3,
            num_thinking_rows=2,
        )
    )
    torch.manual_seed(0)
    x = torch.randn(20, 2, 5) * 0.1
    y = torch.randint(0, 10, [20 if all_train else 10, 2], dtype=torch.float32)

    model(x, y)

    labelled = [s for s in backend.specs if s.role is not None]
    assert len(labelled) == nlayers
    for spec in labelled:
        assert spec.role is AttentionRole.ICL_TRAIN
        assert spec.seq_len_q == spec.seq_len_kv
        assert spec.num_kv_heads == spec.num_heads
