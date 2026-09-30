#  Copyright (c) Prior Labs GmbH 2026.
"""Selective compilation through the existing v3.5 performance option."""

from __future__ import annotations

import dataclasses
import functools
import pickle
from collections.abc import Callable
from typing import Any

import pytest
import torch

from tabpfn.architectures import tabpfn_v3_5 as v35
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.shared.compile_utils import compile_when_enabled
from tests.test_architectures.test_tabpfn_v3_5 import _get_model, _inputs

pytestmark = pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason="Dynamo is not supported"
)

_REGIONS = [
    (v35.CrossAttentionBlock, "forward"),
    (v35.TransformerBlock, "_attention_delta"),
    (v35.TransformerBlock, "_mlp_delta"),
    (v35.TransformerBlock, "forward_cross"),
]


@pytest.fixture
def compiled_regions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Trace real graphs without requiring Inductor or inheriting cached wrappers."""
    for cls, name in _REGIONS:
        original = getattr(cls, name).__wrapped__
        monkeypatch.setattr(cls, name, compile_when_enabled(original))
    real_compile = torch.compile
    regions = []

    def compile_region(fn, **kwargs) -> Callable:
        assert kwargs == {"dynamic": True, "fullgraph": True}
        compiled = real_compile(fn, backend="eager", **kwargs)

        def run(*args, **call_kwargs) -> Any:
            regions.append(fn.__qualname__)
            return compiled(*args, **call_kwargs)

        return run

    monkeypatch.setattr(torch, "compile", compile_region)
    torch._dynamo.reset()
    return regions


@torch.no_grad()
@pytest.mark.parametrize("task_type", ["multiclass", "regression"])
@pytest.mark.parametrize("path", ["full", "chunked", "memory_saving"])
def test__compile_flag__selects_regions_and_can_be_disabled(
    compiled_regions: list[str], monkeypatch: pytest.MonkeyPatch, task_type, path
) -> None:
    model = _get_model(feat_agg_num_blocks=2, dist_embed_num_blocks=2)
    options = PerformanceOptions(
        use_chunkwise_inference=path == "chunked",
        save_peak_memory_factor=2 if path == "memory_saving" else None,
    )
    x, y = _inputs(task_type)

    # Guard the exclusion boundary at execution time, not just by checking which
    # functions were passed to torch.compile.
    def eager_only(fn) -> Callable:
        @functools.wraps(fn)
        def check(*args, **kwargs) -> Any:
            assert not torch.compiler.is_compiling()
            return fn(*args, **kwargs)

        return check

    for cls, name in [
        (v35.TabPFNV3p5, "_preprocess_raw"),
        (v35.TabPFNV3p5, "_process_row_chunk"),
        (v35.ICLTransformerBlock, "forward"),
    ]:
        monkeypatch.setattr(cls, name, eager_only(getattr(cls, name)))

    eager = model(x, y, task_type, performance_options=options)
    assert not compiled_regions
    enabled = dataclasses.replace(options, enable_torch_compile=True)
    compiled = model(x, y, task_type, performance_options=enabled)
    torch.testing.assert_close(compiled, eager)
    assert set(compiled_regions) == {f"{cls.__name__}.{name}" for cls, name in _REGIONS}

    # Disabling the option must bypass even already-created compiled callables.
    calls_before = len(compiled_regions)
    eager_again = model(x, y, task_type, performance_options=options)
    torch.testing.assert_close(eager_again, eager)
    assert len(compiled_regions) == calls_before


@torch.no_grad()
@pytest.mark.parametrize("chunked", [False, True])
def test__compiled_regions__support_kv_cache_and_serialization(
    compiled_regions: list[str], chunked: bool
) -> None:
    model = _get_model(feat_agg_num_blocks=2, dist_embed_num_blocks=2)
    x, y = _inputs("multiclass")
    options = PerformanceOptions(
        enable_torch_compile=True, use_chunkwise_inference=chunked
    )
    expected, cache = model(
        x, y, "multiclass", performance_options=options, return_kv_cache=True
    )
    cached = model(
        x[y.shape[0] :],
        y,
        "multiclass",
        performance_options=options,
        kv_cache=cache,
        x_is_test_only=True,
    )
    torch.testing.assert_close(cached, expected, atol=1e-5, rtol=1e-5)
    assert compiled_regions

    restored = pickle.loads(pickle.dumps(model))  # noqa: S301 -- local test model
    assert all("_torch_compile_cache" not in m.__dict__ for m in restored.modules())
    eager_options = dataclasses.replace(options, enable_torch_compile=False)
    restored_output = restored(x, y, "multiclass", performance_options=eager_options)
    torch.testing.assert_close(restored_output, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("checkpointing", [False, True])
def test__compiled_regions__preserve_gradients(
    compiled_regions: list[str], checkpointing: bool
) -> None:
    model = _get_model(feat_agg_num_blocks=2, dist_embed_num_blocks=2).train()
    x, y = _inputs("regression")
    eager = model(x, y, "regression")
    eager.square().mean().backward()
    gradients = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    compiled = model(
        x,
        y,
        "regression",
        performance_options=PerformanceOptions(
            enable_torch_compile=True, force_recompute_layer=checkpointing
        ),
    )
    compiled.square().mean().backward()
    torch.testing.assert_close(compiled, eager)
    for name, param in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(param.grad, gradients[name])
    assert compiled_regions


@torch.no_grad()
def test__compiled_regions__reuse_graphs_across_dataset_shapes(
    compiled_regions: list[str],
) -> None:
    model = _get_model(feat_agg_num_blocks=2, dist_embed_num_blocks=2)
    options = PerformanceOptions(enable_torch_compile=True)
    graph_counts = []
    torch._dynamo.utils.counters.clear()
    for rows, train, batch, features in [
        (24, 20, 2, 5),
        (31, 25, 3, 7),
        (38, 30, 4, 9),
    ]:
        x = torch.randn(rows, batch, features)
        y = (torch.arange(train) % 5).float().unsqueeze(1).expand(-1, batch)
        eager = model(x, y, "multiclass")
        compiled = model(x, y, "multiclass", performance_options=options)
        torch.testing.assert_close(compiled, eager)
        graph_counts.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    assert compiled_regions
    assert graph_counts[0] > 0
    assert graph_counts[0] == graph_counts[1] == graph_counts[2], graph_counts
