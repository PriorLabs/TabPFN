#  Copyright (c) Prior Labs GmbH 2026.
"""Selective compilation through the existing v3.5 performance option."""

from __future__ import annotations

import dataclasses
import functools
import pickle
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch
from torch.torch_version import TorchVersion

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


@pytest.fixture(
    params=[
        pytest.param(False, id="production"),
        pytest.param(True, id="fullgraph"),
    ]
)
def compiled_regions(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> list[str]:
    """Trace real graphs without requiring Inductor or inheriting cached wrappers."""
    if TorchVersion(torch.__version__) < TorchVersion("2.6"):
        pytest.skip("v3.5 compilation requires PyTorch >= 2.6")
    for cls, name in _REGIONS:
        original = getattr(cls, name).__wrapped__
        monkeypatch.setattr(cls, name, compile_when_enabled(original))
    real_compile = torch.compile
    regions = []

    def compile_region(fn, **kwargs) -> Callable:
        assert kwargs == {"dynamic": True, "fullgraph": False}
        # Production permits graph breaks; the strict variant catches accidental
        # fragmentation of the supported aggregation regions.
        kwargs["fullgraph"] = request.param
        compiled = real_compile(fn, backend="eager", **kwargs)

        def run(*args, **call_kwargs) -> Any:
            regions.append(fn.__qualname__)
            return compiled(*args, **call_kwargs)

        return run

    monkeypatch.setattr(torch, "compile", compile_region)
    torch._dynamo.reset()
    return regions


@torch.no_grad()
def test__compile_flag__unsupported_pytorch__raises_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _get_model()
    x, y = _inputs("multiclass")
    expected = model(x, y, "multiclass")
    monkeypatch.setattr(torch, "__version__", "2.5.0")
    with pytest.raises(ValueError, match=r"v3.5 compilation requires PyTorch >= 2\.6"):
        model(
            x,
            y,
            "multiclass",
            performance_options=PerformanceOptions(enable_torch_compile=True),
        )
    torch.testing.assert_close(model(x, y, "multiclass"), expected)


@pytest.mark.skipif(
    TorchVersion(torch.__version__) < TorchVersion("2.6"),
    reason="v3.5 compilation requires PyTorch >= 2.6",
)
def test__compiled_region__concurrent_first_calls__initialize_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    real_compile = torch.compile

    def compile_region(fn, **kwargs) -> Callable:
        calls.append(fn)
        time.sleep(0.05)  # Release the GIL during wrapper creation to expose races.
        return real_compile(fn, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_region)

    @compile_when_enabled
    def region(layer, x, *, enable_torch_compile=False) -> torch.Tensor:
        del enable_torch_compile
        return layer(x)

    layers = [torch.nn.Linear(4, 4).eval() for _ in range(2)]
    x = torch.randn(3, 4)
    expected = [layer(x) for layer in layers]
    start = threading.Barrier(2)

    def run(layer) -> torch.Tensor:
        start.wait(timeout=10)
        return region(layer, x.clone(), enable_torch_compile=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(run, layers))
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference)
    torch.testing.assert_close(
        region(layers[0], x, enable_torch_compile=True), expected[0]
    )
    assert len(calls) == 1


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
