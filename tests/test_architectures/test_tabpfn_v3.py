"""Tests for the v3 single-file model."""

from __future__ import annotations

import sys

import pytest
import torch

from tabpfn.architectures import tabpfn_v3
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.tabpfn_v3 import TabPFNV3Cache


def _get_model() -> tabpfn_v3.TabPFNV3:
    """Construct v2.5 and base such that they have the same outputs."""
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=10,
        num_buckets=5,
        embed_dim=48,
        nlayers=1,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float64)
    return model


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled() -> None:
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float64)

    output_without_memory_saving = arch(x, y, only_return_standard_out=False)
    output_with_memory_saving = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(save_peak_memory_factor=4),
    )

    msg = "Output keys do not match between implementations"
    assert output_with_memory_saving.keys() == output_without_memory_saving.keys(), msg
    for key in output_with_memory_saving:
        assert torch.allclose(
            output_with_memory_saving[key], output_without_memory_saving[key]
        ), f"Outputs for {key} do not match between implementations."


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_checkpointing_enabled_and_disabled() -> None:
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float64)

    output_without_recomputation = arch(x, y, only_return_standard_out=False)
    output_with_recomputation = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(force_recompute_layer=True),
    )

    msg = "Output keys do not match between implementations"
    assert output_with_recomputation.keys() == output_without_recomputation.keys(), msg
    for key in output_with_recomputation:
        assert torch.allclose(
            output_with_recomputation[key], output_without_recomputation[key]
        ), f"Outputs for {key} do not match between implementations."


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__batch_size_one__padding_still_works() -> None:
    arch = _get_model()

    x = torch.randn(100, 1, 1, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 1], dtype=torch.float64)
    output = arch(x, y)

    assert output.shape == (3, 1, 10)


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward__no_test_set_works_batch_size_one() -> None:
    arch = _get_model()

    x = torch.randn(1, 1, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [1, 1], dtype=torch.float64)

    out = arch(x, y, only_return_standard_out=False)
    assert out["standard"].shape == (0, 1, 10)


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__mem_eff_forward_matches_standard_forward() -> None:
    """Memory-efficient inference path must be numerically identical to standard."""
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float64)

    # Standard path: disable memory-efficient inference via forward argument.
    output_standard = arch(x, y, only_return_standard_out=False)

    # Memory-efficient path: small fixed chunk sizes to force chunking
    # even on this tiny dataset.
    arch.inference_row_chunk_size = 50
    arch.inference_col_chunk_size = 10
    output_mem_eff = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(use_chunkwise_inference=True),
    )

    assert isinstance(output_standard, dict)
    assert isinstance(output_mem_eff, dict)
    assert output_mem_eff.keys() == output_standard.keys(), (
        "Output keys do not match between standard and memory-efficient paths."
    )
    for key in output_mem_eff:
        assert torch.allclose(output_mem_eff[key], output_standard[key], atol=1e-10), (
            f"Outputs for '{key}' differ between standard and memory-efficient "
            "forward passes."
        )


def _get_regression_model() -> tabpfn_v3.TabPFNV3:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=-1,
        num_buckets=100,
        embed_dim=32,
        nlayers=2,
        icl_num_heads=4,
        dist_embed_num_heads=4,
        dist_embed_num_blocks=1,
        feat_agg_num_heads=4,
        feat_agg_num_blocks=1,
        feat_agg_num_cls_tokens=2,
        dist_embed_num_inducing_points=8,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float64)
    return model


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
@pytest.mark.parametrize("use_chunkwise", [False, True])
def test__kv_cache__matches_standard_forward(use_chunkwise: bool) -> None:
    """KV-cache inference must produce identical output to standard forward."""
    arch = _get_regression_model()

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5, dtype=torch.float64) * 0.1
    y = torch.randn(10, dtype=torch.float64)

    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    # Standard forward (no cache)
    out_standard = arch(x, y, performance_options=perf)

    # Build cache
    out_store, cache = arch(x, y, performance_options=perf, return_kv_cache=True)

    assert isinstance(cache, TabPFNV3Cache)
    assert not cache.is_empty()
    assert cache.train_embeddings is not None
    assert len(cache.icl_cache.kv) == 2  # nlayers=2

    # Store-mode output matches standard
    assert torch.allclose(out_standard, out_store, atol=1e-10), (
        "return_kv_cache=True output differs from standard."
    )

    # Use cache for inference
    out_cached = arch(x, y, performance_options=perf, kv_cache=cache)

    assert torch.allclose(out_standard, out_cached, atol=1e-10), (
        "kv_cache inference output differs from standard."
    )


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__multiclass_matches_standard() -> None:
    """KV-cache inference for multiclass produces identical output."""
    arch = _get_model()

    torch.manual_seed(42)
    x = torch.randn(20, 1, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, (10,), dtype=torch.float64)

    perf = PerformanceOptions(use_chunkwise_inference=False)

    out_standard = arch(x, y, performance_options=perf)
    out_store, cache = arch(x, y, performance_options=perf, return_kv_cache=True)
    out_cached = arch(x, y, performance_options=perf, kv_cache=cache)

    assert torch.allclose(out_standard, out_store, atol=1e-10)
    assert torch.allclose(out_standard, out_cached, atol=1e-10)


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__gqa_matches_standard() -> None:
    """KV-cache inference with GQA (num_kv_heads_test) produces identical output."""
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=-1,
        num_buckets=100,
        embed_dim=32,
        nlayers=2,
        icl_num_heads=4,
        icl_num_kv_heads=4,
        icl_num_kv_heads_test=2,
        dist_embed_num_heads=4,
        dist_embed_num_blocks=1,
        feat_agg_num_heads=4,
        feat_agg_num_blocks=1,
        feat_agg_num_cls_tokens=2,
        dist_embed_num_inducing_points=8,
    )
    arch = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    arch.to(torch.float64)

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5, dtype=torch.float64) * 0.1
    y = torch.randn(10, dtype=torch.float64)

    perf = PerformanceOptions(use_chunkwise_inference=False)

    out_standard = arch(x, y, performance_options=perf)
    _, cache = arch(x, y, performance_options=perf, return_kv_cache=True)
    out_cached = arch(x, y, performance_options=perf, kv_cache=cache)

    assert torch.allclose(out_standard, out_cached, atol=1e-10)


@torch.no_grad()
@pytest.mark.parametrize("use_chunkwise", [False, True])
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test__kv_cache__works_under_autocast(
    use_chunkwise: bool, autocast_dtype: torch.dtype
) -> None:
    """KV cache inference works correctly under torch.autocast (fp16/bf16)."""
    arch = _get_regression_model().float()  # model in fp32

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5) * 0.1
    y = torch.randn(10)

    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    # Build cache WITHOUT autocast (fp32 cache)
    _, cache = arch(x, y, performance_options=perf, return_kv_cache=True)
    assert cache is not None

    # Standard forward WITHOUT autocast (reference)
    out_standard = arch(x, y, performance_options=perf)

    # Use cache WITH autocast — this is the scenario that triggered the
    # fp32-cache-under-fp16-input dtype mismatch.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        out_cached_autocast = arch(x, y, performance_options=perf, kv_cache=cache)

    # Also test standard forward under autocast for reference
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        out_standard_autocast = arch(x, y, performance_options=perf)

    # Autocast introduces precision differences; use a loose tolerance
    assert torch.allclose(
        out_standard.float(), out_cached_autocast.float(), atol=1e-2
    ), (
        f"Autocast ({autocast_dtype}) KV-cache output too far from standard "
        f"(max diff: {(out_standard.float() - out_cached_autocast.float()).abs().max().item():.2e})"  # noqa: E501
    )
    assert torch.allclose(
        out_standard_autocast.float(), out_cached_autocast.float(), atol=1e-2
    ), (
        f"Autocast ({autocast_dtype}) KV-cache output differs from autocast standard "
        f"(max diff: {(out_standard_autocast.float() - out_cached_autocast.float()).abs().max().item():.2e})"  # noqa: E501
    )
