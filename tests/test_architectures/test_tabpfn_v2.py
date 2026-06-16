#  Copyright (c) Prior Labs GmbH 2026.
"""Verify that the v2 implementation computes exactly the same outputs as the base."""

from __future__ import annotations

import sys

import pytest
import torch

from tabpfn.architectures import tabpfn_v2
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.tabpfn_v2 import TabPFNV2Cache

TASK_TYPES = ["multiclass", "regression"]


def _build_small_arch(
    seed: int, emsize: int = 192, task_type: str = "multiclass"
) -> tabpfn_v2.TabPFNV2:
    model = tabpfn_v2.get_architecture(
        tabpfn_v2.TabPFNV2Config(
            max_num_classes=10 if task_type == "multiclass" else 0,
            num_buckets=5,
            emsize=emsize,
            nlayers=1,
            nhead=6,
            features_per_group=2,
            seed=seed,
        ),
        cache_trainset_representation=False,
    )
    for param in model.parameters():
        if param.abs().sum() < 1e-6:
            param.data += torch.randn_like(param) * 1e-1
    return model


def _make_targets(
    num_train: int,
    batch_size: int,
    task_type: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Training targets for ``task_type``.

    Class indices in ``[0, 10)`` for multiclass, standard-normal reals for regression.
    """
    if task_type == "multiclass":
        return torch.randint(0, 10, (num_train, batch_size), dtype=dtype)
    return torch.randn(num_train, batch_size, dtype=dtype)


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled(
    task_type: str,
) -> None:
    arch = _build_small_arch(seed=420, task_type=task_type)

    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = _make_targets(97, 2, task_type)

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


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_checkpointing_enabled_and_disabled(
    task_type: str,
) -> None:
    arch = _build_small_arch(seed=420, task_type=task_type)

    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = _make_targets(97, 2, task_type)

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


def _make_kv_cache_data(task_type: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(x_full, y_train, num_train)`` for the KV-cache tests.

    Uses float64 so the cache path (test rows recomputed against the cached K/V) can
    be compared against the standard forward at the tight ``atol=1e-10`` tolerance.
    """
    torch.manual_seed(0)
    num_train, num_test, num_features = 30, 7, 5
    x_full = torch.randn(num_train + num_test, 1, num_features, dtype=torch.float64)
    x_full = x_full * 0.5
    y_train = _make_targets(num_train, 1, task_type, dtype=torch.float64)
    return x_full, y_train, num_train


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
def test__kv_cache__matches_standard_forward(task_type: str) -> None:
    """KV-cache inference must match the standard (full train+test) forward."""
    arch = _build_small_arch(seed=420, task_type=task_type)
    arch.to(torch.float64)
    x_full, y_train, num_train = _make_kv_cache_data(task_type)
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train)

    # Build the cache; the store-mode output must match the standard forward.
    out_store, cache = arch(x_full, y_train, return_kv_cache=True)
    assert isinstance(cache, TabPFNV2Cache)
    assert not cache.is_empty()
    assert len(cache.kv) == 1  # nlayers=1
    assert cache.train_shape == (1, num_train)
    assert cache.feature_cache is not None
    assert torch.allclose(out_standard, out_store, atol=1e-10), (
        "return_kv_cache=True output differs from the standard forward."
    )

    # Use the cache on test-only data.
    out_cached = arch(x_test, y_train, kv_cache=cache, x_is_test_only=True)
    assert out_cached.shape == out_standard.shape
    assert torch.allclose(out_standard, out_cached, atol=1e-10), (
        "kv_cache inference output differs from the standard forward."
    )


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
def test__kv_cache__non_standard_out_matches(task_type: str) -> None:
    """The cache path also returns matching embeddings dicts."""
    arch = _build_small_arch(seed=420, task_type=task_type)
    arch.to(torch.float64)
    x_full, y_train, num_train = _make_kv_cache_data(task_type)
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train, only_return_standard_out=False)
    _, cache = arch(x_full, y_train, return_kv_cache=True)
    out_cached = arch(
        x_test,
        y_train,
        only_return_standard_out=False,
        kv_cache=cache,
        x_is_test_only=True,
    )
    for key in ("standard", "test_embeddings"):
        assert torch.allclose(out_standard[key], out_cached[key], atol=1e-10), (
            f"cache path output for {key} differs from the standard forward."
        )


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
def test__kv_cache__x_is_test_only_requires_populated_cache(task_type: str) -> None:
    arch = _build_small_arch(seed=420, task_type=task_type)
    arch.to(torch.float64)
    x_full, y_train, _ = _make_kv_cache_data(task_type)
    with pytest.raises(ValueError, match="requires a populated kv_cache"):
        arch(x_full, y_train, x_is_test_only=True)
