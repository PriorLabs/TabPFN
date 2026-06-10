#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the v2.5 single-file model."""

from __future__ import annotations

import sys

import pytest
import torch

from tabpfn.architectures import tabpfn_v2_6
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.tabpfn_v2_6 import TabPFNV2p6Cache


def _get_model() -> tabpfn_v2_6.TabPFNV2p6:
    """Construct v2.5 and base such that they have the same outputs."""
    config = tabpfn_v2_6.TabPFNV2p6Config(
        max_num_classes=10,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        features_per_group=3,
        num_thinking_rows=2,
    )
    model = tabpfn_v2_6.get_architecture(config, cache_trainset_representation=False)
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


# --- KV cache tests --------------------------------------------------------------


def _build_small_v2_6(
    *, max_num_classes: int = 10, nlayers: int = 2
) -> tabpfn_v2_6.TabPFNV2p6:
    """Build a small v2.6 architecture (float64) for the KV-cache tests."""
    arch = tabpfn_v2_6.get_architecture(
        tabpfn_v2_6.TabPFNV2p6Config(
            max_num_classes=max_num_classes,
            num_buckets=5,
            emsize=96,
            nlayers=nlayers,
            nhead=6,
            features_per_group=2,
            num_thinking_rows=3,
        ),
    )
    arch.to(torch.float64)
    return arch


def _make_kv_cache_data() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(x_full, y_train, num_train)`` for the KV-cache tests.

    The data deliberately includes NaN and +/-Inf cells in both the train and test
    rows, plus a constant feature, to exercise the cached preprocessing (imputation
    means, standard-scaler statistics, NaN/Inf indicators, constant-feature mask and
    feature-group normalisation parameters).
    """
    torch.manual_seed(0)
    num_train, num_test, num_features = 30, 7, 6
    x_full = torch.randn(num_train + num_test, 1, num_features, dtype=torch.float64)
    x_full = x_full * 0.5
    # NaN cells in a train and a test feature.
    x_full[0:4, :, 0] = torch.nan
    x_full[num_train : num_train + 2, :, 1] = torch.nan
    # +Inf / -Inf cells in a train and a test feature.
    x_full[4:6, :, 2] = float("inf")
    x_full[6:8, :, 2] = float("-inf")
    x_full[num_train + 2 : num_train + 4, :, 4] = float("inf")
    x_full[num_train + 4, :, 4] = float("-inf")
    # Constant feature (removed for all rows).
    x_full[:, :, 3] = 9.0
    y_train = torch.randint(0, 10, (num_train, 1), dtype=torch.float64)
    return x_full, y_train, num_train


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__matches_standard_forward() -> None:
    """KV-cache inference must match the standard (full train+test) forward."""
    arch = _build_small_v2_6()
    x_full, y_train, num_train = _make_kv_cache_data()
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train)

    # Build the cache; the store-mode output must match the standard forward.
    out_store, cache = arch(x_full, y_train, return_kv_cache=True)
    assert isinstance(cache, TabPFNV2p6Cache)
    assert not cache.is_empty()
    assert len(cache.kv) == 2  # nlayers=2
    assert cache.train_shape == (1, num_train)
    assert cache.scaler_cache is not None
    assert cache.feature_state is not None
    assert torch.allclose(out_standard, out_store, atol=1e-10), (
        "return_kv_cache=True output differs from the standard forward."
    )

    # Use the cache on test-only data.
    out_cached = arch(x_test, y_train, kv_cache=cache, x_is_test_only=True)
    assert out_cached.shape == out_standard.shape
    assert torch.allclose(out_standard, out_cached, atol=1e-10), (
        "kv_cache inference output differs from the standard forward."
    )

    # Passing the full train+test tensor (x_is_test_only=False) slices off the train
    # rows internally and must give the same result.
    out_cached_full = arch(x_full, y_train, kv_cache=cache)
    assert torch.allclose(out_standard, out_cached_full, atol=1e-10), (
        "kv_cache inference with the full tensor differs from the standard forward."
    )


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__regression_matches_standard_forward() -> None:
    """KV-cache inference matches the standard forward for the regression head too."""
    arch = _build_small_v2_6(max_num_classes=-1)
    x_full, _, num_train = _make_kv_cache_data()
    y_train = torch.randn(num_train, 1, dtype=torch.float64)
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train)
    _, cache = arch(x_full, y_train, return_kv_cache=True)
    out_cached = arch(x_test, y_train, kv_cache=cache, x_is_test_only=True)
    assert torch.allclose(out_standard, out_cached, atol=1e-10)


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__non_standard_out_matches() -> None:
    """The cache path also returns matching embeddings dicts."""
    arch = _build_small_v2_6()
    x_full, y_train, num_train = _make_kv_cache_data()
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


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__x_is_test_only_requires_populated_cache() -> None:
    arch = _build_small_v2_6()
    x_full, y_train, _ = _make_kv_cache_data()
    with pytest.raises(ValueError, match="requires a populated kv_cache"):
        arch(x_full, y_train, x_is_test_only=True)
