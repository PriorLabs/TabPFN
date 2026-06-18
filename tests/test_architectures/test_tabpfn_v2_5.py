#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the v2.5 single-file model."""

from __future__ import annotations

import sys

import pytest
import torch

from tabpfn.architectures import tabpfn_v2_5
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2p5Cache

TASK_TYPES = ["multiclass", "regression"]


def _make_targets(
    num_train: int,
    batch_size: int,
    task_type: str,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Training targets for ``task_type``.

    Class indices in ``[0, 10)`` for multiclass, standard-normal reals for regression.
    """
    if task_type == "multiclass":
        return torch.randint(0, 10, (num_train, batch_size), dtype=dtype)
    return torch.randn(num_train, batch_size, dtype=dtype)


def _create_small_v2_5(task_type: str = "multiclass") -> tabpfn_v2_5.TabPFNV2p5:
    """Construct a small v2.5 architecture for ``task_type``."""
    configv2 = tabpfn_v2_5.TabPFNV2p5Config(
        max_num_classes=10 if task_type == "multiclass" else -1,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        features_per_group=3,
        num_thinking_rows=2,
    )

    # Get the architectures
    arch_v2_5 = tabpfn_v2_5.get_architecture(
        configv2, cache_trainset_representation=False
    )
    for param in arch_v2_5.parameters():
        if param.abs().sum() < 1e-6:
            param.data += torch.randn_like(param) * 1e-1

    arch_v2_5.to(torch.float64)
    return arch_v2_5


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled(
    task_type: str,
) -> None:
    arch = _create_small_v2_5(task_type)

    x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
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
    arch = _create_small_v2_5(task_type)

    x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
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


def test__thinking_rows__output_has_correct_shape() -> None:
    emsize = 8
    module = tabpfn_v2_5.AddThinkingRows(embedding_size=emsize, num_thinking_rows=5)

    batch_size = 2
    rows = 10
    features = 3
    embedded_input = torch.randn(batch_size, rows, features, emsize)
    single_eval_pos = 7

    output, new_single_eval_pos = module(embedded_input, single_eval_pos)

    assert output.shape == (
        batch_size,
        15,  # rows + num_thinking_rows
        features,
        emsize,
    )
    assert new_single_eval_pos == 12  # original + num_thinking_rows


def test__thinking_rows__tokens_equal_for_each_feature() -> None:
    emsize = 8
    module = tabpfn_v2_5.AddThinkingRows(embedding_size=emsize, num_thinking_rows=5)

    batch_size = 2
    n_rows = 10
    n_features = 3
    embedded_input = torch.randn(batch_size, n_rows, n_features, emsize)
    single_eval_pos = 7

    output, _ = module(embedded_input, single_eval_pos)

    assert output[0, 0, 0, 0] == output[0, 0, 1, 0]
    assert output[0, 0, 0, 0] == output[0, 0, 2, 0]
    assert output[0, 1, 0, 0] == output[0, 1, 1, 0]
    assert output[0, 1, 0, 0] == output[0, 1, 2, 0]


def test__thinking_rows__tokens_different_for_each_row() -> None:
    emsize = 8
    module = tabpfn_v2_5.AddThinkingRows(embedding_size=emsize, num_thinking_rows=5)

    batch_size = 2
    n_rows = 3
    n_features = 3
    embedded_input = torch.randn(batch_size, n_rows, n_features, emsize)
    single_eval_pos = 7

    output, _ = module(embedded_input, single_eval_pos)

    assert not torch.allclose(output[0, 0, 0, 0], output[0, 1, 0, 0])
    assert not torch.allclose(output[0, 0, 0, 0], output[0, 2, 0, 0])
    assert not torch.allclose(output[0, 0, 0, 0], output[0, 1, 0, 1])
    assert not torch.allclose(output[0, 0, 0, 0], output[0, 2, 0, 1])


def test__thinking_rows__save_and_load__output_has_same_value() -> None:
    emsize = 16
    embedded_input = torch.randn(2, 10, 3, emsize)
    single_eval_pos = 7

    module_1 = tabpfn_v2_5.AddThinkingRows(embedding_size=emsize, num_thinking_rows=5)
    module_2 = tabpfn_v2_5.AddThinkingRows(embedding_size=emsize, num_thinking_rows=5)

    output_1, new_pos_1 = module_1(embedded_input, single_eval_pos)
    state = module_1.state_dict()
    module_2.load_state_dict(state)
    output_2, new_pos_2 = module_2(embedded_input, single_eval_pos)

    assert new_pos_1 == new_pos_2
    assert torch.allclose(output_1, output_2)


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__batch_size_one__padding_still_works(task_type: str) -> None:
    arch = _create_small_v2_5(task_type)

    x = torch.randn(100, 1, 1, dtype=torch.float64) * 0.1
    y = _make_targets(97, 1, task_type)
    output = arch(x, y)

    expected_n_out = 10 if task_type == "multiclass" else 5
    assert output.shape == (3, 1, expected_n_out)


# --- KV cache tests --------------------------------------------------------------


def _build_small_v2_5(
    *, task_type: str = "multiclass", nlayers: int = 2
) -> tabpfn_v2_5.TabPFNV2p5:
    """Build a small v2.5 architecture (float64) for the KV-cache tests."""
    arch = tabpfn_v2_5.get_architecture(
        tabpfn_v2_5.TabPFNV2p5Config(
            max_num_classes=10 if task_type == "multiclass" else -1,
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


def _make_kv_cache_data(task_type: str) -> tuple[torch.Tensor, torch.Tensor, int]:
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
    y_train = _make_targets(num_train, 1, task_type)
    return x_full, y_train, num_train


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__matches_standard_forward(task_type: str) -> None:
    """KV-cache inference must match the standard (full train+test) forward."""
    arch = _build_small_v2_5(task_type=task_type)
    x_full, y_train, num_train = _make_kv_cache_data(task_type)
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train)

    # Build the cache; the store-mode output must match the standard forward.
    out_store, cache = arch(x_full, y_train, return_kv_cache=True)
    assert isinstance(cache, TabPFNV2p5Cache)
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


@pytest.mark.parametrize("task_type", TASK_TYPES)
@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__non_standard_out_matches(task_type: str) -> None:
    """The cache path also returns matching embeddings dicts."""
    arch = _build_small_v2_5(task_type=task_type)
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
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__x_is_test_only_requires_populated_cache(task_type: str) -> None:
    arch = _build_small_v2_5(task_type=task_type)
    x_full, y_train, _ = _make_kv_cache_data(task_type)
    with pytest.raises(ValueError, match="requires a populated kv_cache"):
        arch(x_full, y_train, x_is_test_only=True)
