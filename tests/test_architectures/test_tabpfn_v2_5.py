#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the v2.5 single-file model."""

from __future__ import annotations

import sys
from dataclasses import asdict
from typing import cast

import pytest
import torch

from tabpfn import model_loading
from tabpfn.architectures import base, tabpfn_v2_5
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2p5Cache


def _create_identical_small_v2_5_and_base() -> tuple[
    tabpfn_v2_5.TabPFNV2p5, PerFeatureTransformer
]:
    """Construct v2.5 and base such that they have the same outputs."""
    configv2 = tabpfn_v2_5.TabPFNV2p5Config(
        max_num_classes=10,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        features_per_group=3,
        num_thinking_rows=2,
    )
    config_base = base.ModelConfig(
        max_num_classes=10,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        nhid_factor=2,
        features_per_group=3,
        remove_duplicate_features=False,
        nan_handling_enabled=True,
        num_thinking_rows=2,
        seed=42,
    )

    # Get the architectures
    arch_v2_5 = tabpfn_v2_5.get_architecture(
        configv2, cache_trainset_representation=False
    )
    arch_base = base.get_architecture(config_base, cache_trainset_representation=False)
    # Overwrite zero-initialized outputs to make sure we catch differences in
    # attention outputs.
    for param in arch_base.parameters():
        if param.abs().sum() < 1e-6:
            param.data += torch.randn_like(param) * 1e-1

    arch_v2_5.load_state_dict(arch_base.state_dict(), strict=True)

    arch_v2_5.to(torch.float64)
    arch_base.to(torch.float64)

    return arch_v2_5, arch_base


def test__load_state_dict__base_checkpoint_translates_and_round_trips() -> None:
    """Loading a base-architecture state dict should translate keys correctly.

    After loading, re-saving and re-loading the v2.5 state dict (which already
    has v2.5 keys) must leave all weights bit-for-bit identical.
    """
    arch_v2_5, _ = _create_identical_small_v2_5_and_base()
    weights_before = {k: v.clone() for k, v in arch_v2_5.state_dict().items()}
    # Round-trip: save v2.5 keys and load them back into the same model.
    arch_v2_5.load_state_dict(arch_v2_5.state_dict(), strict=True)
    for key, value_before in weights_before.items():
        assert torch.equal(arch_v2_5.state_dict()[key], value_before), (
            f"Weight '{key}' changed after round-trip load_state_dict."
        )


class TestTabPFNv2p5NewVsOldImplementation:
    """Test that the v2.5 architecture computes exactly the same outputs as base."""

    @torch.no_grad()
    @pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
    @pytest.mark.parametrize("model_type", ["regressor", "classifier"])
    def test__forward__v2_5_and_base_have_same_output(self, model_type: str) -> None:
        loaded_models, _, loaded_configs, _ = model_loading.load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which=model_type,
            version="v2.5",
            download_if_not_exists=True,
        )
        arch_base = cast(PerFeatureTransformer, loaded_models[0])
        config_base = loaded_configs[0]

        arch_v2_5 = tabpfn_v2_5.get_architecture(
            tabpfn_v2_5.TabPFNV2p5Config(**asdict(config_base)),
            cache_trainset_representation=False,
        )
        arch_v2_5.load_state_dict(arch_base.state_dict(), strict=True)
        arch_v2_5.to(torch.float64)
        arch_base.to(torch.float64)

        # Create dummy input data
        x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
        y = torch.randint(0, 10, [97, 2], dtype=torch.float64)
        x[0:10, :, 0] = torch.nan
        # We currently don't allow NaNs in the target so we don't test that yet.
        # Include once we allow NaNs in the target.
        # y[0:10, :] = torch.nan

        # Forward pass through both architectures
        output_v2_5 = arch_v2_5(x, y, only_return_standard_out=False)
        output_base = arch_base(x, y, only_return_standard_out=False)

        arch_v2_5.eval()
        arch_base.eval()

        assert output_v2_5.keys(), "No output returned."
        msg = "Output keys do not match between implementations"
        assert output_v2_5.keys() == output_base.keys(), msg
        for key in output_v2_5:
            msg = f"Shapes for {key} do not match between implementations."
            assert output_v2_5[key].shape == output_base[key].shape, msg
            msg = f"Outputs for {key} do not match between implementations."
            assert torch.allclose(output_v2_5[key], output_base[key], atol=1e-6), msg

    @pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
    def test__backward__v2_5_and_base_x_inputs_have_same_gradients(self) -> None:
        arch_v2, arch_base = _create_identical_small_v2_5_and_base()

        # Create dummy input data
        x_v2_5 = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
        x_base = x_v2_5.clone().detach()
        y_v2_5 = torch.randint(0, 10, [97, 2], dtype=torch.float64)
        y_base = y_v2_5.clone().detach()

        x_v2_5.requires_grad = True
        x_base.requires_grad = True

        arch_v2.train()
        arch_base.train()

        # Forward pass and backward pass through both architectures
        arch_v2(x_v2_5, y_v2_5, only_return_standard_out=True).sum().backward()
        arch_base(x_base, y_base, only_return_standard_out=True).sum().backward()

        msg = "Gradients for input x do not match between implementations."
        assert torch.allclose(x_v2_5.grad, x_base.grad), msg


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled() -> None:
    arch, _ = _create_identical_small_v2_5_and_base()

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
    arch, _ = _create_identical_small_v2_5_and_base()

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


def test__batch_size_one__padding_still_works() -> None:
    arch, _ = _create_identical_small_v2_5_and_base()

    x = torch.randn(100, 1, 1, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 1], dtype=torch.float64)
    output = arch(x, y)

    assert output.shape == (3, 1, 10)


# --- KV cache tests --------------------------------------------------------------


def _build_small_v2_5(
    *, max_num_classes: int = 10, nlayers: int = 2
) -> tabpfn_v2_5.TabPFNV2p5:
    """Build a small v2.5 architecture (float64) for the KV-cache tests."""
    arch = tabpfn_v2_5.get_architecture(
        tabpfn_v2_5.TabPFNV2p5Config(
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
    arch = _build_small_v2_5()
    x_full, y_train, num_train = _make_kv_cache_data()
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


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__regression_matches_standard_forward() -> None:
    """KV-cache inference matches the standard forward for the regression head too."""
    arch = _build_small_v2_5(max_num_classes=-1)
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
    arch = _build_small_v2_5()
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
    arch = _build_small_v2_5()
    x_full, y_train, _ = _make_kv_cache_data()
    with pytest.raises(ValueError, match="requires a populated kv_cache"):
        arch(x_full, y_train, x_is_test_only=True)
