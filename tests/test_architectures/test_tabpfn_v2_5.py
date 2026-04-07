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

    output_without_memory_saving = arch(
        x, y, only_return_standard_out=False, save_peak_memory_factor=None
    )
    output_with_memory_saving = arch(
        x, y, only_return_standard_out=False, save_peak_memory_factor=4
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

    output_without_recomputation = arch(
        x, y, only_return_standard_out=False, force_recompute_layer=False
    )
    output_with_recomputation = arch(
        x, y, only_return_standard_out=False, force_recompute_layer=True
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
