#  Copyright (c) Prior Labs GmbH 2026.
"""Verify that the v2 implementation computes exactly the same outputs as the base."""

from __future__ import annotations

import sys
from typing import cast

import pytest
import torch

from tabpfn import model_loading
from tabpfn.architectures import base, tabpfn_v2_sf
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.interface import PerformanceOptions


def _create_identical_small_v2_and_base() -> tuple[
    tabpfn_v2_sf.TabPFNV2, PerFeatureTransformer
]:
    """Construct the v2 and base architectures such that they have the same outputs."""
    configv2 = tabpfn_v2_sf.TabPFNV2Config(
        max_num_classes=10,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        features_per_group=2,
    )
    config_base = base.ModelConfig(
        max_num_classes=10,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        nhid_factor=4,
        features_per_group=2,
        remove_duplicate_features=False,
        nan_handling_enabled=True,
        # Needs to be 420 to match the single-file implementation
        # and avoid special handling of seed=42 in the multi-file
        # implementation.
        seed=420,
    )

    # Get the architectures
    arch_v2 = tabpfn_v2_sf.get_architecture(
        configv2, cache_trainset_representation=False
    )
    arch_base = base.get_architecture(config_base, cache_trainset_representation=False)
    # Overwrite zero-initialized outputs to make sure we catch differences in
    # attention outputs.
    for param in arch_base.parameters():
        if param.abs().sum() < 1e-6:
            param.data += torch.randn_like(param) * 1e-1

    # load_state_dict translates the base-architecture key names automatically.
    arch_v2.load_state_dict(arch_base.state_dict(), strict=True)

    arch_v2.to(torch.float64)
    arch_base.to(torch.float64)

    return arch_v2, arch_base


@torch.no_grad()
def test__load_state_dict__from_base_checkpoint_strict() -> None:
    """A base-architecture state dict can be loaded strictly (all keys consumed)."""
    arch_v2, arch_base = _create_identical_small_v2_and_base()
    # Re-loading the (already translated) base state dict must succeed with strict=True,
    # i.e. the translated keys exactly cover the single-file architecture's parameters.
    result = arch_v2.load_state_dict(arch_base.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


@torch.no_grad()
def test__load_state_dict__native_keys_still_work() -> None:
    """A round-trip of the single-file architecture's own state dict still works."""
    arch_v2, _ = _create_identical_small_v2_and_base()
    arch_v2.load_state_dict(arch_v2.state_dict(), strict=True)


class TestTabPFNv2NewVsOldImplementation:
    """Test that the v2 implementation computes exactly the same outputs as the base."""

    @torch.no_grad()
    @pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
    def test__forward__v2_and_base_have_same_output(self) -> None:
        loaded_models, _, loaded_configs, _ = model_loading.load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download_if_not_exists=True,
        )
        arch_base = cast(PerFeatureTransformer, loaded_models[0])
        # This is the seed used by the v2 implementation. It's important not to use 42,
        # as the base has special handling for this seed.
        arch_base.random_embedding_seed = 420
        config_base = loaded_configs[0]

        arch_v2 = tabpfn_v2_sf.get_architecture(
            tabpfn_v2_sf.TabPFNV2Config(
                max_num_classes=config_base.max_num_classes,
                num_buckets=config_base.num_buckets,
            ),
            cache_trainset_representation=False,
        )
        # load_state_dict translates the base-architecture key names automatically.
        arch_v2.load_state_dict(arch_base.state_dict(), strict=True)

        arch_v2.to(torch.float64)
        arch_base.to(torch.float64)

        # Create dummy input data
        x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
        y = torch.randint(0, 10, [97, 2], dtype=torch.float64)
        # Forward pass through both architectures
        output_v2 = arch_v2(x, y, only_return_standard_out=False)
        output_base = arch_base(x, y, only_return_standard_out=False)

        assert output_v2.keys(), "No output returned."
        msg = "Output keys do not match between implementations"
        assert output_v2.keys() == output_base.keys(), msg
        for key in output_v2:
            msg = f"Outputs for {key} do not match between implementations."
            assert torch.allclose(output_v2[key], output_base[key]), msg

    @pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
    def test__backward__v2_and_base_x_inputs_have_same_gradients(self) -> None:
        # Doing a forward and backward pass is more expensive, so use a smaller model.
        arch_v2, arch_base = _create_identical_small_v2_and_base()

        # Create dummy input data
        x_v2 = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
        x_base = x_v2.clone().detach()
        y_v2 = torch.randint(0, 10, [97, 2], dtype=torch.float32)
        y_base = y_v2.clone().detach()

        x_v2.requires_grad = True
        x_base.requires_grad = True

        # Forward pass and backward pass through both architectures
        arch_v2(x_v2, y_v2, only_return_standard_out=True).sum().backward()
        arch_base(x_base, y_base, only_return_standard_out=True).sum().backward()

        msg = "Gradients for input x do not match between implementations."
        assert torch.allclose(x_v2.grad, x_base.grad), msg


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled() -> None:
    arch, _ = _create_identical_small_v2_and_base()

    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float32)

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
    arch, _ = _create_identical_small_v2_and_base()

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
