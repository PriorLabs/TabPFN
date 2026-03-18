"""Verify that the v2 implementation computes exactly the same outputs as the base."""

from __future__ import annotations

import sys
from typing import cast

import pytest
import torch

from tabpfn import model_loading
from tabpfn.architectures import base, tabpfn_v2
from tabpfn.architectures.base.transformer import PerFeatureTransformer


def _convert_state_dict_base_to_v2(
    base_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert a base architecture state dict to a v2 architecture state dict."""
    n_layers = len(
        [k for k in base_state_dict if k.endswith("self_attn_between_features._w_qkv")]
    )
    base_to_v2_mapping = [
        ("encoder.5.layer.weight", "encoder.5.layer.weight"),
        ("y_encoder.2.layer.weight", "y_encoder.2.layer.weight"),
        ("y_encoder.2.layer.bias", "y_encoder.2.layer.bias"),
        ("decoder_dict.standard.0.weight", "output_projection.0.weight"),
        ("decoder_dict.standard.0.bias", "output_projection.0.bias"),
        ("decoder_dict.standard.2.weight", "output_projection.2.weight"),
        ("decoder_dict.standard.2.bias", "output_projection.2.bias"),
        (
            "feature_positional_embedding_embeddings.weight",
            "feature_positional_embedding_embeddings.weight",
        ),
        (
            "feature_positional_embedding_embeddings.bias",
            "feature_positional_embedding_embeddings.bias",
        ),
    ]
    for i in range(n_layers):
        base_to_v2_mapping.extend(
            [
                (
                    f"transformer_encoder.layers.{i}.mlp.linear1.weight",
                    f"blocks.{i}.mlp.0.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.mlp.linear2.weight",
                    f"blocks.{i}.mlp.2.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_features._w_qkv",
                    f"blocks.{i}.per_sample_attention_between_features.qkv_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_features._w_out",
                    f"blocks.{i}.per_sample_attention_between_features.out_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_items._w_qkv",
                    f"blocks.{i}.per_column_attention_between_cells.qkv_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_items._w_out",
                    f"blocks.{i}.per_column_attention_between_cells.out_projection.weight",
                ),
            ]
        )

    new_state_dict = {
        v2_key: base_state_dict[base_key] for base_key, v2_key in base_to_v2_mapping
    }
    keys_to_delete = []
    keys_to_add = {}
    for key, weight in new_state_dict.items():
        # QKV projection weight is 3, num_heads, output_size, input_size
        if "qkv_projection.weight" in key:
            q_key = key.replace("qkv_projection", "q_projection")
            k_key = key.replace("qkv_projection", "k_projection")
            v_key = key.replace("qkv_projection", "v_projection")
            keys_to_add[q_key] = weight[0].flatten(0, 1)
            keys_to_add[k_key] = weight[1].flatten(0, 1)
            keys_to_add[v_key] = weight[2].flatten(0, 1)
            keys_to_delete.append(key)
        if "out_projection.weight" in key:
            # Out projection is num_heads, head_size, output_size
            # Note that this differs from torch linear weight format
            # (output, input) and we need to transpose the output into the first
            # dim.
            new_state_dict[key] = new_state_dict[key].flatten(0, 1).T
    for key in keys_to_delete:
        new_state_dict.pop(key)
    new_state_dict.update(keys_to_add)
    return new_state_dict


def _create_identical_small_v2_and_base() -> tuple[
    tabpfn_v2.TabPFNV2, PerFeatureTransformer
]:
    """Construct the v2 and base architectures such that they have the same outputs."""
    configv2 = tabpfn_v2.TabPFNV2Config(
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
    arch_v2 = tabpfn_v2.get_architecture(
        configv2, n_out=10, cache_trainset_representation=False
    )
    arch_base = base.get_architecture(
        config_base, n_out=10, cache_trainset_representation=False
    )
    # Overwrite zero-initialized outputs to make sure we catch differences in
    # attention outputs.
    for param in arch_base.parameters():
        if param.abs().sum() < 1e-6:
            param.data += torch.randn_like(param) * 1e-1

    arch_v2.load_state_dict(
        _convert_state_dict_base_to_v2(arch_base.state_dict()), strict=True
    )

    arch_v2.to(torch.float64)
    arch_base.to(torch.float64)

    return arch_v2, arch_base


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

        arch_v2 = tabpfn_v2.get_architecture(
            tabpfn_v2.TabPFNV2Config(
                max_num_classes=config_base.max_num_classes,
                num_buckets=config_base.num_buckets,
            ),
            n_out=config_base.max_num_classes,
            cache_trainset_representation=False,
        )
        arch_v2.load_state_dict(
            _convert_state_dict_base_to_v2(arch_base.state_dict()), strict=True
        )

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
    arch, _ = _create_identical_small_v2_and_base()

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
