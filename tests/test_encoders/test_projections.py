"""Tests for the projections from cell-level tensors to embedding space."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.architectures.encoders import (
    InputNormalizationEncoderStep,
    LinearFeatureGroupEmbedder,
    LinearInputEncoderStep,
    MLPFeatureGroupEmbedder,
    MLPInputEncoderStep,
    NanHandlingEncoderStep,
    RemoveEmptyFeaturesEncoderStep,
    TorchPreprocessingPipeline,
    VariableNumFeaturesEncoderStep,
)


def test_linear_encoder():
    N, B, F, _ = 10, 3, 4, 5
    x = torch.randn([N, B, F])
    x2 = torch.randn([N, B, F])

    encoder = TorchPreprocessingPipeline(
        steps=[
            LinearInputEncoderStep(
                num_features=F * 2, emsize=F, in_keys=("main", "features_2")
            ),
        ]
    )

    out = encoder({"main": x, "features_2": x2}, single_eval_pos=-1)["output"]
    assert out.shape[-1] == F, "Output should have the requested number of features."


def test_linear_projection_forward():
    N, B, num_features, emsize = 10, 3, 7, 5
    x = torch.randn([N, B, num_features])
    projection = LinearFeatureGroupEmbedder(
        num_features_per_group_with_metadata=num_features, emsize=emsize
    )
    out = projection(x)
    assert out.shape == (N, B, emsize)


@pytest.mark.parametrize("num_layers", [2, 3])
def test__MLPInputEncoderStep__embed_each_input_cell(num_layers):
    """Test MLP encoder input/output dimensions."""
    N, B, F = 10, 3, 4
    emsize = 8
    x = torch.randn([N, B, F])

    # Test basic MLP encoder with default hidden_dim (should equal emsize)
    encoder = TorchPreprocessingPipeline(
        steps=[
            MLPInputEncoderStep(
                num_features=F,
                emsize=emsize,
                num_layers=num_layers,
            ),
        ]
    )
    out = encoder({"main": x}, single_eval_pos=-1)["output"]
    assert out.shape == (
        N,
        B,
        emsize,
    ), f"Output shape should be ({N}, {B}, {emsize}), got {out.shape}"

    # Test with explicit hidden_dim
    hidden_dim = 16
    encoder = TorchPreprocessingPipeline(
        steps=[
            MLPInputEncoderStep(
                num_features=F,
                emsize=emsize,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            ),
        ]
    )
    out = encoder({"main": x}, single_eval_pos=-1)["output"]
    assert out.shape == (
        N,
        B,
        emsize,
    ), f"Output shape should be ({N}, {B}, {emsize}), got {out.shape}"


@pytest.mark.parametrize("num_layers", [2, 3])
def test_mlp_projection_forward(num_layers):
    N, B, num_features, emsize = 10, 3, 4, 8
    x = torch.randn([N, B, num_features])
    projection = MLPFeatureGroupEmbedder(
        num_features_per_group_with_metadata=num_features,
        emsize=emsize,
        num_layers=num_layers,
    )
    out = projection(x)
    assert out.shape == (N, B, emsize)


def test_combination():
    N, B, F, fixed_out = 10, 3, 4, 5
    x = torch.randn([N, B, F])
    x[:, 0, 1] = 1.0
    x[:, 2, 1] = 1.0
    domain_indicator = torch.randn([N, B, 1])

    encoder = TorchPreprocessingPipeline(
        steps=[
            RemoveEmptyFeaturesEncoderStep(
                in_keys=("main",),
                out_keys=("main",),
            ),
            NanHandlingEncoderStep(
                in_keys=("main",),
                out_keys=("main", "nan_indicators"),
            ),
            InputNormalizationEncoderStep(
                normalize_on_train_only=True,
                normalize_to_ranking=False,
                normalize_x=True,
                remove_outliers=False,
                in_keys=("main",),
                out_keys=("main",),
            ),
            VariableNumFeaturesEncoderStep(
                num_features=fixed_out,
                in_keys=("main",),
                out_keys=("main",),
            ),
            VariableNumFeaturesEncoderStep(
                num_features=fixed_out,
                normalize_by_used_features=False,
                in_keys=("nan_indicators",),
                out_keys=("nan_indicators",),
            ),
            LinearInputEncoderStep(
                num_features=fixed_out * 2,
                emsize=F,
                in_keys=("main", "nan_indicators"),
                out_keys=("output",),
            ),
        ],
    )

    out = encoder({"main": x, "domain_indicator": domain_indicator}, single_eval_pos=-1)
    assert (out["nan_indicators"] == 0.0).all()
    assert (out["domain_indicator"] == domain_indicator).all()

    out_ref = encoder({"main": x}, single_eval_pos=5)["main"]
    x[:, 1, :] = 100.0
    x[6:, 2, 2] = 100.0
    out = encoder({"main": x}, single_eval_pos=5)["main"]
    assert (out[:, 0, :] == out_ref[:, 0, :]).all(), (
        "Changing one batch should not affect the others."
    )
    assert (out[0:5, 2, 2] == out_ref[0:5, 2, 2]).all(), (
        "Changing unnormalized part of the batch should not affect the others."
    )

    x = torch.randn([N, B, F])
    x[1, 0, 2] = np.inf
    x[1, 0, 3] = -np.inf
    x[0, 1, 0] = np.nan
    x[:, 2, 1] = np.nan

    out = encoder({"main": x, "domain_indicator": domain_indicator}, single_eval_pos=-1)
    _, nan_indicators = out["main"], out["nan_indicators"]

    assert nan_indicators[1, 0, 2] == NanHandlingEncoderStep.inf_indicator
    assert nan_indicators[1, 0, 3] == NanHandlingEncoderStep.neg_inf_indicator
    assert nan_indicators[0, 1, 0] == NanHandlingEncoderStep.nan_indicator
    assert (nan_indicators[:, 2, 1] == NanHandlingEncoderStep.nan_indicator).all()

    assert torch.isclose(
        out["main"].mean(dim=[0, 2]), torch.tensor([0.0]), atol=1e-05
    ).all()

    x_param = torch.nn.parameter.Parameter(x, requires_grad=True)

    s = encoder(
        {"main": x_param, "domain_indicator": domain_indicator}, single_eval_pos=5
    )["main"].sum()
    s.backward()
    assert x_param.grad is not None, (
        "the encoder is not differentiable, i.e. the gradients are None."
    )
    assert not torch.isnan(x_param.grad).any(), (
        "the encoder is not differentiable, i.e. the gradients are nan."
    )
