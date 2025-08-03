from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tabpfn.architectures.base import encoders
from tabpfn.architectures.base.encoders import (
    InputNormalizationEncoderStep,
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    RemoveEmptyFeaturesEncoderStep,
    SequentialEncoder,
    VariableNumFeaturesEncoderStep,
    normalize_data,
    torch_nanmean,
    torch_nanstd,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_torch_nanmean(dtype):
    """Tests that torch_nanmean correctly calculates the mean, ignoring NaNs."""
    # Basic case
    x = torch.tensor([1, 2, 3, 4], dtype=dtype)
    assert torch.isclose(torch_nanmean(x), torch.tensor(2.5, dtype=dtype))

    # Case with NaNs
    x_nan = torch.tensor([1, 2, torch.nan, 4], dtype=dtype)
    assert torch.isclose(torch_nanmean(x_nan), torch.tensor(7 / 3, dtype=dtype))

    # All NaNs case should result in 0.0
    x_all_nan = torch.tensor([torch.nan, torch.nan], dtype=dtype)
    assert torch.isclose(torch_nanmean(x_all_nan), torch.tensor(0.0, dtype=dtype))


def test_torch_nanstd():
    """Tests that torch_nanstd correctly calculates std and edge cases."""
    # Basic case
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    assert torch.isclose(torch_nanstd(x), torch.std(x))

    # Case with NaNs
    x_nan = torch.tensor([1, 2, torch.nan, 4], dtype=torch.float32)
    expected_std = torch.std(torch.tensor([1, 2, 4], dtype=torch.float32))
    assert torch.isclose(torch_nanstd(x_nan), expected_std)

    # Edge Case: Single non-NaN value (std should be 0)
    x_single = torch.tensor([torch.nan, 3, torch.nan], dtype=torch.float32)
    assert torch.isclose(torch_nanstd(x_single), torch.tensor(0.0))

    # Edge Case: Constant values (std should be 0)
    x_const = torch.tensor([5, 5, 5, 5], dtype=torch.float32)
    assert torch.isclose(torch_nanstd(x_const), torch.tensor(0.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize(
    "shape",
    [(10, 3, 4), (5, 1, 2)],  # Sequence, Batch, Features
)
def test_normalize_data_basic(dtype, shape):
    """Tests basic normalization properties."""
    x = torch.randn(shape, dtype=dtype)
    # Ensure there is variance in the data to avoid the constant feature case
    if shape[0] > 1:
        x[0] *= 2
        x[1] *= -1

    x_norm = normalize_data(x)

    # Normalization happens along dim=0 (sequence dimension)
    # So we check that the mean and std along that dim are correct
    mean_of_norm = x_norm.mean(dim=0)
    std_of_norm = x_norm.std(dim=0)

    # Assert that mean is close to 0 and std is close to 1 for each feature
    assert torch.allclose(mean_of_norm, torch.zeros_like(mean_of_norm), atol=1e-3)
    assert torch.allclose(std_of_norm, torch.ones_like(std_of_norm), atol=1e-3)
    assert not torch.isnan(x_norm).any()
    assert not torch.isinf(x_norm).any()


def test_normalize_data_constant_feature():
    """Tests that a constant feature is normalized to zeros without producing NaNs."""
    x = torch.ones(10, 3, 4)  # A tensor of all ones
    x[:, :, 2] = 5.0  # Make one feature constant with value 5
    x_norm = normalize_data(x)

    # The constant feature column should be all zeros
    assert torch.all(x_norm[:, :, 2] == 0)
    assert not torch.isnan(x_norm).any(), "NaNs were produced for a constant feature."


def test_normalize_data_single_sample():
    """Tests that normalizing a single sample results in zeros without errors."""
    x = torch.randn(1, 3, 4)  # Single sample in sequence
    x_norm = normalize_data(x)
    assert torch.all(x_norm == 0)
    assert not torch.isnan(x_norm).any()


def test_input_normalization():
    N, B, F, _ = 10, 3, 4, 5
    x = torch.rand([N, B, F])

    kwargs = {
        "normalize_on_train_only": True,
        "normalize_to_ranking": False,
        "normalize_x": True,
        "remove_outliers": False,
    }

    encoder = SequentialEncoder(
        InputNormalizationEncoderStep(**kwargs), output_key=None
    )

    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert torch.isclose(
        out.var(dim=0), torch.tensor([1.0]), atol=1e-05
    ).all(), "Variance should be 1.0 for all features and batch samples."

    assert torch.isclose(
        out.mean(dim=0), torch.tensor([0.0]), atol=1e-05
    ).all(), "Mean should be 0.0 for all features and batch samples."

    out = encoder({"main": x}, single_eval_pos=5)["main"]
    assert torch.isclose(out[0:5].var(dim=0), torch.tensor([1.0]), atol=1e-03).all(), (
        "Variance should be 1.0 for all features and batch samples if"
        " we only test the normalized positions."
    )

    assert not torch.isclose(out.var(dim=0), torch.tensor([1.0]), atol=1e-05).all(), (
        "Variance should not be 1.0 for all features and batch samples if"
        " we look at the entire batch and only normalize some positions."
    )

    out_ref = encoder({"main": x}, single_eval_pos=5)["main"]
    x[:, 1, :] = 100.0
    x[:, 2, 6:] = 100.0
    out = encoder({"main": x}, single_eval_pos=5)["main"]
    assert (
        out[:, 0, :] == out_ref[:, 0, :]
    ).all(), "Changing one batch should not affeect the others."
    assert (
        out[:, 2, 0:5] == out_ref[:, 2, 0:5]
    ).all(), "Changing unnormalized part of the batch should not affect the others."


def test_remove_empty_feats():
    N, B, F, _ = 10, 3, 4, 5
    x = torch.rand([N, B, F])

    kwargs = {}

    encoder = SequentialEncoder(
        RemoveEmptyFeaturesEncoderStep(**kwargs), output_key=None
    )

    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert (out == x).all(), "Should not change anything if no empty columns."

    x[0, 1, 1] = 0.0
    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert (
        out[:, 1, -1] != 0
    ).all(), "Should not change anything if no column is entirely empty."

    x[:, 1, 1] = 0.0
    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert (
        out[:, 1, -1] == 0
    ).all(), "Empty column should be removed and shifted to the end."
    assert (
        out[:, 1, 1] != 0
    ).all(), "The place of the empty column should be filled with the next column."
    assert (
        out[:, 2, 1] != 0
    ).all(), "Non empty columns should not be changed in their position."


def test_variable_num_features():
    N, B, F, fixed_out = 10, 3, 4, 5
    x = torch.rand([N, B, F])

    kwargs = {"num_features": fixed_out, "normalize_by_used_features": True}

    encoder = SequentialEncoder(
        VariableNumFeaturesEncoderStep(**kwargs), output_key=None
    )

    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert (
        out.shape[-1] == fixed_out
    ), "Features were not extended to the requested number of features."
    assert torch.isclose(
        out[:, :, 0 : x.shape[-1]] / x, torch.tensor([math.sqrt(fixed_out / F)])
    ).all(), "Normalization is not correct."

    x[:, :, -1] = 1.0
    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert (out[:, :, -1] == 0.0).all(), "Constant features should not be normalized."
    assert torch.isclose(
        out[:, :, 0 : x.shape[-1] - 1] / x[:, :, :-1],
        torch.tensor(math.sqrt(fixed_out / (F - 1))),
    ).all(), """Normalization is not correct.
    Constant feature should not count towards number of feats."""

    kwargs["normalize_by_used_features"] = False
    encoder = SequentialEncoder(
        VariableNumFeaturesEncoderStep(**kwargs), output_key=None
    )
    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert (
        out[:, :, : x.shape[-1]] == x
    ).all(), "Features should be unchanged when not normalizing."


def test_nan_handling_encoder():
    N, B, F, _ = 10, 3, 4, 5
    x = torch.randn([N, B, F])
    x[1, 0, 2] = np.inf
    x[1, 0, 3] = -np.inf
    x[0, 1, 0] = np.nan
    x[:, 2, 1] = np.nan

    encoder = SequentialEncoder(NanHandlingEncoderStep(), output_key=None)

    out = encoder({"main": x}, single_eval_pos=-1)
    _, nan_indicators = out["main"], out["nan_indicators"]

    assert nan_indicators[1, 0, 2] == NanHandlingEncoderStep.inf_indicator
    assert nan_indicators[1, 0, 3] == NanHandlingEncoderStep.neg_inf_indicator
    assert nan_indicators[0, 1, 0] == NanHandlingEncoderStep.nan_indicator
    assert (nan_indicators[:, 2, 1] == NanHandlingEncoderStep.nan_indicator).all()

    assert not torch.logical_or(
        torch.isnan(out["main"]), torch.isinf(out["main"])
    ).any()
    assert out["main"].mean() < 1.0
    assert out["main"].mean() > -1.0


def test_linear_encoder():
    N, B, F, _ = 10, 3, 4, 5
    x = torch.randn([N, B, F])
    x2 = torch.randn([N, B, F])

    encoder = SequentialEncoder(
        LinearInputEncoderStep(
            num_features=F * 2,
            emsize=F,
            in_keys=["main", "features_2"],
        ),
        output_key=None,
    )

    out = encoder({"main": x, "features_2": x2}, single_eval_pos=-1)["output"]
    assert out.shape[-1] == F, "Output should have the requested number of features."


def test_combination():
    N, B, F, fixed_out = 10, 3, 4, 5
    x = torch.randn([N, B, F])
    x[:, 0, 1] = 1.0
    x[:, 2, 1] = 1.0
    domain_indicator = torch.randn([N, B, 1])

    encoder = SequentialEncoder(
        RemoveEmptyFeaturesEncoderStep(),
        NanHandlingEncoderStep(),
        InputNormalizationEncoderStep(
            normalize_on_train_only=True,
            normalize_to_ranking=False,
            normalize_x=True,
            remove_outliers=False,
        ),
        VariableNumFeaturesEncoderStep(num_features=fixed_out),
        VariableNumFeaturesEncoderStep(
            num_features=fixed_out,
            normalize_by_used_features=False,
            in_keys=["nan_indicators"],
            out_keys=["nan_indicators"],
        ),
        LinearInputEncoderStep(
            num_features=fixed_out * 2,
            emsize=F,
            in_keys=["main", "nan_indicators"],
            out_keys=["output"],
        ),
        output_key=None,
    )

    out = encoder({"main": x, "domain_indicator": domain_indicator}, single_eval_pos=-1)
    assert (out["nan_indicators"] == 0.0).all()
    assert (out["domain_indicator"] == domain_indicator).all()

    out_ref = encoder({"main": x}, single_eval_pos=5)["main"]
    x[:, 1, :] = 100.0
    x[6:, 2, 2] = 100.0
    out = encoder({"main": x}, single_eval_pos=5)["main"]
    assert (
        out[:, 0, :] == out_ref[:, 0, :]
    ).all(), "Changing one batch should not affeect the others."
    assert (
        out[0:5, 2, 2] == out_ref[0:5, 2, 2]
    ).all(), "Changing unnormalized part of the batch should not affect the others."

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
    assert (
        x_param.grad is not None
    ), "the encoder is not differentiable, i.e. the gradients are None."
    assert not torch.isnan(
        x_param.grad
    ).any(), "the encoder is not differentiable, i.e. the gradients are nan."


def test_multiclass_encoder():
    enc = encoders.MulticlassClassificationTargetEncoder()
    y = torch.tensor([[0, 1, 2, 1, 0], [0, 2, 2, 0, 0]]).T.unsqueeze(-1)
    solution = torch.tensor([[0, 1, 2, 1, 0], [0, 1, 1, 0, 0]]).T.unsqueeze(-1)
    y_enc = enc({"main": y}, single_eval_pos=3)["main"]
    assert (y_enc == solution).all(), f"y_enc: {y_enc}, solution: {solution}"


def test_interface():
    """Test if all encoders can be instantiated and whether they
    treat the test set independently,without interedependency between
    test examples.These tests are only rough and do not test all hyperparameter
    settings and only test the "main" input, e.g. not "nan_indicators".
    """
    # iterate over all subclasses of SeqEncStep and test if they work
    for name, cls in encoders.__dict__.items():
        if (
            isinstance(cls, type)
            and issubclass(cls, encoders.SeqEncStep)
            and cls is not encoders.SeqEncStep
        ):
            num_features = 4
            if cls is encoders.LinearInputEncoderStep:
                obj = cls(num_features=num_features, emsize=16)
            elif cls is encoders.VariableNumFeaturesEncoderStep:
                obj = cls(num_features=num_features)
            elif cls is encoders.InputNormalizationEncoderStep:
                obj = encoders.InputNormalizationEncoderStep(
                    normalize_on_train_only=True,
                    normalize_to_ranking=False,
                    normalize_x=True,
                    remove_outliers=True,
                )
            elif cls is encoders.FrequencyFeatureEncoderStep:
                obj = encoders.FrequencyFeatureEncoderStep(
                    num_features=num_features, num_frequencies=4
                )
            elif cls is encoders.CategoricalInputEncoderPerFeatureEncoderStep:
                continue
            elif cls is encoders.MulticlassClassificationTargetEncoder:
                num_features = 1
                obj = encoders.MulticlassClassificationTargetEncoder()
            else:
                obj = cls()
            x = torch.randn([10, 3, num_features])
            x2 = torch.randn([10, 3, num_features])

            obj({"main": x}, single_eval_pos=len(x), cache_trainset_representation=True)
            transformed_x2 = obj(
                {"main": x2}, single_eval_pos=0, cache_trainset_representation=True
            )
            transformed_x2_shortened = obj(
                {"main": x2[:5]}, single_eval_pos=0, cache_trainset_representation=True
            )
            transformed_x2_inverted = obj(
                {"main": torch.flip(x2, (0,))},
                single_eval_pos=0,
                cache_trainset_representation=True,
            )

            assert (
                transformed_x2["main"][:5] == transformed_x2_shortened["main"]
            ).all(), f"{name} does not work with shortened examples"
            assert (
                torch.flip(transformed_x2["main"], (0,))
                == transformed_x2_inverted["main"]
            ).all(), f"{name} does not work with inverted examples"
