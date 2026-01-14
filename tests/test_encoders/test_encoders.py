"""Tests for the encoders."""

from __future__ import annotations

import math

import numpy as np
import torch

from tabpfn.architectures.encoders import (
    InputNormalizationEncoderStep,
    MulticlassClassificationTargetEncoderStep,
    NanHandlingEncoderStep,
    RemoveEmptyFeaturesEncoderStep,
    SequentialEncoder,
    VariableNumFeaturesEncoderStep,
)


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
    assert torch.isclose(out.var(dim=0), torch.tensor([1.0]), atol=1e-05).all(), (
        "Variance should be 1.0 for all features and batch samples."
    )

    assert torch.isclose(out.mean(dim=0), torch.tensor([0.0]), atol=1e-05).all(), (
        "Mean should be 0.0 for all features and batch samples."
    )

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
    assert (out[:, 0, :] == out_ref[:, 0, :]).all(), (
        "Changing one batch should not affeect the others."
    )
    assert (out[:, 2, 0:5] == out_ref[:, 2, 0:5]).all(), (
        "Changing unnormalized part of the batch should not affect the others."
    )


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
    assert (out[:, 1, -1] != 0).all(), (
        "Should not change anything if no column is entirely empty."
    )

    x[:, 1, 1] = 0.0
    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert (out[:, 1, -1] == 0).all(), (
        "Empty column should be removed and shifted to the end."
    )
    assert (out[:, 1, 1] != 0).all(), (
        "The place of the empty column should be filled with the next column."
    )
    assert (out[:, 2, 1] != 0).all(), (
        "Non empty columns should not be changed in their position."
    )


def test_variable_num_features():
    N, B, F, fixed_out = 10, 3, 4, 5
    x = torch.rand([N, B, F])

    kwargs = {"num_features": fixed_out, "normalize_by_used_features": True}

    encoder = SequentialEncoder(
        VariableNumFeaturesEncoderStep(**kwargs), output_key=None
    )

    out = encoder({"main": x}, single_eval_pos=-1)["main"]
    assert out.shape[-1] == fixed_out, (
        "Features were not extended to the requested number of features."
    )
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
    assert (out[:, :, : x.shape[-1]] == x).all(), (
        "Features should be unchanged when not normalizing."
    )


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


def test_multiclass_encoder():
    enc = MulticlassClassificationTargetEncoderStep()
    y = torch.tensor([[0, 1, 2, 1, 0], [0, 2, 2, 0, 0]]).T.unsqueeze(-1)
    solution = torch.tensor([[0, 1, 2, 1, 0], [0, 1, 1, 0, 0]]).T.unsqueeze(-1)
    y_enc = enc({"main": y}, single_eval_pos=3)["main"]
    assert (y_enc == solution).all(), f"y_enc: {y_enc}, solution: {solution}"
