#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for custom PyTorch operations in encoders/ops.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.preprocessing.steps.utils import mode as np_mode
from tabpfn.preprocessing.torch.ops import (
    mode as torch_mode,
    select_features,
    torch_nanmean,
    torch_nanstd,
)


@pytest.mark.parametrize(
    "x",
    [
        [[1.0, 10.0], [1.0, 20.0], [2.0, 20.0]],  # basic per-column mode
        [[np.nan, 5.0], [1.0, np.inf], [1.0, -np.inf]],  # ignores non-finite
        [[1.0], [1.0], [2.0], [2.0]],  # tie -> smallest value wins
        [[np.nan, 3.0], [np.inf, 3.0]],  # all-non-finite column -> NaN
        [[7.0, 7.0, 9.0]],  # single row
    ],
)
def test__torch_mode__matches_numpy_mode(x: list[list[float]]):
    """The torch and numpy ``mode`` implementations must agree element-wise."""
    arr = np.array(x, dtype=float)
    expected = np.asarray(np_mode(arr), dtype=float)
    got = torch_mode(torch.tensor(arr)).numpy().astype(float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(got[finite], expected[finite])


def test__torch_mode__matches_numpy_mode_1d():
    arr = np.array([np.nan, 4.0, 4.0, np.inf, 2.0], dtype=float)
    assert float(torch_mode(torch.tensor(arr))) == float(np_mode(arr))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test__torch_nanmean__basic(dtype: torch.dtype):
    """Tests that torch_nanmean correctly calculates the mean, ignoring NaNs."""
    x = torch.tensor([1, 2, 3, 4], dtype=dtype)
    assert torch.isclose(torch_nanmean(x), torch.tensor(2.5, dtype=dtype))

    x_nan = torch.tensor([1, 2, torch.nan, 4], dtype=dtype)
    assert torch.isclose(torch_nanmean(x_nan), torch.tensor(7 / 3, dtype=dtype))

    x_all_nan = torch.tensor([torch.nan, torch.nan], dtype=dtype)
    assert torch.isclose(torch_nanmean(x_all_nan), torch.tensor(0.0, dtype=dtype))


def test__torch_nanstd__basic():
    """Tests that torch_nanstd correctly calculates std and edge cases."""
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    assert torch.isclose(torch_nanstd(x), torch.std(x))

    x_nan = torch.tensor([1, 2, torch.nan, 4], dtype=torch.float32)
    expected_std = torch.std(torch.tensor([1, 2, 4], dtype=torch.float32))
    assert torch.isclose(torch_nanstd(x_nan), expected_std)

    x_single_valid = torch.tensor([torch.nan, 3, torch.nan], dtype=torch.float32)
    assert torch.isclose(torch_nanstd(x_single_valid), torch.tensor(0.0))

    x_constant = torch.tensor([5, 5, 5, 5], dtype=torch.float32)
    assert torch.isclose(torch_nanstd(x_constant), torch.tensor(0.0))


def test__select_features__all_selected():
    """Tests select_features returns unchanged tensor when all features are selected."""
    x = torch.randn(10, 3, 4)  # (sequence_length, batch_size, features)
    sel = torch.ones(3, 4, dtype=torch.bool)  # Select all features

    result = select_features(x, sel)
    assert torch.allclose(result, x)
    assert result.shape == x.shape


def test__select_features__batch_size_one():
    """Tests that select_features removes unselected features when batch_size=1."""
    x = torch.randn(10, 1, 5)  # (sequence_length, batch_size=1, features=5)
    sel = torch.tensor([[True, False, True, False, True]])  # Select 3 out of 5 features

    result = select_features(x, sel)
    assert result.shape == (10, 1, 3)
    # Check that the selected features match
    assert torch.allclose(result[:, 0, 0], x[:, 0, 0])
    assert torch.allclose(result[:, 0, 1], x[:, 0, 2])
    assert torch.allclose(result[:, 0, 2], x[:, 0, 4])


def test__select_features__no_features_selected():
    """Tests that select_features handles the case where no features are selected."""
    x = torch.randn(10, 2, 4)
    sel = torch.zeros(2, 4, dtype=torch.bool)  # Select no features

    result = select_features(x, sel)
    assert result.shape == (10, 2, 4)
    assert torch.allclose(result, torch.zeros_like(result))
