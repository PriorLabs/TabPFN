#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for custom PyTorch operations in encoders/ops.py."""

from __future__ import annotations

import pytest
import torch

from tabpfn.preprocessing.torch.ops import select_features, torch_nanmean


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test__torch_nanmean__basic(dtype: torch.dtype):
    """Tests that torch_nanmean correctly calculates the mean, ignoring NaNs."""
    x = torch.tensor([1, 2, 3, 4], dtype=dtype)
    assert torch.isclose(torch_nanmean(x), torch.tensor(2.5, dtype=dtype))

    x_nan = torch.tensor([1, 2, torch.nan, 4], dtype=dtype)
    assert torch.isclose(torch_nanmean(x_nan), torch.tensor(7 / 3, dtype=dtype))

    x_all_nan = torch.tensor([torch.nan, torch.nan], dtype=dtype)
    assert torch.isclose(torch_nanmean(x_all_nan), torch.tensor(0.0, dtype=dtype))


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
