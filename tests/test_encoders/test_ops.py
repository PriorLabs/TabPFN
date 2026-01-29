"""Tests for custom PyTorch operations in encoders/ops.py."""

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.encoders.steps._ops import (
    normalize_data,
    remove_outliers,
    select_features,
    torch_nanmean,
    torch_nanstd,
    torch_nansum,
)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize(
    "shape",
    [(10, 3, 4), (5, 1, 2)],  # Sequence, Batch, Features
)
def test__normalize_data__basic(dtype: torch.dtype, shape: tuple[int, int, int]):
    """Tests basic normalization properties."""
    x = torch.randn(shape, dtype=dtype)
    # Ensure there is variance in the data to avoid the constant feature case
    if shape[0] > 1:
        x[0] *= 2
        x[1] *= -1

    x_norm = normalize_data(x)

    mean_of_norm = x_norm.mean(dim=0)
    std_of_norm = x_norm.std(dim=0)

    # For dtype torch.float16 1e-3 is too much precision and results in
    # randomly failing tests, due to precision. Therefore increase the
    # tolerance.
    atol = 1e-2 if dtype == torch.float16 else 1e-3
    # Assert that mean is close to 0 and std is close to 1 for each feature
    assert torch.allclose(mean_of_norm, torch.zeros_like(mean_of_norm), atol=atol)
    assert torch.allclose(std_of_norm, torch.ones_like(std_of_norm), atol=atol)
    assert not torch.isnan(x_norm).any()
    assert not torch.isinf(x_norm).any()


def test__normalize_data__constant_feature():
    """Tests that a constant feature is normalized to zeros without producing NaNs."""
    x = torch.ones(10, 3, 4)  # A tensor of all ones
    x[:, :, 2] = 5.0  # Make one feature constant with value 5
    x_norm = normalize_data(x)

    # The constant feature column should be all zeros
    assert torch.all(x_norm[:, :, 2] == 0)
    assert not torch.isnan(x_norm).any(), "NaNs were produced for a constant feature."


def test__normalize_data__single_sample():
    """Tests that normalizing a single sample results in zeros without errors."""
    x = torch.randn(1, 3, 4)  # Single sample in sequence
    x_norm = normalize_data(x)
    assert torch.all(x_norm == 0)
    assert not torch.isnan(x_norm).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test__torch_nansum__basic(dtype: torch.dtype):
    """Tests that torch_nansum correctly calculates the sum, treating NaNs as zero."""
    x = torch.tensor([1, 2, 3, 4], dtype=dtype)
    assert torch.isclose(torch_nansum(x), torch.tensor(10.0, dtype=dtype))

    x_nan = torch.tensor([1, 2, torch.nan, 4], dtype=dtype)
    assert torch.isclose(torch_nansum(x_nan), torch.tensor(7.0, dtype=dtype))

    x_all_nan = torch.tensor([torch.nan, torch.nan], dtype=dtype)
    assert torch.isclose(torch_nansum(x_all_nan), torch.tensor(0.0, dtype=dtype))


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


def test__remove_outliers__basic():
    """Tests that remove_outliers clips extreme values based on n_sigma."""
    # Create data with clear outliers
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[2.0, 3.0, 4.0]],
            [[100.0, 3.0, 5.0]],  # Outlier in first feature
        ]
    )  # Shape: (3, 1, 3)

    result, (lower, upper) = remove_outliers(x, n_sigma=1)

    # Check that result has the same shape
    assert result.shape == x.shape

    # Check that bounds are returned
    assert lower.shape == (1, 3)
    assert upper.shape == (1, 3)

    # The outlier should be clipped
    assert result[2, 0, 0] < x[2, 0, 0]


def test__remove_outliers__with_provided_bounds():
    """Tests that remove_outliers uses provided lower and upper bounds."""
    x = torch.tensor(
        [
            [[1.0, 2.0]],
            [[50.0, 20.0]],
        ]
    )  # Shape: (2, 1, 2)

    lower = torch.tensor([[0.0, 1.0]])
    upper = torch.tensor([[4.0, 8.0]])

    result, (returned_lower, returned_upper) = remove_outliers(
        x,
        lower=lower,
        upper=upper,
    )

    # Check that the provided bounds are returned unchanged
    assert torch.allclose(returned_lower, lower)
    assert torch.allclose(returned_upper, upper)

    # Check that outliers are clipped
    assert result[1, 0, 0] < x[1, 0, 0]  # 5.0 should be clipped toward 4.0
    assert result[1, 0, 1] < x[1, 0, 1]  # 10.0 should be clipped toward 8.0
