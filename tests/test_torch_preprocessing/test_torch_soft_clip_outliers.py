"""Tests for TorchSoftClipOutliers."""

from __future__ import annotations

import pytest
import torch

from tabpfn.preprocessing.torch import TorchSoftClipOutliers


def test__fit_transform__basic_clamping():
    """Test that extreme outliers are softly clamped."""
    remover = TorchSoftClipOutliers(n_sigma=1.5)
    # Create data with clear outliers
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [100.0, -100.0],  # Extreme outliers
        ]
    )

    x_transformed = remover(x)

    # Outliers should be clamped closer to the bounds
    assert x_transformed[4, 0] < 100.0  # Upper outlier clamped down
    assert x_transformed[4, 1] > -100.0  # Lower outlier clamped up
    # Non-outliers should be mostly unchanged
    assert torch.allclose(x_transformed[:3], x[:3], atol=1e-5)


def test__fit_transform__nan_handling():
    """Test that NaN values are properly ignored in statistics computation."""
    remover = TorchSoftClipOutliers(n_sigma=2.0)
    x = torch.tensor(
        [
            [1.0, float("nan")],
            [2.0, 2.0],
            [3.0, 3.0],
            [float("nan"), 4.0],
        ]
    )

    x_transformed = remover(x)

    # Should not contain inf
    assert not torch.isinf(x_transformed).any()
    # Non-NaN values should be transformed without error
    assert not torch.isnan(x_transformed[0, 0])
    assert not torch.isnan(x_transformed[1:, 1]).any()


def test__fit__two_pass_robust_statistics():
    """Test that two-pass approach produces more robust bounds."""
    remover = TorchSoftClipOutliers(n_sigma=1.0)
    # Data with a single extreme outlier
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [1000.0]])

    fitted_cache = remover.fit(x)

    lower = fitted_cache["lower"]
    upper = fitted_cache["upper"]

    # The bounds should be computed from data excluding the outlier
    # With the outlier excluded, mean should be ~1.5 and std much smaller
    assert lower is not None
    assert upper is not None
    # Upper bound should be much less than 1000
    assert upper.item() < 100.0


def test__transform__without_fit_raises():
    """Test that transform without fit raises RuntimeError."""
    remover = TorchSoftClipOutliers()
    x = torch.randn(10, 5)

    with pytest.raises(ValueError, match="Invalid fitted cache"):
        remover.transform(x, fitted_cache={})


def test__call__with_num_train_rows():
    """Test that bounds are computed only from training portion."""
    remover = TorchSoftClipOutliers(n_sigma=3.0)
    x_train = torch.randn(50, 5) * 2  # Training data with std ~2
    x_test = torch.randn(50, 5) * 10  # Test data with larger std
    x = torch.cat([x_train, x_test], dim=0)

    x_transformed = remover(x, num_train_rows=50)

    assert x_transformed.shape == x.shape


def test__call__partial_bounds_raises():
    """Test that providing only lower or only upper raises ValueError."""
    remover = TorchSoftClipOutliers()
    x = torch.randn(10, 5)
    lower = torch.zeros(5)

    with pytest.raises(ValueError, match="Invalid fitted cache"):
        remover.transform(x, fitted_cache={"lower": lower})

    with pytest.raises(ValueError, match="Invalid fitted cache"):
        remover.transform(x, fitted_cache={"upper": lower})


def test__call__3d_tensor():
    """Test with 3D tensor (T, B, H) shape commonly used in TabPFN."""
    remover = TorchSoftClipOutliers(n_sigma=3.0)
    x = torch.randn(100, 4, 10)  # T=100, B=4, H=10

    x_transformed = remover(x, num_train_rows=80)

    assert x_transformed.shape == x.shape
    assert not torch.isnan(x_transformed).any()
    assert not torch.isinf(x_transformed).any()
