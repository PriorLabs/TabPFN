"""Tests for TorchRemoveOutliers."""

from __future__ import annotations

import pytest
import torch

from tabpfn.preprocessing.torch import TorchRemoveOutliers


def test__fit_transform__basic_clamping():
    """Test that extreme outliers are softly clamped."""
    remover = TorchRemoveOutliers(n_sigma=2.0)
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
    assert x_transformed[3, 0] < 100.0  # Upper outlier clamped down
    assert x_transformed[3, 1] > -100.0  # Lower outlier clamped up
    # Non-outliers should be mostly unchanged
    assert torch.allclose(x_transformed[:3], x[:3], atol=1e-5)


def test__fit_transform__nan_handling():
    """Test that NaN values are properly ignored in statistics computation."""
    remover = TorchRemoveOutliers(n_sigma=2.0)
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
    remover = TorchRemoveOutliers(n_sigma=1.0)
    # Data with a single extreme outlier
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [1000.0]])

    remover.fit(x)

    # The bounds should be computed from data excluding the outlier
    # With the outlier excluded, mean should be ~1.5 and std much smaller
    assert remover.lower_ is not None
    assert remover.upper_ is not None
    # Upper bound should be much less than 1000
    assert remover.upper_.item() < 100.0


def test__transform__without_fit_raises():
    """Test that transform without fit raises RuntimeError."""
    remover = TorchRemoveOutliers()
    x = torch.randn(10, 5)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        remover.transform(x)


def test__call__with_num_train_rows():
    """Test that bounds are computed only from training portion."""
    remover = TorchRemoveOutliers(n_sigma=3.0)
    x_train = torch.randn(50, 5) * 2  # Training data with std ~2
    x_test = torch.randn(50, 5) * 10  # Test data with larger std
    x = torch.cat([x_train, x_test], dim=0)

    x_transformed = remover(x, num_train_rows=50)

    assert x_transformed.shape == x.shape
    # Bounds should be based on training data, so test data gets more clamped
    assert remover.lower_ is not None
    assert remover.upper_ is not None


def test__call__with_precomputed_bounds():
    """Test using pre-computed lower and upper bounds."""
    remover = TorchRemoveOutliers()
    x = torch.tensor([[0.0], [5.0], [10.0], [15.0]])
    lower = torch.tensor([2.0])
    upper = torch.tensor([12.0])

    x_transformed = remover(x, lower=lower, upper=upper)

    # Values inside bounds should be unchanged or close
    assert torch.allclose(x_transformed[1], x[1], atol=0.1)
    assert torch.allclose(x_transformed[2], x[2], atol=0.1)
    # Values outside bounds should be clamped
    assert x_transformed[0, 0] > x[0, 0]  # 0.0 clamped up toward 2.0
    assert x_transformed[3, 0] < x[3, 0]  # 15.0 clamped down toward 12.0


def test__call__partial_bounds_raises():
    """Test that providing only lower or only upper raises ValueError."""
    remover = TorchRemoveOutliers()
    x = torch.randn(10, 5)
    lower = torch.zeros(5)

    with pytest.raises(ValueError, match="both or neither"):
        remover(x, lower=lower, upper=None)

    with pytest.raises(ValueError, match="both or neither"):
        remover(x, lower=None, upper=lower)


def test__call__3d_tensor():
    """Test with 3D tensor (T, B, H) shape commonly used in TabPFN."""
    remover = TorchRemoveOutliers(n_sigma=3.0)
    x = torch.randn(100, 4, 10)  # T=100, B=4, H=10

    x_transformed = remover(x, num_train_rows=80)

    assert x_transformed.shape == x.shape
    assert not torch.isnan(x_transformed).any()
    assert not torch.isinf(x_transformed).any()
