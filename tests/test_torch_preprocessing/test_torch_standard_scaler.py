"""Tests for TorchStandardScaler."""

from __future__ import annotations

import pytest
import torch

from tabpfn.preprocessing.torch import TorchStandardScaler


class TestTorchStandardScaler:
    """Tests for TorchStandardScaler class."""

    def test__fit_transform__basic(self):
        """Test basic fit_transform produces mean ~0 and std ~1."""
        scaler = TorchStandardScaler()
        x = torch.randn(100, 10)

        x_scaled = scaler(x)

        assert x_scaled.shape == x.shape

        mean = x_scaled.mean(dim=0)
        std = x_scaled.std(dim=0)

        assert torch.allclose(mean, torch.zeros(10), atol=1e-5)
        assert torch.allclose(std, torch.ones(10), atol=1e-1)

    def test__fit__nan_handling(self):
        """Test that NaN values are properly ignored in statistics computation."""
        scaler = TorchStandardScaler()
        x = torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [2.0, 4.0, float("nan")],
                [3.0, 5.0, 5.0],
                [4.0, 6.0, 7.0],
            ]
        )

        x_scaled = scaler(x)

        # Should not contain inf
        assert not torch.isinf(x_scaled).any()

        # NaN values should remain NaN after transformation
        assert torch.isnan(x_scaled[0, 1])
        assert torch.isnan(x_scaled[1, 2])

    def test__fit__all_nan_column(self):
        """Test handling of a column with all NaN values."""
        scaler = TorchStandardScaler()
        x = torch.tensor(
            [
                [1.0, float("nan")],
                [2.0, float("nan")],
                [3.0, float("nan")],
            ]
        )

        x_scaled = scaler(x)

        # Should not produce inf
        assert not torch.isinf(x_scaled).any()

        # First column should be normalized, second column should be all NaN
        assert not torch.isnan(x_scaled[:, 0]).any()
        assert torch.isnan(x_scaled[:, 1]).all()

    def test__fit__constant_feature(self):
        """Test that constant features are handled without producing NaN/inf."""
        scaler = TorchStandardScaler()
        x = torch.tensor(
            [
                [1.0, 5.0, 3.0],
                [2.0, 5.0, 4.0],
                [3.0, 5.0, 5.0],
            ]
        )

        x_scaled = scaler(x)

        assert not torch.isnan(x_scaled).any()
        assert not torch.isinf(x_scaled).any()

        # Constant column (middle) should be all zeros after centering
        assert torch.allclose(x_scaled[:, 1], torch.zeros(3))

    def test__fit__single_sample(self):
        """Test handling of a single sample (edge case)."""
        scaler = TorchStandardScaler()
        x = torch.tensor([[1.0, 2.0, 3.0]])

        x_scaled = scaler(x)

        assert not torch.isnan(x_scaled).any()
        assert not torch.isinf(x_scaled).any()

        # With single sample, std is set to 1, so output should be zeros
        assert torch.allclose(x_scaled, torch.zeros_like(x_scaled))

    def test__forward__with_num_train_rows(self):
        """Test forward method with train/test splitting."""
        scaler = TorchStandardScaler()
        x = torch.randn(150, 10)
        num_train_rows = 100

        x_scaled = scaler(x, num_train_rows=num_train_rows)

        assert x_scaled.shape == x.shape

        # Train portion should have mean ~0 and std ~1
        x_train_scaled = x_scaled[:num_train_rows]
        mean_train = x_train_scaled.mean(dim=0)
        std_train = x_train_scaled.std(dim=0)

        assert torch.allclose(mean_train, torch.zeros(10), atol=1e-5)
        assert torch.allclose(std_train, torch.ones(10), atol=1e-1)

    def test__forward__with_precomputed_statistics(self):
        """Test forward method with pre-computed mean and std."""
        scaler = TorchStandardScaler()
        x = torch.randn(100, 10)

        mean = x.mean(dim=0)
        std = x.std(dim=0)

        x_scaled = scaler.transform(x, fitted_cache={"mean": mean, "std": std})

        expected = (x - mean) / (std + 1e-16)
        assert torch.allclose(x_scaled, expected)

    def test__forward__partial_statistics_raises(self):
        """Test that providing only mean or only std raises ValueError."""
        scaler = TorchStandardScaler()
        x = torch.randn(100, 10)
        mean = x.mean(dim=0)

        with pytest.raises(ValueError, match="Invalid fitted cache"):
            scaler.transform(x, fitted_cache={"mean": mean})

        with pytest.raises(ValueError, match="Invalid fitted cache"):
            scaler.transform(x, fitted_cache={"std": x.std(dim=0)})

    def test__forward__3d_tensor(self):
        """Test with 3D tensor (T, B, H) shape commonly used in TabPFN."""
        scaler = TorchStandardScaler()
        x = torch.randn(100, 4, 10)  # T=100, B=4, H=10

        x_scaled = scaler(x, num_train_rows=80)

        assert x_scaled.shape == x.shape
        assert not torch.isnan(x_scaled).any()
        assert not torch.isinf(x_scaled).any()
