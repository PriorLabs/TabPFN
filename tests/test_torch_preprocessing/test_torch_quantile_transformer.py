"""Tests for TorchQuantileTransformer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import QuantileTransformer

from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.adaptive_quantile_transformer import (
    AdaptiveQuantileTransformer,
)
from tabpfn.preprocessing.torch import (
    TorchPreprocessingPipeline,
    TorchQuantileTransformer,
    TorchQuantileTransformerStep,
)


class TestTorchQuantileTransformerSklearnEquivalence:
    """Tests for sklearn equivalence of TorchQuantileTransformer."""

    @pytest.mark.parametrize(
        ("n_samples", "n_features", "n_quantiles"),
        [
            (50, 3, 20),
            (100, 5, 100),
            (200, 10, 50),
            (500, 2, 200),
            (1000, 8, 100),
            (30, 15, 30),
            (150, 1, 75),
        ],
    )
    def test__fit_transform__matches_sklearn_various_shapes(
        self, n_samples: int, n_features: int, n_quantiles: int
    ):
        """Test sklearn equivalence across various data shapes and quantile counts."""
        rng = np.random.default_rng(42)
        torch.manual_seed(42)

        x_np = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Sklearn transformer
        sklearn_qt = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution="uniform", random_state=42
        )
        sklearn_result = sklearn_qt.fit_transform(x_np)

        # Torch transformer
        torch_qt = TorchQuantileTransformer(n_quantiles=n_quantiles)
        torch_result = torch_qt(x_torch)

        assert torch_result.shape == x_torch.shape
        assert torch.allclose(
            torch_result,
            torch.from_numpy(sklearn_result),
            atol=1e-5,
            rtol=1e-5,
        ), (
            f"Mismatch for shape ({n_samples}, {n_features}) with "
            f"{n_quantiles} quantiles. "
            f"Max diff: {(torch_result - torch.from_numpy(sklearn_result)).abs().max()}"
        )

    @pytest.mark.parametrize(
        ("n_samples", "n_features", "n_quantiles"),
        [
            (100, 3, 50),
            (200, 5, 100),
            (500, 2, 200),
            (150, 8, 75),
            (300, 4, 150),
        ],
    )
    def test__fit_transform__matches_sklearn_with_nans_various_shapes(
        self, n_samples: int, n_features: int, n_quantiles: int
    ):
        """Test NaN handling matches sklearn across various shapes."""
        rng = np.random.default_rng(42)

        x_np = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        # Add some NaN values (roughly 5% of data)
        nan_indices = rng.choice(
            n_samples * n_features,
            size=max(1, n_samples * n_features // 20),
            replace=False,
        )
        x_np.ravel()[nan_indices] = np.nan

        x_torch = torch.from_numpy(x_np)

        # Sklearn transformer
        sklearn_qt = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution="uniform", random_state=42
        )
        sklearn_result = sklearn_qt.fit_transform(x_np)

        # Torch transformer
        torch_qt = TorchQuantileTransformer(n_quantiles=n_quantiles)
        torch_result = torch_qt(x_torch)

        # Check that NaN positions are preserved
        nan_mask = torch.isnan(x_torch)
        assert torch.isnan(torch_result)[nan_mask].all(), (
            "NaN positions should be preserved"
        )

        # Check non-NaN values match sklearn
        non_nan_mask = ~torch.isnan(torch_result)
        sklearn_tensor = torch.from_numpy(sklearn_result)
        assert torch.allclose(
            torch_result[non_nan_mask],
            sklearn_tensor[non_nan_mask],
            atol=1e-5,
            rtol=1e-5,
        ), (
            f"Mismatch for shape ({n_samples}, {n_features}) with {n_quantiles} "
            "quantiles and NaNs. Max diff: "
            f"{(torch_result[non_nan_mask] - sklearn_tensor[non_nan_mask]).abs().max()}"
        )


class TestTorchQuantileTransformer:
    """Tests for TorchQuantileTransformer class."""

    def test__fit_transform__output_range(self):
        """Test that output values are in [0, 1] range."""
        torch.manual_seed(42)

        x = torch.randn(200, 10)
        qt = TorchQuantileTransformer(n_quantiles=100)
        result = qt(x)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test__fit_transform__nan_handling(self):
        """Test that NaN values are properly preserved."""
        qt = TorchQuantileTransformer(n_quantiles=50)
        x = torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [2.0, 4.0, float("nan")],
                [3.0, 5.0, 5.0],
                [4.0, 6.0, 7.0],
                [5.0, 7.0, 9.0],
            ]
        )

        result = qt(x)

        # NaN values should remain NaN
        assert torch.isnan(result[0, 1])
        assert torch.isnan(result[1, 2])

        # Non-NaN values should be in [0, 1]
        non_nan_mask = ~torch.isnan(result)
        assert result[non_nan_mask].min() >= 0.0
        assert result[non_nan_mask].max() <= 1.0

    def test__fit_transform__constant_feature(self):
        """Test that constant features are handled correctly."""
        qt = TorchQuantileTransformer(n_quantiles=50)
        x = torch.tensor(
            [
                [1.0, 5.0, 3.0],
                [2.0, 5.0, 4.0],
                [3.0, 5.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 5.0, 7.0],
            ]
        )

        result = qt(x)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Constant column (middle) should be 0.5
        assert torch.allclose(result[:, 1], torch.full((5,), 0.5))

    def test__forward__with_num_train_rows(self):
        """Test forward method with train/test splitting."""
        torch.manual_seed(42)

        qt = TorchQuantileTransformer(n_quantiles=100)
        x = torch.randn(150, 10)
        num_train_rows = 100

        result = qt(x, num_train_rows=num_train_rows)

        assert result.shape == x.shape

        # Training portion should have values spread across [0, 1]
        train_result = result[:num_train_rows]
        assert train_result.min() >= 0.0
        assert train_result.max() <= 1.0

        # Test values might be slightly outside [0, 1] before clamping,
        # but should still be clamped
        test_result = result[num_train_rows:]
        assert test_result.min() >= 0.0
        assert test_result.max() <= 1.0

    def test__forward__3d_tensor(self):
        """Test with 3D tensor (T, B, H) shape commonly used in TabPFN."""
        torch.manual_seed(42)

        qt = TorchQuantileTransformer(n_quantiles=50)
        x = torch.randn(100, 4, 10)  # T=100, B=4, H=10

        result = qt(x, num_train_rows=80)

        assert result.shape == x.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test__fit__returns_correct_cache_structure(self):
        """Test that fit returns the expected cache structure."""
        qt = TorchQuantileTransformer(n_quantiles=50)
        x = torch.randn(100, 5)

        cache = qt.fit(x)

        assert "quantiles" in cache
        assert "references" in cache
        assert cache["quantiles"].shape[0] == 50
        assert cache["quantiles"].shape[1] == 5
        assert cache["references"].shape[0] == 50
        assert cache["references"][0] == 0.0
        assert cache["references"][-1] == 1.0

    def test__transform__with_precomputed_cache(self):
        """Test transform with pre-computed cache."""
        qt = TorchQuantileTransformer(n_quantiles=50)
        x_train = torch.randn(100, 5)
        x_test = torch.randn(20, 5)

        cache = qt.fit(x_train)
        result = qt.transform(x_test, fitted_cache=cache)

        assert result.shape == x_test.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test__transform__invalid_cache_raises(self):
        """Test that providing invalid cache raises ValueError."""
        qt = TorchQuantileTransformer(n_quantiles=50)
        x = torch.randn(100, 5)

        with pytest.raises(ValueError, match="Invalid fitted cache"):
            qt.transform(x, fitted_cache={"quantiles": torch.randn(50, 5)})

        with pytest.raises(ValueError, match="Invalid fitted cache"):
            qt.transform(x, fitted_cache={"references": torch.linspace(0, 1, 50)})

    def test__fit_transform__n_quantiles_exceeds_samples(self):
        """Test that n_quantiles is adapted when it exceeds number of samples."""
        qt = TorchQuantileTransformer(n_quantiles=1000)
        x = torch.randn(50, 5)

        cache = qt.fit(x)

        # Should adapt to number of samples
        assert cache["quantiles"].shape[0] == 50
        assert cache["references"].shape[0] == 50

    def test__fit_transform__preserves_dtype(self):
        """Test that the output preserves the input dtype."""
        qt = TorchQuantileTransformer(n_quantiles=50)

        # Test float32
        x_f32 = torch.randn(100, 5, dtype=torch.float32)
        result_f32 = qt(x_f32)
        assert result_f32.dtype == torch.float32

        # Test float64
        x_f64 = torch.randn(100, 5, dtype=torch.float64)
        result_f64 = qt(x_f64)
        assert result_f64.dtype == torch.float64

    def test__fit_transform__preserves_device(self):
        """Test that the output preserves the input device."""
        qt = TorchQuantileTransformer(n_quantiles=50)
        x = torch.randn(100, 5)

        result = qt(x)
        assert result.device == x.device

    def test__fit_transform__all_nan_column(self):
        """Test handling of a column with all NaN values."""
        qt = TorchQuantileTransformer(n_quantiles=50)
        x = torch.tensor(
            [
                [1.0, float("nan")],
                [2.0, float("nan")],
                [3.0, float("nan")],
                [4.0, float("nan")],
                [5.0, float("nan")],
            ]
        )

        result = qt(x)

        # First column should be transformed normally
        assert not torch.isnan(result[:, 0]).any()
        assert result[:, 0].min() >= 0.0
        assert result[:, 0].max() <= 1.0

        # Second column (all NaN) should remain all NaN
        assert torch.isnan(result[:, 1]).all()


def _make_schema(
    *,
    num_numericals: int = 0,
    num_categoricals: int = 0,
) -> FeatureSchema:
    """Create a FeatureSchema from modality counts."""
    numerical = [
        Feature(name=None, modality=FeatureModality.NUMERICAL)
    ] * num_numericals
    categorical = [
        Feature(name=None, modality=FeatureModality.CATEGORICAL)
    ] * num_categoricals
    return FeatureSchema(features=numerical + categorical)


class TestTorchQuantileTransformerStepInPipeline:
    """Tests for TorchQuantileTransformerStep used inside TorchPreprocessingPipeline."""

    def test__call__transforms_numerical_columns_only(self):
        """Quantile transformer should only affect numerical columns."""
        torch.manual_seed(42)

        step = TorchQuantileTransformerStep(n_quantiles=50)
        pipeline = TorchPreprocessingPipeline(
            steps=[(step, {FeatureModality.NUMERICAL})]
        )
        schema = _make_schema(num_numericals=3, num_categoricals=2)

        x = torch.randn(100, 1, 5) * 10
        result = pipeline(x, schema, num_train_rows=80)

        # Numerical columns (0-2) should be in [0, 1]
        for col in range(3):
            assert result.x[:, :, col].min() >= 0.0
            assert result.x[:, :, col].max() <= 1.0

        # Categorical columns (3-4) should be unchanged
        assert torch.allclose(result.x[:, :, 3], x[:, :, 3])
        assert torch.allclose(result.x[:, :, 4], x[:, :, 4])

    def test__call__with_nan_values_preserves_nans(self):
        """NaN values should be preserved through the pipeline."""
        torch.manual_seed(42)

        step = TorchQuantileTransformerStep(n_quantiles=50)
        pipeline = TorchPreprocessingPipeline(
            steps=[(step, {FeatureModality.NUMERICAL})]
        )
        schema = _make_schema(num_numericals=3)

        x = torch.randn(100, 1, 3)
        x[5, 0, 0] = float("nan")
        x[10, 0, 1] = float("nan")
        x[20, 0, 2] = float("nan")

        result = pipeline(x, schema, num_train_rows=80)

        assert torch.isnan(result.x[5, 0, 0])
        assert torch.isnan(result.x[10, 0, 1])
        assert torch.isnan(result.x[20, 0, 2])

        non_nan = ~torch.isnan(result.x)
        assert result.x[non_nan].min() >= 0.0
        assert result.x[non_nan].max() <= 1.0


class TestChunkedQuantileTransformerEquivalence:
    """Tests that chunked computation produces identical results to unchunked."""

    def test__fit__column_chunks_match_unchunked(self, monkeypatch):
        """Column-chunked nanquantile fit must match unchunked for 2D and 3D input."""
        torch.manual_seed(42)

        for shape in [(80, 17), (60, 3, 7)]:
            x = torch.randn(*shape)

            # Unchunked reference: chunk_cols larger than any real n_features
            monkeypatch.setattr(
                TorchQuantileTransformer, "_get_fit_chunk_cols", lambda _, _x: 10**9
            )
            cache_unchunked = TorchQuantileTransformer(n_quantiles=30).fit(x)

            # Force chunk_cols=4 so the chunked else-branch is exercised
            monkeypatch.setattr(
                TorchQuantileTransformer, "_get_fit_chunk_cols", lambda _, _x: 4
            )
            cache_chunked = TorchQuantileTransformer(n_quantiles=30).fit(x)

            assert torch.allclose(
                cache_chunked["quantiles"], cache_unchunked["quantiles"], atol=1e-6
            )

    def test__transform__row_chunks_match_unchunked(self, monkeypatch):
        """Row-chunked transform must give identical results to the unchunked path."""
        torch.manual_seed(42)
        x = torch.randn(100, 12, dtype=torch.float32)
        x[3, 2] = float("nan")
        qt = TorchQuantileTransformer(n_quantiles=50)
        cache = qt.fit(x)

        result_full = qt.transform(x, cache)

        monkeypatch.setattr(
            TorchQuantileTransformer, "_get_transform_chunk_size", lambda _, _x: 7
        )
        result_chunked = qt.transform(x, cache)

        assert torch.allclose(result_full, result_chunked, atol=1e-6, equal_nan=True)


class TestTorchQuantileTransformerCategoricalBoundary:
    """Boundary-value equivalence for discrete/categorical data.

    When the input contains only a few unique integer values (e.g. 0-4),
    the min/max training values must be mapped to exactly 0.0 and 1.0,
    matching sklearn's ``QuantileTransformer`` behaviour.
    """

    @pytest.mark.parametrize("n_quantiles", [20, 40, 100])
    def test__categorical_boundary_values__match_sklearn(
        self, n_quantiles: int
    ) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.integers(0, 5, (200, 1)).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        sk = AdaptiveQuantileTransformer(
            n_quantiles=n_quantiles, output_distribution="uniform"
        )
        sk_result = sk.fit_transform(x_np)

        tqt = TorchQuantileTransformer(n_quantiles=n_quantiles)
        cache = tqt.fit(x_torch)
        torch_result = tqt.transform(x_torch, cache).numpy()

        np.testing.assert_allclose(
            sk_result,
            torch_result,
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"Boundary mismatch with n_quantiles={n_quantiles}",
        )
