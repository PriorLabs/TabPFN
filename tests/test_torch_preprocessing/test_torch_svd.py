"""Tests for TorchTruncatedSVD, TorchSafeStandardScaler, and AddSVDFeaturesStep."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import svd_flip as sklearn_svd_flip

from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.add_svd_features_step import (
    AddSVDFeaturesStep as SklearnAddSVDFeaturesStep,
    get_svd_n_components,
)
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
)
from tabpfn.preprocessing.torch.steps import TorchAddSVDFeaturesStep
from tabpfn.preprocessing.torch.torch_svd import (
    TorchSafeStandardScaler,
    TorchTruncatedSVD,
    _svd_flip_stable,
)


class TestSvdFlipStable:
    """Tests for the _svd_flip_stable helper function."""

    def test__svd_flip_stable__matches_sklearn_when_no_ties(self):
        """Without ties, stable flip should match sklearn's v-based flip."""
        torch.manual_seed(42)
        u = torch.randn(10, 5)
        v = torch.randn(5, 8)

        u_torch, v_torch = _svd_flip_stable(u, v)
        u_sklearn, v_sklearn = sklearn_svd_flip(
            u.numpy(), v.numpy(), u_based_decision=False
        )

        assert torch.allclose(u_torch, torch.from_numpy(u_sklearn), atol=1e-6)
        assert torch.allclose(v_torch, torch.from_numpy(v_sklearn), atol=1e-6)

    def test__svd_flip_stable__deterministic_with_ties(self):
        """When multiple elements tie for max abs value, the leftmost wins."""
        # Construct v where row 0 has a tie: columns 2 and 5 both have abs=3
        u = torch.eye(2, dtype=torch.float32)
        v = torch.tensor(
            [
                [1.0, 2.0, -3.0, 0.0, 1.0, 3.0, 0.5],
                [0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        # Leftmost max-abs in row 0 is col 2 (value -3), so sign = -1
        # Leftmost max-abs in row 1 is col 2 (value 2), so sign = +1
        _, v_out = _svd_flip_stable(u, v)

        # Row 0 flipped (sign of col 2 was negative)
        assert v_out[0, 2].item() == 3.0
        assert v_out[0, 5].item() == -3.0  # also flipped
        # Row 1 NOT flipped (sign of col 2 was positive)
        assert v_out[1, 2].item() == 2.0


class TestTorchSafeStandardScaler:
    """Tests for TorchSafeStandardScaler class."""

    def test__fit_transform__matches_sklearn_no_mean(self):
        """Test that scaling matches sklearn's StandardScaler(with_mean=False)."""
        torch.manual_seed(42)
        x_torch = torch.randn(100, 10)
        x_numpy = x_torch.numpy()

        # sklearn
        sklearn_scaler = StandardScaler(with_mean=False)
        x_sklearn = sklearn_scaler.fit_transform(x_numpy)

        # torch
        torch_scaler = TorchSafeStandardScaler()
        x_torch_scaled = torch_scaler(x_torch)

        assert torch.allclose(
            x_torch_scaled, torch.from_numpy(x_sklearn).float(), atol=5e-6
        )

    def test__fit__nan_and_inf_handling(self):
        """Test that NaN and inf values are imputed (matching CPU make_scaler_safe)."""
        scaler = TorchSafeStandardScaler()

        # Test NaN handling: NaN values should be imputed with column means
        x_nan = torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [2.0, 4.0, float("nan")],
                [3.0, 5.0, 5.0],
                [4.0, 6.0, 7.0],
            ]
        )
        x_scaled = scaler(x_nan)
        assert not torch.isinf(x_scaled).any()
        assert not torch.isnan(x_scaled).any(), (
            "NaN values should be imputed with column means, not propagated"
        )

        # Test inf handling: inf should be converted to NaN, then imputed
        x_inf = torch.tensor(
            [
                [1.0, float("inf"), 3.0],
                [2.0, 4.0, float("-inf")],
                [3.0, 5.0, 5.0],
            ]
        )
        x_scaled_inf = scaler(x_inf)
        assert not torch.isinf(x_scaled_inf).any()
        assert not torch.isnan(x_scaled_inf).any(), (
            "inf values should be converted to NaN and then imputed"
        )

    def test__fit__edge_cases(self):
        """Test constant features and single sample don't produce NaN/inf."""
        scaler = TorchSafeStandardScaler()

        # Constant feature
        x_const = torch.tensor(
            [
                [1.0, 5.0, 3.0],
                [2.0, 5.0, 4.0],
                [3.0, 5.0, 5.0],
            ]
        )
        x_scaled = scaler(x_const)
        assert not torch.isnan(x_scaled[:, 0]).any()
        assert not torch.isnan(x_scaled[:, 2]).any()
        assert not torch.isinf(x_scaled).any()

        # Single sample
        x_single = torch.tensor([[1.0, 2.0, 3.0]])
        x_scaled_single = scaler(x_single)
        assert not torch.isnan(x_scaled_single).any()
        assert not torch.isinf(x_scaled_single).any()


class TestTorchTruncatedSVD:
    """Tests for TorchTruncatedSVD class."""

    def test__fit_transform__matches_sklearn_arpack(self):
        """Test that SVD matches sklearn's TruncatedSVD with ARPACK algorithm."""
        torch.manual_seed(42)
        rng = np.random.default_rng(42)

        # Create data with sufficient rank
        x_numpy = rng.standard_normal((50, 20)).astype(np.float32)
        x_torch = torch.from_numpy(x_numpy)

        n_components = 5

        # sklearn with ARPACK (deterministic)
        sklearn_svd = TruncatedSVD(n_components=n_components, algorithm="arpack")
        x_sklearn = sklearn_svd.fit_transform(x_numpy)

        # torch
        torch_svd = TorchTruncatedSVD(n_components=n_components)
        fitted_cache = torch_svd.fit(x_torch)
        x_torch_result = torch_svd.transform(x_torch, fitted_cache)

        # Components match up to per-component sign flips (different LAPACK
        # backends may resolve the SVD sign ambiguity differently).
        torch_comps = fitted_cache["components"]
        sk_comps = torch.from_numpy(sklearn_svd.components_.copy()).float()
        assert torch.allclose(torch_comps.abs(), sk_comps.abs(), atol=1e-5)

        # Transformed data matches up to per-component sign flips.
        sk_transformed = torch.from_numpy(x_sklearn.copy()).float()
        assert torch.allclose(x_torch_result.abs(), sk_transformed.abs(), atol=1e-4)

    def test__fit__n_components_clamped(self):
        """Test that n_components is clamped by both features and samples."""
        svd = TorchTruncatedSVD(n_components=100)

        # Clamped by features
        x_feat = torch.randn(50, 10)
        cache_feat = svd.fit(x_feat)
        assert cache_feat["components"].shape[0] == 10
        assert cache_feat["singular_values"].shape[0] == 10

        # Clamped by samples
        x_samp = torch.randn(5, 20)
        cache_samp = svd.fit(x_samp)
        assert cache_samp["components"].shape[0] == 5
        assert cache_samp["singular_values"].shape[0] == 5

    def test__transform__nan_handling(self):
        """Test that rows with NaN produce NaN output."""
        svd = TorchTruncatedSVD(n_components=3)
        x_train = torch.randn(50, 10)
        fitted_cache = svd.fit(x_train)

        x_test = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [1.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        )

        x_transformed = svd.transform(x_test, fitted_cache)

        # First and third row should not have NaN
        assert not torch.isnan(x_transformed[0]).any()
        assert not torch.isnan(x_transformed[2]).any()

        # Second row (with NaN input) should be all NaN
        assert torch.isnan(x_transformed[1]).all()

    def test__transform__missing_cache_raises(self):
        """Test that transform raises ValueError with invalid cache."""
        svd = TorchTruncatedSVD(n_components=5)
        x = torch.randn(10, 5)

        with pytest.raises(ValueError, match="Invalid fitted cache"):
            svd.transform(x, fitted_cache={})


class TestTorchSVDIntegration:
    """Integration tests for the full SVD pipeline."""

    def test__full_pipeline__matches_sklearn(self):
        """Test the complete pipeline: scale -> SVD -> compare with sklearn."""
        torch.manual_seed(123)
        rng = np.random.default_rng(123)

        # Generate test data
        n_samples, n_features = 100, 30
        n_components = 10

        x_numpy = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        x_torch = torch.from_numpy(x_numpy)

        # sklearn pipeline (as used in reshape_feature_distribution_step.py)
        sklearn_scaler = StandardScaler(with_mean=False)
        sklearn_svd = TruncatedSVD(n_components=n_components, algorithm="arpack")

        x_scaled_sklearn = sklearn_scaler.fit_transform(x_numpy)
        x_transformed_sklearn = sklearn_svd.fit_transform(x_scaled_sklearn)

        # torch pipeline
        torch_scaler = TorchSafeStandardScaler()
        torch_svd = TorchTruncatedSVD(n_components=n_components)

        x_scaled_torch = torch_scaler(x_torch)
        x_transformed_torch = torch_svd(x_scaled_torch)

        # Match up to per-component sign flips
        np.testing.assert_allclose(
            np.abs(x_transformed_torch.numpy()),
            np.abs(x_transformed_sklearn),
            rtol=1e-4,
            atol=1e-4,
        )


class TestAddSVDFeaturesStep:
    """Tests for AddSVDFeaturesStep pipeline step."""

    def test__fit_transform__basic_shape(self):
        """Test that fit_transform produces correct output shapes."""
        torch.manual_seed(42)
        step = TorchAddSVDFeaturesStep("svd_quarter_components")

        # Shape: [num_rows, batch_size, num_cols]
        x = torch.randn(100, 2, 20)
        num_train_rows = 80

        result = step.fit_transform(
            x, column_indices=list(range(20)), num_train_rows=num_train_rows
        )

        # Original columns should be unchanged
        assert result.x.shape == x.shape

        # SVD features should be added
        assert result.added_columns is not None
        assert result.added_columns.shape[0] == 100  # num_rows
        assert result.added_columns.shape[1] == 2  # batch_size
        # get_svd_n_components("svd_quarter_components", 80, 20) = min(9, 5) = 5
        assert result.added_columns.shape[2] == 5

        # Modality should be NUMERICAL
        assert result.added_modality == FeatureModality.NUMERICAL

    def test__fit_transform__n_components_clamped(self):
        """Test that n_components is clamped based on training rows and features."""
        torch.manual_seed(42)

        # Clamped by train rows: 30 rows -> 30//10+1 = 4, 20//2 = 10 → 4
        step_rows = TorchAddSVDFeaturesStep(global_transformer_name="svd")
        x_rows = torch.randn(50, 1, 20)
        result_rows = step_rows.fit_transform(
            x_rows, column_indices=list(range(20)), num_train_rows=30
        )
        # get_svd_n_components("svd", 30, 20) = min(30//10+1=4, 20//2=10) = 4
        assert result_rows.added_columns.shape[2] == 4

        # Clamped by features: 6 features -> 6//2 = 3, 80//10+1 = 9 → 3
        step_feat = TorchAddSVDFeaturesStep(global_transformer_name="svd")
        x_feat = torch.randn(100, 1, 6)
        result_feat = step_feat.fit_transform(
            x_feat, column_indices=list(range(6)), num_train_rows=80
        )
        # get_svd_n_components("svd", 80, 6) = min(80//10+1=9, 6//2=3) = 3
        assert result_feat.added_columns.shape[2] == 3

    def test__fit_transform__matches_sklearn_pipeline(self):
        """Test that SVD features match sklearn's scale + SVD pipeline."""
        torch.manual_seed(42)
        rng = np.random.default_rng(42)

        svd_name = "svd_quarter_components"
        n_rows, n_features = 100, 20
        n_train = 80
        n_components = get_svd_n_components(svd_name, n_train, n_features)

        # Create data
        x_numpy = rng.standard_normal((n_rows, n_features)).astype(np.float32)
        x_torch = torch.from_numpy(x_numpy).unsqueeze(1)  # Add batch dim

        # sklearn pipeline: fit on train, transform all
        sklearn_scaler = StandardScaler(with_mean=False)
        sklearn_svd = TruncatedSVD(n_components=n_components, algorithm="arpack")

        x_train_scaled = sklearn_scaler.fit_transform(x_numpy[:n_train])
        sklearn_svd.fit(x_train_scaled)

        x_all_scaled = sklearn_scaler.transform(x_numpy)
        svd_features_sklearn = sklearn_svd.transform(x_all_scaled)

        # torch pipeline
        step = TorchAddSVDFeaturesStep(svd_name)
        result = step.fit_transform(
            x_torch, column_indices=list(range(n_features)), num_train_rows=n_train
        )

        svd_features_torch = result.added_columns.squeeze(1).numpy()

        # Match up to per-component sign flips
        np.testing.assert_allclose(
            np.abs(svd_features_torch),
            np.abs(svd_features_sklearn),
            rtol=1e-4,
            atol=1e-4,
        )

    def test__fit_transform__subset_of_columns(self):
        """Test that step only operates on specified columns."""
        torch.manual_seed(42)
        step = TorchAddSVDFeaturesStep("svd")

        x = torch.randn(50, 1, 20)
        x_original = x.clone()

        # Only apply to first 10 columns
        column_indices = list(range(10))
        result = step.fit_transform(x, column_indices=column_indices, num_train_rows=40)

        # Original columns should be unchanged (fit_transform clones)
        assert torch.allclose(result.x, x_original)

        # SVD features should be based on only 10 columns
        # get_svd_n_components("svd", 40, 10) = min(40//10+1=5, 10//2=5) = 5
        assert result.added_columns.shape[2] == 5

    def test__fit_transform__with_batch_dimension(self):
        """Test correct handling of batch dimension."""
        torch.manual_seed(42)
        step = TorchAddSVDFeaturesStep("svd_quarter_components")

        batch_size = 4
        n_features = 20
        x = torch.randn(60, batch_size, n_features)

        result = step.fit_transform(
            x, column_indices=list(range(n_features)), num_train_rows=50
        )

        # get_svd_n_components("svd_quarter_components", 50, 20) = min(6, 5) = 5
        assert result.added_columns.shape == (60, batch_size, 5)

    def test__fit_transform__dtype_preservation(self):
        """Test that dtype is preserved through transform."""
        # get_svd_n_components("svd", 20, 10) = min(3, 5) = 3
        step = TorchAddSVDFeaturesStep("svd")

        # float32
        x32 = torch.randn(30, 1, 10, dtype=torch.float32)
        result32 = step.fit_transform(
            x32, column_indices=list(range(10)), num_train_rows=20
        )
        assert result32.added_columns.dtype == torch.float32

        # float64
        step64 = TorchAddSVDFeaturesStep("svd")
        x64 = torch.randn(30, 1, 10, dtype=torch.float64)
        result64 = step64.fit_transform(
            x64, column_indices=list(range(10)), num_train_rows=20
        )
        assert result64.added_columns.dtype == torch.float64


class TestTorchVsSklearnAddSVDFeaturesStep:
    """Equivalence between TorchAddSVDFeaturesStep and sklearn AddSVDFeaturesStep."""

    @pytest.mark.parametrize(
        ("n_rows", "n_features", "n_train", "svd_name", "inject_non_finite"),
        [
            (100, 20, 80, "svd_quarter_components", False),
            (100, 20, 80, "svd", False),
            (200, 50, 150, "svd_quarter_components", False),
            (50, 10, 40, "svd", False),
            # NaN / inf regression: the torch path must impute missing values
            # the same way the CPU pipeline does via make_scaler_safe().
            (100, 20, 80, "svd", True),
        ],
    )
    def test__torch_vs_sklearn__svd_features_match(
        self,
        n_rows: int,
        n_features: int,
        n_train: int,
        svd_name: str,
        inject_non_finite: bool,
    ) -> None:
        """Torch and sklearn SVD features match up to per-component sign flips.

        Different LAPACK backends (MKL on Linux, Accelerate on macOS) may
        produce singular vectors that differ by sign for components with close
        singular values.  We compare absolute values.

        When ``inject_non_finite`` is True the input contains NaN / inf in both
        the train and test portions.  Both paths must produce fully finite output
        (the CPU path achieves this via ``make_scaler_safe``; the torch path via
        mean-imputation inside ``TorchSafeStandardScaler``).
        """
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((n_rows, n_features)).astype(np.float32)

        if inject_non_finite:
            x_np[5, 3] = np.nan
            x_np[10, 7] = np.inf
            x_np[15, 0] = -np.inf
            x_np[n_train + 5, min(12, n_features - 1)] = np.nan  # test row
            x_np[n_train + 10, 1] = np.inf  # test row

        # --- sklearn path ---
        schema = FeatureSchema(
            features=[Feature(name=None, modality=FeatureModality.NUMERICAL)]
            * n_features
        )
        sk_step = SklearnAddSVDFeaturesStep(
            global_transformer_name=svd_name, random_state=42
        )
        sk_step._fit(x_np[:n_train], schema)
        _, sk_svd_features, _ = sk_step._transform(x_np)

        # --- torch path ---
        x_torch = torch.from_numpy(x_np).unsqueeze(1)
        torch_step = TorchAddSVDFeaturesStep(svd_name)
        result = torch_step.fit_transform(
            x_torch, column_indices=list(range(n_features)), num_train_rows=n_train
        )
        torch_svd_features = result.added_columns.squeeze(1).numpy()

        assert sk_svd_features.shape == torch_svd_features.shape

        assert np.all(np.isfinite(sk_svd_features)), (
            "sklearn SVD features should be fully finite"
        )
        assert np.all(np.isfinite(torch_svd_features)), (
            "Torch SVD features should be finite, "
            f"NaN positions: {np.argwhere(np.isnan(torch_svd_features)).tolist()}"
        )

        np.testing.assert_allclose(
            np.abs(torch_svd_features),
            np.abs(sk_svd_features),
            atol=5e-4,
            rtol=5e-4,
            err_msg=f"SVD features differ for {svd_name} with "
            f"n_rows={n_rows}, n_features={n_features}, "
            f"inject_non_finite={inject_non_finite}",
        )


class TestAddSVDFeaturesStepIntegration:
    """Integration tests for AddSVDFeaturesStep with the pipeline."""

    def test__pipeline_integration__adds_columns_correctly(self):
        """Test that SVD features are correctly added via the pipeline."""
        torch.manual_seed(42)

        # get_svd_n_components("svd_quarter_components", 50, 20) = min(6, 5) = 5
        step = TorchAddSVDFeaturesStep("svd_quarter_components")
        pipeline = TorchPreprocessingPipeline(
            steps=[(step, {FeatureModality.NUMERICAL})],
        )

        # Initial metadata: 20 numerical columns
        schema = FeatureSchema(
            features=[Feature(name=None, modality=FeatureModality.NUMERICAL)] * 20
        )

        x = torch.randn(60, 2, 20)

        result = pipeline(x, schema, num_train_rows=50)

        # Should have original 20 + 5 SVD features = 25 columns
        assert result.x.shape == (60, 2, 25)

        # Metadata should be updated
        assert result.feature_schema.num_columns == 25
        # SVD features added to NUMERICAL
        assert len(result.feature_schema.indices_for(FeatureModality.NUMERICAL)) == 25

    def test__pipeline_integration__with_multiple_modalities(self):
        """Test SVD step with mixed modalities."""
        torch.manual_seed(42)

        # get_svd_n_components("svd_quarter_components", 40, 12) = min(5, 3) = 3
        step = TorchAddSVDFeaturesStep("svd_quarter_components")
        # Only apply to numerical features
        pipeline = TorchPreprocessingPipeline(
            steps=[(step, {FeatureModality.NUMERICAL})],
        )

        # 12 numerical + 3 categorical
        schema = FeatureSchema(
            features=[Feature(name=None, modality=FeatureModality.NUMERICAL)] * 12
            + [Feature(name=None, modality=FeatureModality.CATEGORICAL)] * 3
        )

        x = torch.randn(50, 1, 15)

        result = pipeline(x, schema, num_train_rows=40)

        # Should have original 15 + 3 SVD features = 18 columns
        assert result.x.shape == (50, 1, 18)

        # Categorical columns should be unchanged
        assert result.feature_schema.indices_for(FeatureModality.CATEGORICAL) == list(
            range(12, 15)
        )

        # Numerical columns should include original 12 + 3 SVD = 15
        assert len(result.feature_schema.indices_for(FeatureModality.NUMERICAL)) == 15
