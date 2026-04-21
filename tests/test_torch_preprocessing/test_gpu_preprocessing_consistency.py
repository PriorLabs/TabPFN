"""Tests verifying CPU-only vs CPU+GPU preprocessing pipeline consistency.

When ``enable_gpu_preprocessing=True``, the quantile transform, SVD, and shuffle
move from the CPU pipeline to the GPU (torch) pipeline.  These tests verify
that the *combined* output of both paths is numerically identical (within
floating-point tolerance) so the change is behaviour-preserving.
"""

from __future__ import annotations

import sys
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
import torch

from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig
from tabpfn.preprocessing.datamodel import (
    Feature,
    FeatureModality,
    FeatureSchema,
)
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline
from tabpfn.preprocessing.torch.factory import create_gpu_preprocessing_pipeline
from tabpfn.preprocessing.torch.gpu_preprocessing_metadata import (
    compute_effective_n_quantiles,
    is_gpu_quantile_eligible,
)
from tabpfn.utils import infer_random_state

# Type alias for the (X, schema) tuple returned by fixtures and helpers.
_DataWithSchema = tuple[npt.NDArray[np.float64], FeatureSchema]
_ResultWithSchema = tuple[npt.NDArray[np.floating], FeatureSchema]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(n_features: int, n_cat: int) -> FeatureSchema:
    """Build a feature schema with ``n_cat`` CATEGORICAL columns first."""
    features = [Feature(name=None, modality=FeatureModality.CATEGORICAL)] * n_cat + [
        Feature(name=None, modality=FeatureModality.NUMERICAL)
    ] * (n_features - n_cat)
    return FeatureSchema(features=features)


def _make_config(
    pconfig: PreprocessorConfig,
    *,
    fingerprint: bool = True,
    feature_shift_decoder: Literal["shuffle", "rotate"] | None = "shuffle",
    feature_shift_count: int = 3,
    outlier_removal_std: float | None = 12.0,
) -> ClassifierEnsembleConfig:
    return ClassifierEnsembleConfig(
        preprocess_config=pconfig,
        feature_shift_count=feature_shift_count,
        class_permutation=None,
        add_fingerprint_feature=fingerprint,
        polynomial_features="no",
        feature_shift_decoder=feature_shift_decoder,
        subsample_ix=None,
        outlier_removal_std=outlier_removal_std,
        _model_index=0,
    )


def _torch_dtype_to_np_dtype(torch_dtype: torch.dtype) -> np.dtype:
    if torch_dtype == torch.float16:
        return np.float16
    if torch_dtype == torch.float32:
        return np.float32
    if torch_dtype == torch.float64:
        return np.float64
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def _run_cpu_only(
    X: npt.NDArray[np.float64],
    schema: FeatureSchema,
    config: ClassifierEnsembleConfig,
    seed: int,
    torch_dtype: torch.dtype = torch.float32,
) -> _ResultWithSchema:
    """Run the full CPU-only pipeline (+ GPU outlier if applicable)."""
    np_dtype = _torch_dtype_to_np_dtype(torch_dtype)
    static_seed, _ = infer_random_state(seed)
    cpu_pipe = create_preprocessing_pipeline(
        config,
        random_state=static_seed,
        enable_gpu_preprocessing=False,
    )
    cpu_res = cpu_pipe.fit_transform(X.copy(), schema)

    gpu_pipe = create_gpu_preprocessing_pipeline(
        config,
        keep_fitted_cache=False,
        enable_gpu_preprocessing=False,
    )
    if gpu_pipe is not None:
        t = torch.from_numpy(cpu_res.X.astype(np_dtype)).unsqueeze(1)
        gpu_out = gpu_pipe(t, cpu_res.feature_schema)
        return gpu_out.x.squeeze(1).numpy(), gpu_out.feature_schema
    return cpu_res.X.astype(np_dtype), cpu_res.feature_schema


def _run_cpu_plus_gpu(
    X: npt.NDArray[np.float64],
    schema: FeatureSchema,
    config: ClassifierEnsembleConfig,
    seed: int,
    torch_dtype: torch.dtype = torch.float32,
) -> _ResultWithSchema:
    """Run the CPU pipeline with GPU offload for quantile/SVD/shuffle."""
    np_dtype = _torch_dtype_to_np_dtype(torch_dtype)
    static_seed, _ = infer_random_state(seed)
    cpu_pipe = create_preprocessing_pipeline(
        config,
        random_state=static_seed,
        enable_gpu_preprocessing=True,
    )
    cpu_res = cpu_pipe.fit_transform(X.copy(), schema)

    gpu_pipe = create_gpu_preprocessing_pipeline(
        config,
        keep_fitted_cache=False,
        enable_gpu_preprocessing=True,
        feature_schema=cpu_res.feature_schema,
        n_train_samples=cpu_res.X.shape[0],
        random_state=static_seed,
    )
    if gpu_pipe is not None:
        t = torch.from_numpy(cpu_res.X.astype(np_dtype)).unsqueeze(1)
        gpu_out = gpu_pipe(t, cpu_res.feature_schema)
        return gpu_out.x.squeeze(1).numpy(), gpu_out.feature_schema
    return cpu_res.X.astype(np_dtype), cpu_res.feature_schema


def _assert_preprocessing_match(
    X_cpu: npt.NDArray[np.floating],
    schema_cpu: FeatureSchema,
    X_gpu: npt.NDArray[np.floating],
    schema_gpu: FeatureSchema,
    *,
    has_fingerprint_on_gpu: bool = False,
) -> None:
    """Assert preprocessing outputs match.

    All columns must be identical except for the fingerprint column when it
    runs on GPU.  The fingerprint is a SHA-256 hash of row data and is
    extremely sensitive to precision: the CPU path hashes float64 values
    while the GPU path hashes float32 values, producing completely different
    hashes.  At most one column (the fingerprint) is allowed to differ.
    """
    assert X_cpu.shape == X_gpu.shape, f"Shape mismatch: {X_cpu.shape} vs {X_gpu.shape}"

    # Schema: categorical indices must match
    assert schema_cpu.indices_for(FeatureModality.CATEGORICAL) == (
        schema_gpu.indices_for(FeatureModality.CATEGORICAL)
    )

    # Tolerance for non-fingerprint columns.  Most columns match at float32
    # epsilon (~1e-7).  SVD features may differ slightly (up to ~6e-4 for
    # large feature counts) because sklearn uses an iterative algorithm
    # (arpack) while torch uses full SVD.
    dtype = X_cpu.dtype
    atol = 1e-2 if dtype == np.float16 else 5e-3

    if not has_fingerprint_on_gpu:
        np.testing.assert_allclose(X_cpu, X_gpu, atol=atol, rtol=atol)
        return

    # Identify columns with large differences (> atol).  Only the fingerprint
    # column is allowed to have a large difference (> 0.1).
    col_max_diff: npt.NDArray[np.floating] = np.max(np.abs(X_cpu - X_gpu), axis=0)
    large_diff_cols = np.where(col_max_diff > 0.1)[0]

    assert len(large_diff_cols) <= 1, (
        f"Expected at most 1 column with large diff (fingerprint), "
        f"got {len(large_diff_cols)}: cols {large_diff_cols.tolist()}. "
        f"Max diffs per col: {col_max_diff}"
    )

    # All non-fingerprint columns must match within tolerance
    non_fingerprint = np.where(col_max_diff <= 0.1)[0]
    if len(non_fingerprint) > 0:
        np.testing.assert_allclose(
            X_cpu[:, non_fingerprint],
            X_gpu[:, non_fingerprint],
            atol=atol,
            rtol=atol,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> _DataWithSchema:
    """200 rows, 20 features (3 categorical)."""
    rng = np.random.default_rng(42)
    n, f, n_cat = 200, 20, 3
    X = rng.standard_normal((n, f)).astype(np.float64)
    X[:, :n_cat] = rng.integers(0, 5, (n, n_cat)).astype(np.float64)
    schema = _make_schema(f, n_cat)
    return X, schema


@pytest.fixture
def large_feature_data() -> _DataWithSchema:
    """200 rows, 600 features (3 cat) - triggers feature subsampling."""
    rng = np.random.default_rng(42)
    n, f, n_cat = 200, 600, 3
    X = rng.standard_normal((n, f)).astype(np.float64)
    X[:, :n_cat] = rng.integers(0, 5, (n, n_cat)).astype(np.float64)
    schema = _make_schema(f, n_cat)
    return X, schema


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGpuQuantileEligibility:
    def test_eligible(self) -> None:
        assert is_gpu_quantile_eligible("quantile_uni")
        assert is_gpu_quantile_eligible("quantile_uni_coarse")
        assert is_gpu_quantile_eligible("quantile_uni_fine")

    def test_not_eligible(self) -> None:
        assert not is_gpu_quantile_eligible("squashing_scaler_default")
        assert not is_gpu_quantile_eligible("none")
        assert not is_gpu_quantile_eligible("power")
        assert not is_gpu_quantile_eligible("quantile_norm")

    def test_effective_n_quantiles(self) -> None:
        assert compute_effective_n_quantiles("quantile_uni", 200) == 40
        assert compute_effective_n_quantiles("quantile_uni_coarse", 200) == 20


# torch.linalg.svd delegates to platform-specific LAPACK backends (MKL on
# Linux, Accelerate on macOS, OpenBLAS on some Windows builds) that can
# produce different singular vectors for near-degenerate singular values.
# This makes the GPU SVD output non-deterministic across platforms.  On
# macOS the CPU-only (sklearn ARPACK) and GPU (torch full-SVD via
# Accelerate) paths happen to agree, so we test SVD configs there.  On
# other platforms we use configs without SVD to keep the consistency tests
# deterministic.
_IS_MACOS = sys.platform == "darwin"


class TestPipelineConsistency:
    """Compare full CPU-only vs CPU+GPU pipelines."""

    # v2.6 classifier configs (quantile_uni, GPU eligible)
    V26_QUANTILE_NUMERIC = PreprocessorConfig(
        "quantile_uni",
        categorical_name="numeric",
        max_features_per_estimator=680,
    )
    V26_QUANTILE_NUMERIC_APPEND_ORIGINAL = PreprocessorConfig(
        "quantile_uni",
        categorical_name="numeric",
        append_original=True,
        max_features_per_estimator=680,
    )
    # SVD configs — only deterministic on macOS (see note above).
    V26_QUANTILE_SVD = PreprocessorConfig(
        "quantile_uni",
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd_quarter_components",
        max_features_per_estimator=500,
    )
    V26_QUANTILE_ONEHOT = PreprocessorConfig(
        "quantile_uni",
        categorical_name="onehot",
        max_features_per_estimator=680,
    )
    # v2.5 classifier configs (non-quantile)
    V25_SQUASHING_SVD = PreprocessorConfig(
        name="squashing_scaler_default",
        append_original=False,
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd_quarter_components",
        max_features_per_estimator=500,
    )
    V25_SQUASHING_NO_SVD = PreprocessorConfig(
        name="squashing_scaler_default",
        append_original=False,
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name=None,
        max_features_per_estimator=500,
    )
    V25_NONE = PreprocessorConfig(
        name="none",
        categorical_name="numeric",
        max_features_per_estimator=500,
    )

    @pytest.mark.parametrize(
        "torch_dtype",
        [torch.float16, torch.float32, torch.float64],
        ids=["f16", "f32", "f64"],
    )
    @pytest.mark.parametrize(
        "pconfig",
        [
            V26_QUANTILE_NUMERIC,
            V26_QUANTILE_NUMERIC_APPEND_ORIGINAL,
            V26_QUANTILE_ONEHOT,
            pytest.param(
                V26_QUANTILE_SVD,
                marks=pytest.mark.skipif(
                    not _IS_MACOS,
                    reason="torch SVD is non-deterministic across LAPACK backends",
                ),
            ),
        ],
        ids=[
            "v26_quantile_numeric",
            "v26_quantile_numeric_append_original",
            "v26_quantile_onehot",
            "v26_quantile_svd",
        ],
    )
    def test_quantile_configs_match(
        self,
        sample_data: _DataWithSchema,
        pconfig: PreprocessorConfig,
        torch_dtype: torch.dtype,
    ) -> None:
        X, schema = sample_data
        config = _make_config(pconfig)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(
            X, schema, config, seed, torch_dtype=torch_dtype
        )
        X_gpu, schema_gpu = _run_cpu_plus_gpu(
            X, schema, config, seed, torch_dtype=torch_dtype
        )

        # Fingerprint hashes differ when quantile is on GPU because the torch
        # and sklearn quantile transforms produce slightly different boundary
        # values, which changes the SHA-256 hash input.
        quantile_on_gpu = is_gpu_quantile_eligible(pconfig.name)
        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=quantile_on_gpu and config.add_fingerprint_feature,
        )

    @pytest.mark.parametrize(
        "torch_dtype",
        [torch.float16, torch.float32, torch.float64],
        ids=["f16", "f32", "f64"],
    )
    @pytest.mark.parametrize(
        "pconfig",
        [
            pytest.param(
                V25_SQUASHING_SVD,
                marks=pytest.mark.skipif(
                    not _IS_MACOS,
                    reason="torch SVD is non-deterministic across LAPACK backends",
                ),
            ),
            V25_SQUASHING_NO_SVD,
            V25_NONE,
        ],
        ids=["v25_squashing_svd", "v25_squashing_no_svd", "v25_none"],
    )
    def test_non_quantile_configs_svd_shuffle_on_gpu(
        self,
        sample_data: _DataWithSchema,
        pconfig: PreprocessorConfig,
        torch_dtype: torch.dtype,
    ) -> None:
        """For non-quantile transforms, SVD and shuffle still move to GPU."""
        X, schema = sample_data
        config = _make_config(pconfig)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(
            X, schema, config, seed, torch_dtype=torch_dtype
        )
        X_gpu, schema_gpu = _run_cpu_plus_gpu(
            X, schema, config, seed, torch_dtype=torch_dtype
        )

        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=config.add_fingerprint_feature,
        )

    def test_no_fingerprint(self, sample_data: _DataWithSchema) -> None:
        """Without fingerprint, all columns should match exactly."""
        X, schema = sample_data
        pconfig = self.V26_QUANTILE_NUMERIC
        config = _make_config(pconfig, fingerprint=False)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(X, schema, config, seed)
        X_gpu, schema_gpu = _run_cpu_plus_gpu(X, schema, config, seed)
        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=False,
        )

    @pytest.mark.skipif(
        not _IS_MACOS, reason="torch SVD non-deterministic across LAPACK backends"
    )
    def test_no_fingerprint_with_svd(self, sample_data: _DataWithSchema) -> None:
        """Without fingerprint, SVD columns should still match on macOS."""
        X, schema = sample_data
        pconfig = self.V26_QUANTILE_SVD
        config = _make_config(pconfig, fingerprint=False)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(X, schema, config, seed)
        X_gpu, schema_gpu = _run_cpu_plus_gpu(X, schema, config, seed)
        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=False,
        )

    def test_rotate_shuffle(self, sample_data: _DataWithSchema) -> None:
        X, schema = sample_data
        config = _make_config(
            self.V26_QUANTILE_NUMERIC,
            feature_shift_decoder="rotate",
            feature_shift_count=5,
        )
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(X, schema, config, seed)
        X_gpu, schema_gpu = _run_cpu_plus_gpu(X, schema, config, seed)
        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=config.add_fingerprint_feature,
        )

    def test_no_outlier_removal(self, sample_data: _DataWithSchema) -> None:
        X, schema = sample_data
        config = _make_config(self.V26_QUANTILE_NUMERIC, outlier_removal_std=None)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(X, schema, config, seed)
        X_gpu, schema_gpu = _run_cpu_plus_gpu(X, schema, config, seed)
        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=config.add_fingerprint_feature,
        )

    @pytest.mark.skipif(
        not _IS_MACOS, reason="torch SVD non-deterministic across LAPACK backends"
    )
    def test_append_to_original_true_with_svd(
        self, sample_data: _DataWithSchema
    ) -> None:
        """Test append_to_original=True with GPU quantile + SVD (macOS only)."""
        X, schema = sample_data
        pconfig = PreprocessorConfig(
            "quantile_uni",
            append_original=True,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            max_features_per_estimator=500,
        )
        config = _make_config(pconfig)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(X, schema, config, seed)
        X_gpu, schema_gpu = _run_cpu_plus_gpu(X, schema, config, seed)

        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=config.add_fingerprint_feature,
        )

    def test_append_to_original_true_no_svd(self, sample_data: _DataWithSchema) -> None:
        """Test append_to_original=True with GPU quantile, no SVD."""
        X, schema = sample_data
        pconfig = PreprocessorConfig(
            "quantile_uni",
            append_original=True,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name=None,
            max_features_per_estimator=500,
        )
        config = _make_config(pconfig)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(X, schema, config, seed)
        X_gpu, schema_gpu = _run_cpu_plus_gpu(X, schema, config, seed)

        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=config.add_fingerprint_feature,
        )

    @pytest.mark.skipif(
        not _IS_MACOS, reason="torch SVD non-deterministic across LAPACK backends"
    )
    def test_feature_subsampling(self, large_feature_data: _DataWithSchema) -> None:
        """Verify consistency if feature subsampling kicks in (macOS only, uses SVD)."""
        X, schema = large_feature_data
        pconfig = self.V26_QUANTILE_SVD
        config = _make_config(pconfig)
        seed = 42

        X_cpu, schema_cpu = _run_cpu_only(X, schema, config, seed)
        X_gpu, schema_gpu = _run_cpu_plus_gpu(X, schema, config, seed)
        _assert_preprocessing_match(
            X_cpu,
            schema_cpu,
            X_gpu,
            schema_gpu,
            has_fingerprint_on_gpu=config.add_fingerprint_feature,
        )


class TestTestDataConsistency:
    """Verify that test-time transform also matches between paths."""

    @pytest.mark.skipif(
        not _IS_MACOS, reason="torch SVD non-deterministic across LAPACK backends"
    )
    def test_transform_X_test(self, sample_data: _DataWithSchema) -> None:
        """Test data transform with SVD (macOS only)."""
        X, schema = sample_data
        pconfig = PreprocessorConfig(
            "quantile_uni",
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            max_features_per_estimator=500,
        )
        config = _make_config(pconfig)
        seed = 42

        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((50, X.shape[1])).astype(np.float64)
        X_test[:, :3] = rng.integers(0, 5, (50, 3)).astype(np.float64)

        # --- CPU-only path ---
        static_seed, _ = infer_random_state(seed)
        cpu_pipe = create_preprocessing_pipeline(
            config,
            random_state=static_seed,
            enable_gpu_preprocessing=False,
        )
        cpu_fit_res = cpu_pipe.fit_transform(X.copy(), schema)
        X_test_cpu = cpu_pipe.transform(X_test.copy()).X.astype(np.float32)

        # Apply GPU (outlier removal only)
        gpu_pipe_cpu = create_gpu_preprocessing_pipeline(
            config,
            enable_gpu_preprocessing=False,
        )
        cpu_schema = cpu_fit_res.feature_schema
        if gpu_pipe_cpu is not None:
            t = torch.from_numpy(X_test_cpu).unsqueeze(1)
            result = gpu_pipe_cpu(t, cpu_schema)
            X_test_cpu = result.x.squeeze(1).numpy()
            cpu_schema = result.feature_schema

        # --- CPU+GPU path ---
        static_seed, _ = infer_random_state(seed)
        cpu_pipe_gpu = create_preprocessing_pipeline(
            config,
            random_state=static_seed,
            enable_gpu_preprocessing=True,
        )
        cpu_fit_gpu = cpu_pipe_gpu.fit_transform(X.copy(), schema)
        X_test_gpu_cpu = cpu_pipe_gpu.transform(X_test.copy()).X.astype(np.float32)

        gpu_pipe = create_gpu_preprocessing_pipeline(
            config,
            enable_gpu_preprocessing=True,
            feature_schema=cpu_fit_gpu.feature_schema,
            n_train_samples=cpu_fit_gpu.X.shape[0],
            random_state=static_seed,
        )

        # Combine train+test so the GPU pipeline fits on train and transforms
        # test using num_train_rows
        X_combined = np.concatenate([cpu_fit_gpu.X, X_test_gpu_cpu], axis=0)
        t_combined = torch.from_numpy(X_combined.astype(np.float32)).unsqueeze(1)
        gpu_out = gpu_pipe(
            t_combined,
            cpu_fit_gpu.feature_schema,
            num_train_rows=cpu_fit_gpu.X.shape[0],
        )
        X_test_final = gpu_out.x.squeeze(1).numpy()[cpu_fit_gpu.X.shape[0] :]

        _assert_preprocessing_match(
            X_test_cpu,
            cpu_schema,
            X_test_final,
            gpu_out.feature_schema,
            has_fingerprint_on_gpu=True,
        )
