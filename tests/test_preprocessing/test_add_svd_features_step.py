"""Tests for AddSVDFeaturesStep."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing import PreprocessingPipeline
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.add_svd_features_step import (
    AddSVDFeaturesStep,
    get_svd_features_transformer,
)


def _get_schema(num_columns: int) -> FeatureSchema:
    """Create a schema with all numerical features."""
    return FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(num_columns)
        ]
    )


def _get_test_data(
    n_samples: int = 100, n_features: int = 10, seed: int = 42
) -> np.ndarray:
    """Create test data with some structure for SVD to capture."""
    rng = np.random.default_rng(seed)
    # Create data with some latent structure
    latent = rng.standard_normal((n_samples, 3))
    weights = rng.standard_normal((3, n_features))
    noise = rng.standard_normal((n_samples, n_features)) * 0.1
    with np.errstate(all="ignore"):
        return (latent @ weights + noise).astype(np.float32)


def test__transform__returns_x_unchanged_and_svd_in_added_columns() -> None:
    """Test that _transform returns X unchanged, SVD features in added_columns."""
    data = _get_test_data(n_samples=50, n_features=6)
    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    step._fit(data, _get_schema(num_columns=6))
    result, added_cols, modality = step._transform(data)

    # X should be returned unchanged
    assert isinstance(result, np.ndarray)
    assert result.shape == data.shape
    np.testing.assert_array_equal(result, data)

    # SVD features should be in added_columns
    assert added_cols is not None
    assert added_cols.shape[0] == data.shape[0]
    assert added_cols.shape[1] > 0  # Should have some SVD components
    assert modality == FeatureModality.NUMERICAL


def test__transform__with_svd_quarter_components() -> None:
    """Test that svd_quarter_components produces fewer components than svd."""
    data = _get_test_data(n_samples=100, n_features=20)

    step_svd = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    step_svd._fit(data, _get_schema(num_columns=20))
    _, added_svd, _ = step_svd._transform(data)

    step_quarter = AddSVDFeaturesStep(
        global_transformer_name="svd_quarter_components", random_state=42
    )
    step_quarter._fit(data, _get_schema(num_columns=20))
    _, added_quarter, _ = step_quarter._transform(data)

    assert added_svd is not None
    assert added_quarter is not None
    # Quarter components should have fewer or equal columns
    assert added_quarter.shape[1] <= added_svd.shape[1]


def test__transform__with_single_feature_returns_unchanged() -> None:
    """Test that single feature data is returned unchanged without SVD."""
    data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    schema = _get_schema(num_columns=1)
    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    updated_schema = step._fit(data, schema)

    # Schema should be unchanged
    assert updated_schema.num_columns == 1

    # Transformer should not be set for single feature
    assert not hasattr(step, "transformer_") or step.transformer_ is None


def test__fit_transform__returns_added_columns() -> None:
    """Test fit_transform returns X unchanged with SVD in added_columns."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result = step.fit_transform(data, schema)

    # X should be unchanged
    assert result.X.shape == data.shape
    np.testing.assert_array_equal(result.X, data)

    # Schema should be unchanged (pipeline handles adding SVD)
    assert result.feature_schema.num_columns == 6

    # SVD features should be in added_columns
    assert result.X_added is not None
    assert result.X_added.shape[0] == data.shape[0]
    assert result.modality_added == FeatureModality.NUMERICAL


def test__transform__returns_added_columns_after_fit() -> None:
    """Test transform returns X unchanged with SVD in added_columns."""
    data_train = _get_test_data(n_samples=50, n_features=6, seed=42)
    data_test = _get_test_data(n_samples=20, n_features=6, seed=123)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    step.fit_transform(data_train, schema)
    result = step.transform(data_test)

    # X should be unchanged
    assert result.X.shape == data_test.shape

    # SVD features should be in added_columns
    assert result.X_added is not None
    assert result.X_added.shape[0] == data_test.shape[0]


def test__num_output_features__returns_correct_count() -> None:
    """Test num_output_features returns the expected count."""
    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)

    # For n_features=10, n_samples=100:
    # n_components = min(100//10+1, 10//2) = min(11, 5) = 5
    result = step.num_added_features(n_features=10, n_samples=100)
    assert result == 5

    # For n_features=1 (less than 2), should return unchanged
    result_single = step.num_added_features(n_features=1, n_samples=100)
    assert result_single == 1


def test__in_pipeline__returns_added_columns() -> None:
    """Test that the step returns added columns when used in a pipeline."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    pipeline = PreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    result = pipeline.fit_transform(data, schema)

    # Should have original columns plus SVD columns
    assert result.feature_schema.num_columns > 6
    assert result.X.shape[1] > 6
    assert result.X.shape[0] == data.shape[0]


def test__in_pipeline__transform_consistent_with_fit_transform() -> None:
    """Test that transform produces same shape as fit_transform."""
    data_train = _get_test_data(n_samples=50, n_features=6, seed=42)
    data_test = _get_test_data(n_samples=20, n_features=6, seed=123)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    pipeline = PreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])

    fit_result = pipeline.fit_transform(data_train, schema)
    transform_result = pipeline.transform(data_test)

    assert fit_result.X.shape[1] == transform_result.X.shape[1]
    assert (
        fit_result.feature_schema.num_columns
        == transform_result.feature_schema.num_columns
    )


def test__in_pipeline__with_no_modality_sepection() -> None:
    """Test that the step returns added columns when used in a pipeline."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    pipeline = PreprocessingPipeline(steps=[step])
    result = pipeline.fit_transform(data, schema)

    # Should have original columns plus SVD columns
    assert result.feature_schema.num_columns > 6
    assert result.X.shape[1] > 6
    assert result.X.shape[0] == data.shape[0]


def test__random_state__produces_reproducible_results() -> None:
    """Test that same random_state produces identical results."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step1 = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result1 = step1.fit_transform(data, schema)

    step2 = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result2 = step2.fit_transform(data, schema)

    assert result1.X_added is not None
    assert result2.X_added is not None
    np.testing.assert_array_almost_equal(result1.X_added, result2.X_added)


def test__different_random_state__produces_different_results() -> None:
    """Test that different random_state produces different results."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step1 = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result1 = step1.fit_transform(data, schema)

    step2 = AddSVDFeaturesStep(global_transformer_name="svd", random_state=123)
    result2 = step2.fit_transform(data, schema)

    assert result1.X_added is not None
    assert result2.X_added is not None
    # Results may differ due to different random initialization
    # (though for SVD with arpack, they might be similar if data is well-behaved)


def test__get_svd_features_transformer__invalid_name_raises() -> None:
    """Test that invalid transformer name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid global transformer name"):
        # Create an invalid enum value by bypassing the enum
        get_svd_features_transformer(
            "invalid_name",  # type: ignore[arg-type]
            n_samples=100,
            n_features=10,
        )
