"""Tests for NanHandlingPolynomialFeaturesStep."""

from __future__ import annotations

import numpy as np

from tabpfn.preprocessing import PreprocessingPipeline
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.nan_handling_polynomial_features_step import (
    NanHandlingPolynomialFeaturesStep,
)


def _get_schema(num_columns: int) -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(num_columns)
        ]
    )


def test__fit_transform__creates_polynomial_features() -> None:
    """Test that fit_transform creates polynomial features."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    schema = _get_schema(num_columns=3)

    step = NanHandlingPolynomialFeaturesStep(random_state=42)
    result = step.fit_transform(data, schema)

    # X should be standardized but same shape
    assert result.X.shape == (2, 3)

    # Polynomial features should be in added_columns
    assert result.X_added is not None
    assert result.X_added.shape[0] == 2  # Same number of rows
    assert result.X_added.shape[1] > 0  # Some polynomial features created
    assert result.modality_added == FeatureModality.NUMERICAL


def test__fit_transform__max_features_limits_polynomials() -> None:
    """Test that max_features limits the number of polynomial features."""
    data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    schema = _get_schema(num_columns=4)

    max_features = 3
    step = NanHandlingPolynomialFeaturesStep(max_features=max_features, random_state=42)
    result = step.fit_transform(data, schema)

    assert result.X_added is not None
    assert result.X_added.shape == (2, max_features)


def test__fit_transform__empty_data_returns_empty() -> None:
    """Test that empty input data is handled gracefully."""
    # Empty rows
    data = np.array([], dtype=np.float32).reshape(0, 3)
    schema = _get_schema(num_columns=3)

    step = NanHandlingPolynomialFeaturesStep(random_state=42)
    result = step.fit_transform(data, schema)

    assert result.X.shape == (0, 3)
    assert result.X_added is None


def test__fit_transform__single_column() -> None:
    """Test that single column input creates polynomial (squared) features."""
    data = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    schema = _get_schema(num_columns=1)

    step = NanHandlingPolynomialFeaturesStep(random_state=42)
    result = step.fit_transform(data, schema)

    # With 1 column, only 1 polynomial possible (x^2)
    assert result.X_added is not None
    assert result.X_added.shape == (3, 1)


def test__transform__is_deterministic_with_random_state() -> None:
    """Test that same random_state produces same results."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    schema = _get_schema(num_columns=2)

    step1 = NanHandlingPolynomialFeaturesStep(random_state=42)
    result1 = step1.fit_transform(data, schema)

    step2 = NanHandlingPolynomialFeaturesStep(random_state=42)
    result2 = step2.fit_transform(data, schema)

    assert result1.X_added is not None
    assert result2.X_added is not None
    np.testing.assert_array_equal(result1.X_added, result2.X_added)


def test__in_pipeline__returns_correct_schema() -> None:
    """Test that the schema is returned correctly."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    schema = _get_schema(num_columns=2)

    step = NanHandlingPolynomialFeaturesStep(random_state=42, max_features=5)
    pipeline = PreprocessingPipeline(steps=[step])
    result = pipeline.fit_transform(data, schema)

    assert result.feature_schema.num_columns == 5
    expected = [0, 1, 2, 3, 4]
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == expected
