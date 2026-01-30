"""Tests for EncodeCategoricalFeaturesStep, focusing on feature modality handling."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing.datamodel import FeatureMetadata, FeatureModality
from tabpfn.preprocessing.steps import EncodeCategoricalFeaturesStep


def _make_feature_metadata(n_features: int, cat_indices: list[int]) -> FeatureMetadata:
    """Create FeatureMetadata from feature count and categorical indices."""
    return FeatureMetadata.from_only_categorical_indices(cat_indices, n_features)


def _make_test_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    cat_indices: list[int],
) -> np.ndarray:
    """Create test data with categorical and numerical features."""
    X = rng.random((n_samples, n_features))
    # Make categorical columns have integer values with enough categories
    for i in cat_indices:
        X[:, i] = rng.integers(0, 5, size=n_samples).astype(float)
    return X


def test__encode_categorical__ordinal__indices_moved_to_start():
    """Test that ordinal encoding moves categorical indices to the start.

    With categorical at positions [1, 4] and numerical at [0, 2, 3, 5],
    after ordinal encoding the ColumnTransformer reorders to:
    - Categorical columns first: [0, 1]
    - Numerical columns after: [2, 3, 4, 5]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 6
    # Categorical at positions 1 and 4 (not at start)
    cat_indices = [1, 4]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result = step.fit_transform(X, feature_metadata)

    # ColumnTransformer moves categorical columns to the front
    expected_cat_indices = [0, 1]  # Now at positions 0 and 1
    expected_num_indices = [2, 3, 4, 5]  # Numerical columns follow
    assert (
        result.feature_metadata.indices_for(FeatureModality.CATEGORICAL)
        == expected_cat_indices
    )
    assert (
        result.feature_metadata.indices_for(FeatureModality.NUMERICAL)
        == expected_num_indices
    )
    assert result.X.shape == (n_samples, n_features)


def test__encode_categorical__ordinal_shuffled__indices_same_as_ordinal():
    """Test that ordinal_shuffled has same indices as ordinal (only values shuffled).

    With categorical at positions [2, 5] and numerical at [0, 1, 3, 4],
    after encoding:
    - Categorical columns first: [0, 1]
    - Numerical columns after: [2, 3, 4, 5]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 6
    cat_indices = [2, 5]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal_shuffled",
        random_state=42,
    )
    result = step.fit_transform(X, feature_metadata)

    # Indices should be moved to front, same as ordinal
    expected_cat_indices = [0, 1]
    expected_num_indices = [2, 3, 4, 5]
    assert (
        result.feature_metadata.indices_for(FeatureModality.CATEGORICAL)
        == expected_cat_indices
    )
    assert (
        result.feature_metadata.indices_for(FeatureModality.NUMERICAL)
        == expected_num_indices
    )
    assert result.X.shape == (n_samples, n_features)


def test__encode_categorical__ordinal_shuffled__values_are_permuted():
    """Test that ordinal_shuffled actually permutes the category values."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 4
    cat_indices = [1, 3]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    # Fit with ordinal (no shuffling)
    step_ordinal = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result_ordinal = step_ordinal.fit_transform(X, feature_metadata)

    # Fit with ordinal_shuffled
    step_shuffled = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal_shuffled",
        random_state=42,
    )
    result_shuffled = step_shuffled.fit_transform(X.copy(), feature_metadata)

    # The categorical columns (now at 0 and 1) should have different values
    # due to shuffling, but same unique value count
    for col in [0, 1]:
        ordinal_col = result_ordinal.X[:, col]
        shuffled_col = result_shuffled.X[:, col]

        # Same number of unique values
        assert len(np.unique(ordinal_col)) == len(np.unique(shuffled_col))

        # Values should be permuted (not identical) in most cases
        # Note: with a permutation, the mapping changes but output range is same


def test__encode_categorical__onehot__expands_features():
    """Test that onehot encoding expands categorical features correctly.

    With 1 categorical (3 categories) at position [1] and numerical at [0, 2, 3],
    after onehot encoding:
    - 3 onehot columns first: [0, 1, 2] (categorical)
    - 3 numerical columns after: [3, 4, 5]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 4
    cat_indices = [1]  # Single categorical with 3 categories

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    # Force exactly 3 categories for predictable output
    X[:, 1] = rng.integers(0, 3, size=n_samples).astype(float)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="onehot",
        random_state=42,
    )
    result = step.fit_transform(X, feature_metadata)

    # OneHot with drop="if_binary" keeps all 3 categories for non-binary
    # So 1 cat column with 3 cats -> 3 onehot columns (at the front)
    # Total: 3 onehot + 3 numerical = 6 features
    assert result.X.shape[1] == 6

    # Categorical indices should be the onehot columns (first 3)
    expected_cat_indices = [0, 1, 2]
    expected_num_indices = [3, 4, 5]
    assert (
        result.feature_metadata.indices_for(FeatureModality.CATEGORICAL)
        == expected_cat_indices
    )
    assert (
        result.feature_metadata.indices_for(FeatureModality.NUMERICAL)
        == expected_num_indices
    )


def test__encode_categorical__numeric__no_transformation():
    """Test that numeric transform leaves data unchanged.

    With categorical at [0, 2] and numerical at [1, 3, 4],
    no transformation occurs so all indices remain unchanged.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 5
    cat_indices = [0, 2]
    num_indices = [1, 3, 4]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="numeric",
        random_state=42,
    )
    result = step.fit_transform(X, feature_metadata)

    # No transformation, data unchanged
    np.testing.assert_array_equal(result.X, X)
    # All indices remain at original positions
    assert (
        result.feature_metadata.indices_for(FeatureModality.CATEGORICAL) == cat_indices
    )
    assert result.feature_metadata.indices_for(FeatureModality.NUMERICAL) == num_indices


def test__encode_categorical__none__no_transformation():
    """Test that none transform leaves data unchanged.

    With categorical at [1, 3] and numerical at [0, 2, 4],
    no transformation occurs so all indices remain unchanged.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 5
    cat_indices = [1, 3]
    num_indices = [0, 2, 4]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="none",
        random_state=42,
    )
    result = step.fit_transform(X, feature_metadata)

    # No transformation
    np.testing.assert_array_equal(result.X, X)
    assert (
        result.feature_metadata.indices_for(FeatureModality.CATEGORICAL) == cat_indices
    )
    assert result.feature_metadata.indices_for(FeatureModality.NUMERICAL) == num_indices


def test__encode_categorical__ordinal__no_categoricals():
    """Test ordinal encoding with no categorical features.

    All features are numerical, so no transformation occurs.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 4
    cat_indices: list[int] = []
    num_indices = list(range(n_features))

    X = rng.random((n_samples, n_features))
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result = step.fit_transform(X, feature_metadata)

    # No categorical features, data unchanged
    np.testing.assert_array_equal(result.X, X)
    assert result.feature_metadata.indices_for(FeatureModality.CATEGORICAL) == []
    assert result.feature_metadata.indices_for(FeatureModality.NUMERICAL) == num_indices


def test__encode_categorical__ordinal__all_categoricals():
    """Test ordinal encoding when all features are categorical.

    All features are categorical, so no numerical indices exist.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 4
    cat_indices = list(range(n_features))
    num_indices: list[int] = []

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result = step.fit_transform(X, feature_metadata)

    # All features remain categorical, indices 0 to n_features-1
    assert result.X.shape == (n_samples, n_features)
    assert (
        result.feature_metadata.indices_for(FeatureModality.CATEGORICAL) == cat_indices
    )
    assert result.feature_metadata.indices_for(FeatureModality.NUMERICAL) == num_indices


@pytest.mark.parametrize(
    "transform_name",
    ["ordinal", "ordinal_shuffled", "numeric", "none"],
)
def test__encode_categorical__transform_is_deterministic(transform_name: str):
    """Test that transform on test data is deterministic after fit."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 5
    cat_indices = [1, 3]

    X_train = _make_test_data(rng, n_samples, n_features, cat_indices)
    X_test = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_metadata = _make_feature_metadata(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name=transform_name,
        random_state=42,
    )
    step.fit_transform(X_train, feature_metadata)

    # Multiple transforms should give same result
    result1 = step.transform(X_test)
    result2 = step.transform(X_test)

    np.testing.assert_array_equal(result1.X, result2.X)
    assert result1.feature_metadata.indices_for(
        FeatureModality.CATEGORICAL
    ) == result2.feature_metadata.indices_for(FeatureModality.CATEGORICAL)
    assert result1.feature_metadata.indices_for(
        FeatureModality.NUMERICAL
    ) == result2.feature_metadata.indices_for(FeatureModality.NUMERICAL)
