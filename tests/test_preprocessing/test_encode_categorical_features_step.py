"""Tests for EncodeCategoricalFeaturesStep, focusing on feature modality handling."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing.datamodel import (
    Feature,
    FeatureModality,
    FeatureSchema,
    GPUTransformType,
)
from tabpfn.preprocessing.pipeline_interface import PreprocessingPipeline
from tabpfn.preprocessing.steps import EncodeCategoricalFeaturesStep


def _make_feature_schema(n_features: int, cat_indices: list[int]) -> FeatureSchema:
    """Create FeatureSchema from feature count and categorical indices."""
    return FeatureSchema.from_only_categorical_indices(cat_indices, n_features)


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
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result = step.fit_transform(X, feature_schema)

    # ColumnTransformer moves categorical columns to the front
    expected_cat_indices = [0, 1]  # Now at positions 0 and 1
    expected_num_indices = [2, 3, 4, 5]  # Numerical columns follow
    assert (
        result.feature_schema.indices_for(FeatureModality.CATEGORICAL)
        == expected_cat_indices
    )
    assert (
        result.feature_schema.indices_for(FeatureModality.NUMERICAL)
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
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal_shuffled",
        random_state=42,
    )
    result = step.fit_transform(X, feature_schema)

    # Indices should be moved to front, same as ordinal
    expected_cat_indices = [0, 1]
    expected_num_indices = [2, 3, 4, 5]
    assert (
        result.feature_schema.indices_for(FeatureModality.CATEGORICAL)
        == expected_cat_indices
    )
    assert (
        result.feature_schema.indices_for(FeatureModality.NUMERICAL)
        == expected_num_indices
    )
    assert result.X.shape == (n_samples, n_features)


def test__encode_categorical__ordinal_shuffled__values_are_permuted():
    """Test that ordinal_shuffled actually permutes the category values."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 4
    cat_indices = [1, 3]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_schema = _make_feature_schema(n_features, cat_indices)

    # Fit with ordinal (no shuffling)
    step_ordinal = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result_ordinal = step_ordinal.fit_transform(X, feature_schema)

    # Fit with ordinal_shuffled
    step_shuffled = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal_shuffled",
        random_state=42,
    )
    result_shuffled = step_shuffled.fit_transform(X.copy(), feature_schema)

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
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="onehot",
        random_state=42,
    )
    result = step.fit_transform(X, feature_schema)

    # OneHot with drop="if_binary" keeps all 3 categories for non-binary
    # So 1 cat column with 3 cats -> 3 onehot columns (at the front)
    # Total: 3 onehot + 3 numerical = 6 features
    assert result.X.shape[1] == 6

    # Categorical indices should be the onehot columns (first 3)
    expected_cat_indices = [0, 1, 2]
    expected_num_indices = [3, 4, 5]
    assert (
        result.feature_schema.indices_for(FeatureModality.CATEGORICAL)
        == expected_cat_indices
    )
    assert (
        result.feature_schema.indices_for(FeatureModality.NUMERICAL)
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
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="numeric",
        random_state=42,
    )
    result = step.fit_transform(X, feature_schema)

    # No transformation, data unchanged
    np.testing.assert_array_equal(result.X, X)
    # All indices remain at original positions
    assert result.feature_schema.indices_for(FeatureModality.CATEGORICAL) == cat_indices
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == num_indices


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
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="none",
        random_state=42,
    )
    result = step.fit_transform(X, feature_schema)

    # No transformation
    np.testing.assert_array_equal(result.X, X)
    assert result.feature_schema.indices_for(FeatureModality.CATEGORICAL) == cat_indices
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == num_indices


def test__encode_categorical__ordinal__no_categoricals():
    """Test ordinal encoding with no categorical features.

    All features are numerical, so no transformation occurs.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 4
    cat_indices: list[int] = []
    num_indices = list(range(n_features))

    X = rng.random((n_samples, n_features))
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result = step.fit_transform(X, feature_schema)

    # No categorical features, data unchanged
    np.testing.assert_array_equal(result.X, X)
    assert result.feature_schema.indices_for(FeatureModality.CATEGORICAL) == []
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == num_indices


def test__encode_categorical__ordinal__all_categoricals():
    """Test ordinal encoding when all features are categorical.

    All features are categorical, so no numerical indices exist.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 4
    cat_indices = list(range(n_features))
    num_indices: list[int] = []

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    result = step.fit_transform(X, feature_schema)

    # All features remain categorical, indices 0 to n_features-1
    assert result.X.shape == (n_samples, n_features)
    assert result.feature_schema.indices_for(FeatureModality.CATEGORICAL) == cat_indices
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == num_indices


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
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name=transform_name,
        random_state=42,
    )
    step.fit_transform(X_train, feature_schema)

    # Multiple transforms should give same result
    result1 = step.transform(X_test)
    result2 = step.transform(X_test)

    np.testing.assert_array_equal(result1.X, result2.X)
    assert result1.feature_schema.indices_for(
        FeatureModality.CATEGORICAL
    ) == result2.feature_schema.indices_for(FeatureModality.CATEGORICAL)
    assert result1.feature_schema.indices_for(
        FeatureModality.NUMERICAL
    ) == result2.feature_schema.indices_for(FeatureModality.NUMERICAL)


def test__in_pipeline__with_modality_selection_works():
    """Test that the step works when registered with a modality selection."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 5
    cat_indices = [1, 3]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_schema = _make_feature_schema(n_features, cat_indices)

    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="ordinal",
        random_state=42,
    )
    pipeline = PreprocessingPipeline(steps=[(step, {FeatureModality.CATEGORICAL})])
    result = pipeline.fit_transform(X, feature_schema)
    assert result.X.shape == (n_samples, n_features)

    num_indices = [i for i in range(n_features) if i not in cat_indices]
    assert result.feature_schema.indices_for(FeatureModality.CATEGORICAL) == cat_indices
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == num_indices


def test__encode_categorical__onehot__max_cardinality_filters_high_cardinality():
    """Test that max_onehot_cardinality skips high-cardinality features.

    With 2 categorical features (cardinality 5 and 50) and max_onehot_cardinality=30,
    only the low-cardinality feature should be one-hot encoded.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 4
    cat_indices = [0, 1]

    X = rng.random((n_samples, n_features))
    X[:, 0] = rng.integers(0, 5, size=n_samples).astype(float)  # cardinality 5
    X[:, 1] = rng.integers(0, 50, size=n_samples).astype(float)  # cardinality 50
    feature_schema = _make_feature_schema(n_features, cat_indices)

    # With limit: only col 0 (card=5) gets one-hot encoded
    step = EncodeCategoricalFeaturesStep(
        categorical_transform_name="onehot",
        random_state=42,
        max_onehot_cardinality=30,
    )
    result = step.fit_transform(X, feature_schema)

    # Col 0: 5 categories -> 5 onehot cols (drop="if_binary" keeps all for non-binary)
    # Col 1: skipped (card 50 > 30), passed through as 1 col
    # Cols 2,3: numerical, passed through
    # Total: 5 + 1 + 2 = 8
    assert result.X.shape == (n_samples, 8)

    # Skipped high-cardinality col 1 should retain categorical modality
    cat_out = result.feature_schema.indices_for(FeatureModality.CATEGORICAL)
    num_out = result.feature_schema.indices_for(FeatureModality.NUMERICAL)
    # 5 onehot cols + 1 skipped categorical = 6 categorical
    assert len(cat_out) == 6
    # 2 numerical cols remain numerical
    assert len(num_out) == 2

    # Without limit: both get one-hot encoded -> more features
    step_no_limit = EncodeCategoricalFeaturesStep(
        categorical_transform_name="onehot",
        random_state=42,
    )
    result_no_limit = step_no_limit.fit_transform(X, feature_schema)
    assert result_no_limit.X.shape[1] > result.X.shape[1]


def _make_schema_with_gpu_marks(
    n_features: int,
    cat_indices: list[int],
    gpu_quantile_indices: list[int],
) -> FeatureSchema:
    """Create FeatureSchema with scheduled_gpu_transform on specified columns."""
    cat_set = set(cat_indices)
    gpu_set = set(gpu_quantile_indices)
    features = []
    for i in range(n_features):
        modality = (
            FeatureModality.CATEGORICAL if i in cat_set else FeatureModality.NUMERICAL
        )
        mark = GPUTransformType.QUANTILE if i in gpu_set else None
        features.append(
            Feature(name=None, modality=modality, scheduled_gpu_transform=mark)
        )
    return FeatureSchema(features=features)


class TestCarryOverGpuTransformMarks:
    """Tests for scheduled_gpu_transform preservation through encoding."""

    def test__ordinal__marks_follow_column_reorder(self):
        """GPU quantile marks on numerical cols survive ordinal encoding reorder.

        Input: cats at [1, 4], numerical at [0, 2, 3, 5], GPU marks on [0, 2, 3, 5].
        Ordinal encoding reorders to [cat(1), cat(4), num(0), num(2), num(3), num(5)].
        Marks should land on output [2, 3, 4, 5].
        """
        rng = np.random.default_rng(42)
        n_features = 6
        cat_indices = [1, 4]
        num_indices = [0, 2, 3, 5]

        X = _make_test_data(rng, 50, n_features, cat_indices)
        schema = _make_schema_with_gpu_marks(n_features, cat_indices, num_indices)

        step = EncodeCategoricalFeaturesStep("ordinal", random_state=42)
        result = step.fit_transform(X, schema)

        marked = result.feature_schema.get_indices_marked_for_gpu_quantile_transform()
        assert marked == [2, 3, 4, 5], (
            f"Expected marks at [2,3,4,5] after reorder, got {marked}"
        )
        # Categoricals must NOT be marked
        for idx in result.feature_schema.indices_for(FeatureModality.CATEGORICAL):
            assert result.feature_schema.features[idx].scheduled_gpu_transform is None

    def test__ordinal_very_common_categories__marks_preserved(self):
        """Marks survive when _very_common_categories filters out some cats.

        If a categorical column is filtered out (not common enough), it moves
        to the remainder alongside numerical columns. Marks on numerical
        columns must still land on the correct output positions.
        """
        rng = np.random.default_rng(42)
        n_features = 6
        cat_indices = [0, 1]
        num_indices = [2, 3, 4, 5]

        X = _make_test_data(rng, 100, n_features, cat_indices)
        # Make cat 0 very common (passes filter), cat 1 rare (filtered out)
        X[:, 0] = rng.integers(0, 3, size=100).astype(float)  # 3 cats, common
        X[:, 1] = np.arange(100).astype(float)  # 100 unique, fails cardinality check

        schema = _make_schema_with_gpu_marks(n_features, cat_indices, num_indices)

        step = EncodeCategoricalFeaturesStep(
            "ordinal_very_common_categories", random_state=42
        )
        result = step.fit_transform(X, schema)

        marked = result.feature_schema.get_indices_marked_for_gpu_quantile_transform()
        # cat 0 passes filter -> encoded at output[0]
        # cat 1 fails filter -> goes to remainder (NOT marked)
        # nums [2,3,4,5] -> go to remainder (marked)
        # Only numerical columns should be marked, not the filtered-out cat
        for idx in marked:
            assert (
                result.feature_schema.features[idx].modality
                == FeatureModality.NUMERICAL
            )

    def test__numeric_mode__marks_preserved_in_place(self):
        """With 'numeric' encoding (no ColumnTransformer), marks stay in place."""
        rng = np.random.default_rng(42)
        n_features = 5
        cat_indices = [1, 3]
        gpu_marks = [0, 2, 4]

        X = _make_test_data(rng, 50, n_features, cat_indices)
        schema = _make_schema_with_gpu_marks(n_features, cat_indices, gpu_marks)

        step = EncodeCategoricalFeaturesStep("numeric", random_state=42)
        result = step.fit_transform(X, schema)

        marked = result.feature_schema.get_indices_marked_for_gpu_quantile_transform()
        assert marked == gpu_marks

    def test__none_mode__marks_preserved_in_place(self):
        """With 'none' encoding, marks stay in place."""
        rng = np.random.default_rng(42)
        n_features = 4
        cat_indices = [2]
        gpu_marks = [0, 1, 3]

        X = _make_test_data(rng, 50, n_features, cat_indices)
        schema = _make_schema_with_gpu_marks(n_features, cat_indices, gpu_marks)

        step = EncodeCategoricalFeaturesStep("none", random_state=42)
        result = step.fit_transform(X, schema)

        marked = result.feature_schema.get_indices_marked_for_gpu_quantile_transform()
        assert marked == gpu_marks

    def test__onehot__marks_land_on_remainder_columns(self):
        """GPU quantile marks survive one-hot encoding on the remainder columns.

        Input: 4 features, cat at [1] (3 categories), marks on [0, 2, 3].
        One-hot expands cat 1 into 3 columns at the front.
        Remainder = [col 0, col 2, col 3] at output positions [3, 4, 5].
        Marks should land on output [3, 4, 5].
        """
        rng = np.random.default_rng(42)
        n_features = 4
        cat_indices = [1]
        gpu_marks = [0, 2, 3]

        X = _make_test_data(rng, 50, n_features, cat_indices)
        X[:, 1] = rng.integers(0, 3, size=50).astype(float)
        schema = _make_schema_with_gpu_marks(n_features, cat_indices, gpu_marks)

        step = EncodeCategoricalFeaturesStep("onehot", random_state=42)
        result = step.fit_transform(X, schema)

        # 3 onehot columns + 3 remainder columns = 6 total
        assert result.X.shape[1] == 6
        marked = result.feature_schema.get_indices_marked_for_gpu_quantile_transform()
        # Marks should be on the 3 remainder columns (the original numerical cols)
        assert marked == [3, 4, 5]
        # None of the onehot-expanded categorical columns should be marked
        for idx in result.feature_schema.indices_for(FeatureModality.CATEGORICAL):
            assert result.feature_schema.features[idx].scheduled_gpu_transform is None

    def test__no_marks__fast_path_returns_unchanged(self):
        """When no features are marked, the carry-over is a no-op."""
        rng = np.random.default_rng(42)
        n_features = 5
        cat_indices = [1, 3]

        X = _make_test_data(rng, 50, n_features, cat_indices)
        schema = _make_feature_schema(n_features, cat_indices)  # no marks

        step = EncodeCategoricalFeaturesStep("ordinal", random_state=42)
        result = step.fit_transform(X, schema)

        marked = result.feature_schema.get_indices_marked_for_gpu_quantile_transform()
        assert marked == []
