"""Tests for ReshapeFeatureDistributionsStep, focusing on feature modality handling."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.steps import ReshapeFeatureDistributionsStep
from tabpfn.preprocessing.steps.preprocessing_helpers import get_categorical_indices


def _get_numerical_indices(
    feature_modalities: dict[FeatureModality, list[int]],
) -> list[int]:
    """Extract numerical indices from feature modalities dictionary."""
    return feature_modalities.get(FeatureModality.NUMERICAL, [])


def _make_feature_modalities(
    n_features: int, cat_indices: list[int]
) -> dict[FeatureModality, list[int]]:
    """Create a feature_modalities dict from feature count and categorical indices."""
    num_indices = [i for i in range(n_features) if i not in cat_indices]
    return {
        FeatureModality.NUMERICAL: num_indices,
        FeatureModality.CATEGORICAL: cat_indices,
    }


def _make_test_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    cat_indices: list[int],
) -> np.ndarray:
    """Create test data with categorical and numerical features."""
    X = rng.random((n_samples, n_features))
    for i in cat_indices:
        X[:, i] = rng.integers(0, 5, size=n_samples).astype(float)
    return X


def test__reshape__append_false_apply_to_cat_false__cat_at_start():
    """Test append_to_original=False, apply_to_categorical=False.

    Categorical features should be passed through at the start, maintaining
    their relative order but with indices remapped to [0, 1, ...].

    With categorical at [2, 4] and numerical at [0, 1, 3, 5]:
    - Categorical passthrough first: [0, 1]
    - Numerical transformed after: [2, 3, 4, 5]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 6
    # Categorical at positions 2 and 4
    cat_indices = [2, 4]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="safepower",
        apply_to_categorical=False,
        append_to_original=False,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    # Categorical features are passed through first, numerical after
    expected_cat_indices = [0, 1]
    expected_num_indices = [2, 3, 4, 5]
    assert get_categorical_indices(result.feature_modalities) == expected_cat_indices
    assert _get_numerical_indices(result.feature_modalities) == expected_num_indices
    assert result.X.shape == (n_samples, n_features)


def test__reshape__append_false_apply_to_cat_true__no_categoricals():
    """Test append_to_original=False, apply_to_categorical=True.

    When we apply transformation to categorical features, they are no longer
    considered categorical.

    With categorical at [1, 3] and numerical at [0, 2, 4, 5]:
    - All features transformed, all become numerical: [0, 1, 2, 3, 4, 5]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 6
    cat_indices = [1, 3]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="safepower",
        apply_to_categorical=True,
        append_to_original=False,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    # All features are transformed, no categoricals remain, all numerical
    expected_cat_indices: list[int] = []
    expected_num_indices = list(range(n_features))
    assert get_categorical_indices(result.feature_modalities) == expected_cat_indices
    assert _get_numerical_indices(result.feature_modalities) == expected_num_indices
    assert result.X.shape == (n_samples, n_features)


def test__reshape__append_true_apply_to_cat_false__cat_preserved():
    """Test append_to_original=True, apply_to_categorical=False.

    Original features (including categoricals) are passed through first,
    then transformed numerical features are appended.

    With categorical at [1, 4] and numerical at [0, 2, 3, 5]:
    - Original (6 features): cat at [1, 4], num at [0, 2, 3, 5]
    - Appended transformed numerical (4 features): [6, 7, 8, 9]
    - Total numerical: [0, 2, 3, 5, 6, 7, 8, 9]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 6
    cat_indices = [1, 4]
    num_indices = [i for i in range(n_features) if i not in cat_indices]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="none",  # Use 'none' to keep the test simple
        apply_to_categorical=False,
        append_to_original=True,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    # Original features first, then transformed numericals appended
    n_numerical = len(num_indices)
    n_output = n_features + n_numerical

    # Categorical indices stay at original positions [1, 4]
    expected_cat_indices = cat_indices
    # Numerical: original positions + appended transformed positions
    expected_num_indices = num_indices + list(range(n_features, n_output))

    assert get_categorical_indices(result.feature_modalities) == expected_cat_indices
    assert _get_numerical_indices(result.feature_modalities) == expected_num_indices
    assert result.X.shape == (n_samples, n_output)


def test__reshape__append_true_apply_to_cat_true__cat_preserved():
    """Test append_to_original=True, apply_to_categorical=True.

    Original features (including categoricals) are passed through first,
    then ALL transformed features are appended.

    With categorical at [0, 3] and numerical at [1, 2, 4, 5]:
    - Original (6 features): cat at [0, 3], num at [1, 2, 4, 5]
    - Appended transformed all (6 features): [6, 7, 8, 9, 10, 11]
    - Total numerical: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 6
    cat_indices = [0, 3]
    num_indices = [i for i in range(n_features) if i not in cat_indices]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="none",
        apply_to_categorical=True,
        append_to_original=True,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    n_output = n_features + n_features

    # Categorical indices stay at original positions since original is first
    expected_cat_indices = cat_indices
    # Numerical: original positions + all appended transformed positions
    expected_num_indices = num_indices + list(range(n_features, n_output))

    assert get_categorical_indices(result.feature_modalities) == expected_cat_indices
    assert _get_numerical_indices(result.feature_modalities) == expected_num_indices
    assert result.X.shape == (n_samples, n_output)


def test__reshape__no_categoricals__all_numerical():
    """Test with no categorical features.

    All features are numerical, so all indices are numerical: [0, 1, 2, 3, 4]
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 5
    cat_indices: list[int] = []
    num_indices = list(range(n_features))

    X = rng.random((n_samples, n_features))
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="safepower",
        apply_to_categorical=False,
        append_to_original=False,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    assert get_categorical_indices(result.feature_modalities) == []
    assert _get_numerical_indices(result.feature_modalities) == num_indices
    assert result.X.shape == (n_samples, n_features)


def test__reshape__all_categoricals__apply_to_cat_false():
    """Test with all categorical features and apply_to_categorical=False.

    All features are categorical and not transformed, so no numerical indices.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 4
    cat_indices = list(range(n_features))
    num_indices: list[int] = []

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="safepower",
        apply_to_categorical=False,
        append_to_original=False,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    # All features are categorical and passed through
    expected_cat_indices = list(range(n_features))
    assert get_categorical_indices(result.feature_modalities) == expected_cat_indices
    assert _get_numerical_indices(result.feature_modalities) == num_indices
    assert result.X.shape == (n_samples, n_features)


def test__reshape__subsampling__modalities_filtered():
    """Test that feature subsampling correctly filters modalities.

    When features exceed max_features_per_estimator, a random subset is kept.
    Both categorical and numerical indices should be valid after subsampling.
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 600  # More than max_features_per_estimator
    max_features = 500
    cat_indices = [10, 50, 100, 200, 300, 400, 500, 550]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="none",
        apply_to_categorical=False,
        append_to_original=False,
        max_features_per_estimator=max_features,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    # Output should be capped at max_features
    assert result.X.shape == (n_samples, max_features)

    # Categorical indices should be subset and remapped
    result_cat = get_categorical_indices(result.feature_modalities)
    result_num = _get_numerical_indices(result.feature_modalities)

    # The categorical indices should be at the start (due to passthrough first)
    # and should be a subset of the original categorical features
    assert len(result_cat) <= len(cat_indices)
    # All returned categorical indices should be valid
    assert all(0 <= idx < max_features for idx in result_cat)
    # All returned numerical indices should be valid
    assert all(0 <= idx < max_features for idx in result_num)
    # Together they should cover all output features
    assert sorted(result_cat + result_num) == list(range(max_features))


@pytest.mark.parametrize(
    ("append_to_original", "apply_to_categorical"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test__reshape__transform_is_deterministic(
    append_to_original: bool,
    apply_to_categorical: bool,
):
    """Test that transform on test data is deterministic after fit."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 8
    cat_indices = [1, 5]

    X_train = _make_test_data(rng, n_samples, n_features, cat_indices)
    X_test = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="none",
        apply_to_categorical=apply_to_categorical,
        append_to_original=append_to_original,
        random_state=42,
    )
    step.fit(X_train, feature_modalities)

    # Multiple transforms should give same result
    result1 = step.transform(X_test)
    result2 = step.transform(X_test)

    np.testing.assert_array_equal(result1.X, result2.X)
    assert get_categorical_indices(
        result1.feature_modalities
    ) == get_categorical_indices(result2.feature_modalities)
    assert _get_numerical_indices(result1.feature_modalities) == _get_numerical_indices(
        result2.feature_modalities
    )


@pytest.mark.parametrize(
    ("append_to_original", "apply_to_categorical", "expected_cat_at_start"),
    [
        # Not appending, not applying to cat -> cats passthrough first
        pytest.param(False, False, True, id="no_append_no_apply_cat_first"),
        # Not appending, applying to cat -> no cats remain
        pytest.param(False, True, False, id="no_append_apply_cat_none"),
        # Appending, not applying to cat -> cats in original positions
        pytest.param(True, False, False, id="append_no_apply_cat_original_pos"),
        # Appending, applying to cat -> cats in original positions
        pytest.param(True, True, False, id="append_apply_cat_original_pos"),
    ],
)
def test__reshape__categorical_and_numerical_index_positions(
    append_to_original: bool,
    apply_to_categorical: bool,
    expected_cat_at_start: bool,
):
    """Test that categorical and numerical indices are in expected positions.

    Input: 8 features with categorical at [3, 5], numerical at [0, 1, 2, 4, 6, 7]

    Expected outputs by setting:
    - (False, False): cat=[0,1], num=[2,3,4,5,6,7] (cats first, then transformed num)
    - (False, True): cat=[], num=[0,1,2,3,4,5,6,7] (all transformed, all numerical)
    - (True, False): cat=[3,5], num=[0,1,2,4,6,7]+[8..13] (original + appended num)
    - (True, True): cat=[3,5], num=[0,1,2,4,6,7]+[8..15] (original + appended all)
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 8
    # Place categoricals in the middle to test index shuffling
    cat_indices = [3, 5]
    num_indices = [i for i in range(n_features) if i not in cat_indices]

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="none",
        apply_to_categorical=apply_to_categorical,
        append_to_original=append_to_original,
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    result_cat = get_categorical_indices(result.feature_modalities)
    result_num = _get_numerical_indices(result.feature_modalities)

    if not append_to_original and apply_to_categorical:
        # Categoricals are transformed, none remain, all numerical
        assert result_cat == []
        assert result_num == list(range(n_features))
    elif expected_cat_at_start:
        # Categoricals at start [0, 1], numerical after [2, 3, 4, 5, 6, 7]
        assert result_cat == [0, 1]
        assert result_num == list(range(2, n_features))
    elif append_to_original and not apply_to_categorical:
        # Original positions + appended transformed numerical
        n_output = n_features + len(num_indices)
        assert result_cat == cat_indices
        assert result_num == num_indices + list(range(n_features, n_output))
    elif append_to_original and apply_to_categorical:
        # Original positions + appended transformed all
        n_output = n_features + n_features
        assert result_cat == cat_indices
        assert result_num == num_indices + list(range(n_features, n_output))


def test__preprocessing_large_dataset():
    num_samples = 150000
    num_features = 2
    input_shape = (num_samples, num_features)
    rng = np.random.default_rng(42)
    X = rng.random(input_shape)

    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_uni",
        apply_to_categorical=False,
        append_to_original=False,
        max_features_per_estimator=500,
        global_transformer_name=None,
        random_state=42,
    )

    feature_modalities: dict[FeatureModality, list[int]] = {
        FeatureModality.NUMERICAL: [0, 1],
        FeatureModality.CATEGORICAL: [],
    }
    result = preprocessing_step.fit_transform(X, feature_modalities)

    assert result is not None
    X_transformed = result.X
    assert X_transformed.shape == input_shape
    assert X_transformed.dtype == X.dtype


@pytest.mark.parametrize(
    ("append_to_original_setting", "num_features", "expected_output_features"),
    [
        # Test 'auto' mode below the threshold: should append original features
        pytest.param("auto", 10, 20, id="auto_below_threshold_appends"),
        # Test 'auto' mode above the threshold: should NOT append original features
        pytest.param("auto", 600, 500, id="auto_above_threshold_replaces"),
        # If n features more than half of max_features_per_estimator we do not append
        pytest.param("auto", 300, 300, id="auto_below_half_threshold_replaces"),
        # True: always append after capping (600 → capped 500 → doubled)
        pytest.param(True, 600, 1000, id="true_always_appends"),
        # Test False: should never append
        pytest.param(False, 10, 10, id="false_never_appends"),
    ],
)
def test__reshape_step_append_original_logic(
    append_to_original_setting: bool | Literal["auto"],
    num_features: int,
    expected_output_features: int,
):
    """Tests the `append_to_original` logic, including the "auto" mode which
    depends on the APPEND_TO_ORIGINAL_THRESHOLD class constant (500).
    """
    num_samples = 100
    rng = np.random.default_rng(42)
    X = rng.random((num_samples, num_features))

    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_uni",
        append_to_original=append_to_original_setting,
        random_state=42,
        max_features_per_estimator=500,
    )

    feature_modalities: dict[FeatureModality, list[int]] = {
        FeatureModality.NUMERICAL: list(range(num_features)),
        FeatureModality.CATEGORICAL: [],
    }
    result = preprocessing_step.fit_transform(X, feature_modalities)

    assert result.X.shape[0] == num_samples
    assert result.X.shape[1] == expected_output_features


def test__reshape__with_global_transformer():
    """Test that the step works with a global transformer.

    Note: SVD global transformer uses FeatureUnion with passthrough + SVD,
    so output has original_features + svd_components columns.
    SVD n_components = min(n_samples//10+1, n_features//2) = min(11, 4) = 4

    Known limitation: Metadata tracking for categorical indices is not fully
    accurate when using global transformers with ColumnTransformer reordering.
    This will be addressed when refactoring to centralized column orchestration.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 8
    cat_indices = [1, 5]
    n_svd_components = 4  # min(100//10+1, 8//2) = min(11, 4)
    n_output_features = n_features + n_svd_components

    X = _make_test_data(rng, n_samples, n_features, cat_indices)
    feature_modalities = _make_feature_modalities(n_features, cat_indices)

    step = ReshapeFeatureDistributionsStep(
        transform_name="none",
        apply_to_categorical=False,
        append_to_original=False,
        global_transformer_name="svd",
        random_state=42,
    )
    result = step.fit_transform(X, feature_modalities)

    # SVD adds components via FeatureUnion (passthrough + SVD)
    assert result.X.shape == (n_samples, n_output_features)
    # Verify we have some categorical and numerical indices tracked
    # (exact positions may vary due to ColumnTransformer reordering)
    assert len(result.metadata.indices_for(FeatureModality.CATEGORICAL)) == len(
        cat_indices
    )
    assert len(result.metadata.indices_for(FeatureModality.NUMERICAL)) > 0
