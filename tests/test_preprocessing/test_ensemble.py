#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.preprocessing import PowerTransformer

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing import (
    generate_classification_ensemble_configs,
    generate_regression_ensemble_configs,
)
from tabpfn.preprocessing.configs import (
    FeatureSubsamplingMethod,
    PreprocessorConfig,
    SampleSubsamplingMethod,
)
from tabpfn.preprocessing.datamodel import Feature, FeatureModality
from tabpfn.preprocessing.ensemble import (
    DEFAULT_N_ESTIMATORS,
    TabPFNEnsemblePreprocessor,
    _compute_feature_importance_order,
    _compute_majority_downsample_group_counts,
    _draw_balanced_from_pool,
    _fit_importance_ordering,
    _get_subsample_feature_indices,
    _get_subsample_indices_for_estimators,
    _resolve_feature_subsampling_method,
    _resolve_importance_top_k,
    _resolve_sample_subsampling_method,
    _subsample_features_importance_based,
    _subsample_rows_majority_downsample,
    _subsample_rows_stratified,
    scale_n_estimators_for_feature_coverage,
)
from tabpfn.preprocessing.torch import FeatureSchema

skip_on_macos = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="LightGBM requires libomp which is not available on macOS CI",
)


def _get_schema(n_features: int) -> FeatureSchema:
    features = [
        Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL)
        for i in range(n_features)
    ]
    return FeatureSchema(features=features)


def test__get_subsample_indices_for_estimators():
    """Test that different subsample_samples arguments work as expected."""
    common_kwargs = {"num_estimators": 3, "n_samples": 5}

    subsample_samples = [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9]),
    ]
    expected_subsample_indices = [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9]),
        np.array([0, 1, 2, 3, 4]),
    ]
    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_samples,
        rng=np.random.default_rng(42),
        **common_kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index, expected_subsample_index in zip(
        subsample_indices, expected_subsample_indices, strict=True
    ):
        assert subsample_index is not None
        assert (subsample_index == expected_subsample_index).all()

    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=0.5,
        rng=np.random.default_rng(42),
        **common_kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index in subsample_indices:
        assert subsample_index is not None
        assert len(subsample_index) == 3  # int(0.5 * 5) + 1 = 3

    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=2,
        rng=np.random.default_rng(42),
        **common_kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index in subsample_indices:
        assert subsample_index is not None
        assert len(subsample_index) == 2


def test__get_subsample_indices_for_estimators__balanced_coverage():
    """Each row appears exactly the same number of times across estimators.

    Exact balance holds when n_rows % subsample_size == 0: the pool then
    exhausts precisely at estimator boundaries, so refills always start with an
    empty already-selected set and every cycle covers all rows exactly once.
    """
    n_rows = 10
    subsample_size = 5  # 10 % 5 == 0 -> exact balance guaranteed
    num_estimators = 4  # 4 * 5 = 20 draws, 20 / 10 = 2 per row

    indices = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_size,
        num_estimators=num_estimators,
        n_samples=n_rows,
        rng=np.random.default_rng(0),
    )

    assert len(indices) == num_estimators
    for idx in indices:
        assert idx is not None
        assert len(idx) == subsample_size
        assert len(set(idx)) == subsample_size  # no duplicates within one estimator

    counts = np.bincount(np.concatenate(indices), minlength=n_rows)
    assert counts.min() == 2
    assert counts.max() == 2


def test__get_subsample_indices_for_estimators__balanced_coverage_float():
    """Float subsample_samples also produces exact balanced row coverage.

    Uses frac=0.2 so that size = int(0.2 * 20) + 1 = 5, and 20 % 5 == 0,
    ensuring pool cycles align with estimator boundaries.
    """
    n_rows = 20
    num_estimators = 8
    frac = 0.2  # size = int(0.2 * 20) + 1 = 5, 20 % 5 == 0 -> exact balance
    # 8 * 5 = 40 draws, 40 / 20 = 2 per row

    indices = _get_subsample_indices_for_estimators(
        subsample_samples=frac,
        num_estimators=num_estimators,
        n_samples=n_rows,
        rng=np.random.default_rng(1),
    )

    assert len(indices) == num_estimators
    subsample_size = int(frac * n_rows) + 1  # = 5
    for idx in indices:
        assert idx is not None
        assert len(idx) == subsample_size

    counts = np.bincount(np.concatenate(indices), minlength=n_rows)
    assert counts.min() == 2
    assert counts.max() == 2


def test__get_subsample_feature_indices__no_subsampling_needed():
    """Test that None is returned when features fit within the limit."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=10),
        max_features_per_estimator=[15, 15],
        rng=rng,
        feature_subsampling_method=FeatureSubsamplingMethod.RANDOM,
    )

    assert len(result) == 2
    assert result[0] is None
    assert result[1] is None


def test__get_subsample_feature_indices__subsampling_needed():
    """Test that feature indices are generated when subsampling is required."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 20  # Adds 2 features
    pipeline.has_data_dependent_feature_expansion.return_value = False

    pipeline2 = MagicMock()
    pipeline2.num_added_features.return_value = 40  # Adds 2 features
    pipeline2.has_data_dependent_feature_expansion.return_value = False

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline2],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=[80, 80],
        rng=rng,
        feature_subsampling_method=FeatureSubsamplingMethod.BALANCED,
    )

    assert result[0] is not None
    assert len(result[0]) == 60
    assert all(0 <= idx < 100 for idx in result[0])

    assert result[1] is not None
    assert len(result[1]) == 40
    assert all(0 <= idx < 100 for idx in result[1])

    # Assert that each feature is present in at least one of the two estimators.
    assert set(result[0]) | set(result[1]) == set(range(100))


def test__transform_X_test__applies_feature_subsampling() -> None:
    """Regression test: transform_X_test must apply the same feature subsampling
    that was used during fit, otherwise the fitted pipeline's boolean masks will
    have the wrong size for the full-feature test set.
    """
    rng = np.random.default_rng(42)
    n_train = 50
    n_test = 10
    n_features = 20
    max_features = 8  # Force subsampling: 8 < 20

    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(0, 3, n_train)
    X_test = rng.standard_normal((n_test, n_features))

    feature_schema = FeatureSchema.from_only_categorical_indices([], n_features)

    configs = generate_classification_ensemble_configs(
        num_estimators=3,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[
            PreprocessorConfig(
                "none",
                categorical_name="numeric",
                max_features_per_estimator=max_features,
            ),
        ],
        class_shift_method=None,
        n_classes=3,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )

    ensemble_preprocessor = TabPFNEnsemblePreprocessor(
        configs=configs,
        n_samples=n_train,
        feature_schema=feature_schema,
        random_state=0,
        n_preprocessing_jobs=1,
    )

    members = ensemble_preprocessor.fit_transform_ensemble_members(
        X_train=X_train,
        y_train=y_train,
    )

    # All members should have feature_indices set since n_features > max_features.
    for member in members:
        assert member.feature_indices is not None
        assert len(member.feature_indices) == max_features

    # transform_X_test must not raise and must return the correct shape.
    for member in members:
        X_test_transformed = member.transform_X_test(X_test)
        assert X_test_transformed.shape[0] == n_test


def test__get_subsample_feature_indices__random_method():
    """Test that RANDOM method independently subsamples for each estimator."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 20
    pipeline.has_data_dependent_feature_expansion.return_value = False

    pipeline2 = MagicMock()
    pipeline2.num_added_features.return_value = 40
    pipeline2.has_data_dependent_feature_expansion.return_value = False

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline2],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=[80, 80],
        rng=rng,
        feature_subsampling_method=FeatureSubsamplingMethod.RANDOM,
    )

    assert result[0] is not None
    assert len(result[0]) == 60
    assert all(0 <= idx < 100 for idx in result[0])
    # Indices should be sorted
    assert list(result[0]) == sorted(result[0])

    assert result[1] is not None
    assert len(result[1]) == 40
    assert all(0 <= idx < 100 for idx in result[1])
    assert list(result[1]) == sorted(result[1])


def test__get_subsample_feature_indices__constant_and_balanced_method():
    """Test that CONSTANT_AND_BALANCED always includes the first N features."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 20
    pipeline.has_data_dependent_feature_expansion.return_value = False

    rng = np.random.default_rng(42)
    constant_count = 30
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=[80, 80],
        rng=rng,
        feature_subsampling_method=FeatureSubsamplingMethod.CONSTANT_AND_BALANCED,
        constant_feature_count=constant_count,
    )

    for indices in result:
        assert indices is not None
        assert len(indices) == 60
        # The first constant_count features must always be included
        assert set(range(constant_count)).issubset(set(indices))
        # Remaining features come from [constant_count, 100)
        non_constant = set(indices) - set(range(constant_count))
        assert all(constant_count <= idx < 100 for idx in non_constant)
        # Indices should be sorted
        assert list(indices) == sorted(indices)

    # Non-constant features should be balanced: no overlap between the two estimators
    # since 30 + 30 = 60 < 70 non-constant features, the pool suffices without reuse.
    non_constant_0 = set(result[0]) - set(range(constant_count))
    non_constant_1 = set(result[1]) - set(range(constant_count))
    assert len(non_constant_0 & non_constant_1) == 0


def test__get_subsample_feature_indices__constant_and_balanced_budget_less_than_constant():  # noqa: E501
    """Test edge case where budget is less than constant_feature_count."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=[30],
        rng=rng,
        feature_subsampling_method=FeatureSubsamplingMethod.CONSTANT_AND_BALANCED,
        constant_feature_count=50,
    )

    assert result[0] is not None
    assert len(result[0]) == 30
    # Should be the first 30 features
    np.testing.assert_array_equal(result[0], np.arange(30))


def test__get_subsample_feature_indices__no_subsampling_all_concrete_methods():
    """All concrete methods return None when budget covers all features."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    concrete_methods = [
        m for m in FeatureSubsamplingMethod if m is not FeatureSubsamplingMethod.AUTO
    ]
    for method in concrete_methods:
        rng = np.random.default_rng(42)
        result = _get_subsample_feature_indices(
            pipelines=[pipeline],
            n_samples=100,
            feature_schema=_get_schema(n_features=10),
            max_features_per_estimator=[15],
            rng=rng,
            feature_subsampling_method=method,
        )
        assert result[0] is None, f"Expected None for method={method}"


def test__get_subsample_feature_indices__auto_raises_if_unresolved():
    """AUTO passed directly to _get_subsample_feature_indices raises ValueError."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    with pytest.raises(ValueError, match="Unsupported"):
        _get_subsample_feature_indices(
            pipelines=[pipeline],
            n_samples=100,
            feature_schema=_get_schema(n_features=100),
            max_features_per_estimator=[80],
            rng=np.random.default_rng(42),
            feature_subsampling_method=FeatureSubsamplingMethod.AUTO,
        )


def test__get_subsample_feature_indices__invalid_method():
    """Unknown string raises ValueError."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    with pytest.raises(ValueError, match="Unsupported"):
        _get_subsample_feature_indices(
            pipelines=[pipeline],
            n_samples=100,
            feature_schema=_get_schema(n_features=100),
            max_features_per_estimator=[80],
            rng=np.random.default_rng(42),
            feature_subsampling_method="nonexistent",  # type: ignore
        )


def test__get_subsample_feature_indices__balanced_uniformity():
    """8 estimators x 60 features over 100 -> each feature appears 4 or 5 times."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    n_estimators = 8
    n_features = 100
    subsample_size = 60

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline] * n_estimators,
        n_samples=100,
        feature_schema=_get_schema(n_features=n_features),
        max_features_per_estimator=[subsample_size] * n_estimators,
        rng=rng,
        feature_subsampling_method=FeatureSubsamplingMethod.BALANCED,
    )

    assert len(result) == n_estimators
    counts = np.zeros(n_features, dtype=int)
    for indices in result:
        assert indices is not None
        assert len(indices) == subsample_size
        counts[indices] += 1

    # Total slots = 8 * 60 = 480 over 100 features -> perfectly uniform would be 4.8.
    # The pool-refill mechanism allows small deviations, so we check approximate
    # uniformity: each feature appears between 3 and 7 times.
    assert counts.min() >= 3, f"Under-represented feature: min count = {counts.min()}"
    assert counts.max() <= 7, f"Over-represented feature: max count = {counts.max()}"
    # The majority of features should appear 4 or 5 times.
    core_count = np.isin(counts, [4, 5]).sum()
    assert core_count >= n_features * 0.7, (
        f"Expected most features to appear 4 or 5 times, got {core_count}/{n_features}"
    )


def test__get_subsample_feature_indices__balanced_reproducibility():
    """Same /different seed produces identical / different results."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    kwargs = {
        "pipelines": [pipeline, pipeline],
        "n_samples": 100,
        "feature_schema": _get_schema(n_features=100),
        "max_features_per_estimator": [60, 60],
        "feature_subsampling_method": FeatureSubsamplingMethod.BALANCED,
    }

    # Same seed -> identical output.
    result_a = _get_subsample_feature_indices(rng=np.random.default_rng(42), **kwargs)
    result_b = _get_subsample_feature_indices(rng=np.random.default_rng(42), **kwargs)
    for a, b in zip(result_a, result_b, strict=True):
        np.testing.assert_array_equal(a, b)

    # Different seed -> different output.
    result_c = _get_subsample_feature_indices(rng=np.random.default_rng(99), **kwargs)
    any_different = any(
        not np.array_equal(a, c)
        for a, c in zip(result_a, result_c, strict=True)
        if a is not None and c is not None
    )
    assert any_different, "Different seeds should produce different distributions"


def test__end_to_end__balanced_feature_subsampling():
    """Test that features are included the expected number of times."""
    rng = np.random.default_rng(42)
    n_train, n_test, n_features = 50, 10, 100
    n_estimators = 8
    max_features = 50

    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(0, 3, n_train)
    X_test = rng.standard_normal((n_test, n_features))

    feature_schema = FeatureSchema.from_only_categorical_indices([], n_features)

    configs = generate_classification_ensemble_configs(
        num_estimators=n_estimators,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[
            PreprocessorConfig(
                "none",
                categorical_name="numeric",
                max_features_per_estimator=max_features,
            ),
        ],
        class_shift_method=None,
        n_classes=3,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )

    ensemble_preprocessor = TabPFNEnsemblePreprocessor(
        configs=configs,
        n_samples=n_train,
        feature_schema=feature_schema,
        random_state=0,
        n_preprocessing_jobs=1,
        feature_subsampling_method=FeatureSubsamplingMethod.BALANCED,
    )

    members = ensemble_preprocessor.fit_transform_ensemble_members(
        X_train=X_train,
        y_train=y_train,
    )

    assert len(members) == n_estimators

    # Check feature occurrence counts across all members.
    # 8 estimators x 50 features = 400 slots over 100 features → 4 per feature.
    # Perfectly uniform: each feature appears 4 times.
    counts = np.zeros(n_features, dtype=int)
    for member in members:
        assert member.feature_indices is not None
        assert len(member.feature_indices) <= max_features
        counts[member.feature_indices] += 1
        # Transform test data should not raise.
        X_test_transformed = member.transform_X_test(X_test)
        assert X_test_transformed.shape[0] == n_test

    expected_mean = n_estimators * max_features / n_features  # 4.8
    assert counts.min() >= 4, (
        f"Under-represented feature: min count {counts.min()}, "
        f"expected ~{expected_mean:.1f}"
    )


def test__subsample_rows_stratified__maintains_class_proportions():
    """Each estimator subsample should roughly preserve the original class fractions."""
    rng = np.random.default_rng(0)
    # 3 classes with proportions 0.6 / 0.3 / 0.1
    y = np.array([0] * 600 + [1] * 300 + [2] * 100)
    rng.shuffle(y)
    subsample_size = 50
    num_estimators = 10

    result = _subsample_rows_stratified(
        subsample_size=subsample_size,
        y=y,
        num_estimators=num_estimators,
        rng=rng,
    )

    assert result is not None
    assert len(result) == num_estimators
    original_fracs = np.array([0.6, 0.3, 0.1])
    for indices in result:
        assert len(indices) == subsample_size
        y_sub = y[indices]
        counts = np.bincount(y_sub, minlength=3)
        fracs = counts / subsample_size
        # Allow ±10 percentage points deviation.
        np.testing.assert_allclose(fracs, original_fracs, atol=0.1)


def test__subsample_rows_stratified__minority_class_always_included():
    """Minority class must appear in every estimator even under extreme imbalance."""
    rng = np.random.default_rng(42)
    # 999 majority, 1 minority — proportional quota = 0
    y = np.array([0] * 999 + [1] * 1)
    result = _subsample_rows_stratified(
        subsample_size=100,
        y=y,
        num_estimators=10,
        rng=rng,
    )
    assert result is not None
    for indices in result:
        assert len(indices) == 100
        assert 1 in set(y[indices]), "minority class must appear in every estimator"


def test__subsample_rows_stratified__class_allocated_more_slots_than_rows():
    """Oversampling a tiny class must terminate and reuse its rows.

    Largest-remainder allocation can assign a class more slots than it has rows
    (class sizes [2, 98] with subsample_size 99 -> target counts [3, 96]). This
    used to spin forever in _draw_balanced_from_pool because the refill excluded
    all already-drawn slots, leaving the pool permanently empty.
    """
    rng = np.random.default_rng(0)
    y = np.array([0] * 2 + [1] * 98)
    num_estimators = 4

    result = _subsample_rows_stratified(
        subsample_size=99,
        y=y,
        num_estimators=num_estimators,
        rng=rng,
    )

    assert result is not None
    assert len(result) == num_estimators
    for indices in result:
        assert len(indices) == 99
        y_sub = y[indices]
        # 3 slots for class 0: both of its rows plus one duplicate.
        assert (y_sub == 0).sum() == 3
        assert set(indices[y_sub == 0]) == {0, 1}
        assert (y_sub == 1).sum() == 96


def test__draw_balanced_from_pool__size_exceeds_pool_size():
    """Drawing more slots than the pool holds duplicates slots evenly."""
    rng = np.random.default_rng(1)

    slots, _ = _draw_balanced_from_pool(pool=[], size=5, pool_size=2, rng=rng)

    assert len(slots) == 5
    counts = np.bincount(slots, minlength=2)
    # 5 draws over 2 slots split as evenly as possible.
    assert sorted(counts.tolist()) == [2, 3]


def test__subsample_rows_stratified__balanced_coverage():
    """Each row appears approximately the same number of times across estimators."""
    rng = np.random.default_rng(2)
    # Balanced 2-class dataset.
    n_per_class = 50
    y = np.array([0] * n_per_class + [1] * n_per_class)
    subsample_size = 20  # 10 per class
    num_estimators = 10  # 10 * 10 = 100 draws per class, 100/50 = 2 per row

    result = _subsample_rows_stratified(
        subsample_size=subsample_size,
        y=y,
        num_estimators=num_estimators,
        rng=rng,
    )

    assert result is not None
    assert len(result) == num_estimators
    n_rows = len(y)
    counts = np.bincount(np.concatenate(result), minlength=n_rows)
    # Each row should appear approximately 2 times; allow ±1 for pool boundary effects.
    assert counts.min() >= 1
    assert counts.max() <= 3


def test__get_subsample_indices_for_estimators__stratified_dispatch():
    """When y is provided, stratified sampling preserves class proportions."""
    rng = np.random.default_rng(3)
    y = np.array([0] * 80 + [1] * 20)
    n_samples = len(y)
    subsample_size = 40
    num_estimators = 6

    # int subsample_samples
    result = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_size,
        num_estimators=num_estimators,
        n_samples=n_samples,
        rng=rng,
        method=SampleSubsamplingMethod.STRATIFIED,
        y=y,
    )

    assert result is not None
    assert len(result) == num_estimators
    for indices in result:
        assert len(indices) == subsample_size
        counts = np.bincount(y[indices], minlength=2)
        # Natural proportions: 80% class 0, 20% class 1. Allow ±10% tolerance.
        assert abs(counts[0] / subsample_size - 0.8) <= 0.1
        assert abs(counts[1] / subsample_size - 0.2) <= 0.1

    # float subsample_samples
    result_float = _get_subsample_indices_for_estimators(
        subsample_samples=0.4,
        num_estimators=num_estimators,
        n_samples=n_samples,
        rng=np.random.default_rng(4),
        method=SampleSubsamplingMethod.STRATIFIED,
        y=y,
    )
    assert result_float is not None
    expected_size = int(0.4 * n_samples) + 1  # 41
    for indices in result_float:
        assert len(indices) == expected_size
        counts = np.bincount(y[indices], minlength=2)
        assert abs(counts[0] / expected_size - 0.8) <= 0.1
        assert abs(counts[1] / expected_size - 0.2) <= 0.1


# --- Feature importance subsampling tests ---


def test__compute_majority_downsample_group_counts__binary_keeps_minority_whole():
    counts = _compute_majority_downsample_group_counts(
        group_sizes=np.array([900, 100]), subsample_size=300
    )
    np.testing.assert_array_equal(counts, [200, 100])
    assert counts.sum() == 300


def test__compute_majority_downsample_group_counts__keeps_all_non_majority_groups():
    counts = _compute_majority_downsample_group_counts(
        group_sizes=np.array([500, 10, 300, 40]), subsample_size=400
    )
    np.testing.assert_array_equal(counts, [50, 10, 300, 40])
    assert counts.sum() == 400


def test__compute_majority_downsample_group_counts__minority_exceeds_budget():
    with pytest.raises(ValueError, match="greater than the number of non-majority"):
        _compute_majority_downsample_group_counts(
            group_sizes=np.array([600, 400]), subsample_size=100
        )


def test__compute_majority_downsample_group_counts__minority_nearly_fills_budget():
    """The minority is kept whole as long as one row is left for the majority."""
    counts = _compute_majority_downsample_group_counts(
        group_sizes=np.array([900, 99]), subsample_size=100
    )
    np.testing.assert_array_equal(counts, [1, 99])


@pytest.mark.parametrize(
    "group_sizes",
    [
        np.array([10, 10]),
        np.array([10, 2, 10]),
        np.ones(1000, dtype=int),
    ],
)
def test__compute_majority_downsample_group_counts__rejects_tied_majority(
    group_sizes: np.ndarray,
):
    with pytest.raises(ValueError, match="one unique majority target value"):
        _compute_majority_downsample_group_counts(
            group_sizes=group_sizes,
            subsample_size=min(10, int(group_sizes.sum()) - 1),
        )


def test__compute_majority_downsample_group_counts__single_group():
    counts = _compute_majority_downsample_group_counts(
        group_sizes=np.array([100]), subsample_size=30
    )
    np.testing.assert_array_equal(counts, [30])


def test__compute_majority_downsample_group_counts__full_budget_needs_no_majority():
    counts = _compute_majority_downsample_group_counts(
        group_sizes=np.array([10, 10]), subsample_size=20
    )
    np.testing.assert_array_equal(counts, [10, 10])


def test__compute_majority_downsample_group_counts__never_exceeds_group_size():
    rng = np.random.default_rng(0)
    for _ in range(50):
        minority_sizes = rng.integers(1, 50, size=rng.integers(1, 20))
        majority_size = int(minority_sizes.max() + rng.integers(1, 50))
        sizes = np.append(minority_sizes, majority_size)
        rng.shuffle(sizes)
        non_majority_size = int(sizes.sum() - majority_size)
        budget = int(rng.integers(non_majority_size + 1, sizes.sum() + 1))
        counts = _compute_majority_downsample_group_counts(sizes, budget)
        assert counts.sum() == budget
        assert (counts <= sizes).all()
        assert (counts >= 0).all()
        majority_group = int(np.argmax(sizes))
        np.testing.assert_array_equal(
            np.delete(counts, majority_group), np.delete(sizes, majority_group)
        )


def test__subsample_rows_majority_downsample__zero_inflated_regression():
    """Zeros are downsampled while every distinct nonzero target is kept."""
    rng = np.random.default_rng(0)
    n_zeros, n_nonzero = 900, 100
    y = np.concatenate([np.zeros(n_zeros), rng.exponential(size=n_nonzero) + 0.1])
    rng.shuffle(y)
    nonzero_rows = set(np.flatnonzero(y != 0))
    subsample_size = 250

    result = _subsample_rows_majority_downsample(
        subsample_size=subsample_size,
        y=y,
        num_estimators=4,
        rng=rng,
        task_type="regressor",
    )

    assert result is not None
    for indices in result:
        assert len(indices) == subsample_size
        assert len(np.unique(indices)) == subsample_size
        assert set(indices[y[indices] != 0]) == nonzero_rows
        assert (y[indices] == 0).sum() == subsample_size - n_nonzero


def test__subsample_rows_majority_downsample__many_distinct_values_is_fast():
    """Bookkeeping must not loop over every distinct target value per estimator."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(200_000), rng.normal(size=200_000)])
    start = time.perf_counter()
    result = _subsample_rows_majority_downsample(
        subsample_size=250_000,
        y=y,
        num_estimators=8,
        rng=rng,
        task_type="regressor",
    )
    elapsed = time.perf_counter() - start
    assert result is not None
    assert all(len(idx) == 250_000 for idx in result)
    assert elapsed < 10, f"took {elapsed:.1f}s"


def test__subsample_rows_majority_downsample__keeps_all_minority_rows():
    rng = np.random.default_rng(0)
    y = np.array([0] * 950 + [1] * 50)
    rng.shuffle(y)
    minority_rows = set(np.where(y == 1)[0])
    subsample_size = 200
    num_estimators = 8

    result = _subsample_rows_majority_downsample(
        subsample_size=subsample_size,
        y=y,
        num_estimators=num_estimators,
        rng=rng,
        task_type="classifier",
    )

    assert result is not None
    assert len(result) == num_estimators
    for indices in result:
        assert len(indices) == subsample_size
        assert len(np.unique(indices)) == subsample_size, "no duplicate rows"
        assert set(indices[y[indices] == 1]) == minority_rows
        assert (y[indices] == 0).sum() == subsample_size - len(minority_rows)


def test__subsample_rows_majority_downsample__majority_balanced_coverage():
    """Majority rows are drawn round-robin so coverage is even across estimators."""
    rng = np.random.default_rng(1)
    y = np.array([0] * 100 + [1] * 10)
    # 4 estimators x 40 majority slots = 160 draws over 100 majority rows:
    # every row is drawn once in the first pass, 60 of them twice.
    result = _subsample_rows_majority_downsample(
        subsample_size=50,
        y=y,
        num_estimators=4,
        rng=rng,
        task_type="classifier",
    )
    assert result is not None
    majority_counts = np.bincount(
        np.concatenate([idx[y[idx] == 0] for idx in result]), minlength=110
    )[:100]
    assert set(majority_counts.tolist()) == {1, 2}
    assert majority_counts.sum() == 160


def test__subsample_rows_majority_downsample__returns_none_when_no_subsampling_needed():
    y = np.array([0, 0, 1])
    assert (
        _subsample_rows_majority_downsample(
            subsample_size=3,
            y=y,
            num_estimators=2,
            rng=np.random.default_rng(0),
            task_type="classifier",
        )
        is None
    )


def test__subsample_rows_majority_downsample__string_labels():
    rng = np.random.default_rng(3)
    y = np.array(["cat"] * 90 + ["dog"] * 10)
    result = _subsample_rows_majority_downsample(
        subsample_size=30,
        y=y,
        num_estimators=3,
        rng=rng,
        task_type="classifier",
    )
    assert result is not None
    for indices in result:
        assert (y[indices] == "dog").sum() == 10
        assert (y[indices] == "cat").sum() == 20


def test__subsample_rows_majority_downsample__balanced_fallback_for_regression():
    y = np.arange(20)
    with pytest.warns(UserWarning, match="falling back to 'balanced'"):
        result = _subsample_rows_majority_downsample(
            subsample_size=5,
            y=y,
            num_estimators=4,
            rng=np.random.default_rng(0),
            task_type="regressor",
        )

    assert result is not None
    occurrence_counts = np.bincount(np.concatenate(result), minlength=len(y))
    np.testing.assert_array_equal(occurrence_counts, np.ones(len(y), dtype=int))


def test__subsample_rows_majority_downsample__stratified_fallback_for_classifier():
    y = np.array([0] * 10 + [1] * 10)
    with pytest.warns(UserWarning, match="falling back to 'stratified'"):
        result = _subsample_rows_majority_downsample(
            subsample_size=10,
            y=y,
            num_estimators=3,
            rng=np.random.default_rng(0),
            task_type="classifier",
        )

    assert result is not None
    for indices in result:
        assert (y[indices] == 0).sum() == 5
        assert (y[indices] == 1).sum() == 5


@pytest.mark.parametrize("subsample_size", [9, 10])
def test__subsample_rows_majority_downsample__rejects_insufficient_budget(
    subsample_size: int,
):
    y = np.array([0] * 10 + [1] * 6 + [2] * 4)
    with pytest.raises(ValueError, match="greater than the number of non-majority"):
        _subsample_rows_majority_downsample(
            subsample_size=subsample_size,
            y=y,
            num_estimators=3,
            rng=np.random.default_rng(0),
            task_type="classifier",
        )


def test__subsample_rows_majority_downsample__warns_when_class_prior_inverts():
    y = np.array([0] * 500 + [1] * 300 + [2] * 200)
    with pytest.warns(UserWarning, match="changes the class prior"):
        result = _subsample_rows_majority_downsample(
            subsample_size=600,
            y=y,
            num_estimators=2,
            rng=np.random.default_rng(0),
            task_type="classifier",
        )

    assert result is not None
    for indices in result:
        np.testing.assert_array_equal(np.bincount(y[indices]), [100, 300, 200])


def test__resolve_sample_subsampling_method__auto():
    assert (
        _resolve_sample_subsampling_method(
            SampleSubsamplingMethod.AUTO, task_type="classifier"
        )
        == SampleSubsamplingMethod.STRATIFIED
    )
    assert (
        _resolve_sample_subsampling_method(
            SampleSubsamplingMethod.AUTO, task_type="regressor"
        )
        == SampleSubsamplingMethod.BALANCED
    )


def test__resolve_sample_subsampling_method__stratified_rejected_for_regressor():
    with pytest.raises(ValueError, match="only supported for classification"):
        _resolve_sample_subsampling_method(
            SampleSubsamplingMethod.STRATIFIED, task_type="regressor"
        )


def test__resolve_sample_subsampling_method__majority_downsample_allowed_for_regressor():  # noqa: E501
    assert (
        _resolve_sample_subsampling_method(
            SampleSubsamplingMethod.MAJORITY_DOWNSAMPLE, task_type="regressor"
        )
        == SampleSubsamplingMethod.MAJORITY_DOWNSAMPLE
    )


def test__resolve_sample_subsampling_method__accepts_strings():
    resolved = _resolve_sample_subsampling_method(
        "majority_downsample",  # type: ignore[arg-type]
        task_type="classifier",
    )
    assert resolved == SampleSubsamplingMethod.MAJORITY_DOWNSAMPLE


def test__get_subsample_indices_for_estimators__majority_downsample_dispatch():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    result = _get_subsample_indices_for_estimators(
        subsample_samples=300,
        num_estimators=4,
        n_samples=len(y),
        rng=rng,
        method=SampleSubsamplingMethod.MAJORITY_DOWNSAMPLE,
        y=y,
    )
    assert result is not None
    for indices in result:
        assert (y[indices] == 1).sum() == 100
        assert (y[indices] == 0).sum() == 200


def test__get_subsample_indices_for_estimators__class_aware_requires_y():
    with pytest.raises(ValueError, match="requires the targets"):
        _get_subsample_indices_for_estimators(
            subsample_samples=10,
            num_estimators=2,
            n_samples=100,
            rng=np.random.default_rng(0),
            method=SampleSubsamplingMethod.MAJORITY_DOWNSAMPLE,
        )


def test__get_subsample_indices_for_estimators__auto_must_be_resolved():
    with pytest.raises(ValueError, match="must be resolved"):
        _get_subsample_indices_for_estimators(
            subsample_samples=10,
            num_estimators=2,
            n_samples=100,
            rng=np.random.default_rng(0),
            method=SampleSubsamplingMethod.AUTO,
        )


def test__get_subsample_indices_for_estimators__balanced_ignores_y():
    """Explicit 'balanced' ignores the labels even when they are provided."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 999 + [1])
    result = _get_subsample_indices_for_estimators(
        subsample_samples=100,
        num_estimators=3,
        n_samples=len(y),
        rng=rng,
        method=SampleSubsamplingMethod.BALANCED,
        y=y,
    )
    assert result is not None
    # Round-robin over a shuffled pool: 3 x 100 = 300 slots over 1000 rows, so
    # the single minority row cannot appear in every estimator.
    assert sum(1 in set(y[idx]) for idx in result) <= 1


def test__get_subsample_indices_for_estimators__detaches_torch_targets():
    y = torch.tensor([0.0] * 9 + [1.0], requires_grad=True)
    result = _get_subsample_indices_for_estimators(
        subsample_samples=4,
        num_estimators=2,
        n_samples=len(y),
        rng=np.random.default_rng(0),
        method=SampleSubsamplingMethod.MAJORITY_DOWNSAMPLE,
        y=y,
    )

    assert result is not None
    for indices in result:
        assert len(indices) == 4
        assert 9 in indices
    assert y.requires_grad


def test__get_subsample_indices_for_estimators__bfloat16_targets():
    """bfloat16 has no numpy dtype; the labels must be widened, not crash."""
    y = torch.tensor([0.0] * 9 + [1.0], dtype=torch.bfloat16)
    result = _get_subsample_indices_for_estimators(
        subsample_samples=4,
        num_estimators=2,
        n_samples=len(y),
        rng=np.random.default_rng(0),
        method=SampleSubsamplingMethod.STRATIFIED,
        y=y,
    )

    assert result is not None
    for indices in result:
        assert len(indices) == 4
        assert 9 in indices


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.bfloat16])
def test__fit_with_differentiable_input__row_subsampling(dtype: torch.dtype):
    """The differentiable path now stratifies under "auto", like fit() does."""
    rng = np.random.default_rng(0)
    n_majority, n_minority = 180, 20
    X = torch.tensor(rng.normal(size=(n_majority + n_minority, 3)), dtype=torch.float32)
    y_np = np.array([0] * n_majority + [1] * n_minority)
    y = torch.tensor(y_np, dtype=dtype)
    clf = TabPFNClassifier(
        n_estimators=2,
        differentiable_input=True,
        inference_config={"SUBSAMPLE_SAMPLES": 60},
        random_state=0,
    )
    clf.fit_with_differentiable_input(X, y)
    row_indices = clf.ensemble_preprocessor_.subsample_row_indices
    assert row_indices is not None
    for indices in row_indices:
        assert len(indices) == 60
        # Stratified: the minority keeps roughly its 10% share (one slot is
        # reserved per class, the rest allocated by largest remainder) instead
        # of being left to chance as under the previous label-blind sampling.
        assert (y_np[indices] == 1).sum() in (6, 7)


@pytest.mark.parametrize("subsample_samples", [None, [np.array([0, 1])]])
def test__sample_subsampling_method__ignored_without_numeric_subsampling(
    subsample_samples: list[np.ndarray] | None,
):
    configs = generate_regression_ensemble_configs(
        num_estimators=1,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[PreprocessorConfig("none", categorical_name="numeric")],
        target_transforms=[None],
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )

    preprocessor = TabPFNEnsemblePreprocessor(
        configs=configs,
        n_samples=2,
        feature_schema=_get_schema(1),
        random_state=0,
        n_preprocessing_jobs=1,
        subsample_samples=subsample_samples,
        sample_subsampling_method=SampleSubsamplingMethod.STRATIFIED,
        task_type="regressor",
    )

    if subsample_samples is None:
        assert preprocessor.subsample_row_indices is None
    else:
        assert preprocessor.subsample_row_indices is not None
        np.testing.assert_array_equal(
            preprocessor.subsample_row_indices[0], subsample_samples[0]
        )


def test__end_to_end__majority_downsample_row_subsampling():
    """The classifier wires SAMPLE_SUBSAMPLING_METHOD through to the preprocessor."""
    rng = np.random.default_rng(0)
    n_majority, n_minority = 180, 20
    X = rng.normal(size=(n_majority + n_minority, 3))
    y = np.array([0] * n_majority + [1] * n_minority)
    clf = TabPFNClassifier(
        n_estimators=3,
        inference_config={
            "SUBSAMPLE_SAMPLES": 60,
            "SAMPLE_SUBSAMPLING_METHOD": "majority_downsample",
        },
        random_state=0,
    )
    clf.fit(X, y)
    row_indices = clf.ensemble_preprocessor_.subsample_row_indices
    assert row_indices is not None
    assert len(row_indices) == 3
    for indices in row_indices:
        assert len(indices) == 60
        assert (y[indices] == 1).sum() == n_minority
        assert (y[indices] == 0).sum() == 60 - n_minority


def test__end_to_end__majority_downsample_row_subsampling_regressor():
    """A zero-inflated regressor keeps every nonzero target row per estimator."""
    rng = np.random.default_rng(0)
    n_zeros, n_nonzero = 170, 30
    X = rng.normal(size=(n_zeros + n_nonzero, 3))
    y = np.concatenate([np.zeros(n_zeros), rng.exponential(size=n_nonzero) + 0.1])
    reg = TabPFNRegressor(
        n_estimators=3,
        inference_config={
            "SUBSAMPLE_SAMPLES": 80,
            "SAMPLE_SUBSAMPLING_METHOD": "majority_downsample",
        },
        random_state=0,
    )
    reg.fit(X, y)
    row_indices = reg.ensemble_preprocessor_.subsample_row_indices
    assert row_indices is not None
    assert len(row_indices) == 3
    for indices in row_indices:
        assert len(indices) == 80
        assert (y[indices] != 0).sum() == n_nonzero
        assert (y[indices] == 0).sum() == 80 - n_nonzero


def test__subsample_features_importance_based__top_k_always_present():
    """Top-K features must appear in every estimator's selection."""
    rng = np.random.default_rng(0)
    n_features = 20
    importance_order = rng.permutation(n_features)
    top_k = 5
    subsample_sizes = [10, 10, 10]

    result = _subsample_features_importance_based(
        subsample_sizes=subsample_sizes,
        n_total_features=n_features,
        importance_feature_order=importance_order,
        top_k_count=top_k,
        rng=rng,
    )

    top5 = set(importance_order[:top_k])
    for indices in result:
        assert indices is not None
        assert top5.issubset(set(indices))
        assert len(indices) == 10
        assert list(indices) == sorted(indices), "Indices must be sorted"


def test__subsample_features_importance_based__no_subsampling_when_budget_ge_total():
    """Returns None when budget covers all features."""
    rng = np.random.default_rng(0)
    n_features = 10
    importance_order = np.arange(n_features)
    result = _subsample_features_importance_based(
        subsample_sizes=[10, 10],
        n_total_features=n_features,
        importance_feature_order=importance_order,
        top_k_count=5,
        rng=rng,
    )
    assert all(r is None for r in result)


def test__subsample_features_importance_based__budget_less_than_top_k():
    """When budget < top_k, only the most important features are selected."""
    rng = np.random.default_rng(0)
    n_features = 20
    importance_order = np.arange(n_features)  # feature 0 is most important
    result = _subsample_features_importance_based(
        subsample_sizes=[3],
        n_total_features=n_features,
        importance_feature_order=importance_order,
        top_k_count=10,
        rng=rng,
    )
    assert result[0] is not None
    assert len(result[0]) == 3
    assert set(result[0]) == {0, 1, 2}


def test__subsample_features_importance_based__budget_equal_to_top_k():
    """When budget == top_k, exactly the top-k features are returned."""
    rng = np.random.default_rng(0)
    n_features = 20
    importance_order = np.arange(n_features)
    top_k = 8
    result = _subsample_features_importance_based(
        subsample_sizes=[top_k],
        n_total_features=n_features,
        importance_feature_order=importance_order,
        top_k_count=top_k,
        rng=rng,
    )
    assert result[0] is not None
    assert len(result[0]) == top_k
    assert set(result[0]) == set(range(top_k))


def test__subsample_features_importance_based__remaining_budget_balanced_across_estimators():  # noqa: E501
    """Non-top-K slots are filled via balanced round-robin.

    every remaining feature appears roughly equally across estimators sharing the same
    ordering.
    """
    rng = np.random.default_rng(0)
    n_features = 20
    top_k = 5
    budget = 10  # 5 top-K + 5 from remaining 15 features
    n_estimators = 30  # enough passes that balance is visible

    importance_order = np.arange(n_features)
    remaining_features = set(range(top_k, n_features))  # 15 features

    result = _subsample_features_importance_based(
        subsample_sizes=[budget] * n_estimators,
        n_total_features=n_features,
        importance_feature_order=importance_order,
        top_k_count=top_k,
        rng=rng,
    )

    counts = dict.fromkeys(remaining_features, 0)
    for indices in result:
        assert indices is not None
        assert len(indices) == budget
        for idx in indices:
            if idx in remaining_features:
                counts[idx] += 1

    # Each remaining feature should appear at least once across 30 estimators
    # drawing 5 from 15 features (expected ~10 appearances each).
    assert all(c > 0 for c in counts.values()), (
        "Every non-top-K feature must appear at least once"
    )
    # Balanced: max count should be close to min count (within 2x)
    assert max(counts.values()) <= 2 * min(counts.values()), (
        "Balanced pool: feature counts should not differ by more than 2x"
    )


def test__get_subsample_feature_indices__feature_importance_method():
    """GINI_FEATURE_IMPORTANCE method routes correctly and includes top-K."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    rng = np.random.default_rng(42)
    n_features = 50
    top_k = 5
    importance_order = rng.permutation(n_features)

    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline, pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=n_features),
        max_features_per_estimator=[20, 20, 20],
        rng=rng,
        feature_subsampling_method=FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
        importance_feature_order=importance_order,
        importance_top_k_count=top_k,
    )

    top_k_set = set(importance_order[:top_k])
    for indices in result:
        assert indices is not None
        assert len(indices) == 20
        assert top_k_set.issubset(set(indices))
        assert list(indices) == sorted(indices)


def test__get_subsample_feature_indices__feature_importance_none_order_falls_back_to_balanced():  # noqa: E501
    """GINI_FEATURE_IMPORTANCE with importance_feature_order=None falls back to balanced."""  # noqa: E501
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline, pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=50),
        max_features_per_estimator=[20, 20, 20],
        rng=np.random.default_rng(0),
        feature_subsampling_method=FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
        importance_feature_order=None,
    )
    # Should return valid index arrays (balanced fallback), not raise
    assert len(result) == 3
    for indices in result:
        assert indices is not None
        assert len(indices) == 20


def test__resolve_importance_top_k__auto_above_threshold():
    """Auto returns the configured top-k when n_features exceeds the threshold."""
    result = _resolve_importance_top_k(
        "auto", n_total_features=300, auto_top_k=50, auto_min_features=200
    )
    assert result == 50


def test__resolve_importance_top_k__auto_below_threshold():
    """Auto returns n_total_features when n_features is at or below the threshold."""
    result = _resolve_importance_top_k(
        "auto", n_total_features=100, auto_top_k=50, auto_min_features=200
    )
    assert result == 100


def test__resolve_importance_top_k__auto_at_threshold():
    """Auto returns n_total_features when n_features equals the threshold (not strictly above)."""  # noqa: E501
    result = _resolve_importance_top_k(
        "auto", n_total_features=200, auto_top_k=50, auto_min_features=200
    )
    assert result == 200


def test__resolve_importance_top_k__int():
    """Int value is returned as-is."""
    assert _resolve_importance_top_k(42, n_total_features=100) == 42


def test__resolve_importance_top_k__float():
    """Float is resolved as ceil(value * n_total_features), minimum 1."""
    assert _resolve_importance_top_k(0.3, n_total_features=10) == 3  # ceil(3.0)
    assert _resolve_importance_top_k(0.25, n_total_features=10) == 3  # ceil(2.5)
    assert (
        _resolve_importance_top_k(0.01, n_total_features=10) == 1
    )  # floor clamped to 1


def test__resolve_feature_subsampling_method__non_auto_passthrough():
    """Non-AUTO values are returned unchanged regardless of other arguments."""
    for method in FeatureSubsamplingMethod:
        if method is FeatureSubsamplingMethod.AUTO:
            continue
        assert (
            _resolve_feature_subsampling_method(
                method, needs_subsampling=True, n_samples=999_999
            )
            is method
        )
        assert (
            _resolve_feature_subsampling_method(
                method, needs_subsampling=False, n_samples=0
            )
            is method
        )


def test__resolve_feature_subsampling_method__auto_large_dataset_needs_subsampling():
    """AUTO → GINI_FEATURE_IMPORTANCE when subsampling needed and n_samples large."""
    result = _resolve_feature_subsampling_method(
        FeatureSubsamplingMethod.AUTO,
        needs_subsampling=True,
        n_samples=200_000,
        auto_min_samples=100_000,
    )
    assert result is FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE


def test__resolve_feature_subsampling_method__auto_small_dataset():
    """AUTO → BALANCED when n_samples is at or below the threshold."""
    result = _resolve_feature_subsampling_method(
        FeatureSubsamplingMethod.AUTO,
        needs_subsampling=True,
        n_samples=100_000,
        auto_min_samples=100_000,
    )
    assert result is FeatureSubsamplingMethod.BALANCED


def test__resolve_feature_subsampling_method__auto_no_subsampling_needed():
    """AUTO → BALANCED when no feature subsampling is needed, regardless of n_samples."""  # noqa: E501
    result = _resolve_feature_subsampling_method(
        FeatureSubsamplingMethod.AUTO,
        needs_subsampling=False,
        n_samples=999_999,
        auto_min_samples=100_000,
    )
    assert result is FeatureSubsamplingMethod.BALANCED


def test_default_n_estimators__is_unchanged():
    """Pin the package default: `n_estimators="auto"` still means 8 estimators.

    Changing this value silently changes runtime and predictions for every user
    who never touches `n_estimators`, so it should only move deliberately.
    """
    assert DEFAULT_N_ESTIMATORS == 8

    cfg = PreprocessorConfig("none", max_features_per_estimator=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = scale_n_estimators_for_feature_coverage(
            n_estimators="auto",
            n_total_features=10,  # narrow: no coverage scaling in play
            preprocessor_configs=[cfg],
        )
    assert resolved == 8


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test_default_n_estimators__is_the_constructor_default(estimator_cls: type):
    """Both estimators default to `"auto"`, which resolves to DEFAULT_N_ESTIMATORS."""
    assert estimator_cls().n_estimators == "auto"


def test_scale_n_estimators_for_feature_coverage__no_scaling_when_enough_capacity():
    """At capacity (n_estimators * max_features == n_features): no scaling, no warning."""  # noqa: E501
    cfg = PreprocessorConfig("none", max_features_per_estimator=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = scale_n_estimators_for_feature_coverage(
            n_estimators="auto",
            n_total_features=4000,  # exactly DEFAULT_N_ESTIMATORS (8) * 500
            preprocessor_configs=[cfg],
        )
    assert result == DEFAULT_N_ESTIMATORS


def test_scale_n_estimators_for_feature_coverage__scales_up_and_warns():
    """Over capacity: scales to ceil(n_features / max_features) and warns."""
    cfg = PreprocessorConfig("none", max_features_per_estimator=500)
    with pytest.warns(UserWarning, match="Auto-scaling n_estimators"):
        result = scale_n_estimators_for_feature_coverage(
            n_estimators="auto",
            n_total_features=5001,  # non-divisible: also exercises ceil rounding
            preprocessor_configs=[cfg],
        )
    assert result == 11  # ceil(5001 / 500)


def test_scale_n_estimators_for_feature_coverage__uses_min_max_features_across_configs():  # noqa: E501
    """The smallest max_features_per_estimator across configs is the binding budget."""
    small = PreprocessorConfig("none", max_features_per_estimator=500)
    large = PreprocessorConfig("none", max_features_per_estimator=1_000_000)
    with pytest.warns(UserWarning):  # noqa: PT030
        result = scale_n_estimators_for_feature_coverage(
            n_estimators="auto",
            n_total_features=6000,
            preprocessor_configs=[small, large],
        )
    # Bound by min budget (500): ceil(6000 / 500) = 12.
    assert result == 12


@pytest.mark.parametrize("n_estimators", [2, 8])
def test_scale_n_estimators_for_feature_coverage__explicit_value_is_never_scaled(
    n_estimators: int,
):
    """An explicitly passed n_estimators is used as-is, without warning."""
    cfg = PreprocessorConfig("none", max_features_per_estimator=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = scale_n_estimators_for_feature_coverage(
            n_estimators=n_estimators,
            n_total_features=n_estimators * 500,  # exactly covered: no warning
            preprocessor_configs=[cfg],
        )
    assert result == n_estimators


@pytest.mark.parametrize("n_estimators", [2, 8])
def test_scale_n_estimators_for_feature_coverage__explicit_value_warns_if_uncovered(
    n_estimators: int,
):
    """Too small an explicit n_estimators warns but is still used as given."""
    cfg = PreprocessorConfig("none", max_features_per_estimator=500)
    with pytest.warns(UserWarning, match=r"covers at most \d+ of 5001 features"):
        result = scale_n_estimators_for_feature_coverage(
            n_estimators=n_estimators,
            n_total_features=5001,  # needs 11 estimators for full coverage
            preprocessor_configs=[cfg],
        )
    assert result == n_estimators


def test_scale_n_estimators_for_feature_coverage__auto_scaling_disabled():
    """Deprecated auto_scale_n_estimators=False keeps "auto" at the default."""
    cfg = PreprocessorConfig("none", max_features_per_estimator=500)
    with pytest.warns(FutureWarning, match="auto_scale_n_estimators is deprecated"):
        result = scale_n_estimators_for_feature_coverage(
            n_estimators="auto",
            n_total_features=5001,
            preprocessor_configs=[cfg],
            auto_scale_n_estimators=False,
        )
    assert result == DEFAULT_N_ESTIMATORS


def test_scale_n_estimators_for_feature_coverage__auto_scaling_enabled_does_not_warn():
    """The default auto_scale_n_estimators=True emits no deprecation warning."""
    cfg = PreprocessorConfig("none", max_features_per_estimator=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = scale_n_estimators_for_feature_coverage(
            n_estimators="auto",
            n_total_features=10,
            preprocessor_configs=[cfg],
            auto_scale_n_estimators=True,
        )
    assert result == DEFAULT_N_ESTIMATORS


@skip_on_macos
def test___compute_feature_importance_order__classification():
    """Small datasets yield a single valid feature ranking."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 100, 10
    X = rng.standard_normal((n_samples, n_features))
    # Make feature 0 highly predictive
    y = (X[:, 0] > 0).astype(int)

    order = _compute_feature_importance_order(X=X, y=y, task_type="classifier", rng=rng)

    assert order.shape == (n_features,)
    assert set(order) == set(range(n_features)), "All feature indices must appear"
    assert order[0] == 0


@skip_on_macos
def test___compute_feature_importance_order__regression():
    """_compute_feature_importance_order works for regression tasks."""
    rng = np.random.default_rng(1)
    n_samples, n_features = 100, 8
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 2] * 3.0 + rng.standard_normal(n_samples) * 0.1

    order = _compute_feature_importance_order(X=X, y=y, task_type="regressor", rng=rng)

    assert order.shape == (n_features,)
    assert set(order) == set(range(n_features))
    assert order[0] == 2


@skip_on_macos
def test___compute_feature_importance_order__subsamples_large_datasets():
    """max_samples caps the number of rows used for fitting."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 5
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, n_samples)

    order = _compute_feature_importance_order(
        X=X,
        y=y,
        task_type="classifier",
        max_samples=50,
        rng=rng,
    )
    assert order.shape == (n_features,)
    assert set(order) == set(range(n_features))


def _spy_fit_ordering(
    rows_seen: list[int],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """fit_ordering_fn that records how many rows each fit was given."""

    def fit(X_fit: np.ndarray, _y: np.ndarray) -> np.ndarray:
        rows_seen.append(len(X_fit))
        return np.arange(X_fit.shape[1])

    return fit


def test___fit_importance_ordering__small_data_uses_every_row():
    rows_seen: list[int] = []
    order = _fit_importance_ordering(
        X=np.zeros((100, 6)),
        y=np.zeros(100),
        task_type="regressor",
        max_samples=200,
        fit_ordering_fn=_spy_fit_ordering(rows_seen),
        rng=np.random.default_rng(0),
    )

    assert rows_seen == [100]
    assert order.shape == (6,)


def test___fit_importance_ordering__large_data_fits_once_on_max_samples():
    """Above max_samples the rows are subsampled, but only one fit is run."""
    rows_seen: list[int] = []
    order = _fit_importance_ordering(
        X=np.zeros((500, 6)),
        y=np.zeros(500),
        task_type="regressor",
        max_samples=100,
        fit_ordering_fn=_spy_fit_ordering(rows_seen),
        rng=np.random.default_rng(0),
    )

    assert rows_seen == [100]
    assert order.shape == (6,)


@pytest.mark.parametrize("n_samples", [50, 500])
def test__subsample_features_importance_based__covers_all_features(n_samples):
    """Every feature reaches some estimator, on both sides of ``max_samples``."""
    n_features, top_k, size, n_estimators = 12, 4, 8, 2

    for seed in range(20):
        rng = np.random.default_rng(seed)
        order = _fit_importance_ordering(
            X=np.zeros((n_samples, n_features)),
            y=np.zeros(n_samples),
            task_type="regressor",
            max_samples=100,
            fit_ordering_fn=lambda _X, _y: np.arange(n_features),
            rng=rng,
        )
        subsampled = _subsample_features_importance_based(
            [size] * n_estimators, n_features, order, top_k, rng
        )

        # Combined budget covers all features: 2 * (8 - 4) top-up draws == the
        # 8 non-top features, so a shared pool guarantees full coverage.
        covered = set(np.concatenate(subsampled))
        assert covered == set(range(n_features))


def test__end_to_end__feature_importance_skipped_when_no_subsampling_needed():
    """No importance computation when every estimator sees all features."""
    from unittest.mock import patch  # noqa: PLC0415

    rng = np.random.default_rng(8)
    n_train, n_features = 40, 10
    n_estimators = 2
    max_features = n_features  # budget == total → no subsampling needed

    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(0, 2, n_train)

    feature_schema = FeatureSchema.from_only_categorical_indices([], n_features)
    configs = generate_classification_ensemble_configs(
        num_estimators=n_estimators,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[
            PreprocessorConfig(
                "none",
                categorical_name="numeric",
                max_features_per_estimator=max_features,
            ),
        ],
        class_shift_method=None,
        n_classes=2,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )

    with patch(
        "tabpfn.preprocessing.ensemble._compute_feature_importance_order"
    ) as mock_compute:
        TabPFNEnsemblePreprocessor(
            configs=configs,
            n_samples=n_train,
            feature_schema=feature_schema,
            random_state=0,
            n_preprocessing_jobs=1,
            feature_subsampling_method=FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
            importance_top_k_count=n_features,
            X_train=X_train,
            y_train=y_train,
            task_type="classifier",
        )
        mock_compute.assert_not_called()


def test__end_to_end__feature_importance_skipped_when_top_k_equals_all_features():
    """No importance computation when resolved top_k >= n_total_features (all features
    are 'important'), even if subsampling is needed.
    """
    from unittest.mock import patch  # noqa: PLC0415

    rng = np.random.default_rng(9)
    n_train, n_features = 40, 10
    n_estimators = 2
    max_features = n_features - 1  # subsampling IS needed

    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(0, 2, n_train)

    feature_schema = FeatureSchema.from_only_categorical_indices([], n_features)
    configs = generate_classification_ensemble_configs(
        num_estimators=n_estimators,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[
            PreprocessorConfig(
                "none",
                categorical_name="numeric",
                max_features_per_estimator=max_features,
            ),
        ],
        class_shift_method=None,
        n_classes=2,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )

    with patch(
        "tabpfn.preprocessing.ensemble._compute_feature_importance_order"
    ) as mock_compute:
        TabPFNEnsemblePreprocessor(
            configs=configs,
            n_samples=n_train,
            feature_schema=feature_schema,
            random_state=0,
            n_preprocessing_jobs=1,
            feature_subsampling_method=FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
            importance_top_k_count=n_features,  # top_k == n_features → skip LightGBM
            X_train=X_train,
            y_train=y_train,
            task_type="classifier",
        )
        mock_compute.assert_not_called()


@skip_on_macos
def test__end_to_end__feature_importance_subsampling():
    """End-to-end: TabPFNEnsemblePreprocessor with feature_importance subsampling."""
    rng = np.random.default_rng(7)
    n_train, n_features = 60, 30
    n_estimators = 4
    max_features = 15
    top_k = 5

    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(0, 2, n_train)

    feature_schema = FeatureSchema.from_only_categorical_indices([], n_features)

    configs = generate_classification_ensemble_configs(
        num_estimators=n_estimators,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[
            PreprocessorConfig(
                "none",
                categorical_name="numeric",
                max_features_per_estimator=max_features,
            ),
        ],
        class_shift_method=None,
        n_classes=2,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )

    preprocessor = TabPFNEnsemblePreprocessor(
        configs=configs,
        n_samples=n_train,
        feature_schema=feature_schema,
        random_state=0,
        n_preprocessing_jobs=1,
        feature_subsampling_method=FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
        importance_top_k_count=top_k,
        X_train=X_train,
        y_train=y_train,
        task_type="classifier",
    )

    members = preprocessor.fit_transform_ensemble_members(X_train, y_train)
    assert len(members) == n_estimators

    for member in members:
        assert member.feature_indices is not None
        assert len(member.feature_indices) <= max_features


@skip_on_macos
def test___compute_feature_importance_order__lightgbm():
    """LightGBM importance ranks the most predictive feature first."""
    rng = np.random.default_rng(2)
    n_samples, n_features = 200, 10
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 5] > 0).astype(int)

    order = _compute_feature_importance_order(
        X=X,
        y=y,
        task_type="classifier",
        rng=rng,
    )

    assert len(order) == n_features
    assert order[0] == 5

    # With categorical indices — no crash.
    order_cat = _compute_feature_importance_order(
        X=np.abs(X),  # non-negative for LightGBM categorical handling
        y=y,
        task_type="classifier",
        categorical_feature_indices=[0, 1],
        rng=rng,
    )
    assert len(order_cat) == n_features


@skip_on_macos
def test___compute_feature_importance_order__handles_nan():
    """Importance method must tolerate NaN values in X."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 150, 10
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)

    # Inject NaN: ~10% of values, spread across all columns.
    nan_mask = rng.random((n_samples, n_features)) < 0.1
    X[nan_mask] = np.nan

    order = _compute_feature_importance_order(
        X=X,
        y=y,
        task_type="classifier",
        rng=rng,
    )

    assert len(order) > 0
    assert not np.isnan(order).any()


def test__generate_regression_ensemble_configs__target_transforms_not_shared():
    """Members must not share a target_transform instance.

    The transform is fitted in place per member (`_transform_labels_one`), so a
    shared instance would hold only the last member's fitted state, corrupting
    the inverse transform of every other member's predictions at predict time
    whenever members see different training targets (e.g. row subsampling).
    """
    configs = generate_regression_ensemble_configs(
        num_estimators=8,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[
            PreprocessorConfig("none", categorical_name="numeric"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ],
        target_transforms=[None, PowerTransformer()],
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )

    transforms = [
        config.target_transform
        for config in configs
        if config.target_transform is not None
    ]
    assert len(transforms) == 4
    ids = {id(transform) for transform in transforms}
    assert len(ids) == len(transforms), (
        "Ensemble configs share target_transform instances; fitting one member "
        "would clobber the fitted state of the others."
    )
