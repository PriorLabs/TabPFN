from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tabpfn.preprocessing import generate_classification_ensemble_configs
from tabpfn.preprocessing.configs import FeatureSubsamplingMethod, PreprocessorConfig
from tabpfn.preprocessing.datamodel import Feature, FeatureModality
from tabpfn.preprocessing.ensemble import (
    TabPFNEnsemblePreprocessor,
    _get_subsample_feature_indices,
    _get_subsample_indices_for_estimators,
)
from tabpfn.preprocessing.torch import FeatureSchema


def _get_schema(n_features: int) -> FeatureSchema:
    features = [
        Feature(name=None, modality=FeatureModality.NUMERICAL)
        for _ in range(n_features)
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
        subsample_indices, expected_subsample_indices
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


def test__get_subsample_feature_indices__no_subsampling_all_methods():
    """Test that all methods return None when no subsampling is needed."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    for method in FeatureSubsamplingMethod:
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


def test__get_subsample_feature_indices__invalid_method():
    """Test that an invalid method raises ValueError."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0
    pipeline.has_data_dependent_feature_expansion.return_value = False

    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="Unknown feature subsampling method"):
        _get_subsample_feature_indices(
            pipelines=[pipeline],
            n_samples=100,
            feature_schema=_get_schema(n_features=100),
            max_features_per_estimator=[80],
            rng=rng,
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
    for a, b in zip(result_a, result_b):
        np.testing.assert_array_equal(a, b)

    # Different seed -> different output.
    result_c = _get_subsample_feature_indices(rng=np.random.default_rng(99), **kwargs)
    any_different = any(
        not np.array_equal(a, c)
        for a, c in zip(result_a, result_c)
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
