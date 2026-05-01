from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tabpfn.preprocessing import generate_classification_ensemble_configs
from tabpfn.preprocessing.configs import (
    FeatureSubsamplingMethod,
    PreprocessorConfig,
    SVDSupplement,
)
from tabpfn.preprocessing.datamodel import Feature, FeatureModality
from tabpfn.preprocessing.ensemble import (
    TabPFNEnsemblePreprocessor,
    _apply_svd_supplement,
    _compute_svd_supplements,
    _get_subsample_feature_indices,
    _get_subsample_indices_for_estimators,
    _subsample_features_importance_based,
    compute_feature_importance_order,
)
from tabpfn.preprocessing.torch import FeatureSchema
from tabpfn.preprocessing.torch.torch_svd import TorchTruncatedSVD


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


# --- Feature importance subsampling tests ---


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
        importance_feature_orders=[importance_order],
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
        importance_feature_orders=[importance_order],
        top_k_count=5,
        rng=rng,
    )
    assert all(r is None for r in result)


def test__subsample_features_importance_based__top_k_float():
    """Float top_k_count is resolved relative to n_total_features."""
    rng = np.random.default_rng(0)
    n_features = 20
    importance_order = np.arange(n_features)  # feature 0 most important
    # 0.5 * 20 = 10 top features
    result = _subsample_features_importance_based(
        subsample_sizes=[15],
        n_total_features=n_features,
        importance_feature_orders=[importance_order],
        top_k_count=0.5,
        rng=rng,
    )
    assert result[0] is not None
    # Top 10 features (indices 0-9) must all be present
    assert set(range(10)).issubset(set(result[0]))
    assert len(result[0]) == 15


def test__subsample_features_importance_based__top_k_float_invalid():
    """Float top_k_count outside (0, 1] raises ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must be in"):
        _subsample_features_importance_based(
            subsample_sizes=[5],
            n_total_features=10,
            importance_feature_orders=[np.arange(10)],
            top_k_count=1.5,
            rng=rng,
        )


def test__subsample_features_importance_based__budget_less_than_top_k():
    """When budget < top_k, only the most important features are selected."""
    rng = np.random.default_rng(0)
    n_features = 20
    importance_order = np.arange(n_features)  # feature 0 is most important
    result = _subsample_features_importance_based(
        subsample_sizes=[3],
        n_total_features=n_features,
        importance_feature_orders=[importance_order],
        top_k_count=10,
        rng=rng,
    )
    assert result[0] is not None
    assert len(result[0]) == 3
    assert set(result[0]) == {0, 1, 2}


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
        importance_feature_orders=[importance_order],
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
        importance_feature_orders=None,
    )
    # Should return valid index arrays (balanced fallback), not raise
    assert len(result) == 3
    for indices in result:
        assert indices is not None
        assert len(indices) == 20


def test__compute_feature_importance_order__classification():
    """compute_feature_importance_order returns one valid feature ranking per tree."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 100, 10
    X = rng.standard_normal((n_samples, n_features))
    # Make feature 0 highly predictive
    y = (X[:, 0] > 0).astype(int)

    n_estimators = 4
    orders = compute_feature_importance_order(
        X=X, y=y, task_type="classifier", n_estimators=n_estimators, rng=rng
    )

    assert len(orders) == n_estimators
    for order in orders:
        assert order.shape == (n_features,)
        assert set(order) == set(range(n_features)), "All feature indices must appear"
    # Feature 0 should rank first (most important) in the majority of orderings
    assert sum(order[0] == 0 for order in orders) > len(orders) // 2


def test__compute_feature_importance_order__regression():
    """compute_feature_importance_order works for regression tasks."""
    rng = np.random.default_rng(1)
    n_samples, n_features = 100, 8
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 2] * 3.0 + rng.standard_normal(n_samples) * 0.1

    n_estimators = 4
    orders = compute_feature_importance_order(
        X=X, y=y, task_type="regressor", n_estimators=n_estimators, rng=rng
    )

    assert len(orders) == n_estimators
    for order in orders:
        assert order.shape == (n_features,)
        assert set(order) == set(range(n_features))
    assert sum(order[0] == 2 for order in orders) > len(orders) // 2


def test__compute_feature_importance_order__subsamples_large_datasets():
    """max_samples caps the number of rows used for fitting."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 5
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, n_samples)

    n_estimators = 4
    orders = compute_feature_importance_order(
        X=X,
        y=y,
        task_type="classifier",
        n_estimators=n_estimators,
        gini_max_samples=50,
        rng=rng,
    )
    assert len(orders) == n_estimators
    for order in orders:
        assert order.shape == (n_features,)
        assert set(order) == set(range(n_features))


def test__end_to_end__feature_importance_skipped_when_top_k_covers_all():
    """No importance computation when top_k >= n_total_features."""
    from unittest.mock import patch  # noqa: PLC0415

    rng = np.random.default_rng(8)
    n_train, n_features = 40, 10
    n_estimators = 2
    max_features = 8

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
        "tabpfn.preprocessing.ensemble.compute_feature_importance_order"
    ) as mock_compute:
        TabPFNEnsemblePreprocessor(
            configs=configs,
            n_samples=n_train,
            feature_schema=feature_schema,
            random_state=0,
            n_preprocessing_jobs=1,
            feature_subsampling_method=FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
            importance_top_k_count=n_features,  # covers all features → skip
            X_train=X_train,
            y_train=y_train,
            task_type="classifier",
        )
        mock_compute.assert_not_called()


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


def test__compute_permutation_importance_order__classification():
    """Permutation method returns a valid ranking with most predictive feature on top."""  # noqa: E501
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 8
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)

    n_estimators = 4
    orders = compute_feature_importance_order(
        X=X,
        y=y,
        task_type="classifier",
        method=FeatureSubsamplingMethod.PERMUTATION_FEATURE_IMPORTANCE,
        n_estimators=n_estimators,
        rng=rng,
    )

    assert len(orders) == n_estimators
    for order in orders:
        assert order.shape == (n_features,)
        assert set(order) == set(range(n_features))
    assert sum(order[0] == 0 for order in orders) > len(orders) // 2


def test__compute_permutation_importance_order__regression():
    """Permutation method works for regression tasks."""
    rng = np.random.default_rng(1)
    n_samples, n_features = 200, 6
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 3] * 5.0 + rng.standard_normal(n_samples) * 0.1

    n_estimators = 4
    orders = compute_feature_importance_order(
        X=X,
        y=y,
        task_type="regressor",
        method=FeatureSubsamplingMethod.PERMUTATION_FEATURE_IMPORTANCE,
        n_estimators=n_estimators,
        rng=rng,
    )

    assert len(orders) == n_estimators
    for order in orders:
        assert order.shape == (n_features,)
        assert set(order) == set(range(n_features))
    assert sum(order[0] == 3 for order in orders) > len(orders) // 2


def test__end_to_end__permutation_feature_importance_subsampling():
    """End-to-end: TabPFNEnsemblePreprocessor with permutation_feature_importance."""
    rng = np.random.default_rng(9)
    n_train, n_features = 80, 20
    n_estimators = 3
    max_features = 10
    top_k = 4

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
        feature_subsampling_method=FeatureSubsamplingMethod.PERMUTATION_FEATURE_IMPORTANCE,
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


def test__subsample_features_importance_based__different_orderings_yield_different_indices():  # noqa: E501
    """When multiple distinct orderings are given, different estimators get different top-K."""  # noqa: E501
    rng = np.random.default_rng(0)
    n_features = 20
    top_k = 5
    budget = 10

    # Two opposite orderings: first says features 0-4 are top, second says 15-19 are top
    order_a = np.arange(n_features)  # top features: 0,1,2,3,4
    order_b = np.arange(n_features)[::-1].copy()  # top features: 19,18,17,16,15

    result = _subsample_features_importance_based(
        subsample_sizes=[budget, budget],
        n_total_features=n_features,
        importance_feature_orders=[order_a, order_b],
        top_k_count=top_k,
        rng=rng,
    )

    assert result[0] is not None
    assert result[1] is not None
    top_k_a = set(order_a[:top_k])  # {0,1,2,3,4}
    top_k_b = set(order_b[:top_k])  # {15,16,17,18,19}
    # Estimator 0 must include all of order_a's top-K
    assert top_k_a.issubset(set(result[0]))
    # Estimator 1 must include all of order_b's top-K
    assert top_k_b.issubset(set(result[1]))
    # The two selections must differ (no overlap in guaranteed-included features)
    assert top_k_a.isdisjoint(top_k_b), (
        "Top-K sets must be disjoint for opposite orderings"
    )
    assert set(result[0]) != set(result[1]), (
        "Estimators should have different feature sets"
    )


def test__compute_feature_importance_order__gini_large_dataset_yields_diverse_orderings():  # noqa: E501
    """With data > gini_max_samples, independent subsamples produce diverse orderings."""  # noqa: E501
    from tabpfn.constants import GINI_FEATURE_IMPORTANCE_MAX_SAMPLES  # noqa: PLC0415

    rng = np.random.default_rng(42)
    n_samples = GINI_FEATURE_IMPORTANCE_MAX_SAMPLES + 100
    n_features = 10
    # Pure noise so each subsample fit produces a different ranking
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, n_samples)

    n_estimators = 6
    orders = compute_feature_importance_order(
        X=X,
        y=y,
        task_type="classifier",
        n_estimators=n_estimators,
        gini_max_samples=GINI_FEATURE_IMPORTANCE_MAX_SAMPLES,
        rng=rng,
    )

    assert len(orders) == n_estimators
    for order in orders:
        assert order.shape == (n_features,)
        assert set(order) == set(range(n_features))

    # With multiple independent subsamples on noisy data, not all orderings should
    # be identical
    unique_first_features = {order[0] for order in orders}
    assert len(unique_first_features) > 1, (
        "Independent subsamples on noise should produce diverse feature rankings"
    )


def test__compute_feature_importance_order__permutation_folds_yield_diverse_orderings():
    """Permutation method across multiple folds produces diverse per-estimator orderings."""  # noqa: E501
    rng = np.random.default_rng(7)
    n_samples, n_features = 300, 12
    # Pure noise — no single feature dominates, so fold-level rankings vary
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 3, n_samples)

    n_estimators = 6
    orders = compute_feature_importance_order(
        X=X,
        y=y,
        task_type="classifier",
        method=FeatureSubsamplingMethod.PERMUTATION_FEATURE_IMPORTANCE,
        n_estimators=n_estimators,
        rng=rng,
    )

    assert len(orders) == n_estimators
    for order in orders:
        assert order.shape == (n_features,)
        assert set(order) == set(range(n_features))

    # With multiple folds on noisy data, orderings should not all be identical
    unique_orderings = {tuple(o) for o in orders}
    assert len(unique_orderings) > 1, (
        "Permutation importance across folds should produce diverse feature orderings"
    )


# ── SVD supplement tests ──────────────────────────────────────────────────────


def test__compute_svd_supplements__basic():
    """_compute_svd_supplements returns one supplement per estimator."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 80, 20
    n_estimators = 3
    top_k = 5
    budget = 12  # top_k (5) + n_svd (7)

    X_train = rng.standard_normal((n_samples, n_features))
    y_train = rng.integers(0, 2, n_samples)

    orders = compute_feature_importance_order(
        X=X_train,
        y=y_train,
        task_type="classifier",
        n_estimators=n_estimators,
        rng=rng,
    )

    supplements = _compute_svd_supplements(
        X_train=X_train,
        importance_feature_orders=orders,
        top_k_count=top_k,
        subsample_sizes=[budget] * n_estimators,
    )

    assert len(supplements) == n_estimators
    for sup in supplements:
        assert isinstance(sup, SVDSupplement)
        assert len(sup.top_k_indices) == top_k
        assert len(sup.remaining_indices) == n_features - top_k
        assert sup.n_svd_components == budget - top_k
        assert "components" in sup.svd_cache
        assert sup.svd_cache["components"].shape == (
            sup.n_svd_components,
            len(sup.remaining_indices),
        )


def test__apply_svd_supplement__output_shape():
    """_apply_svd_supplement returns array with top_k + n_svd columns."""
    rng = np.random.default_rng(1)
    n_samples, n_features = 50, 15
    top_k, n_svd = 4, 6

    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    top_k_indices = np.arange(top_k)
    remaining_indices = np.arange(top_k, n_features)

    X_rem_t = torch.from_numpy(X[:, remaining_indices].astype(np.float32))
    svd_cache = TorchTruncatedSVD(n_components=n_svd).fit(X_rem_t)

    sup = SVDSupplement(
        top_k_indices=top_k_indices,
        remaining_indices=remaining_indices,
        n_svd_components=n_svd,
        svd_cache=svd_cache,
    )
    result = _apply_svd_supplement(X, sup)

    assert result.shape == (n_samples, top_k + n_svd)
    assert result.dtype == X.dtype  # dtype preserved


def test__apply_svd_supplement__no_svd_returns_top_k_only():
    """When n_svd_components == 0, only top_k columns are returned."""
    rng = np.random.default_rng(2)
    n_samples, n_features = 30, 10
    top_k = 4

    X = rng.standard_normal((n_samples, n_features))
    sup = SVDSupplement(
        top_k_indices=np.arange(top_k),
        remaining_indices=np.arange(top_k, n_features),
        n_svd_components=0,
        svd_cache={},
    )
    result = _apply_svd_supplement(X, sup)
    assert result.shape == (n_samples, top_k)
    np.testing.assert_array_equal(result, X[:, :top_k])


def test__end_to_end__gini_importance_and_svd():
    """End-to-end: TabPFNEnsemblePreprocessor with gini_feature_importance_and_svd."""
    rng = np.random.default_rng(42)
    n_train, n_features = 80, 30
    n_estimators = 3
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
        feature_subsampling_method=FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE_AND_SVD,
        importance_top_k_count=top_k,
        X_train=X_train,
        y_train=y_train,
        task_type="classifier",
    )

    assert preprocessor.svd_supplements is not None
    assert len(preprocessor.svd_supplements) == n_estimators
    assert preprocessor.subsample_feature_indices == [None] * n_estimators

    members = preprocessor.fit_transform_ensemble_members(X_train, y_train)
    assert len(members) == n_estimators

    for _i, member in enumerate(members):
        assert member.svd_supplement is not None
        assert member.feature_indices is None
        # Test data transformation uses the SVD supplement
        X_test = rng.standard_normal((10, n_features))
        X_transformed = member.transform_X_test(X_test)
        assert X_transformed is not None
