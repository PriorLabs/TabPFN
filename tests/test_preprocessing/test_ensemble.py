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
    kwargs = {
        "num_estimators": 3,
        "max_index": 5,
        "static_seed": 42,
    }

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
        **kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index, expected_subsample_index in zip(
        subsample_indices, expected_subsample_indices
    ):
        assert subsample_index is not None
        assert (subsample_index == expected_subsample_index).all()

    subsample_samples = 0.5
    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_samples,
        **kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index in subsample_indices:
        assert subsample_index is not None
        assert len(subsample_index) == 3  # (max_index + 1) * 0.5

    subsample_samples = 2
    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_samples,
        **kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index in subsample_indices:
        assert subsample_index is not None
        assert len(subsample_index) == 2


def test__get_subsample_feature_indices__no_subsampling_needed():
    """Test that None is returned when features fit within the limit."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=10),
        max_features_per_estimator=15,
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

    pipeline2 = MagicMock()
    pipeline2.num_added_features.return_value = 40  # Adds 2 features

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline2],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=80,  # With 2 added features, need to subsample to 6
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
        subsample_samples=None,
        max_index=2,
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
        feature_schema=feature_schema,
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

    pipeline2 = MagicMock()
    pipeline2.num_added_features.return_value = 40

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline2],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=80,
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

    rng = np.random.default_rng(42)
    constant_count = 30
    result = _get_subsample_feature_indices(
        pipelines=[pipeline, pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=80,
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

    rng = np.random.default_rng(42)
    result = _get_subsample_feature_indices(
        pipelines=[pipeline],
        n_samples=100,
        feature_schema=_get_schema(n_features=100),
        max_features_per_estimator=30,
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

    for method in FeatureSubsamplingMethod:
        rng = np.random.default_rng(42)
        result = _get_subsample_feature_indices(
            pipelines=[pipeline],
            n_samples=100,
            feature_schema=_get_schema(n_features=10),
            max_features_per_estimator=15,
            rng=rng,
            feature_subsampling_method=method,
        )
        assert result[0] is None, f"Expected None for method={method}"


def test__get_subsample_feature_indices__invalid_method():
    """Test that an invalid method raises ValueError."""
    pipeline = MagicMock()
    pipeline.num_added_features.return_value = 0

    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="Unknown feature subsampling method"):
        _get_subsample_feature_indices(
            pipelines=[pipeline],
            n_samples=100,
            feature_schema=_get_schema(n_features=100),
            max_features_per_estimator=80,
            rng=rng,
            feature_subsampling_method="nonexistent",  # type: ignore
        )
