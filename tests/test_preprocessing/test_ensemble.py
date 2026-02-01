from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from tabpfn.preprocessing.datamodel import Feature, FeatureModality
from tabpfn.preprocessing.ensemble import (
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
    )

    assert result[0] is not None
    assert len(result[0]) == 60
    assert all(0 <= idx < 100 for idx in result[0])

    assert result[1] is not None
    assert len(result[1]) == 40
    assert all(0 <= idx < 100 for idx in result[1])

    # Assert that each feature is present in at least one of the two estimators.
    assert set(result[0]) | set(result[1]) == set(range(100))
