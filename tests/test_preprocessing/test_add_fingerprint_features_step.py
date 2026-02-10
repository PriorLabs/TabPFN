"""Tests for AddFingerprintFeaturesStep."""

from __future__ import annotations

import numpy as np
import torch

from tabpfn.preprocessing import PreprocessingPipeline
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.add_fingerprint_features_step import (
    AddFingerprintFeaturesStep,
)


def _get_schema(num_columns: int) -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(num_columns)
        ]
    )


def test__transform__returns_x_unchanged_numpy() -> None:
    """Test that _transform returns X unchanged, fingerprint in added_columns."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    step = AddFingerprintFeaturesStep(random_state=42)
    step._fit(data, _get_schema(num_columns=3))
    result, added_cols, modality = step._transform(data)

    # X should be returned unchanged
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)  # Same shape as input
    np.testing.assert_array_equal(result, data)

    # Fingerprint should be available via _get_added_columns
    assert added_cols is not None
    assert added_cols.shape == (2, 1)
    assert modality == FeatureModality.NUMERICAL


def test__transform__returns_x_unchanged_torch() -> None:
    """Test that _transform returns torch tensor unchanged, fingerprint separate."""
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    step = AddFingerprintFeaturesStep(random_state=42)
    step._fit(data.numpy(), _get_schema(num_columns=3))
    result, added_cols, _ = step._transform(data)

    # X should be returned unchanged
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3)

    # Fingerprint should be a torch tensor
    assert isinstance(added_cols, torch.Tensor)
    assert added_cols.shape == (2, 1)


def test__transform__collision_handling_with_duplicate_rows() -> None:
    """Test that duplicate rows get unique fingerprints only when is_test=False."""
    data = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    schema = _get_schema(num_columns=2)
    step = AddFingerprintFeaturesStep(random_state=42)
    step._fit(data, schema)

    # is_test=False: collision handling ensures unique fingerprints
    _, fingerprints_train, _ = step._transform(data, is_test=False)
    assert fingerprints_train is not None
    assert len(np.unique(fingerprints_train)) == 3

    # is_test=True: duplicate rows share the same fingerprint
    _, fingerprints_test, _ = step._transform(data, is_test=True)
    assert fingerprints_test is not None
    assert fingerprints_test[0] == fingerprints_test[1]
    assert fingerprints_test[0] != fingerprints_test[2]


def test__fit_transform__returns_added_columns() -> None:
    """Test fit_transform returns X unchanged with fingerprint in added_columns."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    schema = _get_schema(num_columns=3)

    step = AddFingerprintFeaturesStep(random_state=42)
    result = step.fit_transform(data, schema)

    # X should be unchanged
    assert result.X.shape == (2, 3)
    np.testing.assert_array_equal(result.X, data)

    # Metadata should be unchanged (pipeline handles adding fingerprint)
    assert result.feature_schema.num_columns == 3

    # Fingerprint should be in added_columns
    assert result.X_added is not None
    assert result.X_added.shape == (2, 1)
    assert result.modality_added == FeatureModality.NUMERICAL


def test__transform__returns_added_columns() -> None:
    """Test transform returns X unchanged with fingerprint in added_columns."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    schema = _get_schema(num_columns=3)

    step = AddFingerprintFeaturesStep(random_state=42)
    result = step.fit_transform(data, schema)

    # X should be unchanged
    assert result.X.shape == (2, 3)

    # Fingerprint should be in added_columns
    assert result.X_added is not None
    assert result.X_added.shape == (2, 1)
    assert result.modality_added == FeatureModality.NUMERICAL


def test__transform__no_infinite_loop_with_inf_and_large_floats() -> None:
    """Regression test: rows with inf/large floats must not cause an infinite loop.

    Previously, collision resolution added the counter to float values
    (row + add_to_hash), which is a no-op for inf and floats >= ~1e16 due to
    float64 precision limits, causing the hash to never change.
    """
    data = np.array(
        [
            [np.inf, np.inf],
            [np.inf, np.inf],
            [1e20, 1e20],
            [1e20, 1e20],
            [np.nan, np.inf],
            [np.nan, np.inf],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    schema = _get_schema(num_columns=2)
    step = AddFingerprintFeaturesStep(random_state=42)
    step._fit(data, schema)

    # This must terminate (previously hung forever)
    _, fingerprints, _ = step._transform(data, is_test=False)

    assert fingerprints.shape == (8, 1)
    # All 8 rows should get unique fingerprints (including all-NaN rows,
    # which get different hashes via the hash_counter mechanism)
    assert len(np.unique(fingerprints)) == 8


def test__fit__does_not_modify_metadata() -> None:
    """Test that _fit returns metadata unchanged (pipeline handles added cols)."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    schema = _get_schema(num_columns=3)

    step = AddFingerprintFeaturesStep(random_state=42)
    result_schema = step._fit(data, schema)

    # Metadata should be unchanged - same number of columns
    assert result_schema.num_columns == 3
    assert result_schema.indices_for(FeatureModality.NUMERICAL) == [0, 1, 2]


def test__in_pipeline__returns_added_columns() -> None:
    """Test that the step returns added columns when used in a pipeline."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    schema = _get_schema(num_columns=3)

    step = AddFingerprintFeaturesStep(random_state=42)
    pipeline = PreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    result = pipeline.fit_transform(data, schema)

    assert result.feature_schema.num_columns == 4
    assert result.X.shape == (2, 4)
