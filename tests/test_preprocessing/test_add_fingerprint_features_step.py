"""Tests for AddFingerprintFeaturesStep."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.preprocessing.datamodel import ColumnMetadata, FeatureModality
from tabpfn.preprocessing.steps.add_fingerprint_features_step import (
    AddFingerprintFeaturesStep,
)


@pytest.fixture
def sample_metadata() -> ColumnMetadata:
    """Provides sample column metadata with numerical features."""
    return ColumnMetadata.from_dict({FeatureModality.NUMERICAL: [0, 1, 2]})


def test__transform__returns_x_unchanged_numpy(sample_metadata: ColumnMetadata) -> None:
    """Test that _transform returns X unchanged, fingerprint in added_columns."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    step = AddFingerprintFeaturesStep(random_state=42)
    step._fit(data, sample_metadata)
    result = step._transform(data)

    # X should be returned unchanged
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)  # Same shape as input
    np.testing.assert_array_equal(result, data)

    # Fingerprint should be available via _get_added_columns
    added_cols, modality = step._get_added_columns()
    assert added_cols is not None
    assert added_cols.shape == (2, 1)
    assert modality == FeatureModality.NUMERICAL


def test__transform__returns_x_unchanged_torch(sample_metadata: ColumnMetadata) -> None:
    """Test that _transform returns torch tensor unchanged, fingerprint separate."""
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    step = AddFingerprintFeaturesStep(random_state=42)
    step._fit(data.numpy(), sample_metadata)
    result = step._transform(data)

    # X should be returned unchanged
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3)

    # Fingerprint should be a torch tensor
    added_cols, _modality = step._get_added_columns()
    assert isinstance(added_cols, torch.Tensor)
    assert added_cols.shape == (2, 1)


def test__transform__collision_handling_with_duplicate_rows() -> None:
    """Test that duplicate rows get unique fingerprints only when is_test=False."""
    data = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    metadata = ColumnMetadata.from_dict({FeatureModality.NUMERICAL: [0, 1]})
    step = AddFingerprintFeaturesStep(random_state=42)
    step._fit(data, metadata)

    # is_test=False: collision handling ensures unique fingerprints
    step._transform(data, is_test=False)
    fingerprints_train, _ = step._get_added_columns()
    assert fingerprints_train is not None
    assert len(np.unique(fingerprints_train)) == 3

    # is_test=True: duplicate rows share the same fingerprint
    step._transform(data, is_test=True)
    fingerprints_test, _ = step._get_added_columns()
    assert fingerprints_test is not None
    assert fingerprints_test[0] == fingerprints_test[1]
    assert fingerprints_test[0] != fingerprints_test[2]


def test__fit_transform__returns_added_columns() -> None:
    """Test fit_transform returns X unchanged with fingerprint in added_columns."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    metadata = ColumnMetadata.from_dict({FeatureModality.NUMERICAL: [0, 1, 2]})

    step = AddFingerprintFeaturesStep(random_state=42)
    result = step.fit_transform(data, metadata)

    # X should be unchanged
    assert result.X.shape == (2, 3)
    np.testing.assert_array_equal(result.X, data)

    # Metadata should be unchanged (pipeline handles adding fingerprint)
    assert result.metadata.num_columns == 3

    # Fingerprint should be in added_columns
    assert result.added_columns is not None
    assert result.added_columns.shape == (2, 1)
    assert result.added_modality == FeatureModality.NUMERICAL


def test__transform__returns_added_columns() -> None:
    """Test transform returns X unchanged with fingerprint in added_columns."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    metadata = ColumnMetadata.from_dict({FeatureModality.NUMERICAL: [0, 1, 2]})

    step = AddFingerprintFeaturesStep(random_state=42)
    step.fit(data, metadata)
    result = step.transform(data)

    # X should be unchanged
    assert result.X.shape == (2, 3)

    # Fingerprint should be in added_columns
    assert result.added_columns is not None
    assert result.added_columns.shape == (2, 1)
    assert result.added_modality == FeatureModality.NUMERICAL


def test__fit__does_not_modify_metadata() -> None:
    """Test that _fit returns metadata unchanged (pipeline handles added cols)."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    metadata = ColumnMetadata.from_dict({FeatureModality.NUMERICAL: [0, 1, 2]})

    step = AddFingerprintFeaturesStep(random_state=42)
    result_metadata = step._fit(data, metadata)

    # Metadata should be unchanged - same number of columns
    assert result_metadata.num_columns == 3
    assert result_metadata.indices_for(FeatureModality.NUMERICAL) == [0, 1, 2]
