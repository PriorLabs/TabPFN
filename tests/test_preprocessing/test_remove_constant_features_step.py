"""Tests for RemoveConstantFeaturesStep."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing import PreprocessingPipeline
from tabpfn.preprocessing.datamodel import ColumnMetadata, FeatureModality
from tabpfn.preprocessing.steps.remove_constant_features_step import (
    RemoveConstantFeaturesStep,
)


def test__remove_constant_features_step__drops_constant_numpy() -> None:
    """Remove constant columns for NumPy inputs."""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 5.0, 3.0],
            [1.0, 7.0, 4.0],
        ]
    )
    metadata = ColumnMetadata({FeatureModality.NUMERICAL: [0, 1, 2]})

    step = RemoveConstantFeaturesStep()
    result = step.fit_transform(X, metadata)  # type: ignore[arg-type]

    expected = np.array(
        [
            [2.0, 3.0],
            [5.0, 3.0],
            [7.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(result.X, expected)
    assert result.column_metadata.indices_for(FeatureModality.NUMERICAL) == [0, 1]
    assert result.X_added is None
    assert result.modality_added is None


def test__remove_constant_features_step__raises_when_all_constant() -> None:
    """Raise when all columns are constant."""
    X = np.ones((4, 2))
    metadata = ColumnMetadata.from_dict({FeatureModality.NUMERICAL: [0, 1]})

    step = RemoveConstantFeaturesStep()
    with pytest.raises(TabPFNValidationError, match="All features are constant"):
        step.fit_transform(X, metadata)


def test__remove_constant_features_step__drops_constant_torch() -> None:
    """Remove constant columns for torch inputs."""
    X = torch.tensor(
        [
            [2.0, 0.0, 5.0],
            [2.0, 1.0, 6.0],
            [2.0, 3.0, 6.0],
        ]
    )
    metadata = ColumnMetadata.from_dict({FeatureModality.NUMERICAL: [0, 1, 2]})

    step = RemoveConstantFeaturesStep()
    result = step.fit_transform(X, metadata)  # type: ignore[arg-type]

    expected = torch.tensor(
        [
            [0.0, 5.0],
            [1.0, 6.0],
            [3.0, 6.0],
        ]
    )
    assert isinstance(result.X, torch.Tensor)
    assert torch.equal(result.X, expected)
    assert result.column_metadata.indices_for(FeatureModality.NUMERICAL) == [0, 1]
    assert result.X_added is None
    assert result.modality_added is None


def test__pipeline__remove_constant_features_step() -> None:
    """Test that the pipeline correctly removes constant features."""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 5.0, 3.0],
            [1.0, 7.0, 4.0],
        ]
    )
    metadata = ColumnMetadata({FeatureModality.NUMERICAL: [0, 1, 2]})
    pipeline = PreprocessingPipeline([RemoveConstantFeaturesStep()])
    result = pipeline.fit_transform(X, metadata)  # type: ignore[arg-type]
    assert result.column_metadata.indices_for(FeatureModality.NUMERICAL) == [0, 1]
