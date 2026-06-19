#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for RemoveConstantFeaturesStep."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing import PreprocessingPipeline
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.remove_constant_features_step import (
    RemoveConstantFeaturesStep,
)


def _numerical_metadata(num_features: int) -> FeatureSchema:
    """Create FeatureSchema with numerical features only."""
    return FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(num_features)
        ]
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
    schema = _numerical_metadata(num_features=3)

    step = RemoveConstantFeaturesStep()
    result = step.fit_transform(X, schema)

    expected = np.array(
        [
            [2.0, 3.0],
            [5.0, 3.0],
            [7.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(result.X, expected)
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == [0, 1]
    assert result.X_added is None
    assert result.modality_added is None


def test__remove_constant_features_step__raises_when_all_constant() -> None:
    """Raise when all columns are constant."""
    X = np.ones((4, 2))
    schema = _numerical_metadata(num_features=2)

    step = RemoveConstantFeaturesStep()
    with pytest.raises(TabPFNValidationError, match="All features are constant"):
        step.fit_transform(X, schema)


def test__remove_constant_features_step__drops_constant_torch() -> None:
    """Remove constant columns for torch inputs."""
    X = torch.tensor(
        [
            [2.0, 0.0, 5.0],
            [2.0, 1.0, 6.0],
            [2.0, 3.0, 6.0],
        ]
    )
    schema = _numerical_metadata(num_features=3)

    step = RemoveConstantFeaturesStep()
    result = step.fit_transform(X, schema)  # type: ignore[arg-type]

    expected = torch.tensor(
        [
            [0.0, 5.0],
            [1.0, 6.0],
            [3.0, 6.0],
        ]
    )
    assert isinstance(result.X, torch.Tensor)
    assert torch.equal(result.X, expected)
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == [0, 1]
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
    schema = _numerical_metadata(num_features=3)
    pipeline = PreprocessingPipeline([RemoveConstantFeaturesStep()])
    result = pipeline.fit_transform(X, schema)
    assert result.feature_schema.indices_for(FeatureModality.NUMERICAL) == [0, 1]


def test__pipeline__keeps_non_constant_inf_flagged_column_numpy() -> None:
    """A column flagged ``non_constant_inf`` is kept even when it looks constant.

    The flag (set for passthrough_inf columns carrying >1 distinct non-finite
    value) overrides the constant check, so a finite-constant but flagged column
    survives while an unflagged constant column is dropped.
    """
    X = np.array(
        [
            [5.0, 9.0, 1.0],
            [5.0, 9.0, 2.0],
            [5.0, 9.0, 3.0],
        ]
    )
    schema = FeatureSchema(
        features=[
            Feature(
                name="c0", modality=FeatureModality.NUMERICAL, non_constant_inf=True
            ),
            Feature(name="c1", modality=FeatureModality.NUMERICAL),
            Feature(name="c2", modality=FeatureModality.NUMERICAL),
        ]
    )

    result = PreprocessingPipeline([RemoveConstantFeaturesStep()]).fit_transform(
        X, schema
    )

    # c0 kept by the flag, c1 dropped (constant), c2 kept (varies).
    assert [f.name for f in result.feature_schema.features] == ["c0", "c2"]
    np.testing.assert_array_equal(result.X, X[:, [0, 2]])


def test__pipeline__keeps_non_constant_inf_flagged_column_torch() -> None:
    """The flag is honoured for torch inputs too (covers the tensor OR branch)."""
    X = torch.tensor(
        [
            [5.0, 9.0, 1.0],
            [5.0, 9.0, 2.0],
            [5.0, 9.0, 3.0],
        ]
    )
    schema = FeatureSchema(
        features=[
            Feature(
                name="c0", modality=FeatureModality.NUMERICAL, non_constant_inf=True
            ),
            Feature(name="c1", modality=FeatureModality.NUMERICAL),
            Feature(name="c2", modality=FeatureModality.NUMERICAL),
        ]
    )

    result = PreprocessingPipeline([RemoveConstantFeaturesStep()]).fit_transform(
        X,  # type: ignore[arg-type]
        schema,
    )

    assert [f.name for f in result.feature_schema.features] == ["c0", "c2"]
    assert isinstance(result.X, torch.Tensor)
    assert torch.equal(result.X, X[:, [0, 2]])


def test__pipeline__keeps_column_with_multiple_infinities() -> None:
    """End-to-end: a column carrying +inf and -inf survives constant removal.

    The pipeline NaN's the infinities before the step runs, which would make an
    all-inf column look constant. ``_flag_non_constant_infs`` flags it (it has
    >1 distinct non-finite value), so the step keeps it; a genuinely constant
    finite column is still dropped, and the infinities are restored on the way
    out.
    """
    inf = np.inf
    X = np.array(
        [
            [inf, 9.0, 1.0],
            [-inf, 9.0, 2.0],
            [inf, 9.0, 3.0],
        ]
    )
    # The inf-mask helpers key columns by name, so use unique names (as real
    # schemas do) rather than the ``name=None`` test helper.
    schema = FeatureSchema(
        features=[
            Feature(name=f"input_f{i}", modality=FeatureModality.NUMERICAL)
            for i in range(3)
        ]
    )

    result = PreprocessingPipeline([RemoveConstantFeaturesStep()]).fit_transform(
        X.copy(), schema
    )

    # The constant finite column (index 1) is dropped; the inf-carrying and the
    # varying columns are kept.
    assert result.X.shape == (3, 2)
    # The kept inf column was flagged as non-constant by the helper ...
    assert result.feature_schema.features[0].non_constant_inf is True
    # ... and its infinities round-trip back into the output.
    np.testing.assert_array_equal(result.X[:, 0], np.array([inf, -inf, inf]))
    np.testing.assert_array_equal(result.X[:, 1], np.array([1.0, 2.0, 3.0]))
