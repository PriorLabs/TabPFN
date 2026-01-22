"""Tests for TorchPreprocessingPipeline."""

from __future__ import annotations

from typing_extensions import override

import pytest
import torch

from tabpfn.preprocessing.torch import TorchRemoveOutliersStep
from tabpfn.preprocessing.torch.datamodel import (
    ColumnMetadata,
    FeatureModality,
)
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
)
from tabpfn.preprocessing.torch.steps import TorchStandardScalerStep


class MockStep(TorchPreprocessingStep):
    """Mock step that multiplies selected columns by a factor."""

    def __init__(self, factor: float = 2.0) -> None:
        """Initialize with multiplication factor."""
        super().__init__()
        self.factor = factor
        self.fit_called = False
        self.fitted_columns: list[int] = []

    @override
    def fit(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
    ) -> None:
        """Track that fit was called, then delegate to base class."""
        self.fit_called = True
        self.fitted_columns = column_indices
        super().fit(x, column_indices, num_train_rows)

    @override
    def _fit(self, x: torch.Tensor) -> None:
        """No-op fit for mock."""

    @override
    def _transform(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Multiply columns by factor."""
        return x * self.factor, None, None


def test__call__single_step_transforms_columns():
    """Test pipeline with a single step transforms the correct columns."""
    step = MockStep(factor=3.0)
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = ColumnMetadata(
        indices_by_modality={
            FeatureModality.NUMERICAL: [0, 2],
            FeatureModality.CATEGORICAL: [1],
        },
    )
    x = torch.ones(10, 1, 3)

    result = pipeline(x, metadata, num_train_rows=5)

    # Columns 0 and 2 should be multiplied by 3
    assert torch.allclose(result.x[:, :, 0], torch.full((10, 1), 3.0))
    assert torch.allclose(result.x[:, :, 2], torch.full((10, 1), 3.0))
    # Column 1 should be unchanged
    assert torch.allclose(result.x[:, :, 1], torch.ones(10, 1))
    assert step.fit_called


def test__call__multiple_steps_applied_sequentially():
    """Test that multiple steps are applied in order."""
    step1 = MockStep(factor=2.0)
    step2 = MockStep(factor=3.0)
    pipeline = TorchPreprocessingPipeline(
        steps=[
            (step1, {FeatureModality.NUMERICAL}),
            (step2, {FeatureModality.NUMERICAL}),
        ]
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    result = pipeline(x, metadata, num_train_rows=5)

    # Value should be 1 * 2 * 3 = 6
    assert torch.allclose(result.x, torch.full((10, 1, 1), 6.0))


def test__call__step_skipped_for_empty_indices():
    """Test that steps with no matching columns are skipped."""
    step = MockStep(factor=2.0)
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.TEXT})]  # No TEXT columns in metadata
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0, 1]},
    )
    x = torch.ones(10, 1, 2)

    result = pipeline(x, metadata, num_train_rows=5)

    # Data should be unchanged since step was skipped
    assert torch.allclose(result.x, x)
    assert not step.fit_called


def test__call__2d_input_adds_and_removes_batch_dimension():
    """Test that 2D input gets batch dimension added then removed."""
    step = MockStep(factor=2.0)
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1)  # 2D input

    result = pipeline(x, metadata, num_train_rows=5)

    # Output should also be 2D
    assert result.x.shape == (10, 1)
    assert torch.allclose(result.x, torch.full((10, 1), 2.0))


def test__call__step_targeting_multiple_modalities():
    """Test step that targets multiple modalities at once."""
    step = MockStep(factor=5.0)
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.NUMERICAL, FeatureModality.CATEGORICAL})]
    )
    metadata = ColumnMetadata(
        indices_by_modality={
            FeatureModality.NUMERICAL: [0],
            FeatureModality.CATEGORICAL: [1],
            FeatureModality.TEXT: [2],
        },
    )
    x = torch.ones(10, 1, 3)

    result = pipeline(x, metadata, num_train_rows=5)

    # Columns 0 and 1 should be transformed, column 2 unchanged
    assert torch.allclose(result.x[:, :, 0], torch.full((10, 1), 5.0))
    assert torch.allclose(result.x[:, :, 1], torch.full((10, 1), 5.0))
    assert torch.allclose(result.x[:, :, 2], torch.ones(10, 1))
    assert sorted(step.fitted_columns) == [0, 1]


def test__call__with_real_standard_scaler_step():
    """Test pipeline with a real TorchStandardScalerStep."""
    step = TorchStandardScalerStep()
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0, 1]},
    )
    x = torch.randn(100, 1, 2) * 10 + 5  # Mean ~5, std ~10

    result = pipeline(x, metadata, num_train_rows=80)

    # Training portion should have mean ~0 and std ~1
    train_output = result.x[:80, :, :]
    mean = train_output.mean(dim=0)
    std = train_output.std(dim=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=0.2)


def test__call__no_num_train_rows_fits_on_all_data():
    """Test that when num_train_rows is None, fit uses all data."""
    step = TorchStandardScalerStep()
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    result = pipeline(x, metadata, num_train_rows=None)

    # Output should be zeros: (x - mean) / std = (1 - 1) / 1 = 0
    assert torch.allclose(result.x, torch.zeros((10, 1)))
    # Mean should be 1.0 (all inputs were 1)
    assert step._scaler.mean_ is not None
    assert torch.allclose(step._scaler.mean_, torch.tensor([[1.0]]))
    # Std is set to 1.0 for constant features (to avoid division by zero)
    assert step._scaler.std_ is not None
    assert torch.allclose(step._scaler.std_, torch.tensor([[1.0]]))


def test__call__zero_num_train_rows():
    """Test that fit is skipped when num_train_rows is None."""
    step = TorchRemoveOutliersStep()
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    result = pipeline(x, metadata, num_train_rows=0)

    assert torch.allclose(result.x, x)


def test__call__mismatching_num_columns_raises_error():
    """Test that mismatching num_columns raises an error."""
    step = MockStep(factor=2.0)
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 2)
    with pytest.raises(ValueError, match="Number of columns in input tensor"):
        pipeline(x, metadata, num_train_rows=5)
