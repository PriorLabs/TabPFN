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

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """No-op fit for mock."""
        return {}

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Multiply columns by factor."""
        return x * self.factor, None, None


class MockStepWithCache(TorchPreprocessingStep):
    """Mock step that tracks fit calls and uses a cached mean for subtraction."""

    def __init__(self) -> None:
        """Initialize with fit call counter."""
        super().__init__()
        self.fit_call_count = 0

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute and return the mean as cache."""
        self.fit_call_count += 1
        return {"mean": x.mean(dim=0, keepdim=True)}

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Subtract the cached mean from x."""
        return x - fitted_cache["mean"], None, None


def test__call__single_step_transforms_columns():
    """Test pipeline with a single step transforms the correct columns."""
    step = MockStep(factor=3.0)
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])

    x = torch.ones(10, 1, 3)
    metadata = ColumnMetadata(
        indices_by_modality={
            FeatureModality.NUMERICAL: [0, 2],
            FeatureModality.CATEGORICAL: [1],
        },
    )

    result = pipeline(x, metadata, num_train_rows=5)

    # Columns 0 and 2 should be multiplied by 3
    assert torch.allclose(result.x[:, :, 0], torch.full((10, 1), 3.0))
    assert torch.allclose(result.x[:, :, 2], torch.full((10, 1), 3.0))
    # Column 1 should be unchanged
    assert torch.allclose(result.x[:, :, 1], torch.ones(10, 1))


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

    assert torch.allclose(result.x, x)


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
    pipeline = TorchPreprocessingPipeline(
        steps=[
            (
                MockStep(factor=5.0),
                {FeatureModality.NUMERICAL, FeatureModality.CATEGORICAL},
            )
        ]
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


def test__call__keep_fitted_cache_false_does_not_store_cache():
    """Test that cache is not stored when keep_fitted_cache=False."""
    step = MockStepWithCache()
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.NUMERICAL})],
        keep_fitted_cache=False,
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    pipeline(x, metadata, num_train_rows=5)

    assert pipeline.fitted_cache[0] is None


def test__call__keep_fitted_cache_true_stores_cache():
    """Test that cache is stored when keep_fitted_cache=True."""
    step = MockStepWithCache()
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.NUMERICAL})],
        keep_fitted_cache=True,
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    pipeline(x, metadata, num_train_rows=5)

    assert pipeline.fitted_cache[0] is not None
    assert "mean" in pipeline.fitted_cache[0]


def test__call__use_fitted_cache_true_skips_fit():
    """Test that fit is skipped when use_fitted_cache=True and cache exists."""
    step = MockStepWithCache()
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.NUMERICAL})],
        keep_fitted_cache=True,
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    # First call: fit and store cache
    pipeline(x, metadata, num_train_rows=5)
    assert step.fit_call_count == 1

    # Second call with use_fitted_cache=True: should skip fit
    pipeline(x, metadata, num_train_rows=5, use_fitted_cache=True)
    assert step.fit_call_count == 1  # Still 1, fit was not called again


def test__call__use_fitted_cache_false_refits_even_with_cache():
    """Test that fit is called when use_fitted_cache=False even if cache exists."""
    step = MockStepWithCache()
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.NUMERICAL})],
        keep_fitted_cache=True,
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    # First call: fit and store cache
    pipeline(x, metadata, num_train_rows=5)
    assert step.fit_call_count == 1

    # Second call with use_fitted_cache=False: should refit
    pipeline(x, metadata, num_train_rows=5, use_fitted_cache=False)
    assert step.fit_call_count == 2


def test__call__use_fitted_cache_true_without_stored_cache_refits():
    """Test that fit is called when use_fitted_cache=True but no cache exists."""
    step = MockStepWithCache()
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.NUMERICAL})],
        keep_fitted_cache=False,  # Cache won't be stored
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    with pytest.raises(ValueError, match="only supported if keep_fitted_cache=True"):
        pipeline(x, metadata, num_train_rows=5, use_fitted_cache=True)


def test__call__cached_values_are_used_correctly():
    """Test that cached values produce consistent transforms."""
    step = MockStepWithCache()
    pipeline = TorchPreprocessingPipeline(
        steps=[(step, {FeatureModality.NUMERICAL})],
        keep_fitted_cache=True,
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )

    # First call with data that has mean=5 in training rows
    x1 = torch.full((10, 1, 1), 5.0)
    result1 = pipeline(x1, metadata, num_train_rows=10)
    # After subtracting mean=5, result should be 0
    assert torch.allclose(result1.x, torch.zeros(10, 1, 1))

    # Second call with different data but use cached mean=5
    x2 = torch.full((10, 1, 1), 8.0)
    result2 = pipeline(x2, metadata, num_train_rows=10, use_fitted_cache=True)
    # After subtracting cached mean=5, result should be 3
    assert torch.allclose(result2.x, torch.full((10, 1, 1), 3.0))


def test__call__multiple_steps_cache_stored_independently():
    """Test that cache is stored independently for each step."""
    step1 = MockStepWithCache()
    step2 = MockStepWithCache()
    pipeline = TorchPreprocessingPipeline(
        steps=[
            (step1, {FeatureModality.NUMERICAL}),
            (step2, {FeatureModality.NUMERICAL}),
        ],
        keep_fitted_cache=True,
    )
    metadata = ColumnMetadata(
        indices_by_modality={FeatureModality.NUMERICAL: [0]},
    )
    x = torch.ones(10, 1, 1)

    pipeline(x, metadata, num_train_rows=5)

    # Both steps should have their cache stored
    assert pipeline.fitted_cache[0] is not None
    assert pipeline.fitted_cache[1] is not None
    assert pipeline.fitted_cache[0] != pipeline.fitted_cache[1]
    assert step1.fit_call_count == 1
    assert step2.fit_call_count == 1

    # Second call with use_fitted_cache=True: neither step should refit
    pipeline(x, metadata, num_train_rows=5, use_fitted_cache=True)
    assert step1.fit_call_count == 1
    assert step2.fit_call_count == 1
