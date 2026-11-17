#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for TorchPreprocessingPipeline."""

from __future__ import annotations

import typing
from typing_extensions import override

import pytest
import torch

from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.torch import TorchSoftClipOutliersStep, pipeline_interface
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
    TorchPreprocessingStepResult,
)
from tabpfn.preprocessing.torch.steps import (
    TorchAddSVDFeaturesStep,
    TorchRandomSignFlipStep,
    TorchSelectiveQuantileTransformerStep,
    TorchSelectiveRobustScalerStep,
    TorchSelectiveSafePowerStep,
    TorchSelectiveSquashingScalerStep,
    TorchShuffleFeaturesStep,
    TorchStandardScalerStep,
)


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


def get_test_feature_schema(
    *,
    num_numericals: int = 0,
    num_categoricals: int = 0,
    num_text: int = 0,
) -> FeatureSchema:
    """Create FeatureSchema for tests from modality counts."""
    features = (
        [
            Feature(name=f"num{i}", modality=FeatureModality.NUMERICAL)
            for i in range(num_numericals)
        ]
        + [
            Feature(name=f"cat{i}", modality=FeatureModality.CATEGORICAL)
            for i in range(num_categoricals)
        ]
        + [
            Feature(name=f"txt{i}", modality=FeatureModality.TEXT)
            for i in range(num_text)
        ]
    )
    return FeatureSchema(features=features)


def test__call__single_step_transforms_columns():
    """Test pipeline with a single step transforms the correct columns."""
    step = MockStep(factor=3.0)
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])

    x = torch.ones(10, 1, 3)
    metadata = get_test_feature_schema(num_numericals=2, num_categoricals=1)

    result = pipeline(x, metadata, num_train_rows=5)

    # Columns 0 and 1 should be multiplied by 3
    assert torch.allclose(result.x[:, :, 0], torch.full((10, 1), 3.0))
    assert torch.allclose(result.x[:, :, 1], torch.full((10, 1), 3.0))
    # Column 2 should be unchanged
    assert torch.allclose(result.x[:, :, 2], torch.ones(10, 1))


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
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(num_numericals=2)
    x = torch.ones(10, 1, 2)

    result = pipeline(x, metadata, num_train_rows=5)

    assert torch.allclose(result.x, x)


def test__call__2d_input_adds_and_removes_batch_dimension():
    """Test that 2D input gets batch dimension added then removed."""
    step = MockStep(factor=2.0)
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(
        num_numericals=1,
        num_categoricals=1,
        num_text=1,
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
    metadata = get_test_feature_schema(num_numericals=2)
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
    metadata = get_test_feature_schema(num_numericals=1)
    x = torch.ones(10, 1, 1)

    result = pipeline(x, metadata, num_train_rows=None)

    # Output should be zeros: (x - mean) / std = (1 - 1) / 1 = 0
    assert torch.allclose(result.x, torch.zeros((10, 1)))


def test__call__zero_num_train_rows():
    """Test that fit is skipped when num_train_rows is 0."""
    step = TorchSoftClipOutliersStep()
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = get_test_feature_schema(num_numericals=1)
    x = torch.ones(10, 1, 1)

    result = pipeline(x, metadata, num_train_rows=0)

    assert torch.allclose(result.x, x)


def test__call__mismatching_num_columns_raises_error():
    """Test that mismatching num_columns raises an error."""
    step = MockStep(factor=2.0)
    pipeline = TorchPreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(num_numericals=1)
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
    metadata = get_test_feature_schema(num_numericals=1)

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
    metadata = get_test_feature_schema(num_numericals=1)
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


def _named_numerical_schema(num_features: int) -> FeatureSchema:
    """A numerical-only schema with explicit unique names (needed for inf restore)."""
    return FeatureSchema(
        features=[
            Feature(name=f"c{i}", modality=FeatureModality.NUMERICAL)
            for i in range(num_features)
        ]
    )


class RequireFiniteStep(TorchPreprocessingStep):
    """Step that asserts its input is finite, mimicking inf-intolerant transforms."""

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        assert not torch.isinf(x).any(), "inf reached a transform"
        return {}

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        assert not torch.isinf(x).any(), "inf reached a transform"
        return x, None, None


def test__call__nans_infinities_during_steps_and_restores_them() -> None:
    """+/-inf are hidden from the steps as NaN and written back at the end."""
    pipeline = TorchPreprocessingPipeline(
        steps=[(RequireFiniteStep(), {FeatureModality.NUMERICAL})]
    )
    x = torch.randn(8, 1, 3)
    x[2, 0, 1] = float("inf")
    x[5, 0, 2] = float("-inf")

    result = pipeline(x.clone(), _named_numerical_schema(3), num_train_rows=8)

    assert torch.isposinf(result.x[2, 0, 1])
    assert torch.isneginf(result.x[5, 0, 2])
    # Exactly the originals are restored -- none fabricated elsewhere.
    assert torch.isinf(result.x).sum() == 2


def test__call__finite_input_is_untouched_by_inf_handling() -> None:
    """The inf round-trip is a no-op when the input is already finite."""
    pipeline = TorchPreprocessingPipeline(
        steps=[(RequireFiniteStep(), {FeatureModality.NUMERICAL})]
    )
    x = torch.randn(8, 1, 3)

    result = pipeline(x.clone(), _named_numerical_schema(3), num_train_rows=8)

    assert torch.equal(result.x, x)


def test__call__2d_input_round_trips_infinities() -> None:
    """2D input restores infinities after the batch dim is added and removed."""
    pipeline = TorchPreprocessingPipeline(
        steps=[(RequireFiniteStep(), {FeatureModality.NUMERICAL})]
    )
    x = torch.randn(8, 3)
    x[2, 1] = float("inf")

    result = pipeline(x.clone(), _named_numerical_schema(3), num_train_rows=8)

    assert result.x.shape == (8, 3)
    assert torch.isposinf(result.x[2, 1])


def test__call__restores_infinities_into_shuffled_columns() -> None:
    """Restore matches columns by name, so it survives a shuffle permutation."""
    pipeline = TorchPreprocessingPipeline(
        steps=[
            (RequireFiniteStep(), {FeatureModality.NUMERICAL}),
            (
                TorchShuffleFeaturesStep(
                    shuffle_method="shuffle", shuffle_index=1, random_state=3
                ),
                None,
            ),
        ]
    )
    x = torch.randn(12, 1, 5)
    x[3, 0, 1] = float("inf")
    x[7, 0, 2] = float("-inf")

    result = pipeline(x.clone(), _named_numerical_schema(5), num_train_rows=12)

    names = [f.name for f in result.feature_schema.features]
    assert names != ["c0", "c1", "c2", "c3", "c4"]  # actually permuted
    assert torch.isposinf(result.x[3, 0, names.index("c1")])
    assert torch.isneginf(result.x[7, 0, names.index("c2")])
    assert torch.isinf(result.x).sum() == 2


def test__call__svd_does_not_crash_on_infinities_and_restores_them() -> None:
    """SVD (which crashes on raw inf) runs fine; original columns get inf back."""
    pipeline = TorchPreprocessingPipeline(
        steps=[(TorchAddSVDFeaturesStep(global_transformer_name="svd"), None)]
    )
    x = torch.randn(40, 1, 5)
    x[3, 0, 1] = float("inf")
    x[7, 0, 2] = float("-inf")

    result = pipeline(x.clone(), _named_numerical_schema(5), num_train_rows=40)

    names = [f.name for f in result.feature_schema.features]
    assert torch.isposinf(result.x[3, 0, names.index("c1")])
    assert torch.isneginf(result.x[7, 0, names.index("c2")])


@pytest.mark.parametrize("block_bytes", [None, 12 * 2 * 4])
def test__call__consume_input__same_output_written_into_the_input(
    block_bytes: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if block_bytes is not None:
        for module in ("pipeline_interface", "steps"):
            monkeypatch.setattr(
                f"tabpfn.preprocessing.torch.{module}.IN_PLACE_BLOCK_BYTES", block_bytes
            )
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(12, 2, 7, generator=generator)
    x[0, 0, 3] = 40.0
    schema = FeatureSchema.from_only_categorical_indices([5], num_columns=7)
    pipeline = TorchPreprocessingPipeline(
        steps=[
            (TorchSoftClipOutliersStep(n_sigma=2.0), {FeatureModality.NUMERICAL}),
            (TorchShuffleFeaturesStep("shuffle", 0, random_state=3), None),
        ]
    )
    before = x.clone()
    expected = pipeline(x, feature_schema=schema, num_train_rows=9)
    torch.testing.assert_close(x, before, rtol=0, atol=0)

    actual = pipeline(x, feature_schema=schema, num_train_rows=9, consume_input=True)

    assert actual.x.data_ptr() == x.data_ptr()
    torch.testing.assert_close(actual.x, expected.x, rtol=0, atol=0)
    assert actual.feature_schema == expected.feature_schema


class LegacySignatureStep(MockStep):
    """A step whose `fit_transform` override predates the `in_place` keyword."""

    @override
    def fit_transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
        fitted_cache: dict[str, torch.Tensor] | None = None,
    ) -> TorchPreprocessingStepResult:
        return super().fit_transform(x, column_indices, num_train_rows, fitted_cache)


def test__call__consume_input__step_without_in_place_keyword_runs() -> None:
    schema = get_test_feature_schema(num_numericals=2, num_categoricals=1)
    pipeline = TorchPreprocessingPipeline(
        steps=[(LegacySignatureStep(factor=3.0), {FeatureModality.NUMERICAL})]
    )

    expected = pipeline(torch.ones(10, 1, 3), schema, num_train_rows=5)
    actual = pipeline(
        torch.ones(10, 1, 3), schema, num_train_rows=5, consume_input=True
    )

    torch.testing.assert_close(actual.x, expected.x)


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("indices", [[0, 1, 2, 3, 4, 6], [6, 1, 4, 2]])
def test__fit_transform_column_blocks__on_a_device__match_fit_transform(
    use_cache: bool, indices: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_interface, "IN_PLACE_BLOCK_BYTES", 12 * 2 * 4 * 2)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(12, 2, 7, generator=generator)
    x[0, 0, 3] = 40.0
    x[5, 1, 6] = -30.0
    step = TorchSoftClipOutliersStep(n_sigma=2.0)
    expected = step.fit_transform(x.clone(), indices, num_train_rows=9)
    cache = expected.fitted_cache if use_cache else None

    actual = step._fit_transform_column_blocks(
        x, indices, 9, cache, torch.device("cpu")
    )

    assert actual.x.data_ptr() == x.data_ptr()
    assert torch.equal(actual.x, expected.x)
    assert actual.fitted_cache is not None
    assert expected.fitted_cache is not None
    for key, value in expected.fitted_cache.items():
        assert torch.equal(actual.fitted_cache[key], value)


@pytest.mark.parametrize("block_width", [1, 2, 5])
def test__fit_transform__in_place_column_blocks__equal_the_whole_fit(
    block_width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, batch, columns = 301, 2, 13
    monkeypatch.setattr(
        pipeline_interface, "IN_PLACE_BLOCK_BYTES", block_width * rows * batch * 4
    )
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(rows, batch, columns, generator=generator) * 3 + 1
    x[::7, 0, 4] = float("nan")
    x[3, 1, 9] = 60.0
    x[11, 0, 2] = -45.0
    indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12]
    step = TorchSoftClipOutliersStep(n_sigma=2.0)
    whole = step.fit_transform(x, indices, num_train_rows=200)

    blocks = step.fit_transform(x.clone(), indices, num_train_rows=200, in_place=True)

    assert whole.fitted_cache is not None
    assert blocks.fitted_cache is not None
    for key, value in whole.fitted_cache.items():
        assert torch.equal(blocks.fitted_cache[key], value)
    assert torch.equal(blocks.x.nan_to_num(), whole.x.nan_to_num())
    assert torch.equal(blocks.x.isnan(), whole.x.isnan())


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("cpu", "cpu", True),
        ("cpu", "cpu:0", True),
        ("mps", "mps:0", True),
        ("mps:0", "mps", True),
        ("cuda:0", "cuda:1", False),
        ("cuda:1", "cuda:1", True),
        ("cpu", "cuda", False),
        ("mps", "cpu", False),
    ],
)
def test__same_device__index_none_is_the_current_one(
    a: str, b: str, *, same: bool
) -> None:
    assert pipeline_interface._same_device(torch.device(a), torch.device(b)) is same


def test__extract_inf_masks__scan_device__matches_the_host_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_interface, "DEVICE_BLOCK_BYTES", 3 * 2 * 5 * 4)
    x = torch.randn(10, 2, 5, generator=torch.Generator().manual_seed(0))
    x[7, 1, 2] = float("inf")
    x[0, 0, 4] = -float("inf")
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=5)
    expected_x = x.clone()
    expected = pipeline_interface._extract_inf_masks(expected_x, schema)

    actual = pipeline_interface._extract_inf_masks(
        x, schema, scan_device=torch.device("cpu")
    )

    assert len(actual) == 2
    assert list(actual) == list(expected)
    for name, mask in expected.items():
        torch.testing.assert_close(actual[name], mask, rtol=0, atol=0)
    torch.testing.assert_close(x, expected_x, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test__call__host_input_with_compute_device__matches_the_device_input() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 1, 9, generator=generator)
    x[0, 0, 3] = 40.0
    x[9, 0, 2] = float("inf")
    schema = FeatureSchema.from_only_categorical_indices([5], num_columns=9)
    pipeline = TorchPreprocessingPipeline(
        steps=[
            (TorchSoftClipOutliersStep(n_sigma=2.0), {FeatureModality.NUMERICAL}),
            (TorchShuffleFeaturesStep("shuffle", 0, random_state=3), None),
        ]
    )
    device = torch.device("cuda")
    expected = pipeline(x.to(device), feature_schema=schema, num_train_rows=50)

    actual = pipeline(
        x,
        feature_schema=schema,
        num_train_rows=50,
        consume_input=True,
        compute_device=device,
    )

    assert actual.x.device.type == "cpu"
    torch.testing.assert_close(actual.x, expected.x.cpu(), equal_nan=True)
    assert actual.feature_schema == expected.feature_schema


def test__fit_transform_rows_on_device__row_blocks__match_fit_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_interface, "DEVICE_BLOCK_BYTES", 3 * 2 * 7 * 4)
    x = torch.randn(10, 2, 7, generator=torch.Generator().manual_seed(0))
    step = TorchShuffleFeaturesStep("shuffle", 0, random_state=3)
    expected = step.fit_transform(x.clone(), list(range(7)), num_train_rows=6)

    actual = pipeline_interface._fit_transform_rows_on_device(
        step, x, list(range(7)), 6, None, torch.device("cpu")
    )

    assert actual.x.data_ptr() == x.data_ptr()
    torch.testing.assert_close(actual.x, expected.x, rtol=0, atol=0)
    assert actual.schema_permutation == expected.schema_permutation


def _blockwise_pipeline(
    *, keep_fitted_cache: bool = False
) -> TorchPreprocessingPipeline:
    """One step of each kind the block runners handle, on 10 numerical columns."""
    return TorchPreprocessingPipeline(
        keep_fitted_cache=keep_fitted_cache,
        steps=[
            (TorchSelectiveQuantileTransformerStep(20, [0, 1, 2]), None),
            (TorchSelectiveSafePowerStep([3, 4]), None),
            (TorchSelectiveSquashingScalerStep(3.0, [5, 6]), None),
            (TorchSelectiveRobustScalerStep([7, 8]), None),
            (TorchRandomSignFlipStep(random_state=0), {FeatureModality.NUMERICAL}),
            (TorchShuffleFeaturesStep("shuffle", 0, random_state=3), None),
        ],
    )


def _wide_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 1, 10, generator=generator) * torch.linspace(0.5, 20, 10)
    x[3, 0, 4] = float("nan")
    return x


_TARGETS = [0, 2, 3, 5, 7, 8, 9]

_ROUNDING = {"rtol": 1e-6, "atol": 1e-6}
"""How far a fit may move with the columns fitted beside it: reductions over rows can
round differently on some platforms, and safepower's lambda search stops within its
tolerance."""

_FITS_UP_TO_ROUNDING = (
    TorchSelectiveQuantileTransformerStep,
    TorchSelectiveSafePowerStep,
)


@pytest.mark.parametrize(
    "step",
    [
        TorchSelectiveQuantileTransformerStep(20, _TARGETS),
        TorchSelectiveSafePowerStep(_TARGETS),
        TorchSelectiveSquashingScalerStep(3.0, _TARGETS),
        TorchSelectiveRobustScalerStep(_TARGETS),
        TorchSoftClipOutliersStep(n_sigma=3.0),
    ],
    ids=["quantile", "safepower", "squashing", "robust", "soft_clip"],
)
def test__fit_transform_column_blocks__columnwise_step__matches_fit_transform(
    step: TorchPreprocessingStep, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Three columns of 64 float32 rows per block: three blocks for seven targets.
    monkeypatch.setattr(pipeline_interface, "IN_PLACE_BLOCK_BYTES", 3 * 64 * 4)
    x = _wide_input()
    expected = step.fit_transform(x.clone(), _TARGETS, num_train_rows=50)

    actual = step._fit_transform_column_blocks(
        x, _TARGETS, 50, None, torch.device("cpu")
    )

    tolerance = (
        _ROUNDING if isinstance(step, _FITS_UP_TO_ROUNDING) else {"rtol": 0, "atol": 0}
    )
    assert actual.x.data_ptr() == x.data_ptr()
    torch.testing.assert_close(actual.x, expected.x, equal_nan=True, **tolerance)
    assert actual.fitted_cache is not None
    assert expected.fitted_cache is not None
    for key, value in expected.fitted_cache.items():
        torch.testing.assert_close(actual.fitted_cache[key], value, **tolerance)


def test__fit_transform_rows_on_device__sign_flip__matches_fit_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_interface, "DEVICE_BLOCK_BYTES", 10 * 10 * 4)
    x = _wide_input()
    step = TorchRandomSignFlipStep(random_state=0)
    expected = step.fit_transform(x.clone(), _TARGETS, num_train_rows=50)

    actual = pipeline_interface._fit_transform_rows_on_device(
        step, x, _TARGETS, 50, None, torch.device("cpu")
    )

    torch.testing.assert_close(actual.x, expected.x, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test__call__host_table__matches_the_device_table() -> None:
    x = _wide_input()
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=10)
    device = torch.device("cuda")
    expected = _blockwise_pipeline()(x.to(device), schema, num_train_rows=50)

    actual = _blockwise_pipeline()(
        x.clone(),
        schema,
        num_train_rows=50,
        consume_input=True,
        compute_device=device,
    )

    assert actual.x.device.type == "cpu"
    torch.testing.assert_close(actual.x, expected.x.cpu(), equal_nan=True, **_ROUNDING)


@pytest.mark.parametrize("use_fitted_cache", [False, True])
def test__call__staged_on_the_cpu__matches_the_whole_table(
    monkeypatch: pytest.MonkeyPatch, *, use_fitted_cache: bool
) -> None:
    """The staged path, with the CPU standing in for a separate compute device."""
    monkeypatch.setattr(pipeline_interface, "_same_device", lambda _a, _b: False)
    monkeypatch.setattr(pipeline_interface, "IN_PLACE_BLOCK_BYTES", 2 * 64 * 4)
    monkeypatch.setattr(pipeline_interface, "DEVICE_BLOCK_BYTES", 10 * 10 * 4)
    x = _wide_input()
    x[7, 0, 1] = float("inf")
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=10)
    whole = _blockwise_pipeline(keep_fitted_cache=True)
    staged = _blockwise_pipeline(keep_fitted_cache=True)
    expected = whole(x.clone(), schema, num_train_rows=50)
    actual = staged(
        x.clone(),
        schema,
        num_train_rows=50,
        consume_input=True,
        compute_device=torch.device("cpu"),
    )
    if use_fitted_cache:
        x_test = x[50:] * 1.5
        expected = whole(
            x_test.clone(), schema, num_train_rows=0, use_fitted_cache=True
        )
        actual = staged(
            x_test.clone(),
            schema,
            num_train_rows=0,
            use_fitted_cache=True,
            consume_input=True,
            compute_device=torch.device("cpu"),
        )

    # Safepower's fit moves slightly with the columns fitted beside it.
    torch.testing.assert_close(actual.x, expected.x, equal_nan=True, **_ROUNDING)
    assert actual.feature_schema == expected.feature_schema


def test__call__staged__rejects_a_step_without_a_block_runner_before_any_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = TorchPreprocessingPipeline(
        steps=[
            (TorchSelectiveRobustScalerStep([0, 1]), None),
            (TorchAddSVDFeaturesStep(random_state=0), None),
        ]
    )
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_interface, "_run_step", lambda step, *_, **__: ran.append(step)
    )
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=10)

    with pytest.raises(
        ValueError, match=r"TorchAddSVDFeaturesStep cannot run by .*pull/1427"
    ):
        pipeline(
            _wide_input(),
            schema,
            num_train_rows=50,
            consume_input=True,
            # A device unlike `x`'s; nothing is moved there before the check.
            compute_device=torch.device("meta"),
        )
    assert ran == []


def test__run_step__staged__moves_wide_blocks_and_computes_in_narrow_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_interface, "DEVICE_BLOCK_BYTES", 6 * 64 * 4)
    monkeypatch.setattr(pipeline_interface, "IN_PLACE_BLOCK_BYTES", 2 * 64 * 4)
    step = TorchSoftClipOutliersStep(n_sigma=3.0)
    widths: list[int] = []
    fit, transform = step._fit, step._transform

    def spy_fit(x: torch.Tensor) -> dict[str, torch.Tensor]:
        widths.append(x.shape[-1])
        return fit(x)

    def spy_transform(x: torch.Tensor, **kwargs: typing.Any) -> typing.Any:
        widths.append(x.shape[-1])
        return transform(x, **kwargs)

    moved: list[int] = []
    run_blocks = step._fit_transform_column_blocks

    def spy_blocks(x: torch.Tensor, *args: typing.Any) -> typing.Any:
        moved.append(x.shape[-1])
        return run_blocks(x, *args)

    monkeypatch.setattr(step, "_fit", spy_fit)
    monkeypatch.setattr(step, "_transform", spy_transform)
    monkeypatch.setattr(step, "_fit_transform_column_blocks", spy_blocks)

    pipeline_interface._run_step(
        step,
        _wide_input(),
        _TARGETS,
        50,
        None,
        consume_input=True,
        compute_device=torch.device("cpu"),
    )

    assert moved == [6, 1]
    assert max(widths) == 2


def test__run_step__mps__computes_safepower_on_the_cpu() -> None:
    """MPS has no float64, so safepower's blocks stay on the CPU."""
    step = TorchSelectiveSafePowerStep(_TARGETS)
    x = _wide_input()
    expected = step.fit_transform(x.clone(), _TARGETS, num_train_rows=50)

    actual = pipeline_interface._run_step(
        step,
        x,
        _TARGETS,
        50,
        None,
        consume_input=True,
        compute_device=torch.device("mps"),
    )

    torch.testing.assert_close(actual.x, expected.x, rtol=0, atol=0, equal_nan=True)
