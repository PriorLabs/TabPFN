#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for AddSVDFeaturesStep."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing import PreprocessingPipeline
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.add_svd_features_step import (
    AddSVDFeaturesStep,
    _pin_layout,
    get_svd_component_pool_size,
    get_svd_features_transformer,
    get_svd_n_extra_random_components,
)


def _get_schema(num_columns: int) -> FeatureSchema:
    """Create a schema with all numerical features."""
    return FeatureSchema(
        features=[
            Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL)
            for i in range(num_columns)
        ]
    )


def _get_test_data(
    n_samples: int = 100, n_features: int = 10, seed: int = 42
) -> np.ndarray:
    """Create test data with some structure for SVD to capture."""
    rng = np.random.default_rng(seed)
    # Create data with some latent structure
    latent = rng.standard_normal((n_samples, 3))
    weights = rng.standard_normal((3, n_features))
    noise = rng.standard_normal((n_samples, n_features)) * 0.1
    with np.errstate(all="ignore"):
        return (latent @ weights + noise).astype(np.float32)


def test__transform__returns_x_unchanged_and_svd_in_added_columns() -> None:
    """Test that _transform returns X unchanged, SVD features in added_columns."""
    data = _get_test_data(n_samples=50, n_features=6)
    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    step._fit(data, _get_schema(num_columns=6))
    result, added_cols, modality = step._transform(data)

    # X should be returned unchanged
    assert isinstance(result, np.ndarray)
    assert result.shape == data.shape
    np.testing.assert_array_equal(result, data)

    # SVD features should be in added_columns
    assert added_cols is not None
    assert added_cols.shape[0] == data.shape[0]
    assert added_cols.shape[1] > 0  # Should have some SVD components
    assert modality == FeatureModality.NUMERICAL


def test__transform__with_svd_quarter_components() -> None:
    """Test that svd_quarter_components produces fewer components than svd."""
    data = _get_test_data(n_samples=100, n_features=20)

    step_svd = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    step_svd._fit(data, _get_schema(num_columns=20))
    _, added_svd, _ = step_svd._transform(data)

    step_quarter = AddSVDFeaturesStep(
        global_transformer_name="svd_quarter_components", random_state=42
    )
    step_quarter._fit(data, _get_schema(num_columns=20))
    _, added_quarter, _ = step_quarter._transform(data)

    assert added_svd is not None
    assert added_quarter is not None
    # Quarter components should have fewer or equal columns
    assert added_quarter.shape[1] <= added_svd.shape[1]


def test__transform__with_single_feature_returns_unchanged() -> None:
    """Test that single feature data is returned unchanged without SVD."""
    data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    schema = _get_schema(num_columns=1)
    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    updated_schema = step._fit(data, schema)

    # Schema should be unchanged
    assert updated_schema.num_columns == 1

    # Transformer should not be set for single feature
    assert not hasattr(step, "transformer_") or step.transformer_ is None


def test__fit_transform__returns_added_columns() -> None:
    """Test fit_transform returns X unchanged with SVD in added_columns."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result = step.fit_transform(data, schema)

    # X should be unchanged
    assert result.X.shape == data.shape
    np.testing.assert_array_equal(result.X, data)

    # Schema should be unchanged (pipeline handles adding SVD)
    assert result.feature_schema.num_columns == 6

    # SVD features should be in added_columns
    assert result.X_added is not None
    assert result.X_added.shape[0] == data.shape[0]
    assert result.modality_added == FeatureModality.NUMERICAL


def test__transform__returns_added_columns_after_fit() -> None:
    """Test transform returns X unchanged with SVD in added_columns."""
    data_train = _get_test_data(n_samples=50, n_features=6, seed=42)
    data_test = _get_test_data(n_samples=20, n_features=6, seed=123)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    step.fit_transform(data_train, schema)
    result = step.transform(data_test)

    # X should be unchanged
    assert result.X.shape == data_test.shape

    # SVD features should be in added_columns
    assert result.X_added is not None
    assert result.X_added.shape[0] == data_test.shape[0]


def test__num_output_features__returns_correct_count() -> None:
    """Test num_output_features returns the expected count."""
    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)

    # For n_features=10, n_samples=100:
    # n_components = min(100//10+1, 10//2) = min(11, 5) = 5
    result = step.num_added_features(100, _get_schema(num_columns=10))
    assert result == 10 // 2

    # For n_features=1 (less than 2), should return unchanged
    result_single = step.num_added_features(
        n_samples=100, feature_schema=_get_schema(num_columns=1)
    )
    assert result_single == 0


def test__in_pipeline__returns_added_columns() -> None:
    """Test that the step returns added columns when used in a pipeline."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    pipeline = PreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])
    result = pipeline.fit_transform(data, schema)

    # Should have original columns plus SVD columns
    assert result.feature_schema.num_columns > 6
    assert result.X.shape[1] > 6
    assert result.X.shape[0] == data.shape[0]


def test__in_pipeline__transform_consistent_with_fit_transform() -> None:
    """Test that transform produces same shape as fit_transform."""
    data_train = _get_test_data(n_samples=50, n_features=6, seed=42)
    data_test = _get_test_data(n_samples=20, n_features=6, seed=123)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    pipeline = PreprocessingPipeline(steps=[(step, {FeatureModality.NUMERICAL})])

    fit_result = pipeline.fit_transform(data_train, schema)
    transform_result = pipeline.transform(data_test)

    assert fit_result.X.shape[1] == transform_result.X.shape[1]
    assert (
        fit_result.feature_schema.num_columns
        == transform_result.feature_schema.num_columns
    )


def test__in_pipeline__with_no_modality_selection() -> None:
    """Test that the step returns added columns when used in a pipeline."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    pipeline = PreprocessingPipeline(steps=[step])
    result = pipeline.fit_transform(data, schema)

    # Should have original columns plus SVD columns
    assert result.feature_schema.num_columns > 6
    assert result.X.shape[1] > 6
    assert result.X.shape[0] == data.shape[0]


def test__random_state__produces_reproducible_results() -> None:
    """Test that same random_state produces identical results."""
    data = _get_test_data(n_samples=50, n_features=6)
    schema = _get_schema(num_columns=6)

    step1 = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result1 = step1.fit_transform(data, schema)

    step2 = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result2 = step2.fit_transform(data, schema)

    assert result1.X_added is not None
    assert result2.X_added is not None
    np.testing.assert_array_almost_equal(result1.X_added, result2.X_added)


def test__refit__after_no_op_produces_svd_features() -> None:
    data_1feat = _get_test_data(n_samples=50, n_features=1)
    data_6feat = _get_test_data(n_samples=50, n_features=6)
    schema_1 = _get_schema(num_columns=1)
    schema_6 = _get_schema(num_columns=6)

    # Fit on 1 feature (is_no_op=True), then re-fit on 6 features.
    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result_noop = step.fit_transform(data_1feat, schema_1)
    assert result_noop.X_added is None  # no-op: no SVD features

    result_refit = step.fit_transform(data_6feat, schema_6)

    # Fresh step fit only on 6 features.
    step_fresh = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    result_fresh = step_fresh.fit_transform(data_6feat, schema_6)

    assert result_refit.X_added is not None, "is_no_op was not reset on re-fit"
    assert result_fresh.X_added is not None
    np.testing.assert_array_almost_equal(result_refit.X_added, result_fresh.X_added)


def test__get_svd_features_transformer__invalid_name_raises() -> None:
    """Test that invalid transformer name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid global transformer name"):
        # Create an invalid enum value by bypassing the enum
        get_svd_features_transformer(
            "invalid_name",  # type: ignore[arg-type]
            n_samples=100,
            n_features=10,
        )


# --- extra random components (SVD_EXTRA_RANDOM_COMPONENT_FRACTION) -----------


@pytest.mark.parametrize(
    ("n_top", "pool_size", "fraction", "expected"),
    [
        (10, 40, 0.0, 0),  # off
        (10, 40, 0.5, 5),  # half again
        (10, 40, 1.0, 10),  # twice as many
        (10, 40, 0.25, 3),  # rounds up: ceil(2.5)
        (1, 40, 0.5, 1),  # any positive fraction adds at least one
        (10, 12, 0.5, 2),  # clamped: only 2 components left below the top-k
        (10, 10, 0.5, 0),  # nothing below the top-k to draw from
    ],
)
def test__get_svd_n_extra_random_components__counts(
    n_top: int, pool_size: int, fraction: float, expected: int
) -> None:
    assert get_svd_n_extra_random_components(n_top, pool_size, fraction) == expected


def test__extra_random_components__adds_half_as_many_features_again() -> None:
    """0.5 appends ceil(k/2) more components, and the count is predicted upfront."""
    data = _get_test_data(n_samples=500, n_features=40)
    schema = _get_schema(num_columns=40)
    # k = min(500//10+1, 40//2) = min(51, 20) = 20
    n_top = 20

    plain = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    extra = AddSVDFeaturesStep(
        global_transformer_name="svd",
        random_state=42,
        extra_random_component_fraction=0.5,
    )

    plain_added = plain.fit_transform(data, schema).X_added
    extra_added = extra.fit_transform(data, schema).X_added

    assert plain_added is not None
    assert extra_added is not None
    assert plain_added.shape[1] == n_top
    assert extra_added.shape[1] == n_top + n_top // 2
    # The budget calculation must agree with what the step actually appends.
    assert extra.num_added_features(500, schema) == extra_added.shape[1]


def test__extra_random_components__off_by_default_and_fits_only_top_k() -> None:
    """Default must be the cheap path: only the top-k are decomposed."""
    data = _get_test_data(n_samples=500, n_features=40)
    schema = _get_schema(num_columns=40)

    step = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    step.fit_transform(data, schema)

    assert step.component_indices_ is None
    assert step.transformer_.steps[1][1].n_components == 20


def test__extra_random_components__keeps_the_top_k_and_appends_below_them() -> None:
    """The top-k block is untouched; the extras come from deeper in the spectrum."""
    data = _get_test_data(n_samples=500, n_features=40)
    schema = _get_schema(num_columns=40)
    n_top = 20

    plain = AddSVDFeaturesStep(global_transformer_name="svd", random_state=42)
    extra = AddSVDFeaturesStep(
        global_transformer_name="svd",
        random_state=42,
        extra_random_component_fraction=0.5,
    )
    plain_added = plain.fit_transform(data, schema).X_added
    extra_added = extra.fit_transform(data, schema).X_added

    assert plain_added is not None
    assert extra_added is not None
    # Only close, not identical: arpack is iterative, so asking it for the whole
    # spectrum instead of the top-k perturbs the top-k it converges to slightly.
    np.testing.assert_allclose(
        extra_added[:, :n_top], plain_added, rtol=1e-3, atol=1e-4
    )

    indices = extra.component_indices_
    assert indices is not None
    np.testing.assert_array_equal(indices[:n_top], np.arange(n_top))
    assert (indices[n_top:] >= n_top).all()
    assert indices[n_top:].max() < min(500, 40) - 1 + 1  # inside the pool
    assert len(set(indices.tolist())) == len(indices)  # drawn without replacement


def test__extra_random_components__match_a_full_spectrum_reference() -> None:
    """The appended columns are the selected components, exactly."""
    data = _get_test_data(n_samples=300, n_features=30)
    schema = _get_schema(num_columns=30)

    step = AddSVDFeaturesStep(
        global_transformer_name="svd",
        random_state=7,
        extra_random_component_fraction=0.5,
    )
    added = step.fit_transform(data, schema).X_added
    assert added is not None
    assert step.component_indices_ is not None

    reference = get_svd_features_transformer(
        "svd", 300, 30, random_state=7, n_components=min(300, 30) - 1
    )
    reference.fit(_pin_layout(data))
    expected = reference.transform(_pin_layout(data))[:, step.component_indices_]

    np.testing.assert_allclose(added, expected, rtol=1e-6, atol=1e-6)


def test__extra_random_components__differ_per_seed() -> None:
    """Each ensemble member draws its own tail components; that is the point."""
    data = _get_test_data(n_samples=500, n_features=40)
    schema = _get_schema(num_columns=40)

    def indices_for(seed: int) -> list[int]:
        step = AddSVDFeaturesStep(
            global_transformer_name="svd",
            random_state=seed,
            extra_random_component_fraction=0.5,
        )
        step.fit_transform(data, schema)
        assert step.component_indices_ is not None
        return step.component_indices_.tolist()

    assert indices_for(1) != indices_for(2)
    assert indices_for(1) == indices_for(1)  # reproducible for a given seed


def test__extra_random_components__no_pool_below_top_k_is_a_noop() -> None:
    """When the top-k already exhaust the spectrum there is nothing to add."""
    # n_samples=6 -> pool = min(6, 6) - 1 = 5; k = min(6//10+1, 6//2) = 1.
    # Tiny table: verify the step still fits and never asks for more than the pool.
    data = _get_test_data(n_samples=6, n_features=6)
    schema = _get_schema(num_columns=6)

    step = AddSVDFeaturesStep(
        global_transformer_name="svd",
        random_state=42,
        extra_random_component_fraction=0.5,
    )
    added = step.fit_transform(data, schema).X_added
    assert added is not None
    assert added.shape[1] == step.num_added_features(6, schema)
    assert added.shape[1] <= get_svd_component_pool_size(6, 6)


def test__extra_random_components__single_feature_still_a_noop() -> None:
    """The fewer-than-two-features guard wins over the extra components."""
    data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    schema = _get_schema(num_columns=1)

    step = AddSVDFeaturesStep(
        global_transformer_name="svd",
        random_state=42,
        extra_random_component_fraction=0.5,
    )
    result = step.fit_transform(data, schema)

    assert result.X_added is None
    assert step.num_added_features(4, schema) == 0
