from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing_extensions import override

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
)

from tabpfn.preprocessing import steps
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingPipeline,
    PreprocessingStep,
)
from tabpfn.preprocessing.steps import (
    AddFingerprintFeaturesStep,
    DifferentiableZNormStep,
    ReshapeFeatureDistributionsStep,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import (
    OrderPreservingColumnTransformer,
)


def _get_preprocessing_steps() -> list[Callable[..., PreprocessingStep],]:
    defaults: list[Callable[..., PreprocessingStep]] = [
        cls
        for cls in steps.__dict__.values()
        if (
            isinstance(cls, type)
            and issubclass(cls, PreprocessingStep)
            and cls is not PreprocessingStep
            and cls is not DifferentiableZNormStep  # works on torch tensors
        )
    ]
    extras: list[Callable[..., PreprocessingStep]] = [
        partial(
            ReshapeFeatureDistributionsStep,
            transform_name="none",
            append_to_original=True,
            apply_to_categorical=False,
        )
    ]
    return defaults + extras


def _get_random_data(
    rng: np.random.Generator, n_samples: int, n_features: int, cat_inds: list[int]
) -> np.ndarray:
    x = rng.random((n_samples, n_features))
    x[:, cat_inds] = rng.integers(0, 3, size=(n_samples, len(cat_inds))).astype(float)
    return x


def _make_metadata(n_features: int, cat_inds: list[int]) -> FeatureSchema:
    return FeatureSchema.from_only_categorical_indices(cat_inds, n_features)


def test__preprocessing_steps__transform__is_idempotent():
    """Test that calling transform multiple times on the same data
    gives the same result. This ensures transform is deterministic
    and doesn't have internal state changes.
    """
    rng = np.random.default_rng(42)
    n_samples = 20
    n_features = 4
    cat_inds = [1, 3]
    feature_schema = _make_metadata(n_features, cat_inds)
    for cls in _get_preprocessing_steps():
        x = _get_random_data(rng, n_samples, n_features, cat_inds)
        x2 = _get_random_data(rng, n_samples, n_features, cat_inds)

        obj = cls()
        obj.fit_transform(x, feature_schema)

        # Calling transform multiple times should give the same result
        result1 = obj.transform(x2)
        result2 = obj.transform(x2)

        assert np.allclose(result1.X, result2.X), f"Transform not idempotent for {cls}"
        assert result1.feature_schema.indices_for(
            FeatureModality.CATEGORICAL
        ) == result2.feature_schema.indices_for(FeatureModality.CATEGORICAL)


def test__preprocessing_steps__transform__no_sample_interdependence():
    """Test that preprocessing steps don't have
    interdependence between samples during transform. Each sample should be
    transformed independently based only on parameters learned during fit.
    """
    rng = np.random.default_rng(42)
    n_samples = 20
    n_features = 4
    cat_inds = [1, 3]
    feature_schema = _make_metadata(n_features, cat_inds)
    for cls in _get_preprocessing_steps():
        x = _get_random_data(rng, n_samples, n_features, cat_inds)
        x2 = _get_random_data(rng, n_samples, n_features, cat_inds)

        obj = cls()
        obj.fit_transform(x, feature_schema)

        # Test 1: Shuffling samples should give correspondingly shuffled results
        result_normal = obj.transform(x2)
        result_reversed = obj.transform(x2[::-1])
        assert np.allclose(result_reversed.X[::-1], result_normal.X), (
            f"Transform depends on sample order for {cls}"
        )

        # Test 2: Transforming a subset should match the subset of full transformation
        result_full = obj.transform(x2)
        result_subset = obj.transform(x2[:4])
        assert np.allclose(result_full.X[:4], result_subset.X), (
            f"Transform depends on other samples in batch for {cls}"
        )

        # Test 3: Categorical features should remain the same
        assert result_full.feature_schema.indices_for(
            FeatureModality.CATEGORICAL
        ) == result_subset.feature_schema.indices_for(FeatureModality.CATEGORICAL)


def test__pipeline__handles_added_columns_from_fingerprint_step():
    """Test that the pipeline correctly handles added_columns from steps.

    The fingerprint step returns X unchanged and provides the fingerprint
    via added_columns. The pipeline should concatenate this and update schema.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 10, 3
    X = rng.random((n_samples, n_features))
    schema = FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(n_features)
        ]
    )

    # Create pipeline with fingerprint step
    fingerprint_step = AddFingerprintFeaturesStep(random_state=42)
    pipeline = PreprocessingPipeline(steps=[fingerprint_step])

    result = pipeline.fit_transform(X, schema)

    # Pipeline should have concatenated the fingerprint column
    assert result.X.shape == (n_samples, n_features + 1)

    # Metadata should track the new column
    assert result.feature_schema.num_columns == n_features + 1
    assert (
        len(result.feature_schema.indices_for(FeatureModality.NUMERICAL))
        == n_features + 1
    )

    # Original columns should be preserved
    np.testing.assert_array_equal(result.X[:, :n_features], X)


def test__pipeline__transform_also_handles_added_columns():
    """Test that pipeline.transform also correctly handles added_columns."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 10, 3
    X_train = rng.random((n_samples, n_features))
    X_test = rng.random((5, n_features))
    schema = FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(n_features)
        ]
    )

    # Create and fit pipeline
    fingerprint_step = AddFingerprintFeaturesStep(random_state=42)
    pipeline = PreprocessingPipeline(steps=[fingerprint_step])
    pipeline.fit_transform(X_train, schema)

    # Transform test data
    result = pipeline.transform(X_test)

    # Should also have the fingerprint column
    assert result.X.shape == (5, n_features + 1)


# TODO: Ideally we don't allow for this in no preprocessing step!
def test__pipeline__raises_error_when_modality_step_changes_column_count():
    """Test that pipeline raises error if modality-registered step changes columns."""

    class BadStep(PreprocessingStep):
        """A step that incorrectly returns more columns than it received."""

        @override
        def _fit(self, X: np.ndarray, metadata: FeatureSchema) -> FeatureSchema:
            return metadata

        @override
        def _transform(
            self, X: np.ndarray, *, is_test: bool = False
        ) -> tuple[np.ndarray, None, None]:
            # Incorrectly return more columns
            return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), None, None

    rng = np.random.default_rng(42)
    X = rng.random((10, 3))
    schema = FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL) for _ in range(3)
        ]
    )

    # Register step with modalities - should raise error
    bad_step = BadStep()
    pipeline = PreprocessingPipeline(steps=[(bad_step, {FeatureModality.NUMERICAL})])

    with pytest.raises(ValueError, match="received 3 columns but returned 4"):
        pipeline.fit_transform(X, schema)


# This is a test for the OrderPreservingColumnTransformer, which is not used currently
# But might be used in the future, therefore I'll leave it in.
@pytest.mark.skip
def test__order_preserving_column_transformer():
    """Should raise AssertionError if column sets overlap."""
    ordinal_enc1 = OrdinalEncoder()
    ordinal_enc2 = OrdinalEncoder()
    onehotencoder1 = OneHotEncoder()

    # Test assertion raised due to too many transformers
    multiple_transformers = [
        ("ordinal_enc1", ordinal_enc1, ["a", "b"]),
        ("ordinal_enc2", ordinal_enc2, ["c", "d"]),
    ]

    with pytest.raises(
        AssertionError,
        match="OrderPreservingColumnTransformer only supports up to one transformer",
    ):
        OrderPreservingColumnTransformer(transformers=multiple_transformers)

    # Test assertion, due to unsupported encoder type (OneHotEncoder)
    incompatible_transformer = [("onehot", onehotencoder1, ["a", "b"])]

    with pytest.raises(AssertionError, match="are instances of OneToOneFeatureMixin"):
        OrderPreservingColumnTransformer(transformers=incompatible_transformer)

        # --- Mock dataset ---
    mock_data_df = pd.DataFrame(
        {
            "a": [10, 20, 30, 40],
            "b": ["x", "y", "x", "z"],
        }
    )

    # Test if normal column transformer shuffles column order,
    # while the OrderPreserving restores the original order
    non_overlapping_ordinal_encoder = [("ordinal_enc1", ordinal_enc1, ["b"])]

    vanilla_transformer = ColumnTransformer(
        transformers=non_overlapping_ordinal_encoder, remainder=FunctionTransformer()
    )

    vanilla_output = vanilla_transformer.fit_transform(mock_data_df)

    # Vanilla transformer shuffles column order
    assert not np.array_equal(mock_data_df.iloc[:, 0].values, vanilla_output[:, 0])

    preserving_transformer = OrderPreservingColumnTransformer(
        transformers=non_overlapping_ordinal_encoder, remainder=FunctionTransformer()
    )

    # OrderPreserving transformer does not shuffle column order
    preserved_output = preserving_transformer.fit_transform(mock_data_df)
    np.testing.assert_equal(mock_data_df.iloc[:, 0].values, preserved_output[:, 0])
