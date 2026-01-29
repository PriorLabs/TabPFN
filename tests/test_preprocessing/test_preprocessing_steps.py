from __future__ import annotations

from collections.abc import Callable
from functools import partial

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
from tabpfn.preprocessing.pipeline_interfaces import FeaturePreprocessingTransformerStep
from tabpfn.preprocessing.steps import (
    DifferentiableZNormStep,
    ReshapeFeatureDistributionsStep,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import (
    OrderPreservingColumnTransformer,
)


def _get_preprocessing_steps() -> list[
    Callable[..., FeaturePreprocessingTransformerStep],
]:
    defaults: list[Callable[..., FeaturePreprocessingTransformerStep]] = [
        cls
        for cls in steps.__dict__.values()
        if (
            isinstance(cls, type)
            and issubclass(cls, FeaturePreprocessingTransformerStep)
            and cls is not FeaturePreprocessingTransformerStep
            and cls is not DifferentiableZNormStep  # works on torch tensors
        )
    ]
    extras: list[Callable[..., FeaturePreprocessingTransformerStep]] = [
        partial(
            ReshapeFeatureDistributionsStep,
            transform_name="none",
            append_to_original=True,
            global_transformer_name="svd",
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


def test__preprocessing_steps__transform__is_idempotent():
    """Test that calling transform multiple times on the same data
    gives the same result. This ensures transform is deterministic
    and doesn't have internal state changes.
    """
    rng = np.random.default_rng(42)
    n_samples = 20
    n_features = 4
    cat_inds = [1, 3]
    for cls in _get_preprocessing_steps():
        x = _get_random_data(rng, n_samples, n_features, cat_inds)
        x2 = _get_random_data(rng, n_samples, n_features, cat_inds)

        obj = cls().fit(x, cat_inds)

        # Calling transform multiple times should give the same result
        result1 = obj.transform(x2)
        result2 = obj.transform(x2)

        assert np.allclose(result1.X, result2.X), f"Transform not idempotent for {cls}"
        assert result1.categorical_features == result2.categorical_features


def test__preprocessing_steps__transform__no_sample_interdependence():
    """Test that preprocessing steps don't have
    interdependence between samples during transform. Each sample should be
    transformed independently based only on parameters learned during fit.
    """
    rng = np.random.default_rng(42)
    n_samples = 20
    n_features = 4
    cat_inds = [1, 3]
    for cls in _get_preprocessing_steps():
        x = _get_random_data(rng, n_samples, n_features, cat_inds)
        x2 = _get_random_data(rng, n_samples, n_features, cat_inds)

        obj = cls().fit(x, cat_inds)

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
        assert result_full.categorical_features == result_subset.categorical_features


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
