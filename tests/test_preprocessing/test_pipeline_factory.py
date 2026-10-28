#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for create_preprocessing_pipeline."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from tabpfn.preprocessing.configs import EnsembleConfig, PreprocessorConfig
from tabpfn.preprocessing.datamodel import FeatureSchema
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline


def _fit_transform(
    X: np.ndarray, polynomial_features: Literal["no", "all"] | int
) -> np.ndarray:
    config = EnsembleConfig(
        preprocess_config=PreprocessorConfig("none"),
        add_fingerprint_feature=False,
        polynomial_features=polynomial_features,
        feature_shift_count=0,
        feature_shift_decoder=None,
        outlier_removal_std=None,
        _model_index=0,
        passthrough_inf=False,
    )
    pipeline = create_preprocessing_pipeline(config, random_state=42)
    schema = FeatureSchema.from_only_categorical_indices([], X.shape[1])
    return np.asarray(pipeline.fit_transform(X, schema).X)


@pytest.mark.parametrize("polynomial_features", ["no", "all", 3])
def test__constant_columns__do_not_change_output(
    polynomial_features: Literal["no", "all"] | int,
) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 3))
    X_with_constants = np.hstack([X, np.zeros((9, 3)), np.full((9, 2), 5.0)])

    np.testing.assert_allclose(
        _fit_transform(X, polynomial_features),
        _fit_transform(X_with_constants, polynomial_features),
    )


def test__polynomial_features__constant_and_all_nan_products_are_removed() -> None:
    signs = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
    nan_in_first_half = np.array([np.nan, np.nan, np.nan, 1.0, 2.0, 3.0])
    nan_in_second_half = np.array([4.0, 5.0, 6.0, np.nan, np.nan, np.nan])
    X = np.column_stack([signs, nan_in_first_half, nan_in_second_half])

    X_out = _fit_transform(X, "all")

    assert not np.isnan(X_out).all(axis=0).any()
    non_nan_values = [col[~np.isnan(col)] for col in X_out.T]
    assert all(len(np.unique(values)) > 1 for values in non_nan_values)
