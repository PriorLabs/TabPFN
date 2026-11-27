#  Copyright (c) Prior Labs GmbH 2026.

"""The feature gather fused into each estimator's working copy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.preprocessing import pipeline_interface
from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingPipeline,
    gather_columns,
)
from tabpfn.preprocessing.steps import RemoveConstantFeaturesStep
from tabpfn.preprocessing.transform import _fit_preprocessing_one


@pytest.mark.parametrize("source", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test__gather_columns__blocks__match_fancy_indexing_and_cast(
    monkeypatch: pytest.MonkeyPatch, source: type, order: str, dtype: type
) -> None:
    monkeypatch.setattr(pipeline_interface, "GATHER_BLOCK_BYTES", 3 * 50 * 4)
    rng = np.random.default_rng(0)
    X = np.asarray(rng.standard_normal((50, 17)) * 10, dtype=source, order=order)
    if source is not np.int64:
        X[3, 4] = np.nan
    indices = rng.permutation(17)[:11]

    got = gather_columns(X, indices, dtype)

    assert got.flags.c_contiguous
    np.testing.assert_array_equal(got, X[:, indices].astype(dtype))


def _schema(n: int, n_cat: int) -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(
                name=f"f{i}",
                modality=FeatureModality.CATEGORICAL
                if i < n_cat
                else FeatureModality.NUMERICAL,
            )
            for i in range(n)
        ]
    )


def _config(pconfig: PreprocessorConfig) -> ClassifierEnsembleConfig:
    return ClassifierEnsembleConfig(
        preprocess_config=pconfig,
        feature_shift_count=2,
        class_permutation=None,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder="shuffle",
        outlier_removal_std=None,
        _model_index=0,
        estimator_id=0,
        passthrough_inf=False,
    )


@pytest.mark.parametrize(
    "pconfig",
    [
        PreprocessorConfig("none", categorical_name="ordinal_shuffled"),
        PreprocessorConfig("quantile_uni_coarse", categorical_name="numeric"),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test__fit_preprocessing_one__feature_subset__matches_the_subset_array(
    pconfig: PreprocessorConfig, dtype: type
) -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 12)).astype(dtype)
    X[:, :2] = rng.integers(0, 4, (120, 2))
    y = rng.integers(0, 2, 120)
    schema = _schema(12, 2)
    indices = np.array([7, 0, 3, 11, 1, 5])
    config = _config(pconfig)

    def run(
        X_in: np.ndarray, feature_indices: np.ndarray | None, s: FeatureSchema
    ) -> tuple:
        pipeline = create_preprocessing_pipeline(config, random_state=3)
        return _fit_preprocessing_one(
            0,
            config,
            X_in,
            y,
            feature_schema=s,
            pipeline=pipeline,
            feature_indices=feature_indices,
            may_keep_input=False,
        )

    fused = run(X, indices, schema)
    subset = run(X[:, indices], None, schema.slice_for_indices(indices.tolist()))

    assert fused[3].dtype == subset[3].dtype
    np.testing.assert_array_equal(fused[3], subset[3])
    np.testing.assert_array_equal(fused[4], subset[4])
    assert fused[5] == subset[5]


def test__pipeline_fit_transform__feature_indices__torch_matches_numpy() -> None:
    X = np.random.default_rng(0).normal(size=(50, 6)).astype(np.float32)
    X[:, 3] = 2.0
    schema = FeatureSchema.from_only_categorical_indices([], 3)
    keep = np.array([4, 0, 3])

    def run(data: np.ndarray | torch.Tensor) -> tuple[np.ndarray, int]:
        result = PreprocessingPipeline([RemoveConstantFeaturesStep()]).fit_transform(
            data, schema, feature_indices=keep
        )
        return np.asarray(result.X), result.feature_schema.num_columns

    expected, expected_columns = run(X)
    actual, actual_columns = run(torch.from_numpy(X))

    assert actual.shape == expected.shape == (50, 2)
    assert actual_columns == expected_columns == 2
    np.testing.assert_array_equal(actual, expected)
