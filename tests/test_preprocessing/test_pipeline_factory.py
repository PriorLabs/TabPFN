#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for create_preprocessing_pipeline."""

from __future__ import annotations

import dataclasses
import itertools
import tracemalloc
import typing
from typing import Literal

import numpy as np
import pytest

from tabpfn.preprocessing.configs import EnsembleConfig, PreprocessorConfig
from tabpfn.preprocessing.datamodel import FeatureSchema
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingPipeline,
    step_owns_input,
)
from tabpfn.preprocessing.presets import (
    v2_5_classifier_preprocessor_configs,
    v2_5_regressor_preprocessor_configs,
    v2_classifier_preprocessor_configs,
    v2_regressor_preprocessor_configs,
)
from tabpfn.preprocessing.steps import (
    EncodeCategoricalFeaturesStep,
    ReshapeFeatureDistributionsStep,
)
from tabpfn.preprocessing.torch.gpu_preprocessing_metadata import (
    get_gpu_transform_type,
)


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
        estimator_id=0,
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


def _preprocessor_configs() -> list[PreprocessorConfig]:
    """The preset configs, and one per transform name with and without SVD."""
    presets = [
        *v2_classifier_preprocessor_configs(),
        *v2_regressor_preprocessor_configs(),
        *v2_5_classifier_preprocessor_configs(),
        *v2_5_regressor_preprocessor_configs(),
    ]
    names = typing.get_args(typing.get_type_hints(PreprocessorConfig)["name"])
    swept = [
        PreprocessorConfig(name, global_transformer_name=svd)
        for name, svd in itertools.product(names, [None, "svd"])
    ]
    differentiable = [PreprocessorConfig("none", differentiable=True)]
    return presets + swept + differentiable


def _keeps_float32_by_step_types(
    config: EnsembleConfig, *, enable_gpu_preprocessing: bool
) -> bool:
    """Whether the pipeline keeps a float32 input, as decided from its step types."""
    pconfig = config.preprocess_config
    schedule_gpu_transform = (
        get_gpu_transform_type(pconfig.name) if enable_gpu_preprocessing else None
    )
    adds_svd = (
        not pconfig.differentiable
        and not enable_gpu_preprocessing
        and pconfig.global_transformer_name not in (None, "None")
    )
    adds_fingerprint = config.add_fingerprint_feature and not enable_gpu_preprocessing
    return (
        config.polynomial_features == "no"
        and not pconfig.differentiable
        and (schedule_gpu_transform is not None or pconfig.name == "none")
        and not adds_svd
        and not adds_fingerprint
    )


@pytest.mark.parametrize("enable_gpu_preprocessing", [False, True])
@pytest.mark.parametrize("add_fingerprint_feature", [False, True])
@pytest.mark.parametrize("polynomial_features", ["no", 3])
def test__create_preprocessing_pipeline__keep_float32__matches_step_types(
    enable_gpu_preprocessing: bool,
    add_fingerprint_feature: bool,
    polynomial_features: Literal["no"] | int,
) -> None:
    for pconfig in _preprocessor_configs():
        config = EnsembleConfig(
            preprocess_config=pconfig,
            add_fingerprint_feature=add_fingerprint_feature,
            polynomial_features=polynomial_features,
            feature_shift_count=0,
            feature_shift_decoder="shuffle",
            outlier_removal_std=None,
            _model_index=0,
            estimator_id=0,
            passthrough_inf=False,
        )

        pipeline = create_preprocessing_pipeline(
            config,
            random_state=0,
            enable_gpu_preprocessing=enable_gpu_preprocessing,
        )

        expected = _keeps_float32_by_step_types(
            config, enable_gpu_preprocessing=enable_gpu_preprocessing
        )
        assert pipeline.keep_float32 == expected, dataclasses.asdict(pconfig)


def test__reshape_step__no_arithmetic__follows_a_changed_transform_name() -> None:
    step = ReshapeFeatureDistributionsStep(transform_name="none")
    pipeline = PreprocessingPipeline([step])
    assert step.no_arithmetic
    assert pipeline.keep_float32

    step.transform_name = "quantile_uni"

    assert not step.no_arithmetic
    assert not pipeline.keep_float32


def test__keep_float32_pipeline__runs_on_a_finite_input_itself() -> None:
    """Nothing writes into a finite float32 input, so a fit or transform that may
    keep it does not copy it; by default they copy.
    """
    pipeline = PreprocessingPipeline(
        [ReshapeFeatureDistributionsStep(transform_name="none")]
    )
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=3)

    fitted = pipeline.fit_transform(X, schema, may_keep_input=True)
    transformed = pipeline.transform(X, may_keep_input=True)

    assert np.shares_memory(fitted.X, X)
    assert np.shares_memory(transformed.X, X)
    assert not np.shares_memory(pipeline.transform(X).X, X)
    assert fitted.X.dtype == transformed.X.dtype == np.float32


def test__keep_float32_pipeline__fit__copies_the_callers_input() -> None:
    """A fitted model may keep its training data, so the caller must opt in."""
    pipeline = PreprocessingPipeline(
        [ReshapeFeatureDistributionsStep(transform_name="none")]
    )
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=3)

    result = pipeline.fit_transform(X, schema)

    assert not np.shares_memory(result.X, X)


@pytest.mark.parametrize(
    "layout", ["negative_strides", "fortran_order"], ids=lambda layout: layout
)
def test__keep_float32_pipeline__copies_an_input_that_is_not_c_ordered(
    layout: str,
) -> None:
    """The output is C-ordered, as the copy is."""
    pipeline = PreprocessingPipeline(
        [ReshapeFeatureDistributionsStep(transform_name="none")]
    )
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    X = X[::-1] if layout == "negative_strides" else np.asfortranarray(X)
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=3)
    pipeline.fit_transform(X, schema)

    result = pipeline.transform(X, may_keep_input=True)

    assert not np.shares_memory(result.X, X)
    assert result.X.flags.c_contiguous


def test__keep_float32_pipeline__copies_an_input_with_infinities() -> None:
    """Infinities are replaced in place, so such an input is copied first."""
    pipeline = PreprocessingPipeline(
        [ReshapeFeatureDistributionsStep(transform_name="none")]
    )
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    X[0, 0] = np.inf
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=3)

    result = pipeline.fit_transform(X, schema, may_keep_input=True)

    assert not np.shares_memory(result.X, X)
    assert np.isinf(X[0, 0])


def test__keep_float32_pipeline__a_steps_new_array__is_written_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step after one that returned a new array may write into that array."""
    owned: list[bool] = []
    encode = EncodeCategoricalFeaturesStep._ordinal_fit_transform

    def spy(ct: typing.Any, X: np.ndarray) -> np.ndarray:
        owned.append(step_owns_input())
        return encode(ct, X)

    monkeypatch.setattr(
        EncodeCategoricalFeaturesStep, "_ordinal_fit_transform", staticmethod(spy)
    )
    pipeline = PreprocessingPipeline(
        [
            # Moves the categorical column to the front, into a new array.
            ReshapeFeatureDistributionsStep(transform_name="none"),
            EncodeCategoricalFeaturesStep("ordinal", random_state=0),
        ]
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    X[:, 2] = rng.integers(0, 3, size=X.shape[0])
    schema = FeatureSchema.from_only_categorical_indices([2], num_columns=4)

    pipeline.fit_transform(X, schema, may_keep_input=True)

    assert owned == [True]


def test__keep_float32_pipeline__encodes_a_finite_input_without_a_float64_table() -> (
    None
):
    """Ordinal codes go into a float32 copy, not a float64 table cast back."""
    pipeline = PreprocessingPipeline(
        [
            EncodeCategoricalFeaturesStep("ordinal", random_state=0),
            ReshapeFeatureDistributionsStep(transform_name="none"),
        ]
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20_000, 20)).astype(np.float32)
    X[:, 0] = rng.integers(0, 5, size=X.shape[0])
    before = X.copy()
    schema = FeatureSchema.from_only_categorical_indices([0], num_columns=20)

    tracemalloc.start()
    try:
        result = pipeline.fit_transform(X, schema, may_keep_input=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # One float32 copy and the encoder's own bookkeeping; a float64 table is two.
    assert peak < 2 * X.nbytes
    assert result.X.dtype == np.float32
    np.testing.assert_array_equal(X, before)
