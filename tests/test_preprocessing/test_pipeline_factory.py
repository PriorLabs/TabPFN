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
    PreprocessingStep,
    SharedColumnsTable,
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
    RemoveConstantFeaturesStep,
    ReshapeFeatureDistributionsStep,
)
from tabpfn.preprocessing.torch.gpu_preprocessing_metadata import (
    GPUTransformType,
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
    """Infinities are replaced in place, so such an input is copied first, and
    gathered columns holding one are not shared.
    """
    pipeline = PreprocessingPipeline(
        [ReshapeFeatureDistributionsStep(transform_name="none")]
    )
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    X[0, 0] = np.inf
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=3)

    result = pipeline.fit_transform(X, schema, may_keep_input=True)

    assert not np.shares_memory(result.X, X)
    assert np.isinf(X[0, 0])

    X[:, 1] = np.arange(20) % 4 + 10
    encoding = PreprocessingPipeline(
        [EncodeCategoricalFeaturesStep("ordinal_shuffled", random_state=0)]
    )
    indices = np.array([0, 1])
    encoding.fit_transform(
        X,
        FeatureSchema.from_only_categorical_indices([1], num_columns=2),
        feature_indices=indices,
    )
    gathered = encoding.transform(X, feature_indices=indices, may_keep_input=True).X
    assert isinstance(gathered, np.ndarray)
    assert np.isinf(gathered).sum() == 1


class _NewArrayStep(PreprocessingStep):
    """Returns its input as a new array."""

    no_arithmetic = True

    def _fit(self, X: np.ndarray, feature_schema: FeatureSchema) -> FeatureSchema:
        del X
        return feature_schema

    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, None, None]:
        del is_test
        return np.array(X), None, None


class _OwnershipSpy(PreprocessingStep):
    """Records whether the pipeline lets it write into its input."""

    no_arithmetic = True

    def __init__(self) -> None:
        super().__init__()
        self.owned: list[bool] = []

    def _fit(self, X: np.ndarray, feature_schema: FeatureSchema) -> FeatureSchema:
        del X
        self.owned.append(step_owns_input())
        return feature_schema

    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, None, None]:
        del is_test
        return X, None, None


@pytest.mark.parametrize("new_array_first", [False, True])
def test__keep_float32_pipeline__a_steps_new_array__is_the_next_steps_to_write(
    *, new_array_first: bool
) -> None:
    spy = _OwnershipSpy()
    pipeline = PreprocessingPipeline(
        [_NewArrayStep(), spy] if new_array_first else [spy]
    )
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    schema = FeatureSchema.from_only_categorical_indices([], num_columns=3)

    pipeline.fit_transform(X, schema, may_keep_input=True)

    assert spy.owned == [new_array_first]


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


@pytest.mark.parametrize(
    "encoding",
    [
        "ordinal",
        "ordinal_shuffled",
        "ordinal_common_categories",
        "ordinal_very_common_categories_shuffled",
        "numeric",
    ],
)
def test__keep_float32_pipeline__may_keep_input__shares_the_columns_it_leaves(
    encoding: str,
) -> None:
    """Columns the steps only move are read from `X`; the encoded ones are held."""

    def pipeline() -> PreprocessingPipeline:
        return PreprocessingPipeline(
            [
                RemoveConstantFeaturesStep(),
                ReshapeFeatureDistributionsStep(transform_name="none", random_state=0),
                EncodeCategoricalFeaturesStep(encoding, random_state=0),
            ]
        )

    rng = np.random.default_rng(0)
    X, X_test = (
        rng.standard_normal((rows, 9)).astype(np.float32) for rows in (3000, 500)
    )
    for table in (X, X_test):
        table[:, 2] = rng.integers(0, 5, len(table))
        table[:, 6] = rng.integers(0, 3, len(table))
        table[:, 4] = 7.0  # constant, so removed
    X_test[3, 2] = np.nan
    X_test[4, 2] = 9.0  # a category unseen at fit
    schema = FeatureSchema.from_only_categorical_indices([2, 6], num_columns=9)
    copying, sharing = pipeline(), pipeline()

    expected = copying.fit_transform(X.copy(), schema)
    actual = sharing.fit_transform(X, schema, may_keep_input=True)

    assert isinstance(actual.X, SharedColumnsTable)
    assert actual.X.base is X
    assert actual.X.own.shape[1] == (0 if encoding == "numeric" else 2)
    assert actual.feature_schema == expected.feature_schema
    np.testing.assert_array_equal(np.asarray(actual.X), expected.X)
    shared = sharing.transform(X_test, may_keep_input=True).X
    assert isinstance(shared, SharedColumnsTable)
    assert isinstance(sharing.transform(X_test).X, np.ndarray)
    np.testing.assert_array_equal(
        np.asarray(shared), copying.transform(X_test.copy()).X
    )


def test__keep_float32_pipeline__feature_indices__shares_the_gathered_columns() -> None:
    """A pipeline reading some of `X`'s columns shares its own copy of them at
    predict, and codes them in place at fit.
    """
    pipeline = PreprocessingPipeline(
        [
            RemoveConstantFeaturesStep(),
            ReshapeFeatureDistributionsStep(transform_name="none", random_state=0),
            EncodeCategoricalFeaturesStep("ordinal_shuffled", random_state=0),
        ]
    )
    rng = np.random.default_rng(0)
    X, X_test = (
        rng.standard_normal((rows, 9)).astype(np.float32) for rows in (3000, 500)
    )
    for table in (X, X_test):
        table[:, 2] = rng.integers(10, 15, len(table))
        table[:, 6] = rng.integers(20, 23, len(table))
    indices = np.array([0, 2, 3, 6, 8])
    schema = FeatureSchema.from_only_categorical_indices([1, 3], num_columns=5)
    fitted = pipeline.fit_transform(
        X, schema, feature_indices=indices, may_keep_input=True
    ).X
    assert isinstance(fitted, np.ndarray)
    before = X_test.copy()

    shared = pipeline.transform(X_test, feature_indices=indices, may_keep_input=True).X

    assert isinstance(shared, SharedColumnsTable)
    assert not np.shares_memory(shared.base, X_test)
    assert shared.own.shape[1] == 2
    np.testing.assert_array_equal(X_test, before)
    np.testing.assert_array_equal(
        np.asarray(shared), pipeline.transform(X_test[:, indices].copy()).X
    )


def test__keep_float32_pipeline__may_keep_input__appending_reshape__matches() -> None:
    """A reshape that appends the original columns hands sklearn an array."""

    def pipeline() -> PreprocessingPipeline:
        return PreprocessingPipeline(
            [
                RemoveConstantFeaturesStep(),
                ReshapeFeatureDistributionsStep(
                    transform_name="none",
                    append_to_original=True,
                    schedule_gpu_transform=GPUTransformType.QUANTILE,
                ),
                EncodeCategoricalFeaturesStep("ordinal_shuffled", random_state=0),
            ]
        )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    X[:, 1] = rng.integers(0, 4, size=200)
    schema = FeatureSchema.from_only_categorical_indices([1], num_columns=6)
    copying, sharing = pipeline(), pipeline()

    expected = copying.fit_transform(X.copy(), schema)
    actual = sharing.fit_transform(X, schema, may_keep_input=True)

    np.testing.assert_array_equal(np.asarray(actual.X), expected.X)
    np.testing.assert_array_equal(
        np.asarray(sharing.transform(X, may_keep_input=True).X),
        copying.transform(X.copy()).X,
    )


@pytest.mark.parametrize("encoding", ["ordinal", "ordinal_shuffled"])
def test__keep_float32_pipeline__may_keep_input__all_categorical__leaves_x(
    encoding: str,
) -> None:
    """With every column encoded, the encoding still writes into its own columns."""

    def pipeline() -> PreprocessingPipeline:
        return PreprocessingPipeline(
            [
                RemoveConstantFeaturesStep(),
                ReshapeFeatureDistributionsStep(transform_name="none", random_state=0),
                EncodeCategoricalFeaturesStep(encoding, random_state=0),
            ]
        )

    rng = np.random.default_rng(0)
    # Category values that are not their codes, so writing codes changes them.
    X, X_test = (
        (10 * rng.integers(1, 6, (rows, 3))).astype(np.float32) for rows in (200, 40)
    )
    X_before, X_test_before = X.copy(), X_test.copy()
    schema = FeatureSchema.from_only_categorical_indices([0, 1, 2], num_columns=3)
    copying, sharing = pipeline(), pipeline()

    expected = copying.fit_transform(X.copy(), schema)
    actual = sharing.fit_transform(X, schema, may_keep_input=True)
    shared = sharing.transform(X_test, may_keep_input=True).X

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(X_test, X_test_before)
    np.testing.assert_array_equal(np.asarray(actual.X), expected.X)
    np.testing.assert_array_equal(
        np.asarray(shared), copying.transform(X_test.copy()).X
    )
