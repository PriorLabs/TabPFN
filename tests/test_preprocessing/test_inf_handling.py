#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for infinity passthrough handling.

Infinities are passed through preprocessing by the pipeline itself rather than
by dedicated steps: :func:`_extract_inf_masks` records and NaN's them before the
steps run, and :func:`_restore_inf_masks` writes them back afterwards, mapping
renamed/derived columns back to their source via :attr:`Feature.ancestor`. This
module covers those helpers, the round-trip through a real (renaming) step, and
the propagation of ``passthrough_inf`` through validation and the ensemble.
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.errors import TabPFNValidationError
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import (
    PreprocessingPipeline,
    clean_data,
    generate_classification_ensemble_configs,
    generate_regression_ensemble_configs,
)
from tabpfn.preprocessing.clean import (
    PANDAS_FASTER_THAN_MIXED_PATH,
    _inf_masks_numpy_numeric_,
    _inf_masks_pandas_only,
    _is_single_float_block,
    fix_dtypes,
    numeric_columns,
    process_text_na_dataframe,
)
from tabpfn.preprocessing.configs import PreprocessorConfig
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.pipeline_interface import (
    _extract_inf_masks,
    _restore_inf_masks,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder
from tabpfn.preprocessing.steps.reshape_feature_distribution_step import (
    ReshapeFeatureDistributionsStep,
)
from tabpfn.validation import ensure_compatible_fit_inputs_sklearn


def _numerical_schema(num_features: int) -> FeatureSchema:
    """Create FeatureSchema with numerical features only."""
    return FeatureSchema(
        features=[
            Feature(name=f"input_f{i}", modality=FeatureModality.NUMERICAL)
            for i in range(num_features)
        ]
    )


def _reshape_pipeline(*, append_to_original: bool = False) -> PreprocessingPipeline:
    """A one-step pipeline whose step renames the columns it transforms."""
    return PreprocessingPipeline(
        [
            ReshapeFeatureDistributionsStep(
                transform_name="quantile_uni_coarse",
                apply_to_categorical=False,
                append_to_original=append_to_original,
            )
        ]
    )


# --- _extract_inf_masks / _restore_inf_masks ------------------------------------


def test__extract_inf_masks__records_per_feature_and_nans_in_place() -> None:
    """Only features with infinities are recorded; the infs become NaN in X."""
    X = np.array([[1.0, np.inf, 3.0], [4.0, 5.0, -np.inf], [7.0, 8.0, 9.0]])
    schema = _numerical_schema(num_features=3)

    masks = _extract_inf_masks(X, schema)

    assert set(masks) == {"input_f1", "input_f2"}
    np.testing.assert_array_equal(masks["input_f1"], np.array([np.inf, 0.0, 0.0]))
    np.testing.assert_array_equal(masks["input_f2"], np.array([0.0, -np.inf, 0.0]))
    # Infinities are replaced with NaN, finite entries untouched.
    assert np.isnan(X[0, 1])
    assert np.isnan(X[1, 2])
    assert not np.isinf(X).any()
    assert X[0, 0] == 1.0
    assert X[2, 1] == 8.0


def test__extract_inf_masks__noop_on_finite_input() -> None:
    """Finite input yields an empty mapping and leaves X unchanged."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = X.copy()

    masks = _extract_inf_masks(X, _numerical_schema(num_features=2))

    assert masks == {}
    np.testing.assert_array_equal(X, original)


def test__extract_inf_masks__records_tensor_for_torch_input() -> None:
    """The recorded mask is a tensor when the input is a torch tensor."""
    X = torch.tensor([[1.0, float("inf"), 3.0], [4.0, 5.0, float("-inf")]])

    masks = _extract_inf_masks(X, _numerical_schema(num_features=3))

    assert set(masks) == {"input_f1", "input_f2"}
    assert all(isinstance(m, torch.Tensor) for m in masks.values())
    assert torch.isnan(X[0, 1])
    assert torch.isnan(X[1, 2])


def test__restore_inf_masks__matches_by_name() -> None:
    """Infinities are written back into the column with the recorded name."""
    X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    schema = _numerical_schema(num_features=3)
    masks = {
        "input_f1": np.array([np.inf, 0.0]),
        "input_f2": np.array([0.0, -np.inf]),
    }

    _restore_inf_masks(X, schema, masks)

    assert np.isposinf(X[0, 1])
    assert np.isneginf(X[1, 2])


def test__restore_inf_masks__matches_by_ancestor_and_one_to_many() -> None:
    """A renamed column restores via ancestor; many columns can share a source."""
    X = np.full((2, 2), np.nan)
    # Both columns derive from the same input feature.
    schema = FeatureSchema(
        features=[
            Feature(
                name="reshape_0",
                modality=FeatureModality.NUMERICAL,
                ancestor="input_f0",
            ),
            Feature(
                name="reshape_0_copy",
                modality=FeatureModality.NUMERICAL,
                ancestor="input_f0",
            ),
        ]
    )
    masks = {"input_f0": np.array([np.inf, 0.0])}

    _restore_inf_masks(X, schema, masks)

    assert np.isposinf(X[0, 0])
    assert np.isposinf(X[0, 1])


# --- end-to-end pipeline round-trip (through a renaming step) -------------------


_ROUND_TRIP_DATA = [
    [1.0, np.inf, 3.0],
    [4.0, 5.0, -np.inf],
    [7.0, 8.0, 9.0],
    [2.0, 1.0, 0.5],
    [3.0, 2.0, 1.5],
]


def test__pipeline__round_trips_infinities_through_renaming_step() -> None:
    """Infinities survive a step that renames the columns it transforms."""
    X = np.array(_ROUND_TRIP_DATA)
    result = _reshape_pipeline().fit_transform(X, _numerical_schema(num_features=3))

    # The transformed columns are renamed, but the infs land at their origin.
    assert [f.name for f in result.feature_schema.features] == [
        "reshape_0",
        "reshape_1",
        "reshape_2",
    ]
    assert np.isposinf(result.X[0, 1])
    assert np.isneginf(result.X[1, 2])


def test__pipeline__round_trips_infinities_for_torch_input() -> None:
    """Torch input round-trips even though the sklearn step returns numpy.

    The recorded (torch) mask is coerced to the output array kind before the
    infinities are written back.
    """
    X = torch.tensor(_ROUND_TRIP_DATA)
    result = _reshape_pipeline().fit_transform(X, _numerical_schema(num_features=3))

    assert np.isposinf(np.asarray(result.X)[0, 1])
    assert np.isneginf(np.asarray(result.X)[1, 2])


def test__pipeline__append_to_original_restores_into_every_descendant() -> None:
    """With append_to_original, both the original and its copy get the inf back."""
    X = np.array(
        [[1.0, np.inf, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 3.0, 4.0]]
    )
    result = _reshape_pipeline(append_to_original=True).fit_transform(
        X, _numerical_schema(num_features=3)
    )

    names = [f.name for f in result.feature_schema.features]
    # Original f1 (kept) and its appended reshaped copy both carry the inf.
    orig_idx = names.index("input_f1")
    posinf_cols = {c for _, c in np.argwhere(np.isposinf(result.X))}
    assert orig_idx in posinf_cols
    assert len(posinf_cols) == 2  # original + one appended copy


def test__pipeline__predict_restores_test_pattern_not_train() -> None:
    """Predict restores the test data's infinities, not the fitted train pattern.

    The mask is recomputed each call, so a test set whose inf pattern differs
    from train restores correctly (regression test for the old fitted-mask bug).
    """
    pipeline = _reshape_pipeline()
    schema = _numerical_schema(num_features=3)

    X_train = np.array(
        [[1.0, np.inf, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 3.0, 4.0]]
    )
    pipeline.fit_transform(X_train.copy(), schema)

    # Different inf pattern at predict time: inf at [2, 0].
    X_test = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [np.inf, 8.0, 9.0], [2.0, 3.0, 4.0]]
    )
    result = pipeline.transform(X_test.copy())

    assert np.isposinf(result.X[2, 0])  # the test infinity survives ...
    assert not np.isinf(result.X[0]).any()  # ... no train infinity is fabricated


# --- validation gating and ensemble propagation --------------------------------


def _preprocessed_X_trains(configs, X_train, y_train) -> list[np.ndarray]:
    """Run the ensemble preprocessor and return each member's preprocessed X_train."""
    feature_schema = FeatureSchema.from_only_categorical_indices([], X_train.shape[1])
    preprocessor = TabPFNEnsemblePreprocessor(
        configs=configs,
        n_samples=X_train.shape[0],
        feature_schema=feature_schema,
        random_state=0,
        n_preprocessing_jobs=1,
    )
    members = preprocessor.fit_transform_ensemble_members(
        X_train=X_train, y_train=y_train
    )
    return [np.asarray(m.X_train) for m in members]


def test__classifier_preprocessing__identical_without_inf_regardless_of_flag() -> None:
    """On finite input, passthrough_inf does not change classifier preprocessing."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 5))
    y_train = rng.integers(0, 3, 40)

    common = {
        "num_estimators": 3,
        "add_fingerprint_feature": False,
        "polynomial_features": "no",
        "feature_shift_decoder": None,
        "preprocessor_configs": [
            PreprocessorConfig("none", categorical_name="numeric")
        ],
        "class_shift_method": None,
        "n_classes": 3,
        "random_state": 0,
        "num_models": 1,
        "outlier_removal_std": None,
    }
    with_inf = generate_classification_ensemble_configs(**common, passthrough_inf=True)
    without_inf = generate_classification_ensemble_configs(**common)

    out_with = _preprocessed_X_trains(with_inf, X_train, y_train)
    out_without = _preprocessed_X_trains(without_inf, X_train, y_train)

    assert len(out_with) == len(out_without)
    for a, b in zip(out_with, out_without, strict=True):
        np.testing.assert_array_equal(a, b)


def test__regressor_preprocessing__identical_without_inf_regardless_of_flag() -> None:
    """On finite input, passthrough_inf does not change regressor preprocessing."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 5))
    y_train = rng.standard_normal(40)

    common = {
        "num_estimators": 3,
        "add_fingerprint_feature": False,
        "polynomial_features": "no",
        "feature_shift_decoder": None,
        "preprocessor_configs": [
            PreprocessorConfig("none", categorical_name="numeric")
        ],
        "target_transforms": [None],
        "random_state": 0,
        "num_models": 1,
        "outlier_removal_std": None,
    }
    with_inf = generate_regression_ensemble_configs(**common, passthrough_inf=True)
    without_inf = generate_regression_ensemble_configs(**common)

    out_with = _preprocessed_X_trains(with_inf, X_train, y_train)
    out_without = _preprocessed_X_trains(without_inf, X_train, y_train)

    assert len(out_with) == len(out_without)
    for a, b in zip(out_with, out_without, strict=True):
        np.testing.assert_array_equal(a, b)


# --- end-to-end full-pipeline inf passthrough ----------------------------------

# (label, preprocessor config, add_fingerprint_feature, polynomial_features).
# Chosen to exercise the renaming reshape step, GPU-schedulable presets, SVD and
# fingerprint appended columns, one-hot expansion, and multi-output transforms.
_E2E_PRESETS = [
    ("none", PreprocessorConfig("none", categorical_name="numeric"), False, "no"),
    (
        "quantile_uni_coarse",
        PreprocessorConfig("quantile_uni_coarse", categorical_name="numeric"),
        False,
        "no",
    ),
    (
        "squashing_scaler_default",
        PreprocessorConfig("squashing_scaler_default", categorical_name="numeric"),
        False,
        "no",
    ),
    (
        "norm_and_kdi",
        PreprocessorConfig("norm_and_kdi", categorical_name="numeric"),
        False,
        "no",
    ),
    (
        "svd+fingerprint+poly",
        PreprocessorConfig(
            "quantile_uni_coarse",
            categorical_name="numeric",
            global_transformer_name="svd",
        ),
        True,
        3,
    ),
    (
        "onehot",
        PreprocessorConfig("quantile_uni_coarse", categorical_name="onehot"),
        False,
        "no",
    ),
]


def _infinity_rows(X: np.ndarray) -> tuple[set[int], set[int]]:
    """Return (rows with any +inf, rows with any -inf) in ``X``."""
    return (
        set(np.where(np.isposinf(X).any(axis=1))[0].tolist()),
        set(np.where(np.isneginf(X).any(axis=1))[0].tolist()),
    )


@pytest.mark.parametrize(
    ("label", "preprocessor_config", "add_fp", "poly"), _E2E_PRESETS
)
def test__full_pipeline__passes_infinities_through_to_preprocessed_output(
    label: str,
    preprocessor_config: PreprocessorConfig,
    add_fp: bool,
    poly,
) -> None:
    """Infinities survive the full factory pipeline at their original rows.

    Runs the real ensemble preprocessor (poly -> remove-constant -> reshape ->
    encode -> SVD -> fingerprint -> shuffle) with ``passthrough_inf=True``. The
    output adds/reorders/renames columns, so we assert the robust invariant: the
    set of rows carrying a +/-inf is exactly the input's, never fabricated
    elsewhere. The +inf at row 3 and -inf at row 7 must reach the model input.
    """
    del label
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 5))
    y_train = rng.integers(0, 3, 40)
    X_train[3, 1] = np.inf
    X_train[7, 2] = -np.inf

    configs = generate_classification_ensemble_configs(
        num_estimators=2,
        add_fingerprint_feature=add_fp,
        polynomial_features=poly,
        feature_shift_decoder="shuffle",  # exercise the column permutation
        preprocessor_configs=[preprocessor_config],
        class_shift_method=None,
        n_classes=3,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
        passthrough_inf=True,
    )

    members = _preprocessed_X_trains(configs, X_train, y_train)

    assert members  # sanity: the preprocessor produced ensemble members
    for out in members:
        pos_rows, neg_rows = _infinity_rows(out)
        assert pos_rows == {3}
        assert neg_rows == {7}


def test__full_pipeline__omits_infinities_when_passthrough_disabled_is_unused() -> None:
    """Without infinities present the full pipeline output is finite.

    The handling is unconditional but a no-op on finite input, so a finite run
    produces no fabricated infinities regardless of the flag.
    """
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 5))
    y_train = rng.integers(0, 3, 40)

    configs = generate_classification_ensemble_configs(
        num_estimators=2,
        add_fingerprint_feature=True,
        polynomial_features="no",
        feature_shift_decoder="shuffle",
        preprocessor_configs=[
            PreprocessorConfig(
                "quantile_uni_coarse",
                categorical_name="numeric",
                global_transformer_name="svd",
            )
        ],
        class_shift_method=None,
        n_classes=3,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
        passthrough_inf=True,
    )

    for out in _preprocessed_X_trains(configs, X_train, y_train):
        assert not np.isinf(out).any()


def test__regressor_full_pipeline__passes_infinities_through() -> None:
    """The regressor config path also preserves infinities end-to-end (with SVD)."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 5))
    y_train = rng.standard_normal(40)
    X_train[3, 1] = np.inf
    X_train[7, 2] = -np.inf

    configs = generate_regression_ensemble_configs(
        num_estimators=2,
        add_fingerprint_feature=True,
        polynomial_features="no",
        feature_shift_decoder="shuffle",
        preprocessor_configs=[
            PreprocessorConfig(
                "quantile_uni_coarse",
                categorical_name="numeric",
                global_transformer_name="svd",
            )
        ],
        target_transforms=[None],
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
        passthrough_inf=True,
    )

    for out in _preprocessed_X_trains(configs, X_train, y_train):
        pos_rows, neg_rows = _infinity_rows(out)
        assert pos_rows == {3}
        assert neg_rows == {7}


def test__generate_classification_configs__propagates_passthrough_inf() -> None:
    """The flag reaches every generated classifier config."""
    common = {
        "num_estimators": 3,
        "add_fingerprint_feature": False,
        "polynomial_features": "no",
        "feature_shift_decoder": None,
        "preprocessor_configs": [PreprocessorConfig("none")],
        "class_shift_method": None,
        "n_classes": 2,
        "random_state": 0,
        "num_models": 1,
        "outlier_removal_std": None,
    }

    enabled = generate_classification_ensemble_configs(**common, passthrough_inf=True)
    disabled = generate_classification_ensemble_configs(**common)

    assert all(c.passthrough_inf for c in enabled)
    assert not any(c.passthrough_inf for c in disabled)


def test__generate_regression_configs__propagates_passthrough_inf() -> None:
    """The flag reaches every generated regressor config."""
    common = {
        "num_estimators": 3,
        "add_fingerprint_feature": False,
        "polynomial_features": "no",
        "feature_shift_decoder": None,
        "preprocessor_configs": [PreprocessorConfig("none")],
        "target_transforms": [None],
        "random_state": 0,
        "num_models": 1,
        "outlier_removal_std": None,
    }

    enabled = generate_regression_ensemble_configs(**common, passthrough_inf=True)
    disabled = generate_regression_ensemble_configs(**common)

    assert all(c.passthrough_inf for c in enabled)
    assert not any(c.passthrough_inf for c in disabled)


def _classifier_with_resolved_config(*, passthrough_inf: bool) -> TabPFNClassifier:
    """Classifier whose ``inference_config_`` is resolved as ``fit()`` would, but
    without loading weights -- the validation helpers read it directly.
    """
    estimator = TabPFNClassifier()
    estimator.inference_config_ = dataclasses.replace(
        InferenceConfig.get_default("multiclass", ModelVersion.V2),
        PASSTHROUGH_INF=passthrough_inf,
    )
    return estimator


def test__fit_validation__accepts_infinities_when_passthrough_enabled() -> None:
    """Input validation lets infinities through when ``PASSTHROUGH_INF=True``."""
    X = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]])
    y = np.array([0, 1, 0])
    estimator = _classifier_with_resolved_config(passthrough_inf=True)

    X_out, _, _, _ = ensure_compatible_fit_inputs_sklearn(X, y, estimator=estimator)

    assert np.isposinf(X_out[0, 1])


def test__fit_validation__rejects_infinities_when_passthrough_disabled() -> None:
    """Input validation rejects infinities when ``PASSTHROUGH_INF=False``."""
    X = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]])
    y = np.array([0, 1, 0])
    estimator = _classifier_with_resolved_config(passthrough_inf=False)

    with pytest.raises(TabPFNValidationError):
        ensure_compatible_fit_inputs_sklearn(X, y, estimator=estimator)


@pytest.mark.parametrize("passthrough_inf", [True, False])
def test__classifier_fit_predict__handles_infinities_per_passthrough_flag(
    passthrough_inf: bool,
) -> None:
    """End-to-end fit/predict with infinities in X.

    With the flag enabled the fit must succeed (infinities are NaN'd through
    preprocessing and written back); with it disabled they are rejected at
    validation.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5))
    y = (X[:, 0] > 0).astype(int)
    X[3, 1] = np.inf
    X[7, 2] = -np.inf

    model = TabPFNClassifier(
        n_estimators=1, inference_config={"PASSTHROUGH_INF": passthrough_inf}
    )
    if passthrough_inf:
        # Passing infs through preprocessing must not silently produce invalid
        # values (NaN from e.g. inf-inf); errstate turns any such op into a hard
        # error. Scoped to this branch: the reject branch trips sklearn's own
        # finite check (it sums +inf/-inf -> NaN before raising its error).
        with np.errstate(invalid="raise"):
            model.fit(X, y)
            predictions = model.predict(X)
        assert predictions.shape == (X.shape[0],)
        assert np.isfinite(np.asarray(predictions)).all()
    else:
        with pytest.raises(TabPFNValidationError):
            model.fit(X, y)


@pytest.mark.parametrize("passthrough_inf", [True, False])
def test__regressor_fit_predict__handles_infinities_per_passthrough_flag(
    passthrough_inf: bool,
) -> None:
    """End-to-end regressor fit/predict with infinities in X (see classifier twin)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5))
    y = X[:, 0] + rng.standard_normal(60) * 0.1
    X[3, 1] = np.inf
    X[7, 2] = -np.inf

    model = TabPFNRegressor(
        n_estimators=1, inference_config={"PASSTHROUGH_INF": passthrough_inf}
    )
    if passthrough_inf:
        # Passing infs through preprocessing must not silently produce invalid
        # values (NaN from e.g. inf-inf); errstate turns any such op into a hard
        # error. Scoped to this branch: the reject branch trips sklearn's own
        # finite check (it sums +inf/-inf -> NaN before raising its error).
        with np.errstate(invalid="raise"):
            model.fit(X, y)
            predictions = model.predict(X)
        assert predictions.shape == (X.shape[0],)
        assert np.isfinite(np.asarray(predictions)).all()
    else:
        with pytest.raises(TabPFNValidationError):
            model.fit(X, y)


# All v2.x architectures (download weights on first use). v2.6 has no
# fit_with_cache support yet -- it's only a guard (regressor.py / classifier.py:
# "fit_with_cache is not supported for TabPFN v2.6 yet."). strict xfail makes the
# day that guard is lifted fail loudly, prompting this marker's removal.
_FIT_WITH_CACHE_VERSIONS = [
    ModelVersion.V2,
    ModelVersion.V2_5,
    pytest.param(
        ModelVersion.V2_6,
        marks=pytest.mark.xfail(
            raises=ValueError, strict=True, reason="fit_with_cache guard for v2.6"
        ),
    ),
    ModelVersion.V3,
]


def _nonfinite_tensors(
    obj: object, path: str = "kv_cache", _seen: set[int] | None = None
) -> list[str]:
    """Paths to every tensor in ``obj`` (recursing arbitrarily) that holds inf/NaN.

    Walks attributes, dicts, and sequences so it covers any architecture's cache
    layout without naming fields. The cache build refits the standard scaler; if
    raw +/-inf reached it the cached stats would be poisoned (the bug this guards).
    """
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return []
    _seen.add(id(obj))

    if isinstance(obj, torch.Tensor):
        return [] if torch.isfinite(obj).all() else [f"{path} {tuple(obj.shape)}"]

    items: Iterable[tuple[Any, Any]]
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(obj, "__dict__"):
        items = vars(obj).items()
    else:
        return []

    return [
        bad
        for key, value in items
        for bad in _nonfinite_tensors(value, f"{path}.{key}", _seen)
    ]


def _assert_kv_caches_finite(model: TabPFNClassifier | TabPFNRegressor) -> None:
    """Every tensor in every per-member KV cache must be finite (no inf/NaN)."""
    bad = _nonfinite_tensors(getattr(model.executor_, "kv_caches", []))
    assert not bad, f"non-finite tensors in kv cache: {bad}"


@pytest.mark.parametrize("model_version", _FIT_WITH_CACHE_VERSIONS)
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_cache__inf_in_train_does_not_degenerate_clean_test(
    estimator_cls: type[TabPFNClassifier] | type[TabPFNRegressor],
    model_version: ModelVersion,
) -> None:
    """Regression test for the ``fit_with_cache`` + ``passthrough_inf`` cache bug.

    The v3 architecture used to build its inference KV cache by re-fitting the
    standard scaler on the *raw* train tensor, whose passed-through +/-inf
    poisoned the cached mean/std (``torch_nanmean``/``torch_nanstd`` ignore NaN
    but not inf). At predict time, standardising the (finite) test rows with
    those statistics produced ``(finite - inf)/inf = NaN`` for every cell of the
    affected columns, NaN logits, and ``nan_to_num`` then collapsed every
    prediction to a constant (ROC-AUC 0.5 / degenerate regression output).

    The scaler statistics are now fitted on the imputed train rows and reused for
    the cache, so the train and test rows are standardised identically. Predicting
    on fully finite test data must yield finite, non-constant outputs.
    """
    rng = np.random.default_rng(0)
    n_features = 8
    X_train = rng.standard_normal((120, n_features))
    signal = X_train[:, 0].copy()  # copy before corruption (column 0 may get inf'd)
    X_test = rng.standard_normal((40, n_features))

    # Corrupt ~5% of train cells with +inf, hitting most columns (the test set
    # stays finite, mirroring the real-world failure).
    n_corrupt = round(0.05 * X_train.size)
    flat = rng.choice(X_train.size, size=n_corrupt, replace=False)
    rows, cols = np.unravel_index(flat, X_train.shape)
    X_train[rows, cols] = np.inf

    if estimator_cls is TabPFNClassifier:
        y_train = (signal > 0).astype(int)
        model = estimator_cls.create_default_for_version(
            model_version,
            n_estimators=1,
            fit_mode="fit_with_cache",
            inference_config={"PASSTHROUGH_INF": True},
        )
        model.fit(X_train, y_train)
        _assert_kv_caches_finite(model)
        proba = model.predict_proba(X_test)
        assert np.isfinite(proba).all()
        # Degenerate (poisoned-cache) output is identical for every row.
        assert proba[:, 1].std() > 1e-3
    else:
        y_train = signal + rng.standard_normal(120) * 0.1
        model = estimator_cls.create_default_for_version(
            model_version,
            n_estimators=1,
            fit_mode="fit_with_cache",
            inference_config={"PASSTHROUGH_INF": True},
        )
        model.fit(X_train, y_train)
        _assert_kv_caches_finite(model)
        preds = model.predict(X_test)
        assert np.isfinite(preds).all()
        assert np.asarray(preds).std() > 1e-3


def test__clean_data__handles_infinities_on_categoricals() -> None:
    """MRE For a bug triggered in "tabpfn.preprocessing.clean.clean_data()"
    where categoricals containing infs crash the ordinal encoder.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 5))

    # setup a categorical column
    X[:, 0] = rng.integers(0, 3, size=X.shape[0])
    # add an inf
    X[0, 0] = np.inf
    schema = FeatureSchema(
        features=[Feature(name="f0", modality=FeatureModality.CATEGORICAL)]
        + [
            Feature(name=f"f{idx}", modality=FeatureModality.CATEGORICAL)
            for idx in range(X.shape[-1] - 1)
        ]
    )
    X_clean, *_ = clean_data(X, schema, passthrough_inf=True)
    assert np.all(np.isinf(X_clean) == np.isinf(X))


# --- process_text_na_dataframe inf-mask fast path -------------------------------
#
# `process_text_na_dataframe` records +/-inf positions before ordinal encoding and
# restores them afterwards. The recording splits work by dtype: numeric columns are
# tested directly with numpy (the fast path), while non-numeric columns (object /
# string / categorical that may hold a python ``float('inf')``) fall back to the
# slower whole-column pandas comparison. Both must agree with the original
# pure-pandas semantics: an element is +inf iff ``X == np.inf`` (NA -> False).


def _numpy_split_inf_masks(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The per-block numpy path (numeric columns extracted into one float array)."""
    pos_inf = np.zeros(X.shape, dtype=bool)
    neg_inf = np.zeros(X.shape, dtype=bool)
    numeric_col_mask = numeric_columns(X)
    _inf_masks_numpy_numeric_(X, numeric_col_mask, pos_inf, neg_inf)
    return pos_inf, neg_inf


def _median_runtime(fn) -> float:
    """Median wall-clock of ``fn`` over 11 runs after a warm-up call."""
    fn()  # warm up
    samples = []
    for _ in range(11):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _restored_inf_masks(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run the real cleaning step and read back where infinities landed.

    Output columns align positionally with ``X``'s columns, so the returned masks
    can be compared cell-for-cell against :func:`_pandas_inf_reference`.
    """
    out = process_text_na_dataframe(
        X.copy(),  # the function mutates its input
        ord_encoder=get_ordinal_encoder(),
        fit_encoder=True,
        passthrough_inf=True,
    )
    return np.isposinf(out), np.isneginf(out)


def _inf_dtype_frames() -> dict[str, pd.DataFrame]:
    """DataFrames exercising every dtype branch of the inf-mask split."""
    inf, ninf = np.inf, -np.inf
    return {
        # numeric-only -> fast numpy path only
        "all_float64": pd.DataFrame(
            {"a": [1.0, inf, 3.0, 4.0], "b": [ninf, 2.0, 3.0, 4.0]}
        ),
        # object column holding a python float('inf') -> slow pandas path
        "object_and_float": pd.DataFrame(
            {
                "a": pd.Series([inf, 1.0, 2.0, 3.0], dtype="object"),
                "b": [1.0, 2.0, inf, 4.0],
            }
        ),
        # string column never holds infinities; its comparison is a nullable
        # boolean that must coerce to all-False, not crash the bool indexing
        "string_and_float": pd.DataFrame(
            {
                "s": pd.array(["x", "y", "z", "w"], dtype="string"),
                "n": [inf, 2.0, ninf, 4.0],
            }
        ),
        # categorical is non-numeric -> slow path; one category is +inf
        "categorical_and_float": pd.DataFrame(
            {
                "c": pd.Series([inf, 1.0, 2.0, 1.0]).astype("category"),
                "n": [1.0, 2.0, 3.0, ninf],
            }
        ),
        # no numeric columns at all -> only the slow pandas path runs
        "all_object": pd.DataFrame(
            {
                "a": pd.Series([inf, 1.0, 2.0, 3.0], dtype="object"),
                "b": pd.Series([1.0, ninf, 2.0, 3.0], dtype="object"),
            }
        ),
        # finite numeric frame -> the fast path must fabricate no infinities
        "finite_only": pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]}
        ),
    }


@pytest.mark.parametrize("frame_id", list(_inf_dtype_frames()))
def test__process_text_na_dataframe__inf_masks_match_pandas_reference(
    frame_id: str,
) -> None:
    """The numpy fast path reproduces the original pure-pandas +/-inf semantics.

    Regression guard for the dtype-split mask computation: for each dtype mix the
    restored infinities must land in exactly the cells the whole-frame pandas
    comparison would mark, with no fabricated or dropped infinities.
    """
    X = _inf_dtype_frames()[frame_id]
    ref_pos, ref_neg = _inf_masks_pandas_only(X)

    got_pos, got_neg = _restored_inf_masks(X)

    np.testing.assert_array_equal(got_pos, ref_pos)
    np.testing.assert_array_equal(got_neg, ref_neg)


def test__process_text_na_dataframe__nullable_numeric_na_not_seen_as_inf() -> None:
    """Nullable extension numerics survive the fast path via the real fit pipeline.

    Nullable ``Int64``/``Float64`` reach this step only after ``fix_dtypes`` has
    cast them to ``float64`` (pd.NA -> NaN, +/-inf preserved). The numpy fast path
    then tests them with ``np.float64`` values, so a missing entry (NaN) must never
    be mistaken for an infinity while a genuine inf is carried through.
    """
    raw = pd.DataFrame(
        {
            "f": pd.array([1.0, np.inf, pd.NA, 4.0], dtype="Float64"),
            "i": pd.array([1, 2, 3, pd.NA], dtype="Int64"),  # ints are never inf
            "g": [-np.inf, 2.0, 3.0, 4.0],
        }
    )
    X = fix_dtypes(raw, cat_indices=None)

    got_pos, got_neg = _restored_inf_masks(X)

    expected_pos = np.zeros(X.shape, dtype=bool)
    expected_neg = np.zeros(X.shape, dtype=bool)
    expected_pos[1, 0] = True  # +inf in column 'f'
    expected_neg[0, 2] = True  # -inf in column 'g'
    np.testing.assert_array_equal(got_pos, expected_pos)
    np.testing.assert_array_equal(got_neg, expected_neg)


def test__is_single_float_block__distinguishes_consolidated_from_fragmented() -> None:
    """The fast-path predicate fires only for a single contiguous float block."""
    rng = np.random.default_rng(0)
    consolidated = pd.DataFrame(rng.standard_normal((4, 3)))
    assert _is_single_float_block(consolidated)

    # fix_dtypes assigns column-by-column, fragmenting into one block per column.
    fragmented = fix_dtypes(rng.standard_normal((4, 3)), cat_indices=None)
    assert not _is_single_float_block(fragmented)

    # A mixed frame is never a single float block.
    mixed = pd.DataFrame({"n": [1.0, 2.0], "s": pd.array(["a", "b"], dtype="string")})
    assert not _is_single_float_block(mixed)


def test__process_text_na_dataframe__single_float_block_round_trips_infs() -> None:
    """The consolidated-float fast path restores +/-inf at their original cells."""
    X_np = np.arange(12, dtype="float64").reshape(4, 3)
    X_np[0, 1] = np.inf
    X_np[3, 2] = -np.inf
    X = pd.DataFrame(X_np)
    assert _is_single_float_block(X)  # the branch under test is actually taken

    ref_pos, ref_neg = _inf_masks_pandas_only(X)
    got_pos, got_neg = _restored_inf_masks(X)

    np.testing.assert_array_equal(got_pos, ref_pos)
    np.testing.assert_array_equal(got_neg, ref_neg)
    assert got_pos[0, 1]
    assert got_neg[3, 2]


@pytest.mark.skipif(
    PANDAS_FASTER_THAN_MIXED_PATH, reason="native pandas <3.0.0 is faster and in use"
)
def test__inf_mask__per_block_path_not_slower_than_pandas_on_fragmented() -> None:
    """On a fragmented frame the per-block numpy path beats pure pandas.

    A wide numeric frame shaped like a real post-``fix_dtypes`` input (one block
    per column) is the layout that reaches ``process_text_na_dataframe`` in
    practice; there the per-block numpy path beats the whole-frame pandas
    comparison (~1.5x locally). The assertion is a no-regression guard with
    generous slack so it survives CI noise; marked ``slow`` to run at merge time.
    """
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((2000, 300))
    X_np[0, 0] = np.inf
    X = fix_dtypes(X_np, cat_indices=None)
    assert not _is_single_float_block(X)  # routed to the per-block path

    # Same result, so the speed comparison is apples-to-apples.
    np.testing.assert_array_equal(
        np.stack(_numpy_split_inf_masks(X)), np.stack(_inf_masks_pandas_only(X))
    )

    numpy_median = _median_runtime(lambda: _numpy_split_inf_masks(X))
    pandas_median = _median_runtime(lambda: _inf_masks_pandas_only(X))

    # ~1.5x faster locally; allow 25% slack so only a genuine regression fails.
    assert numpy_median < pandas_median * 1.25, (
        f"per-block inf-mask path regressed: {numpy_median * 1e3:.2f}ms vs "
        f"pandas {pandas_median * 1e3:.2f}ms"
    )


def test__inf_mask__pandas_path_not_slower_than_per_block_on_single_block() -> None:
    """On a consolidated float frame the fast pandas path beats per-block numpy.

    A single contiguous float block compares element-wise on that block and
    ``to_numpy()`` is a view, so pandas beats extracting the columns into a fresh
    array (~3x locally). Guards the fast-pandas branch added for this layout.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((2000, 300)))
    X.iloc[0, 0] = np.inf
    assert _is_single_float_block(X)  # routed to the fast pandas path

    np.testing.assert_array_equal(
        np.stack(_inf_masks_pandas_only(X)), np.stack(_numpy_split_inf_masks(X))
    )

    pandas_median = _median_runtime(lambda: _inf_masks_pandas_only(X))
    numpy_median = _median_runtime(lambda: _numpy_split_inf_masks(X))

    # ~3x faster locally; allow 25% slack so only a genuine regression fails.
    assert pandas_median < numpy_median * 1.25, (
        f"single-block fast-pandas path regressed: {pandas_median * 1e3:.2f}ms vs "
        f"per-block {numpy_median * 1e3:.2f}ms"
    )


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__estimator_fit_predict__handles_infinities_on_all_dtypes(
    estimator_cls: type[TabPFNClassifier] | type[TabPFNRegressor],
) -> None:
    """The model should support infs in all dtypes."""
    rng = np.random.default_rng(0)

    features = rng.normal(size=4)
    if estimator_cls is TabPFNClassifier:
        y = (features > 0).astype(int)
    else:
        y = features + rng.standard_normal(features.shape[0]) * 0.1

    # dataframe with inf features in first row
    features[0] = np.inf
    X = pd.DataFrame(
        {
            "num": features,
            # ints cannot be inf-valued
            "int": [np.inf, 3, 1, 2],
            "cat": [np.inf, 2, 1, 2],
            "cat_obj": [np.inf, "1", "1", "2"],
            "txt": [np.inf, "foo", "bar", "baz"],
        }
    )

    # setup categorical columns
    X = X.astype(
        {
            col: pd.CategoricalDtype(set(X[col]))
            for col in X.columns
            if col.startswith("cat")
        }
    )

    model = estimator_cls(
        n_estimators=1,
        inference_config={"PASSTHROUGH_INF": True},
        categorical_features_indices=[0],
    )
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (X.shape[0],)
    assert np.isfinite(np.asarray(predictions)).all()


# --- CUDA end-to-end (real GPU hardware) ---------------------------------------

_CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a CUDA device")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__estimator_fit_predict_on_cuda__passes_infinities_through(
    estimator_cls: type[TabPFNClassifier] | type[TabPFNRegressor],
) -> None:
    """End-to-end fit/predict with infinities on a real CUDA device.

    Exercises the GPU (torch) preprocessing pipeline on actual hardware -- where
    SVD/quantile would crash on raw +/-inf -- so it covers the CPU->GPU inf
    round-trip that the CPU-only and CPU-tensor torch tests cannot. Skipped where
    no GPU is present.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5))
    X[3, 1] = np.inf
    X[7, 2] = -np.inf
    if estimator_cls is TabPFNClassifier:
        y = (X[:, 0] > 0).astype(int)
    else:
        y = X[:, 0] + rng.standard_normal(60) * 0.1

    model = estimator_cls(
        n_estimators=2, inference_config={"PASSTHROUGH_INF": True}, device="cuda"
    )
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (X.shape[0],)
    assert np.isfinite(np.asarray(predictions)).all()


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a CUDA device")
def test__classifier_fit_on_cuda__rejects_infinities_without_passthrough() -> None:
    """On CUDA too, infinities are rejected at validation when passthrough is off."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5))
    y = (X[:, 0] > 0).astype(int)
    X[3, 1] = np.inf

    model = TabPFNClassifier(
        n_estimators=1, inference_config={"PASSTHROUGH_INF": False}, device="cuda"
    )
    with pytest.raises(TabPFNValidationError):
        model.fit(X, y)
