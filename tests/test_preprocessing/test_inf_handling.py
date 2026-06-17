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

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing import (
    PreprocessingPipeline,
    generate_classification_ensemble_configs,
    generate_regression_ensemble_configs,
)
from tabpfn.preprocessing.configs import PreprocessorConfig
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.pipeline_interface import (
    _extract_inf_masks,
    _restore_inf_masks,
)
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


def test__fit_validation__accepts_infinities_when_passthrough_enabled() -> None:
    """Input validation lets infinities through when ``passthrough_inf=True``."""
    X = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]])
    y = np.array([0, 1, 0])
    estimator = TabPFNClassifier(passthrough_inf=True)

    X_out, _, _, _ = ensure_compatible_fit_inputs_sklearn(X, y, estimator=estimator)

    assert np.isposinf(X_out[0, 1])


def test__fit_validation__rejects_infinities_when_passthrough_disabled() -> None:
    """Input validation rejects infinities when ``passthrough_inf=False``."""
    X = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]])
    y = np.array([0, 1, 0])
    estimator = TabPFNClassifier(passthrough_inf=False)

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

    model = TabPFNClassifier(n_estimators=1, passthrough_inf=passthrough_inf)
    if passthrough_inf:
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (X.shape[0],)
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

    model = TabPFNRegressor(n_estimators=1, passthrough_inf=passthrough_inf)
    if passthrough_inf:
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (X.shape[0],)
    else:
        with pytest.raises(TabPFNValidationError):
            model.fit(X, y)
