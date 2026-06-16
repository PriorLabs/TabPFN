#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for infinity passthrough handling.

Covers the :class:`InfToNanStep` / :class:`RestoreInfStep` preprocessing steps,
their wiring into the pipeline via ``EnsembleConfig.passthrough_inf``, and the
propagation of the flag through the ensemble config generators.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing import (
    PreprocessingPipeline,
    generate_classification_ensemble_configs,
    generate_regression_ensemble_configs,
)
from tabpfn.preprocessing.configs import (
    ClassifierEnsembleConfig,
    PreprocessorConfig,
)
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline
from tabpfn.preprocessing.steps.inf_handling import InfToNanStep, RestoreInfStep
from tabpfn.validation import ensure_compatible_fit_inputs_sklearn


def _numerical_schema(num_features: int) -> FeatureSchema:
    """Create FeatureSchema with numerical features only."""
    return FeatureSchema(
        features=[
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(num_features)
        ]
    )


def _step_types(pipeline: PreprocessingPipeline) -> list[str]:
    """Return the class names of the pipeline's steps, unwrapping modality tuples."""
    names = []
    for step in pipeline.steps:
        unwrapped = step[0] if isinstance(step, tuple) else step
        names.append(type(unwrapped).__name__)
    return names


def _classifier_config(*, passthrough_inf: bool) -> ClassifierEnsembleConfig:
    return ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig("none"),
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_count=0,
        feature_shift_decoder=None,
        outlier_removal_std=None,
        _model_index=0,
        class_permutation=None,
        passthrough_inf=passthrough_inf,
    )


def test__inf_to_nan_step__replaces_inf_with_nan() -> None:
    """Infinite values are replaced with NaN, finite values untouched."""
    X = np.array(
        [
            [1.0, np.inf, 3.0],
            [4.0, 5.0, -np.inf],
            [7.0, 8.0, 9.0],
        ]
    )
    schema = _numerical_schema(num_features=3)

    result = InfToNanStep().fit_transform(X, schema)

    assert np.isnan(result.X[0, 1])
    assert np.isnan(result.X[1, 2])
    assert not np.isinf(result.X).any()
    # Finite entries are preserved.
    assert result.X[0, 0] == 1.0
    assert result.X[2, 1] == 8.0
    assert result.X_added is None
    assert result.modality_added is None


def test__inf_to_nan_step__records_inf_mask_per_feature() -> None:
    """Only features containing infinities get an ``inf_mask``."""
    X = np.array(
        [
            [1.0, np.inf, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    schema = _numerical_schema(num_features=3)

    InfToNanStep().fit_transform(X, schema)

    # Feature 0 and 2 are finite -> no mask.
    assert schema.features[0].inf_mask is None
    assert schema.features[2].inf_mask is None
    # Feature 1 has an inf in the first row -> mask records the value, 0 elsewhere.
    mask = schema.features[1].inf_mask
    assert mask is not None
    np.testing.assert_array_equal(mask, np.array([np.inf, 0.0]))


def test__inf_to_nan_step__noop_without_infinities() -> None:
    """All-finite input is returned unchanged with no recorded ``inf_mask``."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    schema = _numerical_schema(num_features=2)

    result = InfToNanStep().fit_transform(X, schema)

    np.testing.assert_array_equal(
        result.X, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    assert all(feat.inf_mask is None for feat in schema.features)
    assert result.X_added is None
    assert result.modality_added is None


def test__restore_inf_step__restores_recorded_infinities() -> None:
    """Round-trip: NaN'd infinities are written back to their original cells."""
    X = np.array(
        [
            [1.0, np.inf, 3.0],
            [4.0, 5.0, -np.inf],
            [7.0, 8.0, 9.0],
        ]
    )
    original = X.copy()
    # Share the same schema (hence the same Feature objects) across both steps so
    # the recorded inf_mask is visible to the restore step.
    schema = _numerical_schema(num_features=3)

    nan_result = InfToNanStep().fit_transform(X, schema)
    restore_result = RestoreInfStep().fit_transform(nan_result.X, schema)

    np.testing.assert_array_equal(restore_result.X, original)
    assert restore_result.X_added is None
    assert restore_result.modality_added is None


def test__restore_inf_step__noop_without_recorded_infinities() -> None:
    """Restore is a no-op when no feature carries an ``inf_mask``."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    schema = _numerical_schema(num_features=2)

    result = RestoreInfStep().fit_transform(X, schema)

    np.testing.assert_array_equal(result.X, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test__pipeline__wraps_with_inf_steps_when_passthrough_enabled() -> None:
    """``passthrough_inf=True`` brackets the pipeline with the inf steps."""
    pipeline = create_preprocessing_pipeline(
        _classifier_config(passthrough_inf=True),
        random_state=0,
    )
    names = _step_types(pipeline)
    assert names[0] == "InfToNanStep"
    assert names[-1] == "RestoreInfStep"


def test__pipeline__omits_inf_steps_when_passthrough_disabled() -> None:
    """``passthrough_inf=False`` leaves the inf steps out entirely."""
    pipeline = create_preprocessing_pipeline(
        _classifier_config(passthrough_inf=False),
        random_state=0,
    )
    names = _step_types(pipeline)
    assert "InfToNanStep" not in names
    assert "RestoreInfStep" not in names


def test__pipeline__round_trip_restores_infinities() -> None:
    """An end-to-end passthrough pipeline preserves infinite values."""
    X = np.array(
        [
            [1.0, np.inf, 3.0],
            [4.0, 5.0, -np.inf],
            [7.0, 8.0, 9.0],
            [2.0, 1.0, 0.5],
        ]
    )
    schema = _numerical_schema(num_features=3)
    pipeline = PreprocessingPipeline([InfToNanStep(), RestoreInfStep()])

    result = pipeline.fit_transform(X.copy(), schema)

    assert np.isposinf(result.X[0, 1])
    assert np.isneginf(result.X[1, 2])


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

    Regression test for a crash where ``passthrough_inf=True`` survived input
    validation but the infinities then reached the ordinal encoder in
    ``clean_data`` and raised. With the flag enabled the fit must succeed; with it
    disabled the infinities are rejected at validation.
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
