#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the InferenceConfig."""

from __future__ import annotations

import io
from dataclasses import asdict, replace

import numpy as np
import pytest
import sklearn.datasets
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures import tabpfn_v2_5
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import (
    DEFAULT_SOFTMAX_TEMPERATURE,
    InferenceConfig,
    cpu_sample_limit,
)
from tabpfn.preprocessing import PreprocessorConfig


def test__save_and_load__loaded_value_equal_to_saved() -> None:
    config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2_5
    )

    with io.BytesIO() as buffer:
        torch.save(asdict(config), buffer)
        buffer.seek(0)
        loaded_config = InferenceConfig(**torch.load(buffer, weights_only=False))

    assert loaded_config == config


def test__override_with_user_input__dict_of_overrides__sets_values_correctly() -> None:
    config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2
    )
    overrides = {
        "PREPROCESS_TRANSFORMS": [
            {
                "name": "adaptive",
                "append_original": "auto",
                "categorical_name": "ordinal_very_common_categories_shuffled",
                "global_transformer_name": "svd",
            }
        ],
        "POLYNOMIAL_FEATURES": "all",
    }
    new_config = config.override_with_user_input_and_resolve_auto(overrides)
    assert new_config is not config
    assert new_config != config
    assert isinstance(new_config.PREPROCESS_TRANSFORMS[0], PreprocessorConfig)
    assert new_config.PREPROCESS_TRANSFORMS[0].name == "adaptive"
    assert new_config.POLYNOMIAL_FEATURES == "all"


def test__override_with_user_input__config_override__replaces_entire_config() -> None:
    config = InferenceConfig.get_default(
        task_type="regression", model_version=ModelVersion.V2
    )
    override_config = InferenceConfig(
        PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="adaptive")],
        POLYNOMIAL_FEATURES="all",
    )
    new_config = config.override_with_user_input_and_resolve_auto(override_config)
    assert new_config is not config
    assert new_config != config
    assert new_config == override_config


def test__override_with_user_input__override_is_None__returns_copy_of_config() -> None:
    config = InferenceConfig.get_default(
        task_type="regression", model_version=ModelVersion.V2_5
    )
    new_config = config.override_with_user_input_and_resolve_auto(user_config=None)
    assert new_config is not config
    assert new_config == config


def _make_classifier_specs() -> ClassifierModelSpecs:
    config = tabpfn_v2_5.TabPFNV2p5Config(
        emsize=8,
        features_per_group=1,
        max_num_classes=10,
        nhead=2,
        nlayers=2,
        num_buckets=100,
    )
    model = tabpfn_v2_5.get_architecture(
        config=config, cache_trainset_representation=False
    )
    inference_config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2_5
    )
    return ClassifierModelSpecs(
        model=model,
        architecture_config=config,
        inference_config=inference_config,
    )


def _make_regressor_specs(max_num_classes: int = 10) -> RegressorModelSpecs:
    config = tabpfn_v2_5.TabPFNV2p5Config(
        emsize=8,
        features_per_group=1,
        max_num_classes=max_num_classes,
        nhead=2,
        nlayers=2,
        num_buckets=100,
    )
    model = tabpfn_v2_5.get_architecture(
        config=config, cache_trainset_representation=False
    )
    borders = torch.linspace(-3, 3, config.num_buckets + 1)
    norm_criterion = FullSupportBarDistribution(borders)
    inference_config = InferenceConfig.get_default(
        task_type="regression", model_version=ModelVersion.V2_5
    )
    return RegressorModelSpecs(
        model=model,
        architecture_config=config,
        inference_config=inference_config,
        norm_criterion=norm_criterion,
    )


def test__classifier_get_inference_config__before_fit__returns_config() -> None:
    specs = _make_classifier_specs()
    clf = TabPFNClassifier(model_path=specs, device="cpu")
    assert not hasattr(clf, "inference_config_")
    config = clf.get_inference_config()
    assert isinstance(config, InferenceConfig)
    assert config == specs.inference_config


def test__classifier_get_inference_config__returns_deepcopy() -> None:
    specs = _make_classifier_specs()
    clf = TabPFNClassifier(model_path=specs, device="cpu")
    config = clf.get_inference_config()
    assert config is not clf.inference_config_
    config.PREPROCESS_TRANSFORMS.clear()
    assert len(clf.inference_config_.PREPROCESS_TRANSFORMS) > 0


def test__classifier_get_inference_config__with_override__applies_override() -> None:
    specs = _make_classifier_specs()
    clf = TabPFNClassifier(
        model_path=specs,
        device="cpu",
        inference_config={"POLYNOMIAL_FEATURES": "all"},
    )
    config = clf.get_inference_config()
    assert config.POLYNOMIAL_FEATURES == "all"
    assert specs.inference_config.POLYNOMIAL_FEATURES == "no"


def test__regressor_get_inference_config__before_fit__returns_config() -> None:
    specs = _make_regressor_specs()
    reg = TabPFNRegressor(model_path=specs, device="cpu")
    assert not hasattr(reg, "inference_config_")
    config = reg.get_inference_config()
    assert isinstance(config, InferenceConfig)
    assert config == specs.inference_config


def test__regressor_get_inference_config__returns_deepcopy() -> None:
    specs = _make_regressor_specs()
    reg = TabPFNRegressor(model_path=specs, device="cpu")
    config = reg.get_inference_config()
    assert config is not reg.inference_config_
    config.PREPROCESS_TRANSFORMS.clear()
    assert len(reg.inference_config_.PREPROCESS_TRANSFORMS) > 0


def test__regressor_get_inference_config__with_override__applies_override() -> None:
    specs = _make_regressor_specs()
    reg = TabPFNRegressor(
        model_path=specs,
        device="cpu",
        inference_config={"POLYNOMIAL_FEATURES": "all"},
    )
    config = reg.get_inference_config()
    assert config.POLYNOMIAL_FEATURES == "all"
    assert specs.inference_config.POLYNOMIAL_FEATURES == "no"


def test__cpu_sample_limit__v3__returns_5000() -> None:
    assert cpu_sample_limit(ModelVersion.V3) == 5000


def test__cpu_sample_limit__pre_v3_versions__return_1000() -> None:
    assert cpu_sample_limit(ModelVersion.V2) == 1000
    assert cpu_sample_limit(ModelVersion.V2_5) == 1000
    assert cpu_sample_limit(ModelVersion.V2_6) == 1000


# =============================================================================
# SOFTMAX_TEMPERATURE
# =============================================================================


def test__softmax_temperature__omitted_from_config__falls_back_to_legacy_default() -> (
    None
):
    """Checkpoints predating the field must keep the temperature they shipped with.

    They carry no `SOFTMAX_TEMPERATURE` key, so the class default is the only thing
    standing between them and a silent behavior change.
    """
    config = InferenceConfig(PREPROCESS_TRANSFORMS=[])
    assert config.SOFTMAX_TEMPERATURE == DEFAULT_SOFTMAX_TEMPERATURE == 0.9

    for model_version in (ModelVersion.V2, ModelVersion.V2_5):
        for task_type in ("multiclass", "regression"):
            default = InferenceConfig.get_default(
                task_type=task_type,  # type: ignore[arg-type]
                model_version=model_version,
            )
            assert default.SOFTMAX_TEMPERATURE == DEFAULT_SOFTMAX_TEMPERATURE


def test__equals_ignoring_softmax_temperature__only_temperature_differs__is_equal() -> (
    None
):
    config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2_5
    )
    other = replace(config, SOFTMAX_TEMPERATURE=0.5)

    assert other != config
    assert other.equals_ignoring_softmax_temperature(config)
    assert config.equals_ignoring_softmax_temperature(other)


def test__equals_ignoring_softmax_temperature__other_field_differs__is_not_equal() -> (
    None
):
    config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2_5
    )
    other = replace(config, SOFTMAX_TEMPERATURE=0.5, POLYNOMIAL_FEATURES="all")

    assert not other.equals_ignoring_softmax_temperature(config)


def _classifier_specs(temperature: float | None = None) -> ClassifierModelSpecs:
    specs = _make_classifier_specs()
    if temperature is not None:
        specs.inference_config = replace(
            specs.inference_config, SOFTMAX_TEMPERATURE=temperature
        )
    return specs


def _with_temperature(
    specs: ClassifierModelSpecs, temperature: float
) -> ClassifierModelSpecs:
    """The same weights as `specs`, with a different declared temperature."""
    return ClassifierModelSpecs(
        model=specs.model,
        architecture_config=specs.architecture_config,
        inference_config=replace(
            specs.inference_config, SOFTMAX_TEMPERATURE=temperature
        ),
    )


@pytest.fixture(scope="module")
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    return sklearn.datasets.make_classification(
        n_samples=30,
        n_classes=3,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )


def test__classifier_softmax_temperature__auto__taken_from_checkpoint(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(model_path=_classifier_specs(0.42), device="cpu")
    clf.fit(X, y)

    assert clf.softmax_temperature == "auto"
    assert clf.softmax_temperature_ == 0.42
    assert clf.softmax_temperatures_ == [0.42]
    assert clf.get_inference_config().SOFTMAX_TEMPERATURE == 0.42


def test__classifier_softmax_temperature__explicit__overrides_checkpoint(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(0.42), device="cpu", softmax_temperature=1.5
    )
    clf.fit(X, y)

    assert clf.softmax_temperature_ == 1.5
    assert clf.softmax_temperatures_ == [1.5]
    assert clf.get_inference_config().SOFTMAX_TEMPERATURE == 1.5


def test__classifier_softmax_temperature__inference_config__overrides_checkpoint(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(0.42),
        device="cpu",
        inference_config={"SOFTMAX_TEMPERATURE": 1.5},
    )
    clf.fit(X, y)

    assert clf.softmax_temperature_ == 1.5
    assert clf.softmax_temperatures_ == [1.5]


def test__classifier_softmax_temperature__argument_wins_over_inference_config(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(0.42),
        device="cpu",
        softmax_temperature=1.5,
        inference_config={"SOFTMAX_TEMPERATURE": 0.7},
    )
    clf.fit(X, y)

    assert clf.softmax_temperature_ == 1.5
    assert clf.softmax_temperatures_ == [1.5]


def test__classifier_softmax_temperature__differing_checkpoints__kept_per_model(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Checkpoints declaring different temperatures can be ensembled.

    The rest of the inference config still has to match; only the temperature is
    allowed to differ, and each model keeps its own.
    """
    X, y = classification_data
    specs = [_classifier_specs(0.5), _classifier_specs(1.2)]

    clf = TabPFNClassifier(model_path=specs, device="cpu", n_estimators=2)
    clf.fit(X, y)

    assert clf.softmax_temperatures_ == [0.5, 1.2]
    assert clf.softmax_temperature_ == 0.5
    assert clf.predict_proba(X).shape == (X.shape[0], 3)


def test__classifier_softmax_temperature__per_model__changes_predictions(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """A single model's temperature has to reach the predictions of the ensemble."""
    X, y = classification_data
    base = [_classifier_specs(1.0), _classifier_specs(1.0)]

    def probas_for(second_model_temperature: float) -> np.ndarray:
        clf = TabPFNClassifier(
            model_path=[base[0], _with_temperature(base[1], second_model_temperature)],
            device="cpu",
            n_estimators=2,
            random_state=0,
        )
        clf.fit(X, y)
        return clf.predict_proba(X)

    unchanged = probas_for(1.0)
    sharpened = probas_for(0.3)

    assert not np.allclose(unchanged, sharpened)


def test__classifier_softmax_temperature__equal_per_model__matches_argument(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Checkpoints that agree must behave exactly like passing that temperature."""
    X, y = classification_data
    base = [_classifier_specs(1.0), _classifier_specs(1.0)]

    from_checkpoints = TabPFNClassifier(
        model_path=[_with_temperature(spec, 0.7) for spec in base],
        device="cpu",
        n_estimators=2,
        random_state=0,
    )
    from_checkpoints.fit(X, y)

    from_argument = TabPFNClassifier(
        model_path=base,
        device="cpu",
        n_estimators=2,
        random_state=0,
        softmax_temperature=0.7,
    )
    from_argument.fit(X, y)

    np.testing.assert_allclose(
        from_checkpoints.predict_proba(X), from_argument.predict_proba(X)
    )


def test__logits_to_probabilities__temperature_per_estimator__scales_each_estimator(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The per-estimator temperature tensor scales dim 0 of the raw logits."""
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(), device="cpu", n_estimators=2, random_state=0
    )
    clf.fit(X, y)

    raw_logits = torch.from_numpy(clf.predict_raw_logits(X)).float()
    temperatures = torch.tensor([0.5, 2.0])

    probas = clf.logits_to_probabilities(raw_logits, softmax_temperature=temperatures)

    expected = torch.stack(
        [
            torch.softmax(raw_logits[i] / temperatures[i], dim=-1)
            for i in range(raw_logits.shape[0])
        ]
    ).mean(dim=0)
    torch.testing.assert_close(probas, expected)


def test__regressor_softmax_temperature__auto__taken_from_checkpoint() -> None:
    X, y = sklearn.datasets.make_regression(
        n_samples=30, n_features=4, noise=10.0, random_state=0
    )
    specs = _make_regressor_specs(max_num_classes=0)
    specs.inference_config = replace(specs.inference_config, SOFTMAX_TEMPERATURE=0.42)

    reg = TabPFNRegressor(model_path=specs, device="cpu")
    reg.fit(X, y)

    assert reg.softmax_temperature == "auto"
    assert reg.softmax_temperature_ == 0.42
    assert reg.softmax_temperatures_ == [0.42]


def test__regressor_softmax_temperature__per_model__changes_predictions() -> None:
    X, y = sklearn.datasets.make_regression(
        n_samples=30, n_features=4, noise=10.0, random_state=0
    )
    base = _make_regressor_specs(max_num_classes=0)

    def predictions_for(second_model_temperature: float) -> np.ndarray:
        second = RegressorModelSpecs(
            model=base.model,
            architecture_config=base.architecture_config,
            inference_config=replace(
                base.inference_config, SOFTMAX_TEMPERATURE=second_model_temperature
            ),
            norm_criterion=base.norm_criterion,
        )
        reg = TabPFNRegressor(
            model_path=[base, second], device="cpu", n_estimators=2, random_state=0
        )
        reg.fit(X, y)
        return reg.predict(X)

    unchanged = predictions_for(base.inference_config.SOFTMAX_TEMPERATURE)
    sharpened = predictions_for(0.3)

    assert not np.allclose(unchanged, sharpened)
