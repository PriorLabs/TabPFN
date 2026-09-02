#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the InferenceConfig."""

from __future__ import annotations

import inspect
import io
import warnings
from dataclasses import asdict, fields, replace

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
    OVERRIDABLE_FIELDS,
    InferenceConfig,
    cpu_sample_limit,
)
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.preprocessing.ensemble import DEFAULT_N_ESTIMATORS


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
    with pytest.warns(
        FutureWarning, match="deprecated and will be removed in a future version"
    ):
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


def test__equals_ignoring_overridable_fields__only_those_differ__is_equal() -> None:
    config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2_5
    )
    other = replace(config, SOFTMAX_TEMPERATURE=0.5, N_ESTIMATORS=3)

    assert other != config
    assert other.equals_ignoring_overridable_fields(config)
    assert config.equals_ignoring_overridable_fields(other)


def test__equals_ignoring_overridable_fields__other_field_differs__is_not_equal() -> (
    None
):
    config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2_5
    )
    other = replace(
        config, SOFTMAX_TEMPERATURE=0.5, N_ESTIMATORS=3, POLYNOMIAL_FEATURES="all"
    )

    assert not other.equals_ignoring_overridable_fields(config)


def test__overridable_fields__name_real_fields_and_arguments() -> None:
    """The table drives both error messages and the resolution, so it must line up.

    Each key is a config field and each argument is the estimator argument that
    sets it, which `_resolve_overrides` reads off the estimator by name.
    """
    field_names = {f.name for f in fields(InferenceConfig)}
    for field, (plural, argument) in OVERRIDABLE_FIELDS.items():
        assert field in field_names
        assert plural
        assert argument in inspect.signature(TabPFNClassifier).parameters
        assert argument in inspect.signature(TabPFNRegressor).parameters


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


@pytest.mark.parametrize(
    "inference_config",
    [
        {"SOFTMAX_TEMPERATURE": 0.7},
        InferenceConfig.get_default("multiclass", ModelVersion.V2_5),
    ],
)
def test__classifier_softmax_temperature__named_twice__raises(
    classification_data: tuple[np.ndarray, np.ndarray],
    inference_config: dict | InferenceConfig,
) -> None:
    """Which of the two would win is not apparent from the call, so neither does.

    A hand-built `InferenceConfig` always carries a temperature, so passing one
    counts as naming it.
    """
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(0.42),
        device="cpu",
        softmax_temperature=1.5,
        inference_config=inference_config,
    )
    with pytest.raises(ValueError, match="`softmax_temperature` was given twice"):
        clf.fit(X, y)


def test__classifier_softmax_temperature__inference_config_without_it__is_allowed(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Only naming the temperature clashes, not passing an `inference_config`."""
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(0.42),
        device="cpu",
        softmax_temperature=1.5,
        inference_config={"POLYNOMIAL_FEATURES": "all"},
    )
    clf.fit(X, y)

    assert clf.softmax_temperature_ == 1.5
    assert clf.get_inference_config().POLYNOMIAL_FEATURES == "all"


def test__classifier_softmax_temperature__reaches_the_predictions(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The temperature a checkpoint declares must behave like passing that value."""
    X, y = classification_data
    base = _make_classifier_specs()

    def probas_for(specs: ClassifierModelSpecs, **kwargs: object) -> np.ndarray:
        clf = TabPFNClassifier(
            model_path=specs, device="cpu", n_estimators=2, random_state=0, **kwargs
        )
        clf.fit(X, y)
        return clf.predict_proba(X)

    from_checkpoint = probas_for(_with_temperature(base, 0.3))
    from_argument = probas_for(base, softmax_temperature=0.3)
    untouched = probas_for(_with_temperature(base, 1.0))

    np.testing.assert_allclose(from_checkpoint, from_argument)
    assert not np.allclose(from_checkpoint, untouched)


def test__classifier_softmax_temperature__checkpoints_disagree__raises(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """One temperature is applied to the whole ensemble, so two of them is an error."""
    X, y = classification_data
    specs = [_classifier_specs(0.5), _classifier_specs(1.2)]

    clf = TabPFNClassifier(model_path=specs, device="cpu", n_estimators=2)
    with pytest.raises(ValueError, match="different softmax temperatures"):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"softmax_temperature": 0.7},
        {"inference_config": {"SOFTMAX_TEMPERATURE": 0.7}},
    ],
)
def test__classifier_softmax_temperature__disagreeing_checkpoints_and_override__is_used(
    classification_data: tuple[np.ndarray, np.ndarray],
    kwargs: dict,
) -> None:
    """Naming a temperature is what the error tells the user to do, so it must work."""
    X, y = classification_data
    specs = [_classifier_specs(0.5), _classifier_specs(1.2)]

    clf = TabPFNClassifier(model_path=specs, device="cpu", n_estimators=2, **kwargs)
    clf.fit(X, y)

    assert clf.softmax_temperature_ == 0.7
    assert clf.predict_proba(X).shape == (X.shape[0], 3)


def test__classifier_softmax_temperature__checkpoints_agree__is_used(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    specs = [_classifier_specs(0.5), _classifier_specs(0.5)]

    clf = TabPFNClassifier(model_path=specs, device="cpu", n_estimators=2)
    clf.fit(X, y)

    assert clf.softmax_temperature_ == 0.5


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


def test__regressor_softmax_temperature__reaches_the_predictions() -> None:
    """The temperature a checkpoint declares must behave like passing that value."""
    X, y = sklearn.datasets.make_regression(
        n_samples=30, n_features=4, noise=10.0, random_state=0
    )
    base = _make_regressor_specs(max_num_classes=0)

    def predictions_for(temperature: float, **kwargs: object) -> np.ndarray:
        specs = RegressorModelSpecs(
            model=base.model,
            architecture_config=base.architecture_config,
            inference_config=replace(
                base.inference_config, SOFTMAX_TEMPERATURE=temperature
            ),
            norm_criterion=base.norm_criterion,
        )
        reg = TabPFNRegressor(
            model_path=specs, device="cpu", n_estimators=2, random_state=0, **kwargs
        )
        reg.fit(X, y)
        return reg.predict(X)

    from_checkpoint = predictions_for(0.3)
    from_argument = predictions_for(1.0, softmax_temperature=0.3)
    untouched = predictions_for(1.0)

    np.testing.assert_allclose(from_checkpoint, from_argument)
    assert not np.allclose(from_checkpoint, untouched)


def test__regressor_softmax_temperature__checkpoints_disagree__raises() -> None:
    X, y = sklearn.datasets.make_regression(
        n_samples=30, n_features=4, noise=10.0, random_state=0
    )
    base = _make_regressor_specs(max_num_classes=0)
    other = RegressorModelSpecs(
        model=base.model,
        architecture_config=base.architecture_config,
        inference_config=replace(base.inference_config, SOFTMAX_TEMPERATURE=1.2),
        norm_criterion=base.norm_criterion,
    )

    reg = TabPFNRegressor(model_path=[base, other], device="cpu", n_estimators=2)
    with pytest.raises(ValueError, match="different softmax temperatures"):
        reg.fit(X, y)


# =============================================================================
# Deprecation of the `InferenceConfig` object form of `inference_config`
# =============================================================================


def test__inference_config__object_form__warns_it_is_deprecated(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The object form is the one that silently discards checkpoint values."""
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(),
        device="cpu",
        inference_config=InferenceConfig.get_default("multiclass", ModelVersion.V2_5),
    )

    with pytest.warns(
        FutureWarning, match="deprecated and will be removed in a future version"
    ):
        clf.fit(X, y)


def test__inference_config__dict_form__does_not_warn(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The recommended form is not deprecated."""
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs(),
        device="cpu",
        inference_config={"POLYNOMIAL_FEATURES": "all"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        clf.fit(X, y)


def test__inference_config__not_given__does_not_warn(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(model_path=_classifier_specs(), device="cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        clf.fit(X, y)


def test__inference_config__object_form__still_replaces_the_whole_config(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Deprecated, not yet changed: the object still wins over the checkpoint."""
    X, y = classification_data
    specs = _classifier_specs(0.42)
    specs.inference_config = replace(
        specs.inference_config, MAX_NUMBER_OF_FEATURES=1234
    )
    user_config = InferenceConfig.get_default("multiclass", ModelVersion.V2_5)

    clf = TabPFNClassifier(model_path=specs, device="cpu", inference_config=user_config)
    with pytest.warns(FutureWarning):
        clf.fit(X, y)

    resolved = clf.get_inference_config()
    assert resolved.SOFTMAX_TEMPERATURE == DEFAULT_SOFTMAX_TEMPERATURE
    assert resolved.MAX_NUMBER_OF_FEATURES == user_config.MAX_NUMBER_OF_FEATURES != 1234


def test__inference_config__dict_form__keeps_the_fields_it_does_not_name(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The migration the warning recommends: a dict leaves the checkpoint alone."""
    X, y = classification_data
    specs = _classifier_specs(0.42)
    specs.inference_config = replace(
        specs.inference_config, MAX_NUMBER_OF_FEATURES=1234
    )

    clf = TabPFNClassifier(
        model_path=specs, device="cpu", inference_config={"POLYNOMIAL_FEATURES": "all"}
    )
    clf.fit(X, y)

    resolved = clf.get_inference_config()
    assert resolved.POLYNOMIAL_FEATURES == "all"
    assert resolved.SOFTMAX_TEMPERATURE == 0.42
    assert resolved.MAX_NUMBER_OF_FEATURES == 1234


def test__inference_config__asdict_of_a_config__matches_the_object_form(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The escape hatch the warning names, for callers that do want a full replace."""
    X, y = classification_data
    specs = _classifier_specs(0.42)
    user_config = replace(
        InferenceConfig.get_default("multiclass", ModelVersion.V2_5),
        POLYNOMIAL_FEATURES="all",
    )

    from_object = TabPFNClassifier(
        model_path=specs, device="cpu", inference_config=user_config
    )
    with pytest.warns(FutureWarning):
        from_object.fit(X, y)

    from_dict = TabPFNClassifier(
        model_path=specs, device="cpu", inference_config=asdict(user_config)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        from_dict.fit(X, y)

    assert from_dict.get_inference_config() == from_object.get_inference_config()


# =============================================================================
# N_ESTIMATORS
# =============================================================================


def _classifier_specs_with_n_estimators(n_estimators: int) -> ClassifierModelSpecs:
    specs = _make_classifier_specs()
    specs.inference_config = replace(specs.inference_config, N_ESTIMATORS=n_estimators)
    return specs


def test__n_estimators__omitted_from_config__falls_back_to_legacy_default() -> None:
    """Checkpoints predating the field must keep the count they shipped with."""
    config = InferenceConfig(PREPROCESS_TRANSFORMS=[])
    assert config.N_ESTIMATORS == DEFAULT_N_ESTIMATORS == 8

    for model_version in (ModelVersion.V2, ModelVersion.V2_5):
        for task_type in ("multiclass", "regression"):
            default = InferenceConfig.get_default(
                task_type=task_type,  # type: ignore[arg-type]
                model_version=model_version,
            )
            assert default.N_ESTIMATORS == DEFAULT_N_ESTIMATORS


def test__classifier_n_estimators__auto__taken_from_checkpoint(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs_with_n_estimators(3), device="cpu"
    )
    clf.fit(X, y)

    assert clf.n_estimators == "auto"
    assert clf.n_estimators_ == 3
    assert clf.get_inference_config().N_ESTIMATORS == 3


def test__classifier_n_estimators__auto__unset_checkpoint_runs_legacy_default(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(model_path=_make_classifier_specs(), device="cpu")
    clf.fit(X, y)

    assert clf.n_estimators_ == DEFAULT_N_ESTIMATORS


def test__classifier_n_estimators__explicit__overrides_checkpoint(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs_with_n_estimators(3),
        device="cpu",
        n_estimators=5,
    )
    clf.fit(X, y)

    assert clf.n_estimators_ == 5


def test__classifier_n_estimators__explicit__stays_out_of_the_config(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """`inference_config_` is persisted into checkpoints by `save_tabpfn_model`.

    It therefore has to keep describing the model, not this run's compute budget --
    fine-tuning runs a handful of estimators on purpose, and that handful must not
    end up declared by every checkpoint it writes.
    """
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs_with_n_estimators(3),
        device="cpu",
        n_estimators=5,
    )
    clf.fit(X, y)

    assert clf.n_estimators_ == 5
    assert clf.get_inference_config().N_ESTIMATORS == 3


def test__classifier_n_estimators__inference_config__overrides_checkpoint(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs_with_n_estimators(3),
        device="cpu",
        inference_config={"N_ESTIMATORS": 6},
    )
    clf.fit(X, y)

    assert clf.n_estimators_ == 6


def test__classifier_n_estimators__named_twice__raises(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Which of the two would win is not apparent from the call, so neither does."""
    X, y = classification_data
    clf = TabPFNClassifier(
        model_path=_classifier_specs_with_n_estimators(3),
        device="cpu",
        n_estimators=5,
        inference_config={"N_ESTIMATORS": 6},
    )
    with pytest.raises(ValueError, match="`n_estimators` was given twice"):
        clf.fit(X, y)


def test__classifier_n_estimators__checkpoints_disagree__raises(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """One count is applied to the whole ensemble, so two of them is an error."""
    X, y = classification_data
    specs = [
        _classifier_specs_with_n_estimators(3),
        _classifier_specs_with_n_estimators(4),
    ]

    clf = TabPFNClassifier(model_path=specs, device="cpu")
    with pytest.raises(ValueError, match="different numbers of estimators"):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 2},
        {"inference_config": {"N_ESTIMATORS": 2}},
    ],
)
def test__classifier_n_estimators__disagreeing_checkpoints_and_override__is_used(
    classification_data: tuple[np.ndarray, np.ndarray],
    kwargs: dict,
) -> None:
    """Naming a count is what the error tells the user to do, so it must work."""
    X, y = classification_data
    specs = [
        _classifier_specs_with_n_estimators(3),
        _classifier_specs_with_n_estimators(4),
    ]

    clf = TabPFNClassifier(model_path=specs, device="cpu", **kwargs)
    clf.fit(X, y)

    assert clf.n_estimators_ == 2


def test__classifier_n_estimators__checkpoints_agree__is_used(
    classification_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = classification_data
    specs = [
        _classifier_specs_with_n_estimators(3),
        _classifier_specs_with_n_estimators(3),
    ]

    clf = TabPFNClassifier(model_path=specs, device="cpu")
    clf.fit(X, y)

    assert clf.n_estimators_ == 3


def test__regressor_n_estimators__auto__taken_from_checkpoint() -> None:
    X, y = sklearn.datasets.make_regression(
        n_samples=30, n_features=4, noise=10.0, random_state=0
    )
    specs = _make_regressor_specs(max_num_classes=0)
    specs.inference_config = replace(specs.inference_config, N_ESTIMATORS=3)

    reg = TabPFNRegressor(model_path=specs, device="cpu")
    reg.fit(X, y)

    assert reg.n_estimators == "auto"
    assert reg.n_estimators_ == 3


def test__regressor_n_estimators__explicit__overrides_checkpoint() -> None:
    X, y = sklearn.datasets.make_regression(
        n_samples=30, n_features=4, noise=10.0, random_state=0
    )
    specs = _make_regressor_specs(max_num_classes=0)
    specs.inference_config = replace(specs.inference_config, N_ESTIMATORS=3)

    reg = TabPFNRegressor(model_path=specs, device="cpu", n_estimators=5)
    reg.fit(X, y)

    assert reg.n_estimators_ == 5
    assert reg.get_inference_config().N_ESTIMATORS == 3
