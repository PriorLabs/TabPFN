"""Tests for the InferenceConfig."""

from __future__ import annotations

import io
from dataclasses import asdict

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures import base
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.architectures.base.config import ModelConfig
from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
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
    config = ModelConfig(
        emsize=8,
        features_per_group=1,
        max_num_classes=10,
        nhead=2,
        nlayers=2,
        remove_duplicate_features=True,
        num_buckets=100,
    )
    model = base.get_architecture(config=config, cache_trainset_representation=False)
    inference_config = InferenceConfig.get_default(
        task_type="multiclass", model_version=ModelVersion.V2_5
    )
    return ClassifierModelSpecs(
        model=model,
        architecture_config=config,
        inference_config=inference_config,
    )


def _make_regressor_specs() -> RegressorModelSpecs:
    config = ModelConfig(
        emsize=8,
        features_per_group=1,
        max_num_classes=10,
        nhead=2,
        nlayers=2,
        remove_duplicate_features=True,
        num_buckets=100,
    )
    model = base.get_architecture(config=config, cache_trainset_representation=False)
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
