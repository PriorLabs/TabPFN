"""Tests for the InferenceConfig."""

from __future__ import annotations

import io
from dataclasses import asdict

import torch

from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import PreprocessorConfig


def test__save_and_load__loaded_value_equal_to_saved() -> None:
    config = InferenceConfig.get_default(task_type="multiclass", model_version="latest")

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
    new_config = config.override_with_user_input(overrides)
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
    new_config = config.override_with_user_input(override_config)
    assert new_config is not config
    assert new_config != config
    assert new_config == override_config


def test__override_with_user_input__override_is_None__returns_copy_of_config() -> None:
    config = InferenceConfig.get_default(task_type="regression", model_version="latest")
    new_config = config.override_with_user_input(user_config=None)
    assert new_config is not config
    assert new_config == config
