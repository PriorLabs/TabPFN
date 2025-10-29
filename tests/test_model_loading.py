from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, overload
from typing_extensions import override
from unittest.mock import patch

import pytest
import torch
from pydantic.dataclasses import dataclass
from torch import Tensor

from tabpfn import model_loading
from tabpfn.architectures import ARCHITECTURES, base
from tabpfn.architectures.base.config import ModelConfig
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    ArchitectureModule,
)
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import PreprocessorConfig


def test__load_model__no_architecture_name_in_checkpoint__loads_base_architecture(
    tmp_path: Path,
) -> None:
    config = _get_minimal_base_architecture_config()
    model = base.get_architecture(config, n_out=10, cache_trainset_representation=True)
    checkpoint = {"state_dict": model.state_dict(), "config": asdict(config)}
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    loaded_model, _, loaded_config, _ = model_loading.load_model(path=checkpoint_path)
    assert isinstance(loaded_model, PerFeatureTransformer)
    assert isinstance(loaded_config, ModelConfig)


def _get_minimal_base_architecture_config() -> ModelConfig:
    return ModelConfig(
        emsize=8,
        features_per_group=1,
        max_num_classes=10,
        nhead=2,
        nlayers=2,
        remove_duplicate_features=True,
        num_buckets=1000,
    )


class FakeArchitectureModule(ArchitectureModule):
    @override
    def parse_config(
        self, config: dict[str, Any]
    ) -> tuple[ArchitectureConfig, dict[str, Any]]:
        return FakeConfig(**config), {}

    @override
    def get_architecture(
        self,
        config: ArchitectureConfig,
        *,
        n_out: int,
        cache_trainset_representation: bool,
    ) -> Architecture:
        return DummyArchitecture()


@dataclass
class FakeConfig(ArchitectureConfig):
    key_a: str = "a_value"


class DummyArchitecture(Architecture):
    """The interface that all architectures must implement.

    Architectures are PyTorch modules, which is then wrapped by e.g.
    TabPFNClassifier or TabPFNRegressor to form the complete model.
    """

    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[True] = True,
        categorical_inds: list[list[int]] | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[False],
        categorical_inds: list[list[int]] | None = None,
    ) -> dict[str, Tensor]: ...

    @override
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
    ) -> Tensor | dict[str, Tensor]:
        raise NotImplementedError()


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_model__architecture_name_in_checkpoint__loads_specified_architecture(
    tmp_path: Path,
) -> None:
    config_dict = {
        "max_num_classes": 10,
        "num_buckets": 100,
    }
    checkpoint = {
        "state_dict": {},
        "config": config_dict,
        "architecture_name": "fake_arch",
    }
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    loaded_model, _, loaded_config, _ = model_loading.load_model(path=checkpoint_path)
    assert isinstance(loaded_model, DummyArchitecture)
    assert isinstance(loaded_config, FakeConfig)


def test__load_v2_checkpoint__returns_v2_default_preprocessings(
    tmp_path: Path,
) -> None:
    arch_config = _get_minimal_base_architecture_config()
    model = base.get_architecture(
        arch_config, n_out=10, cache_trainset_representation=True
    )
    # v2 checkpoints have no "architecture_name" key
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": asdict(arch_config),
    }
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    _, _, _, inference_config = model_loading.load_model_criterion_config(
        model_path=[checkpoint_path, checkpoint_path],
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2",
        download_if_not_exists=False,
    )

    assert inference_config.PREPROCESS_TRANSFORMS == "v2_default"


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_post_v2_ckpt_without_inference_config__returns_none_preprocessings(
    tmp_path: Path,
) -> None:
    arch_config = {"max_num_classes": 10, "num_buckets": 100}
    checkpoint = {
        "state_dict": {},
        "config": arch_config,
        "architecture_name": "fake_arch",
    }
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    _, _, _, inference_config = model_loading.load_model_criterion_config(
        model_path=[checkpoint_path, checkpoint_path],
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2",
        download_if_not_exists=False,
    )

    assert inference_config.PREPROCESS_TRANSFORMS is None


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_multiple_models_with_difference_inference_configs__raises(
    tmp_path: Path,
) -> None:
    arch_config = {"max_num_classes": 10, "num_buckets": 100}
    checkpoint_1 = {
        "state_dict": {},
        "config": arch_config,
        "architecture_name": "fake_arch",
        "inference_config": asdict(
            InferenceConfig(
                PREPROCESS_TRANSFORMS=[
                    PreprocessorConfig(
                        "quantile_uni_coarse",
                        append_original="auto",
                        categorical_name="ordinal_very_common_categories_shuffled",
                        global_transformer_name="svd",
                        subsample_features=-1,
                    )
                ]
            )
        ),
    }
    checkpoint_1_path = tmp_path / "checkpoint1.ckpt"
    torch.save(checkpoint_1, checkpoint_1_path)
    checkpoint_2 = {
        "state_dict": {},
        "config": arch_config,
        "architecture_name": "fake_arch",
        "inference_config": asdict(
            InferenceConfig(
                PREPROCESS_TRANSFORMS=[
                    PreprocessorConfig(
                        "none",
                        categorical_name="numeric",
                        subsample_features=-1,
                    )
                ]
            )
        ),
    }
    checkpoint_2_path = tmp_path / "checkpoint2.ckpt"
    torch.save(checkpoint_2, checkpoint_2_path)

    with pytest.raises(ValueError, match="Inference configs for different models"):
        model_loading.load_model_criterion_config(
            model_path=[checkpoint_1_path, checkpoint_2_path],
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download_if_not_exists=False,
        )
