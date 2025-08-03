from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, overload
from typing_extensions import override
from unittest.mock import patch

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


def test__load_model__no_architecture_name_in_checkpoint__loads_base_architecture(
    tmp_path: Path,
) -> None:
    config = _get_minimal_base_architecture_config()
    model = base.get_architecture(config, n_out=10, cache_trainset_representation=True)
    checkpoint = {"state_dict": model.state_dict(), "config": asdict(config)}
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    loaded_model, _, loaded_config = model_loading.load_model(path=checkpoint_path)
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

    loaded_model, _, loaded_config = model_loading.load_model(path=checkpoint_path)
    assert isinstance(loaded_model, DummyArchitecture)
    assert isinstance(loaded_config, FakeConfig)
