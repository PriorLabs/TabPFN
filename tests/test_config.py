from __future__ import annotations

from dataclasses import asdict, field
from typing import Any
from typing_extensions import override

import pytest
from pydantic.dataclasses import dataclass

from tabpfn.architectures import ARCHITECTURES
from tabpfn.architectures.base.config import ModelConfig
from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    ArchitectureModule,
)


@pytest.mark.parametrize(("architecture"), [ARCHITECTURES["base"]])
def test__parse_config__no_unused_keys__returns_empty_dict(
    architecture: ArchitectureModule,
) -> None:
    config, unused_config = architecture.parse_config(
        {"max_num_classes": 3, "num_buckets": 10}
    )
    assert isinstance(config, ArchitectureConfig)
    assert unused_config == {}


@pytest.mark.parametrize(("architecture"), [ARCHITECTURES["base"]])
def test__parse_config__unused_keys__returns_unused_config(
    architecture: ArchitectureModule,
) -> None:
    config, unused_config = architecture.parse_config(
        {
            "max_num_classes": 3,
            "num_buckets": 10,
            "unexpected_key": "value",
            "unexpected_sub_object": {"another_key": "value_2"},
        }
    )
    assert isinstance(config, ArchitectureConfig)
    assert unused_config == {
        "unexpected_key": "value",
        "unexpected_sub_object": {"another_key": "value_2"},
    }


@dataclass
class FakeConfig(ArchitectureConfig):
    a: int = 1
    b: FakeSubConfig = field(default_factory=lambda: FakeSubConfig())


@dataclass
class FakeSubConfig:
    c: int = 2


class FakeArchitectureModule(ArchitectureModule):
    @override
    def parse_config(
        self, config: dict[str, Any]
    ) -> tuple[ArchitectureConfig, dict[str, Any]]:
        parsed_config = FakeConfig(**config)
        return parsed_config, parsed_config.get_unused_config(config)

    @override
    def get_architecture(
        self,
        config: ArchitectureConfig,
        *,
        n_out: int,
        cache_trainset_representation: bool,
    ) -> Architecture:
        raise NotImplementedError()


def test__parse_config__nested_config__no_unused_keys__returns_empty_dict() -> None:
    config, unused_config = FakeArchitectureModule().parse_config(
        {"max_num_classes": 10, "num_buckets": 100, "a": 1, "b": {"c": 3}}
    )
    assert isinstance(config, FakeConfig)
    assert unused_config == {}


def test__parse_config__nested_config__unused_keys__returns_unused_keys() -> None:
    config, unused_config = FakeArchitectureModule().parse_config(
        {
            "max_num_classes": 10,
            "num_buckets": 100,
            "a": 1,
            "b": {"c": 3, "extra": "value"},
        }
    )
    assert isinstance(config, FakeConfig)
    assert unused_config == {"b": {"extra": "value"}}


def test__base_config__upgrade__no_old_keys__does_nothing() -> None:
    config = asdict(
        ModelConfig(
            emsize=16,
            features_per_group=1,
            max_num_classes=1,
            nhead=2,
            remove_duplicate_features=False,
            num_buckets=1000,
        )
    )

    assert ModelConfig.upgrade_config(config) == config


def test__base_config__attention_type__old_and_new_set__raises_value_error() -> None:
    config = asdict(
        ModelConfig(
            emsize=16,
            features_per_group=1,
            max_num_classes=1,
            nhead=2,
            remove_duplicate_features=False,
            num_buckets=1000,
        )
    )
    config["attention_type"] = "full"
    config["item_attention_type"] = "full"
    config["feature_attention_type"] = "full"

    with pytest.raises(
        ValueError, match="Can't have both old and new attention types set"
    ):
        ModelConfig.upgrade_config(config)
