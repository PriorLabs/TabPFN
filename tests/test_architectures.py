"""Tests for tabpfn.architectures.__init__."""

from __future__ import annotations

from typing import Any
from typing_extensions import override
from unittest.mock import patch

import pytest

from tabpfn.architectures import ARCHITECTURES, register_architecture
from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    ArchitectureModule,
)


class _FakeArchitectureModule(ArchitectureModule):
    @override
    def parse_config(
        self, config: dict[str, Any]
    ) -> tuple[ArchitectureConfig, dict[str, Any]]:
        raise NotImplementedError

    @override
    def get_architecture(
        self,
        config: ArchitectureConfig,
        *,
        n_out: int,
        cache_trainset_representation: bool,
    ) -> Architecture:
        raise NotImplementedError


@patch.dict(ARCHITECTURES, clear=True)
def test__register_architecture__new_name__adds_to_dict() -> None:
    module = _FakeArchitectureModule()
    register_architecture("test_arch", module)
    assert ARCHITECTURES["test_arch"] is module


@patch.dict(ARCHITECTURES, clear=True)
def test__register_architecture__same_module_twice__succeeds() -> None:
    module = _FakeArchitectureModule()
    register_architecture("test_arch", module)
    register_architecture("test_arch", module)
    assert ARCHITECTURES["test_arch"] is module


@patch.dict(ARCHITECTURES, clear=True)
def test__register_architecture__different_module_same_name__raises_value_error() -> (
    None
):
    module_a = _FakeArchitectureModule()
    module_b = _FakeArchitectureModule()
    register_architecture("test_arch", module_a)
    with pytest.raises(ValueError, match="There is already a different architecture"):
        register_architecture("test_arch", module_b)
    assert ARCHITECTURES["test_arch"] is module_a


@patch.dict(ARCHITECTURES, clear=True)
def test__register_architecture__multiple_different_names__all_registered() -> None:
    module_a = _FakeArchitectureModule()
    module_b = _FakeArchitectureModule()
    register_architecture("arch_a", module_a)
    register_architecture("arch_b", module_b)
    assert ARCHITECTURES["arch_a"] is module_a
    assert ARCHITECTURES["arch_b"] is module_b
