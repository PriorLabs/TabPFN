"""Tests for the v2.5 single-file model."""

from __future__ import annotations

import sys

import pytest
import torch

from tabpfn.architectures import tabpfn_v2_6
from tabpfn.architectures.interface import PerformanceOptions


def _get_model() -> tabpfn_v2_6.TabPFNV2p6:
    """Construct v2.5 and base such that they have the same outputs."""
    config = tabpfn_v2_6.TabPFNV2p6Config(
        max_num_classes=10,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        features_per_group=3,
        num_thinking_rows=2,
    )
    model = tabpfn_v2_6.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float64)
    return model


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled() -> None:
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float64)

    output_without_memory_saving = arch(x, y, only_return_standard_out=False)
    output_with_memory_saving = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(save_peak_memory_factor=4),
    )

    msg = "Output keys do not match between implementations"
    assert output_with_memory_saving.keys() == output_without_memory_saving.keys(), msg
    for key in output_with_memory_saving:
        assert torch.allclose(
            output_with_memory_saving[key], output_without_memory_saving[key]
        ), f"Outputs for {key} do not match between implementations."


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_checkpointing_enabled_and_disabled() -> None:
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float64)

    output_without_recomputation = arch(x, y, only_return_standard_out=False)
    output_with_recomputation = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(force_recompute_layer=True),
    )

    msg = "Output keys do not match between implementations"
    assert output_with_recomputation.keys() == output_without_recomputation.keys(), msg
    for key in output_with_recomputation:
        assert torch.allclose(
            output_with_recomputation[key], output_without_recomputation[key]
        ), f"Outputs for {key} do not match between implementations."


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__batch_size_one__padding_still_works() -> None:
    arch = _get_model()

    x = torch.randn(100, 1, 1, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [97, 1], dtype=torch.float64)
    output = arch(x, y)

    assert output.shape == (3, 1, 10)


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward__no_test_set_works_batch_size_one() -> None:
    arch = _get_model()

    x = torch.randn(1, 1, 20, dtype=torch.float64) * 0.1
    y = torch.randint(0, 10, [1, 1], dtype=torch.float64)

    out = arch(x, y, only_return_standard_out=False)
    assert out["standard"].shape == (0, 1, 10)
