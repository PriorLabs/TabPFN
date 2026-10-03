#  Copyright (c) Prior Labs GmbH 2026.

"""Reductions must work when the criterion and the logits are on different devices.

`mean` already re-homed its buffer; `median`, `mode` and `icdf` indexed
`borders` directly and raised. That can happen whenever an estimator is
reassembled and only part of it is moved — e.g. restoring a fitted regressor
and placing the weights but not the bar distribution.
"""

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution


def _second_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _reductions(bd: FullSupportBarDistribution, logits: torch.Tensor) -> dict:
    return {
        "mean": bd.mean(logits),
        "median": bd.median(logits),
        "mode": bd.mode(logits),
        "icdf": bd.icdf(logits, 0.9),
        "quantile": bd.quantile(logits),
    }


@pytest.fixture
def borders() -> torch.Tensor:
    return torch.linspace(-3, 3, 101)


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.randn(64, 100, generator=torch.Generator().manual_seed(0))


def test_reductions_run_with_borders_on_another_device(borders, logits):
    device = _second_device()
    if device is None:
        pytest.skip("needs a second device")

    expected = _reductions(FullSupportBarDistribution(borders.clone()), logits)
    # Borders left on CPU, logits on the compute device.
    got = _reductions(FullSupportBarDistribution(borders.clone()), logits.to(device))

    for name, value in got.items():
        assert value.device.type == device.type, name
        torch.testing.assert_close(
            value.cpu().float(), expected[name].float(), rtol=0, atol=1e-4, msg=name
        )


def test_co_located_results_are_unchanged(borders, logits):
    """The re-homing must be a no-op when the devices already match."""
    bd = FullSupportBarDistribution(borders)
    first = _reductions(bd, logits)
    second = _reductions(bd, logits)
    for name, value in first.items():
        assert torch.equal(value, second[name]), name
