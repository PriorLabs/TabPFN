#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.shared import bar_distribution
from tests.utils import get_pytest_devices_with_mps_marked_slow


def _make_full_support_distribution(
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[bar_distribution.FullSupportBarDistribution, torch.Tensor]:
    borders = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=dtype, device=device)
    dist = bar_distribution.FullSupportBarDistribution(borders)
    logits = torch.tensor([0.25, 0.5, 0.25], dtype=dtype, device=device).log()
    return dist, logits


def test_cdf_out_of_bounds():
    logits = torch.tensor([0.05, 0.05, 0.1, 0.3, 0.5]).log()
    d = bar_distribution.BarDistribution(
        borders=torch.tensor([0.0, 1.0, 2.0, 2.5, 4.0, 6.0])
    )

    # outside range
    assert d.cdf(logits, torch.tensor([-1.0]))[0].item() == pytest.approx(0.0)
    assert d.cdf(logits, torch.tensor([7.0]))[0].item() == pytest.approx(1.0)

    # on borders
    assert d.cdf(logits, torch.tensor([0.0]))[0].item() == pytest.approx(0.0)
    assert d.cdf(logits, torch.tensor([1.0]))[0].item() == pytest.approx(0.05)
    assert d.cdf(logits, torch.tensor([2.5]))[0].item() == pytest.approx(0.2)
    assert d.cdf(logits, torch.tensor([6.0]))[0].item() == pytest.approx(1.0)

    # inside bucket
    assert d.cdf(logits, torch.tensor([1.5]))[0].item() == pytest.approx(0.075)


def test_move_to_larger():
    logits = torch.arange(99).float()
    old_d = bar_distribution.BarDistribution(borders=torch.linspace(0, 1, 100))
    new_d = bar_distribution.BarDistribution(borders=torch.linspace(-1, 2, 1000))
    new_logits = old_d.get_probs_for_different_borders(logits, new_d.borders).log()
    assert old_d.median(logits).item() == pytest.approx(new_d.median(new_logits).item())
    assert old_d.mean(logits).item() == pytest.approx(
        new_d.mean(new_logits).item(), abs=1e-4
    )


def test_full_support_cdf_and_icdf_checkpoints():
    dist, logits = _make_full_support_distribution()

    assert dist.icdf(logits, 0.125).item() == pytest.approx(-2.0)
    assert dist.icdf(logits, 0.875).item() == pytest.approx(2.0)

    ys = torch.tensor([float("-inf"), -2.0, -1.0, 0.0, 1.0, 2.0, float("inf")])
    expected = torch.tensor([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
    assert torch.allclose(dist.cdf(logits, ys), expected)


@pytest.mark.parametrize(
    "left_prob",
    [0.0, 0.001, 0.125, 0.249, 0.25, 0.5, 0.75, 0.875, 0.999, 1.0],
)
def test_full_support_cdf_icdf_round_trip(left_prob: float):
    dist, logits = _make_full_support_distribution(dtype=torch.float64)
    batch_logits = torch.stack(
        (logits, torch.tensor([0.1, 0.3, 0.6], dtype=logits.dtype).log())
    )

    values = dist.icdf(batch_logits, left_prob)
    actual = dist.cdf(batch_logits, values.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(
        actual,
        torch.full_like(actual, left_prob),
        atol=1e-12,
        rtol=1e-12,
    )


def test_full_support_inherited_quantiles_and_border_translation():
    dist, logits = _make_full_support_distribution()
    batch_logits = logits.expand(2, -1)

    assert torch.equal(dist.median(batch_logits), dist.icdf(batch_logits, 0.5))
    assert torch.equal(
        dist.quantile(batch_logits, center_prob=0.75),
        torch.stack(
            (dist.icdf(batch_logits, 0.125), dist.icdf(batch_logits, 0.875)),
            dim=-1,
        ),
    )
    assert torch.equal(
        dist.ucb(batch_logits, best_f=0.0, rest_prob=0.125),
        dist.icdf(batch_logits, 0.875),
    )
    assert torch.equal(
        dist.ucb(batch_logits, best_f=0.0, rest_prob=0.125, maximize=False),
        dist.icdf(batch_logits, 0.125),
    )

    new_borders = torch.tensor([-3.0, -2.0, 0.0, 2.0, 3.0])
    translated = dist.get_probs_for_different_borders(batch_logits, new_borders)
    expected = torch.tensor([0.125, 0.375, 0.375, 0.125]).expand(2, -1)
    assert torch.allclose(translated, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", get_pytest_devices_with_mps_marked_slow())
def test_full_support_sample_preserves_shape_device_and_dtype(
    device: str,
    dtype: torch.dtype,
):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64 tensors")

    dist, logits = _make_full_support_distribution(dtype=dtype, device=device)
    batch_logits = logits.expand(2, 3, -1).contiguous()

    scalar_sample = dist.sample(logits)
    samples = dist.sample(batch_logits)

    assert scalar_sample.shape == logits.shape[:-1]
    assert scalar_sample.device.type == torch.device(device).type
    assert scalar_sample.dtype == dtype
    assert torch.isfinite(scalar_sample)
    assert samples.shape == batch_logits.shape[:-1]
    assert samples.device.type == torch.device(device).type
    assert samples.dtype == dtype
    assert torch.isfinite(samples).all()


def test_full_support_sample_matches_tail_mass_and_cdf():
    torch.manual_seed(7)
    dist, logits = _make_full_support_distribution(dtype=torch.float64)
    samples = dist.sample(logits.repeat(20_000, 1))

    outside = ((samples < dist.borders[0]) | (samples > dist.borders[-1])).double()
    assert outside.mean().item() == pytest.approx(0.25, abs=0.015)

    ys = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=samples.dtype)
    empirical_cdf = torch.stack([(samples <= y).double().mean() for y in ys])
    expected_cdf = torch.tensor(
        [0.125, 0.25, 0.5, 0.75, 0.875],
        dtype=samples.dtype,
    )
    assert torch.allclose(empirical_cdf, expected_cdf, atol=0.015, rtol=0.0)


def test_average_bar_distributions_into_different_one():
    num_bars = [100, 80, 10, 5]
    logits = [torch.arange(nb - 1).float() for nb in num_bars]
    bar_dists = [
        bar_distribution.BarDistribution(borders=torch.linspace(-1, -0.5, num_bars[0])),
        bar_distribution.BarDistribution(borders=torch.linspace(0, 2, num_bars[1])),
        bar_distribution.BarDistribution(borders=torch.linspace(1, 3, num_bars[2])),
        bar_distribution.BarDistribution(borders=torch.linspace(2, 3, num_bars[3])),
    ]

    new_d = bar_distribution.BarDistribution(borders=torch.linspace(-1, 2, 100))
    new_logits = new_d.average_bar_distributions_into_this(bar_dists, logits)

    assert new_d.cdf(new_logits, torch.tensor([-1.0])).item() == pytest.approx(0.0)
    assert new_d.cdf(new_logits, torch.tensor([0.0])).item() == pytest.approx(0.25)
    assert new_d.cdf(new_logits, torch.tensor([3.0])).item() == pytest.approx(1.0)

    new_small_d = bar_distribution.BarDistribution(borders=torch.linspace(-1, 2, 10))
    new_small_logits = new_small_d.average_bar_distributions_into_this(
        bar_dists, logits
    )

    assert new_small_d.cdf(
        new_small_logits, torch.tensor([-1.0])
    ).item() == pytest.approx(0.0)
    assert new_small_d.cdf(
        new_small_logits, torch.tensor([0.0])
    ).item() == pytest.approx(0.25)
    assert new_small_d.cdf(
        new_small_logits, torch.tensor([2.0])
    ).item() == pytest.approx(1.0)
    pos = torch.tensor([new_small_d.borders[-2]])
    assert new_small_d.cdf(new_small_logits, pos).item() == pytest.approx(
        sum(bd.cdf(lo, pos)[0].item() for bd, lo in zip(bar_dists, logits, strict=True))
        / 4
    )
