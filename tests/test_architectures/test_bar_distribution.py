#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import math

import pytest
import torch

from tabpfn.architectures.shared import bar_distribution


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


def test_halfnormal_with_p_weight_before_respects_dtype():
    """The scale must be built in `range_max`'s dtype, not always float32.

    A hardcoded `torch.tensor(1.0)` evaluates `icdf` in float32 and caps the
    scale at ~1e-8 relative accuracy, which is a float32-sized error inside an
    otherwise float64 tail computation (PRI-361). float32 callers must be
    bit-identical to before.
    """
    half_normal = bar_distribution.FullSupportBarDistribution

    width_32 = torch.tensor(0.016)
    reference_32 = width_32 / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
        torch.tensor(0.5)
    )
    assert torch.equal(
        half_normal.halfnormal_with_p_weight_before(width_32).scale, reference_32
    )
    # A plain Python float keeps the default dtype, so it is unaffected too.
    assert torch.equal(
        half_normal.halfnormal_with_p_weight_before(0.016).scale, reference_32
    )

    # In float64 the scale must place exactly half the mass below `range_max`.
    width_64 = torch.tensor(0.016, dtype=torch.float64)
    scale_64 = half_normal.halfnormal_with_p_weight_before(width_64).scale
    assert scale_64.dtype == torch.float64
    mass_below = torch.erf(width_64 / (scale_64 * math.sqrt(2.0)))
    assert mass_below.item() == pytest.approx(0.5, abs=1e-15)


def test_move_to_larger():
    logits = torch.arange(99).float()
    old_d = bar_distribution.BarDistribution(borders=torch.linspace(0, 1, 100))
    new_d = bar_distribution.BarDistribution(borders=torch.linspace(-1, 2, 1000))
    new_logits = old_d.get_probs_for_different_borders(logits, new_d.borders).log()
    assert old_d.median(logits).item() == pytest.approx(new_d.median(new_logits).item())
    assert old_d.mean(logits).item() == pytest.approx(
        new_d.mean(new_logits).item(), abs=1e-4
    )


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
