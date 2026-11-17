#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the fitted tail of the nonnegative tail readouts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from tabpfn.pareto_tail import (
    TAIL_MIN_POSITIVES,
    ParetoTail,
    capped_mean_excess,
    fit_pareto_tail,
)


def _gpd_survival(t: float, sigma: float, xi: float) -> float:
    if xi == 0:
        return float(np.exp(-t / sigma))
    return max(1 + xi * t / sigma, 0.0) ** (-1 / xi)


@pytest.mark.parametrize("xi", [-0.4, 0.0, 0.5, 1.0, 1.7])
@pytest.mark.parametrize("limit", [0.3, 4.0, 50.0])
def test__capped_mean_excess__integrates_the_survival_function(
    xi: float, limit: float
) -> None:
    sigma = 2.0
    # For xi = -0.4 the support ends at 5, so the limit 50 lies beyond it.
    expected, _ = quad(_gpd_survival, 0.0, limit, args=(sigma, xi), limit=200)

    np.testing.assert_allclose(
        capped_mean_excess(sigma, xi, limit), expected, rtol=1e-9
    )


def test__pareto_tail__mean_excess_is_capped_at_the_cap() -> None:
    tail = ParetoTail(threshold=1.0, cap=5.0, n_excesses=30, sigma=2.0, xi=0.5)

    np.testing.assert_allclose(tail.mean_excess, capped_mean_excess(2.0, 0.5, 4.0))
    assert tail.mean_excess < 2.0 / (1 - 0.5)


def test__fit_pareto_tail__default_threshold_is_the_k_plus_first_largest() -> None:
    y = np.concatenate([np.zeros(500), np.arange(1.0, 2001.0)])

    tail = fit_pareto_tail(y, threshold=None, cap=None, fix_shape=True)

    # k = round(0.05 * 2000) = 100: u is the 101st largest positive.
    assert tail is not None
    assert tail.n_excesses == 100
    assert tail.threshold == 1900.0
    assert tail.xi == 0.0
    np.testing.assert_allclose(tail.sigma, np.mean(np.arange(1.0, 101.0)))


@pytest.mark.parametrize(("n_pos", "k"), [(200, 30), (40_000, 500)])
def test__fit_pareto_tail__clips_the_number_of_excesses(n_pos: int, k: int) -> None:
    y = np.arange(1.0, n_pos + 1.0)

    tail = fit_pareto_tail(y, threshold=None, cap=None, fix_shape=True)

    assert tail is not None
    assert tail.n_excesses == k


def test__fit_pareto_tail__uses_a_given_threshold() -> None:
    y = np.arange(0.0, 101.0)

    tail = fit_pareto_tail(y, threshold=89.5, cap=None, fix_shape=True)

    assert tail is not None
    assert tail.n_excesses == 11
    np.testing.assert_allclose(tail.sigma, np.mean(np.arange(90.0, 101.0)) - 89.5)


@pytest.mark.parametrize(("cap", "expected"), [(None, 200.0), (150.0, 150.0)])
def test__fit_pareto_tail__cap_defaults_to_twice_the_largest_target(
    cap: float | None, expected: float
) -> None:
    y = np.arange(0.0, 101.0)

    tail = fit_pareto_tail(y, threshold=None, cap=cap, fix_shape=True)

    assert tail is not None
    assert tail.cap == expected


def test__fit_pareto_tail__cap_at_or_below_the_threshold__raises() -> None:
    y = np.arange(0.0, 101.0)

    with pytest.raises(ValueError, match="cap"):
        fit_pareto_tail(y, threshold=89.5, cap=89.5, fix_shape=True)


def test__fit_pareto_tail__recovers_a_heavy_tail_shape() -> None:
    u = np.random.default_rng(0).random(500)
    excesses = 2.0 * ((1 - u) ** -0.4 - 1) / 0.4
    y = np.concatenate([np.full(10_000, 0.5), 10.0 + excesses])

    tail = fit_pareto_tail(y, threshold=10.0, cap=np.inf, fix_shape=False)

    assert tail is not None
    assert tail.xi == pytest.approx(0.4, abs=0.15)
    assert tail.mean_excess == pytest.approx(excesses.mean(), rel=0.2)


def test__fit_pareto_tail__needs_enough_positive_targets() -> None:
    y = np.concatenate([np.zeros(1000), np.arange(1.0, TAIL_MIN_POSITIVES)])

    assert fit_pareto_tail(y, threshold=None, cap=None, fix_shape=False) is None


def test__fit_pareto_tail__ties_at_the_threshold__fit_no_tail() -> None:
    """Counts whose largest values tie leave no excess above the threshold."""
    y = np.concatenate([np.zeros(100), np.ones(100), np.full(100, 2.0)])

    assert fit_pareto_tail(y, threshold=None, cap=None, fix_shape=False) is None
