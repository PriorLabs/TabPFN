#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the per-member mean readouts of nonnegative targets."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.mean_readout import (
    POT_MIN_POSITIVES,
    MemberMeans,
    ParetoTail,
    fit_pareto_tail,
)

# Borders in the standardized frame; with y_mean=0 and y_std=1 they are also the
# target values. The end buckets are the half-normal tails.
BORDERS = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])


def _logits(*probs_per_row: dict[int, float]) -> torch.Tensor:
    probs = torch.zeros(len(probs_per_row), len(BORDERS) - 1, dtype=torch.float64)
    for row, bucket_probs in enumerate(probs_per_row):
        for bucket, p in bucket_probs.items():
            probs[row, bucket] = p
    return probs.log()


def _mean(
    readout: str,
    logits: torch.Tensor,
    *,
    y_max: float = 3.0,
    tail: ParetoTail | None = None,
    borders: np.ndarray = BORDERS,
) -> np.ndarray:
    means = MemberMeans(readout, y_mean=0.0, y_std=1.0, y_max=y_max, tail=tail)  # type: ignore[arg-type]
    means.add(borders, logits)
    return means.mean()


def test__single_clamp__reads_each_bucket_as_its_mean_of_max_y_0() -> None:
    logits = _logits({2: 1.0}, {3: 1.0}, {2: 0.5, 3: 0.5})

    # [-1, 0] -> 0; [0, 1] -> 0.5; uniform on [-1, 1] -> E[max(Y, 0)] = 1/4.
    np.testing.assert_allclose(_mean("single_clamp", logits), [0.0, 0.5, 0.25])


def test__single_clamp__reads_a_straddling_bucket_by_its_part_above_zero() -> None:
    borders = np.array([-3.0, -2.0, -0.5, 1.5, 2.0, 3.0, 3.5, 4.0])
    logits = _logits({2: 1.0})

    # Uniform on [-0.5, 1.5]: E[max(Y, 0)] = (1.5^2 / 2) / 2.
    np.testing.assert_allclose(
        _mean("single_clamp", logits, borders=borders), [1.5**2 / 4]
    )


def test__single_clamp__averages_the_member_means() -> None:
    means = MemberMeans("single_clamp", y_mean=0.0, y_std=1.0, y_max=3.0, tail=None)
    means.add(BORDERS, _logits({3: 1.0}))
    means.add(BORDERS + 1.0, _logits({3: 1.0}))

    np.testing.assert_allclose(means.mean(), [(0.5 + 1.5) / 2])


def test__member_means__returns_target_units() -> None:
    means = MemberMeans("single_clamp", y_mean=10.0, y_std=4.0, y_max=30.0, tail=None)
    means.add(BORDERS, _logits({3: 1.0}))

    # The bucket [0, 1] is [10, 14] in target units.
    np.testing.assert_allclose(means.mean(), [12.0])


def test__single_trunc__drops_the_mass_above_the_largest_target() -> None:
    logits = _logits({3: 0.5, 5: 0.5}, {3: 0.5, 4: 0.5})

    # y_max = 2: the bucket [2, 3] is dropped; [1, 2] is kept.
    np.testing.assert_allclose(
        _mean("single_trunc", logits, y_max=2.0), [0.5, (0.5 + 1.5) / 2]
    )


def test__single_trunc__keeps_the_share_of_a_straddling_bucket_below_the_cap() -> None:
    logits = _logits({3: 0.5, 4: 0.5})

    # y_max = 1.5: half of [1, 2] stays, uniform on [1, 1.5].
    expected = (0.5 * 0.5 + 0.25 * 1.25) / 0.75
    np.testing.assert_allclose(_mean("single_trunc", logits, y_max=1.5), [expected])


def test__single_pot__adds_the_tail_probability_times_the_mean_excess() -> None:
    tail = ParetoTail(threshold=1.0, n_excesses=30, sigma=2.0, xi=0.5)
    logits = _logits({3: 0.5, 4: 0.5}, {3: 1.0}, {4: 0.5, 5: 0.5})

    means = _mean("single_pot", logits, y_max=2.0, tail=tail)

    # Row 0: E[min(Y, 1)] = 0.75, P(Y > 1) = 0.5. Row 1: no mass above u.
    # Row 2: [2, 3] lies above y_max and is dropped, so all kept mass is above u.
    np.testing.assert_allclose(means, [0.75 + 0.5 * 4.0, 0.5, 1.0 + 4.0])


def test__single_pot__without_a_tail_is_the_truncated_mean() -> None:
    logits = _logits({3: 0.5, 5: 0.5})

    np.testing.assert_allclose(_mean("single_pot", logits, y_max=2.0), [0.5])


def _excess_sample(xi: float, sigma: float, n: int, seed: int = 0) -> np.ndarray:
    u = np.random.default_rng(seed).random(n)
    return sigma * ((1 - u) ** -xi - 1) / xi


def test__fit_pareto_tail__default_threshold_is_the_k_plus_first_largest() -> None:
    y = np.concatenate([np.zeros(500), np.arange(1.0, 2001.0)])

    tail = fit_pareto_tail(y, threshold=None, fix_shape=True)

    # k = round(0.05 * 2000) = 100: u is the 101st largest positive.
    assert tail is not None
    assert tail.n_excesses == 100
    assert tail.threshold == 1900.0
    assert tail.xi == 0.0
    np.testing.assert_allclose(tail.sigma, np.mean(np.arange(1.0, 101.0)))


@pytest.mark.parametrize(("n_pos", "k"), [(200, 30), (40_000, 500)])
def test__fit_pareto_tail__clips_the_number_of_excesses(n_pos: int, k: int) -> None:
    y = np.arange(1.0, n_pos + 1.0)

    tail = fit_pareto_tail(y, threshold=None, fix_shape=True)

    assert tail is not None
    assert tail.n_excesses == k


def test__fit_pareto_tail__uses_a_given_threshold() -> None:
    y = np.arange(0.0, 101.0)

    tail = fit_pareto_tail(y, threshold=89.5, fix_shape=True)

    assert tail is not None
    assert tail.n_excesses == 11
    np.testing.assert_allclose(tail.sigma, np.mean(np.arange(90.0, 101.0)) - 89.5)


def test__fit_pareto_tail__recovers_a_heavy_tail_shape() -> None:
    excesses = _excess_sample(xi=0.4, sigma=2.0, n=500)
    y = np.concatenate([np.full(10_000, 0.5), 10.0 + excesses])

    tail = fit_pareto_tail(y, threshold=10.0, fix_shape=False)

    assert tail is not None
    assert tail.xi == pytest.approx(0.4, abs=0.15)
    assert tail.mean_excess == pytest.approx(excesses.mean(), rel=0.2)


def test__fit_pareto_tail__needs_enough_positive_targets() -> None:
    y = np.concatenate([np.zeros(1000), np.arange(1.0, POT_MIN_POSITIVES)])

    assert fit_pareto_tail(y, threshold=None, fix_shape=False) is None
