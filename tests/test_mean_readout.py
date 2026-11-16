#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the per-member mean readouts of nonnegative targets."""

from __future__ import annotations

import numpy as np
import torch

from tabpfn.mean_readout import MemberMeans

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
    borders: np.ndarray = BORDERS,
) -> np.ndarray:
    means = MemberMeans(readout, y_mean=0.0, y_std=1.0, y_max=y_max)  # type: ignore[arg-type]
    means.add(borders, logits)
    return means.mean()


def test__single_trunc__reads_each_bucket_as_its_mean_of_max_y_0() -> None:
    logits = _logits({2: 1.0}, {3: 1.0}, {2: 0.5, 3: 0.5})

    # [-1, 0] -> 0; [0, 1] -> 0.5; uniform on [-1, 1] -> E[max(Y, 0)] = 1/4.
    np.testing.assert_allclose(_mean("single_trunc", logits), [0.0, 0.5, 0.25])


def test__single_trunc__reads_a_straddling_bucket_by_its_part_above_zero() -> None:
    borders = np.array([-3.0, -2.0, -0.5, 1.5, 2.0, 3.0, 3.5, 4.0])
    logits = _logits({2: 1.0})

    # Uniform on [-0.5, 1.5]: E[max(Y, 0)] = (1.5^2 / 2) / 2.
    np.testing.assert_allclose(
        _mean("single_trunc", logits, borders=borders), [1.5**2 / 4]
    )


def test__single_trunc__averages_the_member_means() -> None:
    means = MemberMeans("single_trunc", y_mean=0.0, y_std=1.0, y_max=3.0)
    means.add(BORDERS, _logits({3: 1.0}))
    means.add(BORDERS + 1.0, _logits({3: 1.0}))

    np.testing.assert_allclose(means.mean(), [(0.5 + 1.5) / 2])


def test__member_means__returns_target_units() -> None:
    means = MemberMeans("single_trunc", y_mean=10.0, y_std=4.0, y_max=30.0)
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
