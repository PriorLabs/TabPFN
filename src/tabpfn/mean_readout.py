#  Copyright (c) Prior Labs GmbH 2026.

"""Mean readouts of nonnegative regression targets from each ensemble member.

The default mean averages the members' probabilities on one shared grid, whose
bucket at 0 can straddle 0. These readouts take each member's mean on its own
borders instead, and average the member means:

- "single_clamp": E[max(Y, 0)].
- "single_trunc": E[max(Y, 0) | Y <= y_max], with y_max the largest training
  target. The mass above y_max is dropped and the rest renormalized.
- "single_pot": peaks over threshold. Above a threshold u, the tail comes from a
  generalized Pareto distribution (GPD) fitted to the training targets:
  E[Y] = E_trunc[min(Y, u)] + nu * sigma / (1 - xi), with
  nu = P(u < Y <= y_max) / P(Y <= y_max) from the member.
- "single_pot_exp": "single_pot" with the GPD shape xi fixed at 0.

`RegressorTuningConfig.rescale_pot_tail` scales nu by one factor fitted on the
tuning holdout rows.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import torch
from scipy.optimize import minimize

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

MeanReadout = Literal[
    "ensemble", "single_clamp", "single_trunc", "single_pot", "single_pot_exp"
]

POT_READOUTS = ("single_pot", "single_pot_exp")

POT_MIN_POSITIVES = 50
"""Fewer positive training targets than this give no tail fit; the "single_pot"
readouts then return the truncated mean."""

POT_TAIL_FRACTION = 0.05
POT_MIN_EXCESSES = 30
POT_MAX_EXCESSES = 500

XI_PRIOR_SD = 0.3
XI_BOUNDS = (-0.5, 0.9)

ROWS_PER_CHUNK = 4096
"""Rows per float64 softmax, which holds rows x buckets values at once."""


@dataclasses.dataclass(frozen=True)
class ParetoTail:
    """A GPD of the training targets' excesses over ``threshold``."""

    threshold: float
    n_excesses: int
    sigma: float
    xi: float
    probability_scale: float = 1.0
    """The factor on the members' tail probability nu."""

    @property
    def mean_excess(self) -> float:
        """E[Y - u | Y > u] under the GPD."""
        return self.sigma / (1 - self.xi)


def fit_pareto_tail(
    y: np.ndarray, *, threshold: float | None, fix_shape: bool
) -> ParetoTail | None:
    """Fit a GPD by MAP to the positive targets above the threshold.

    Args:
        y: The nonnegative training targets.
        threshold: The threshold u. If None, u is the (k+1)-th largest positive
            target, with k = clip(round(0.05 * n_pos), 30, 500).
        fix_shape: Fix xi at 0, an exponential tail. Otherwise xi has a
            N(0, 0.3^2) prior and lies in [-0.5, 0.9].

    Returns:
        The fitted tail, or None with fewer than `POT_MIN_POSITIVES` positive
        targets, or when no target lies above the threshold.
    """
    positives = np.sort(y[y > 0])[::-1].astype(np.float64)
    if len(positives) < POT_MIN_POSITIVES:
        return None
    if threshold is None:
        k = int(
            np.clip(
                round(POT_TAIL_FRACTION * len(positives)),
                POT_MIN_EXCESSES,
                POT_MAX_EXCESSES,
            )
        )
        threshold = float(positives[k])
        excesses = positives[:k] - threshold
    else:
        excesses = positives[positives > threshold] - threshold
    if len(excesses) == 0:
        return None
    mean_excess = float(excesses.mean())
    if fix_shape:
        return ParetoTail(threshold, len(excesses), sigma=mean_excess, xi=0.0)

    def objective(params: np.ndarray) -> float:
        sigma, xi = float(np.exp(params[0])), float(params[1])
        return (
            _gpd_negative_log_likelihood(sigma, xi, excesses)
            + 0.5 * (xi / XI_PRIOR_SD) ** 2
        )

    best = None
    for xi0 in (0.0, 0.3, -0.2, 0.6):
        result = minimize(
            objective,
            np.array([np.log(mean_excess * max(1 - xi0, 0.1)), xi0]),
            method="L-BFGS-B",
            bounds=[
                (np.log(mean_excess) - 10, np.log(mean_excess) + 10),
                XI_BOUNDS,
            ],
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    assert best is not None
    return ParetoTail(
        threshold,
        len(excesses),
        sigma=float(np.exp(best.x[0])),
        xi=float(best.x[1]),
    )


def _gpd_negative_log_likelihood(
    sigma: float, xi: float, excesses: np.ndarray
) -> float:
    z = excesses / sigma
    if abs(xi) < 1e-8:
        return float(np.sum(np.log(sigma) + z))
    arg = 1 + xi * z
    if np.any(arg <= 0):
        return np.inf
    return float(np.sum(np.log(sigma) + (1 + 1 / xi) * np.log(arg)))


class MemberMeans:
    """Averages one readout over the ensemble members of one prediction.

    Members arrive as their borders, in the estimator's standardized frame, and
    their logits over those borders.
    """

    def __init__(
        self,
        readout: MeanReadout,
        *,
        y_mean: float,
        y_std: float,
        y_max: float,
        tail: ParetoTail | None,
    ) -> None:
        self.readout = readout
        self.y_mean = y_mean
        self.y_std = y_std
        self.y_max = y_max
        self.tail = tail
        self._sum: torch.Tensor | None = None
        self._n_members = 0

    def add(self, borders: np.ndarray | torch.Tensor, logits: torch.Tensor) -> None:
        """Add one member's readout statistics."""
        b = torch.as_tensor(np.ascontiguousarray(borders), dtype=torch.float64)
        if self.readout in POT_READOUTS:
            threshold = self.y_max if self.tail is None else self.tail.threshold
            stats = _pot_member_stats(
                b,
                logits,
                y_mean=self.y_mean,
                y_std=self.y_std,
                y_max=self.y_max,
                threshold=threshold,
            )
        else:
            zero = (0.0 - self.y_mean) / self.y_std
            cap = (
                (self.y_max - self.y_mean) / self.y_std
                if self.readout == "single_trunc"
                else None
            )
            mean_z = _clamped_member_mean(b, logits, zero=zero, cap=cap)
            stats = (mean_z * self.y_std + self.y_mean)[:, None]
        self._sum = stats if self._sum is None else self._sum + stats
        self._n_members += 1

    def _member_average(self) -> torch.Tensor:
        assert self._sum is not None
        return self._sum / self._n_members

    def mean(self) -> np.ndarray:
        """The readout's mean in target units, of shape [n_samples]."""
        stats = self._member_average()
        if self.readout not in POT_READOUTS:
            return stats[:, 0].numpy()
        truncated, below_threshold, tail_probability = stats.unbind(-1)
        if self.tail is None:
            return truncated.numpy()
        return (
            below_threshold
            + self.tail.probability_scale * tail_probability * self.tail.mean_excess
        ).numpy()

    def tail_probability(self) -> np.ndarray:
        """The members' average P(u < Y <= y_max) / P(Y <= y_max), without the
        tail's `probability_scale`.
        """
        assert self.readout in POT_READOUTS
        return self._member_average()[:, 2].numpy()


def _softmax_products(logits: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """softmax(logits) @ columns in float64 on the CPU, of shape [rows, M]."""
    logits = logits.detach().to("cpu", torch.float64)
    return torch.cat(
        [torch.softmax(chunk, -1) @ columns for chunk in logits.split(ROWS_PER_CHUNK)]
    )


def _clamped_member_mean(
    borders: torch.Tensor, logits: torch.Tensor, *, zero: float, cap: float | None
) -> torch.Tensor:
    """E[max(Y, zero)] of one member, or E[max(Y, zero) | Y <= cap] with `cap`.

    Interior buckets are uniform and the end buckets half-normal, as in
    `FullSupportBarDistribution.mean`. A clamped end bucket is read at
    max(zero, its mean). A zero-width bucket holds its border.
    """
    lower, upper = borders[:-1], borders[1:]
    widths = upper - lower
    safe_widths = torch.where(widths > 0, widths, 1.0)
    values = (lower + upper) / 2
    left, right = (
        FullSupportBarDistribution.halfnormal_with_p_weight_before(w).mean
        if w > 0
        else 0.0
        for w in (widths[0], widths[-1])
    )
    values[0], values[-1] = borders[1] - left, borders[-2] + right
    straddle = ((upper**2 - zero**2) / 2 + zero * (zero - lower)) / safe_widths
    inside = torch.where(lower >= zero, values, straddle)
    clamped = torch.where(upper <= zero, zero, torch.where(widths > 0, inside, values))
    clamped[0], clamped[-1] = values[0].clamp(min=zero), values[-1].clamp(min=zero)
    if cap is None:
        return _softmax_products(logits, clamped[:, None])[:, 0]
    # A bucket straddling the cap keeps its share below it, uniform there.
    keep = torch.where(
        lower >= cap,
        0.0,
        torch.where(upper > cap, (cap - lower) / safe_widths, 1.0),
    ).nan_to_num(0.0)
    clamped = torch.where(upper > cap, ((lower + cap) / 2).clamp(min=zero), clamped)
    kept_mean, kept_mass = _softmax_products(
        logits, torch.stack([keep * clamped, keep], dim=-1)
    ).unbind(-1)
    return kept_mean / kept_mass


def _pot_member_stats(
    borders: torch.Tensor,
    logits: torch.Tensor,
    *,
    y_mean: float,
    y_std: float,
    y_max: float,
    threshold: float,
) -> torch.Tensor:
    """One member's E[max(Y, 0) | Y <= y_max], E[min(max(Y, 0), u) | Y <= y_max]
    and P(u < Y <= y_max) / P(Y <= y_max), in target units, of shape [rows, 3].

    Buckets are uniform in target units on the member's borders; the end buckets
    too. Buckets with a non-finite border carry no mass. P(Y <= y_max) includes
    the mass at or below 0.
    """
    u = threshold
    zero = -y_mean / y_std
    at_zero = (borders - zero).abs() <= 1e-9 * max(1.0, abs(zero))
    raw = torch.where(at_zero, 0.0, borders * y_std + y_mean)
    finite = torch.isfinite(raw[:-1]) & torch.isfinite(raw[1:])
    lo = torch.where(finite, raw[:-1], 0.0)
    hi = torch.where(finite, raw[1:], 0.0)
    width = hi - lo
    point = width <= 0
    safe_width = torch.where(point, 1.0, width)

    def integral(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Integral of max(y, 0) over [x0, x1], x0 <= x1."""
        return (x1.clamp(min=0) ** 2 - x0.clamp(min=0) ** 2) / 2

    top = hi.clamp(max=y_max)
    below_max = torch.where(
        point, (lo <= y_max).double(), ((top - lo) / safe_width).clamp(0, 1)
    )
    truncated = torch.where(
        point,
        lo.clamp(min=0) * (lo <= y_max),
        integral(lo.clamp(max=y_max), top) / safe_width,
    )
    below_threshold = torch.where(
        point,
        lo.clamp(min=0).clamp(max=u) * (lo <= y_max),
        (
            integral(lo.clamp(max=u), hi.clamp(max=u))
            + u * (top - lo.clamp(min=u)).clamp(min=0)
        )
        / safe_width,
    )
    above_threshold = torch.where(
        point,
        ((lo > u) & (lo <= y_max)).double(),
        (top - lo.clamp(min=u)).clamp(min=0) / safe_width,
    )
    columns = torch.stack(
        [below_max, truncated, below_threshold, above_threshold], dim=-1
    )
    columns = torch.where(finite[:, None], columns, 0.0)
    mass, *expectations = _softmax_products(logits, columns).unbind(-1)
    safe_mass = torch.where(mass > 0, mass, 1.0)
    return torch.stack(expectations, dim=-1) / safe_mass[:, None]
