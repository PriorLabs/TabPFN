#  Copyright (c) Prior Labs GmbH 2026.

"""The fitted tail of the "nonnegative_and_pareto_tail" and
"nonnegative_and_exponential_tail" readouts.

Above a threshold u, these readouts replace the predicted distribution with a
generalized Pareto distribution (GPD) fitted to the training targets above u,
capped at an upper threshold M:
E[Y] = E[min(Y, u)] + nu * E[min(Y - u, M - u) | Y > u], with nu = P(Y > u) read
from the predicted distribution.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import minimize

TAIL_MIN_POSITIVES = 50
"""Fewer positive training targets than this give no tail fit; the readouts then
return the "nonnegative_and_capped" mean."""

TAIL_FRACTION = 0.05
TAIL_MIN_EXCESSES = 30
TAIL_MAX_EXCESSES = 500

XI_PRIOR_SD = 0.3
XI_BOUNDS = (-0.5, 0.9)


@dataclasses.dataclass(frozen=True)
class ParetoTail:
    """A GPD of the training targets' excesses over `threshold`, capped at `cap`."""

    threshold: float
    cap: float
    n_excesses: int
    sigma: float
    xi: float
    probability_scale: float = 1.0
    """The factor on the predicted tail probability nu."""

    @property
    def mean_excess(self) -> float:
        """E[min(Y, M) - u | Y > u] under the GPD."""
        return capped_mean_excess(self.sigma, self.xi, self.cap - self.threshold)


def capped_mean_excess(sigma: float, xi: float, limit: float) -> float:
    """E[min(X, limit)] for X ~ GPD(sigma, xi), the integral of its survival
    function (1 + xi * t / sigma)^(-1/xi) from 0 to `limit`.
    """
    if xi < 0:
        # The support ends at -sigma / xi.
        limit = min(limit, -sigma / xi)
    z = limit / sigma
    if abs(xi) < 1e-8:
        return sigma * -np.expm1(-z)
    if abs(xi - 1) < 1e-8:
        return sigma * np.log1p(z)
    # At the end of the support, log1p(-1) = -inf, which reads 0 as it should.
    with np.errstate(divide="ignore"):
        return sigma / (xi - 1) * np.expm1((1 - 1 / xi) * np.log1p(xi * z))


def fit_pareto_tail(
    y: np.ndarray, *, threshold: float | None, cap: float | None, fix_shape: bool
) -> ParetoTail | None:
    """Fit a GPD by MAP to the positive targets above the threshold.

    Args:
        y: The nonnegative training targets.
        threshold: The threshold u. If None, u is the (k+1)-th largest positive
            target, with k = clip(round(0.05 * n_pos), 30, 500).
        cap: The upper threshold M > u, where the tail is capped. If None, M is
            twice the largest target.
        fix_shape: Fix xi at 0, an exponential tail. Otherwise xi has a
            N(0, 0.3^2) prior and lies in [-0.5, 0.9].

    Returns:
        The fitted tail, or None with fewer than `TAIL_MIN_POSITIVES` positive
        targets, or when no target lies above the threshold.
    """
    positives = np.sort(y[y > 0])[::-1].astype(np.float64)
    if len(positives) < TAIL_MIN_POSITIVES:
        return None
    if threshold is None:
        k = int(
            np.clip(
                round(TAIL_FRACTION * len(positives)),
                TAIL_MIN_EXCESSES,
                TAIL_MAX_EXCESSES,
            )
        )
        threshold = float(positives[k])
        excesses = positives[:k] - threshold
    else:
        excesses = positives[positives > threshold] - threshold
    if cap is None:
        cap = 2 * float(positives[0])
    if cap <= threshold:
        raise ValueError(
            f"The tail cap {cap} must lie above the tail threshold {threshold}."
        )
    # Ties at the threshold, e.g. in count data, leave every excess at 0.
    if not np.any(excesses > 0):
        return None
    mean_excess = float(excesses.mean())
    if fix_shape:
        return ParetoTail(threshold, cap, len(excesses), sigma=mean_excess, xi=0.0)

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
        cap,
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
