#  Copyright (c) Prior Labs GmbH 2026.

"""Mean readouts of nonnegative regression targets from each ensemble member.

The default mean averages the members' probabilities on one shared grid, whose
bucket at 0 can straddle 0. These readouts take each member's mean on its own
borders instead, and average the member means:

- "single_clamp": E[max(Y, 0)].
- "single_trunc": E[max(Y, 0) | Y <= y_max], with y_max the largest training
  target. The mass above y_max is dropped and the rest renormalized.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

MeanReadout = Literal["ensemble", "single_clamp", "single_trunc"]

ROWS_PER_CHUNK = 4096
"""Rows per float64 softmax, which holds rows x buckets values at once."""


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
    ) -> None:
        self.readout = readout
        self.y_mean = y_mean
        self.y_std = y_std
        self.y_max = y_max
        self._sum: torch.Tensor | None = None
        self._n_members = 0

    def add(self, borders: np.ndarray | torch.Tensor, logits: torch.Tensor) -> None:
        """Add one member's mean."""
        b = torch.as_tensor(np.ascontiguousarray(borders), dtype=torch.float64)
        zero = (0.0 - self.y_mean) / self.y_std
        cap = (
            (self.y_max - self.y_mean) / self.y_std
            if self.readout == "single_trunc"
            else None
        )
        mean_z = _clamped_member_mean(b, logits, zero=zero, cap=cap)
        mean = mean_z * self.y_std + self.y_mean
        self._sum = mean if self._sum is None else self._sum + mean
        self._n_members += 1

    def mean(self) -> np.ndarray:
        """The readout's mean in target units, of shape [n_samples]."""
        assert self._sum is not None
        return (self._sum / self._n_members).numpy()


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
