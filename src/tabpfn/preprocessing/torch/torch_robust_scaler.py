#  Copyright (c) Prior Labs GmbH 2026.

"""Torch implementation of the ``robust`` preset."""

from __future__ import annotations

import torch
from scipy.stats import norm

from tabpfn.preprocessing.steps.utils import subsample_row_indices

# Rows the median and quartiles are fitted on, as for the quantile transform.
ROBUST_FIT_SUBSAMPLE = 100_000
# Interquartile range of the standard normal, sklearn's ``unit_variance`` divisor.
_NORMAL_IQR = float(norm.ppf(0.75) - norm.ppf(0.25))
# Input bytes per ``torch.nanquantile`` call; its sort needs several times more.
_FIT_BLOCK_BYTES = 128 * 1024 * 1024


class TorchRobustScaler:
    """NaN-aware ``RobustScaler(unit_variance=True)``: median center, IQR scale.

    Mirrors the CPU ``robust`` preset. The statistics are fitted on at most
    ``subsample`` rows chosen with ``random_state``; ``subsample=None`` uses all
    rows. ``+-inf`` is ignored in the fit and stays ``+-inf`` in the transform.
    """

    def __init__(
        self, subsample: int | None = ROBUST_FIT_SUBSAMPLE, random_state: int = 0
    ) -> None:
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit ``center`` and ``scale`` of shape ``x.shape[1:]`` on ``x``."""
        rows = subsample_row_indices(x.shape[0], self.subsample, self.random_state)
        if rows is not None:
            x = x[torch.from_numpy(rows).to(x.device)]
        dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        flat = x.reshape(x.shape[0], -1)
        qs = torch.tensor([0.25, 0.5, 0.75], device=x.device, dtype=dtype)
        bytes_per_col = torch.finfo(dtype).bits // 8 * max(flat.shape[0], 1)
        n_cols = max(1, _FIT_BLOCK_BYTES // bytes_per_col)
        quantiles = torch.cat(
            [
                torch.nanquantile(_inf_to_nan(flat[:, i : i + n_cols].to(dtype)), qs, 0)
                for i in range(0, flat.shape[1], n_cols)
            ],
            dim=1,
        ).reshape(3, *x.shape[1:])
        iqr = quantiles[2] - quantiles[0]
        # sklearn's ``_handle_zeros_in_scale``.
        iqr = torch.where(iqr < 10 * torch.finfo(dtype).eps, 1.0, iqr)
        return {"center": quantiles[1], "scale": iqr / _NORMAL_IQR}

    def transform(
        self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return ``(x - center) / scale`` in the dtype of ``x``."""
        center, scale = fitted_cache["center"], fitted_cache["scale"]
        return ((x.to(center.dtype) - center) / scale).to(x.dtype)


def _inf_to_nan(x: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(torch.isinf(x), float("nan"))
