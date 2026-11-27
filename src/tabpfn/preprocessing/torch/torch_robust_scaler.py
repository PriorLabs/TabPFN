#  Copyright (c) Prior Labs GmbH 2026.

"""Torch version of the `robust` preset, sklearn's `RobustScaler(unit_variance=True)`.

Like the rest of `preprocessing/torch`, `fit` returns its state instead of storing
it on the instance.
"""

from __future__ import annotations

import torch

from tabpfn.preprocessing.torch.torch_squashing_scaler import (
    _FIT_BLOCK_BYTES,
    _block_size,
)

# The interquartile range of a standard normal, `norm.ppf(0.75) - norm.ppf(0.25)`.
# With `unit_variance=True`, sklearn divides the scale by it.
_NORMAL_IQR = 1.3489795003921634


class TorchRobustScaler:
    """Centers on the median and scales by the interquartile range, per column.

    Matches `sklearn.preprocessing.RobustScaler(unit_variance=True)`. NaNs are
    ignored when fitting and kept when transforming. A scale below `10 * eps` is set
    to 1. Quantiles interpolate linearly, as `numpy.nanpercentile` does.
    """

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes each column's center and scale over the rows of `x`."""
        compute_dtype = x.dtype if x.dtype == torch.float64 else torch.float32
        qs = torch.tensor([0.25, 0.5, 0.75], device=x.device, dtype=compute_dtype)
        columns = x.shape[-1]
        block = _block_size(x, dim=x.ndim - 1, budget_bytes=_FIT_BLOCK_BYTES)
        # Each column's statistics depend only on that column, so splitting the
        # columns into blocks does not change them.
        quantiles = torch.cat(
            [
                torch.nanquantile(
                    x[..., start : start + block].to(compute_dtype), qs, dim=0
                )
                for start in range(0, columns, block)
            ],
            dim=-1,
        )
        lower, center, upper = quantiles[0], quantiles[1], quantiles[2]
        scale = upper - lower
        # Use float64's eps, as sklearn does: the CPU pipeline gives it float64 input.
        eps = torch.finfo(torch.float64).eps
        scale = torch.where(scale < 10 * eps, torch.ones_like(scale), scale)
        scale = scale / _NORMAL_IQR
        return {"center": center.to(x.dtype), "scale": scale.to(x.dtype)}

    def transform(
        self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Returns `(x - center) / scale`."""
        return (x - fitted_cache["center"]).div_(fitted_cache["scale"])
