#  Copyright (c) Prior Labs GmbH 2026.

"""Torch counterpart of the `robust` preset: `RobustScaler(unit_variance=True)`.

The state is returned explicitly from `fit` rather than stored on the instance,
matching the rest of `preprocessing/torch`.
"""

from __future__ import annotations

import torch

from tabpfn.preprocessing.torch.torch_squashing_scaler import (
    _FIT_BLOCK_BYTES,
    _TRANSFORM_BLOCK_BYTES,
    _block_size,
)

# `norm.ppf(0.75) - norm.ppf(0.25)`: the interquartile range of a standard normal, which
# `unit_variance=True` divides the scale by.
_NORMAL_IQR = 1.3489795003921634


class TorchRobustScaler:
    """Centers on the median and scales by the interquartile range, per column.

    Matches `sklearn.preprocessing.RobustScaler(unit_variance=True)`: NaNs are ignored
    when fitting and kept when transforming; a scale below `10 * eps` becomes 1.
    Quantiles interpolate linearly, as `numpy.nanpercentile` does.
    """

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-column center and scale over dim 0 of `x`, of shape `x.shape[1:]`."""
        compute_dtype = x.dtype if x.dtype == torch.float64 else torch.float32
        qs = torch.tensor([0.25, 0.5, 0.75], device=x.device, dtype=compute_dtype)
        columns = x.shape[-1]
        block = _block_size(x, dim=x.ndim - 1, budget_bytes=_FIT_BLOCK_BYTES)
        # Per-column statistics are independent, so the blocks cannot change them.
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
        # float64's eps, as sklearn sees float64 input: the CPU pipeline casts to it.
        eps = torch.finfo(torch.float64).eps
        scale = torch.where(scale < 10 * eps, torch.ones_like(scale), scale)
        scale = scale / _NORMAL_IQR
        return {"center": center.to(x.dtype), "scale": scale.to(x.dtype)}

    def transform(
        self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """`(x - center) / scale`, written a block of rows at a time."""
        center = fitted_cache["center"]
        scale = fitted_cache["scale"]
        out = torch.empty_like(x)
        block = _block_size(x, dim=0, budget_bytes=_TRANSFORM_BLOCK_BYTES)
        for start in range(0, x.shape[0], block):
            out_block = out[start : start + block]
            torch.sub(x[start : start + block], center, out=out_block)
            out_block.div_(scale)
        return out
