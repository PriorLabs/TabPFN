#  Copyright (c) Prior Labs GmbH 2026.

"""Torch implementation of SquashingScaler with NaN handling.

Mirrors the CPU
:class:`tabpfn.preprocessing.steps.squashing_scaler_transformer.SquashingScaler`,
which is itself adapted from skrub:
  https://github.com/skrub-data/skrub
The algorithmic logic (robust median-centering with quartile scaling, min-max
fallback, soft-clip) is derived from skrub's ``SquashingScaler``.

Original skrub attribution:
  Copyright (c) 2018-2023, The dirty_cat developers, 2023-2026 the skrub developers.
  All rights reserved.
  SPDX-License-Identifier: BSD-3-Clause

The state is returned explicitly from ``fit`` rather than stored on the
instance, matching the rest of ``preprocessing/torch``.
"""

from __future__ import annotations

import torch


def _replace_inf_with_nan(x: torch.Tensor) -> torch.Tensor:
    """Replace ±inf with NaN so percentile/min/max see only finite values."""
    return torch.where(torch.isinf(x), torch.full_like(x, float("nan")), x)


class TorchSquashingScaler:
    """Squashing scaler for PyTorch tensors with NaN/inf handling.

    Per-column behavior, picked at fit time:

    * **zero columns** (``nanmax == nanmin``): finite values become ``0``.
    * **minmax columns** (``q_lower == q_upper`` but range is non-zero): scaled
      as ``2 * (x - median) / (max - min + eps)``.
    * **robust columns** (general case): scaled as
      ``(x - median) / (q_upper - q_lower)``.

    All three branches then pass through the soft clip
    ``z / sqrt(1 + (z / max_absolute_value) ** 2)``. ``±inf`` inputs are mapped
    to ``±max_absolute_value`` and NaNs are preserved.
    """

    def __init__(
        self,
        max_absolute_value: float = 3.0,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        super().__init__()
        if not (0.0 <= quantile_range[0] < quantile_range[1] <= 100.0):
            raise ValueError(
                "quantile_range must satisfy 0 <= lower < upper <= 100, got "
                f"{quantile_range!r}.",
            )
        if not (max_absolute_value > 0):
            raise ValueError(
                f"max_absolute_value must be positive, got {max_absolute_value!r}.",
            )
        self.max_absolute_value = max_absolute_value
        self.quantile_range = quantile_range

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute per-column scaling state from training rows.

        Args:
            x: Input tensor with shape ``[T, ...]`` where ``T`` is the number of
                training rows. Statistics are reduced over dim 0; remaining
                dims (e.g. ``[batch, n_cols]``) define the cache shape.

        Returns:
            Cache dict with keys ``center``, ``scale``, ``zero_mask`` (each of
            shape ``x.shape[1:]``).
        """
        feature_shape = x.shape[1:]
        device = x.device
        dtype = x.dtype

        if x.shape[0] <= 1:
            return {
                "center": torch.zeros(feature_shape, device=device, dtype=dtype),
                "scale": torch.ones(feature_shape, device=device, dtype=dtype),
                "zero_mask": torch.ones(feature_shape, device=device, dtype=torch.bool),
            }

        x_finite = _replace_inf_with_nan(x)

        # nanquantile cannot share its sort with nanmin/nanmax across the same
        # call, so they're computed separately. The dominant cost is the single
        # nanquantile call covering all three quartiles at once.
        col_min = torch.amin(
            torch.where(
                torch.isnan(x_finite), torch.full_like(x_finite, float("inf")), x_finite
            ),
            dim=0,
        )
        col_max = torch.amax(
            torch.where(
                torch.isnan(x_finite),
                torch.full_like(x_finite, float("-inf")),
                x_finite,
            ),
            dim=0,
        )
        # All-NaN columns yield ±inf above; surface them as NaN so the masks
        # below treat them as the "general" path (output stays NaN).
        all_nan = torch.isnan(x_finite).all(dim=0)
        col_min = torch.where(all_nan, torch.full_like(col_min, float("nan")), col_min)
        col_max = torch.where(all_nan, torch.full_like(col_max, float("nan")), col_max)

        # torch.nanquantile requires float32 or float64; upcast (e.g. from
        # float16) just for the quantile computation, then cast results back.
        quantile_dtype = (
            dtype if dtype in (torch.float32, torch.float64) else torch.float32
        )
        lower_q, upper_q = self.quantile_range
        qs = torch.tensor(
            [lower_q / 100.0, 0.5, upper_q / 100.0],
            device=device,
            dtype=quantile_dtype,
        )
        quantiles = torch.nanquantile(x_finite.to(quantile_dtype), qs, dim=0).to(dtype)
        q_lower, q_median, q_upper = quantiles[0], quantiles[1], quantiles[2]

        zero_mask = col_max == col_min
        minmax_mask = (q_lower == q_upper) & ~zero_mask
        robust_mask = ~(zero_mask | minmax_mask)

        eps = torch.finfo(dtype).tiny
        center = torch.zeros(feature_shape, device=device, dtype=dtype)
        scale = torch.ones(feature_shape, device=device, dtype=dtype)

        center = torch.where(robust_mask, q_median, center)
        scale = torch.where(robust_mask, q_upper - q_lower, scale)

        center = torch.where(minmax_mask, q_median, center)
        # minmax: x_out = 2 * (x - median) / (max - min + eps)
        # = (x - center) / scale  with  scale = (max - min + eps) / 2
        scale = torch.where(minmax_mask, (col_max - col_min + eps) / 2.0, scale)

        return {
            "center": center.to(dtype=dtype),
            "scale": scale.to(dtype=dtype),
            "zero_mask": zero_mask,
        }

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the fitted scaling and soft-clip.

        Args:
            x: Input tensor with shape compatible with the cache's feature
                shape (the cache broadcasts over leading dims).
            fitted_cache: Cache returned by ``fit``.

        Returns:
            Transformed tensor with the same shape as ``x``. NaNs are
            preserved; ``±inf`` map to ``±max_absolute_value``.
        """
        for key in ("center", "scale", "zero_mask"):
            if key not in fitted_cache:
                raise ValueError(
                    "Invalid fitted cache. Must contain 'center', 'scale', "
                    f"and 'zero_mask'. Missing: {key}.",
                )

        center = fitted_cache["center"]
        scale = fitted_cache["scale"]
        zero_mask = fitted_cache["zero_mask"]

        pos_inf = torch.isposinf(x)
        neg_inf = torch.isneginf(x)
        nan_mask = torch.isnan(x)

        # Replace ±inf with NaN so the scale ops never produce 0 * inf = nan
        # for what was originally a finite outlier, and so soft-clip operates
        # on the centered/scaled finite distribution.
        x_finite = _replace_inf_with_nan(x)

        x_scaled = (x_finite - center) / scale

        # Broadcast zero_mask up to x's shape so we can zero-out finite
        # entries column-wise without touching NaNs.
        zero_broadcast = zero_mask.expand_as(x_scaled)
        x_scaled = torch.where(
            zero_broadcast & ~nan_mask,
            torch.zeros_like(x_scaled),
            x_scaled,
        )

        b = self.max_absolute_value
        x_clipped = x_scaled / torch.sqrt(1.0 + (x_scaled / b) ** 2)

        x_clipped = torch.where(pos_inf, torch.full_like(x_clipped, b), x_clipped)
        return torch.where(neg_inf, torch.full_like(x_clipped, -b), x_clipped)

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
    ) -> torch.Tensor:
        """Apply squashing scaling with optional train/test splitting.

        Convenience wrapper for ``fit`` + ``transform`` with no state retained
        on the instance, matching the other torch preprocessing helpers.

        Args:
            x: Input tensor of shape ``[T, ...]`` where ``T`` is the number of
                rows. Statistics are computed only over the first
                ``num_train_rows`` rows when provided.
            num_train_rows: Position to split train and test data. If None,
                statistics are computed from all rows.

        Returns:
            Transformed tensor with the same shape as ``x``.
        """
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x
        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)
