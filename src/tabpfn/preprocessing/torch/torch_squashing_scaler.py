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

The state is returned explicitly from `fit` rather than stored on the
instance, matching the rest of `preprocessing/torch`.
"""

from __future__ import annotations

import torch

# Bytes of input one `fit` column block covers. The sort inside
# `torch.nanquantile` needs a large multiple of what it is given -- it double-buffers
# both the sorted values and their int64 indices -- so bounding the block by bytes is
# what bounds fit's peak at every shape. Sits at the knee: larger blocks cost time as
# well as memory, smaller ones start paying for the extra launches.
_FIT_BLOCK_BYTES = 128 * 1024 * 1024

# Below this much input, `fit` reduces in a single pass instead. Blocking the sort
# costs wall time on older torch, so it is only worth doing once the unblocked peak is
# large enough to be a problem.
_FIT_UNBLOCKED_BYTES = 512 * 1024 * 1024

# Bytes of output `transform` produces at a time. Its output has to exist in full;
# none of its intermediates do. Small end of the useful range, because wall time is
# flat across that range while what the intermediates cost scales with the block.
_TRANSFORM_BLOCK_BYTES = 32 * 1024 * 1024


def _scalar(value: float, like: torch.Tensor) -> torch.Tensor:
    """A 0-dim tensor of `like`'s dtype and device, to broadcast into `where`.

    Broadcasting one number costs nothing, where `torch.full_like` would allocate a
    tensor the size of `x` to carry it.
    """
    return torch.full((), value, dtype=like.dtype, device=like.device)


def _replace_inf_with_nan(x: torch.Tensor) -> torch.Tensor:
    """Replace ±inf with NaN so percentile/min/max see only finite values."""
    return torch.where(torch.isinf(x), _scalar(float("nan"), x), x)


def _block_size(x: torch.Tensor, dim: int, budget_bytes: int) -> int:
    """Slices of `dim` whose data is about `budget_bytes`."""
    per_slice = x.element_size()
    for index, size in enumerate(x.shape):
        if index != dim:
            per_slice *= size
    return max(1, budget_bytes // max(1, per_slice))


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

        col_min, col_max, quantiles = self._column_statistics(x, qs, quantile_dtype)
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

    def _column_statistics(
        self,
        x: torch.Tensor,
        qs: torch.Tensor,
        quantile_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-column min, max and quantiles, reduced a block of columns at a time.

        Every statistic is reduced over dim 0 and so is independent per column.

        What must not happen here is a whole `_replace_inf_with_nan` copy of `x`:
        that copy is the input to `nanquantile`, so keeping it to a block is what
        keeps the sort to a block.

        Returns:
            min, max, quantiles as torch.Tensor
        """
        columns = x.shape[-1]
        block = _block_size(x, dim=x.ndim - 1, budget_bytes=_FIT_BLOCK_BYTES)
        if block >= columns or x.element_size() * x.nelement() <= _FIT_UNBLOCKED_BYTES:
            return self._column_statistics_block(x, qs, quantile_dtype)

        parts = [
            self._column_statistics_block(
                x[..., start : start + block], qs, quantile_dtype
            )
            for start in range(0, columns, block)
        ]
        return (
            torch.cat([part[0] for part in parts], dim=-1),
            torch.cat([part[1] for part in parts], dim=-1),
            torch.cat([part[2] for part in parts], dim=-1),
        )

    def _column_statistics_block(
        self,
        x: torch.Tensor,
        qs: torch.Tensor,
        quantile_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """min, max and quantiles for one block of columns."""
        x_finite = _replace_inf_with_nan(x)

        is_nan = torch.isnan(x_finite)
        col_min = torch.amin(
            torch.where(is_nan, _scalar(float("inf"), x_finite), x_finite), dim=0
        )
        col_max = torch.amax(
            torch.where(is_nan, _scalar(float("-inf"), x_finite), x_finite), dim=0
        )
        # All-NaN columns yield ±inf above; surface them as NaN so the masks
        # in `fit` treat them as the "general" path (output stays NaN).
        all_nan = is_nan.all(dim=0)
        del is_nan
        col_min = torch.where(all_nan, _scalar(float("nan"), col_min), col_min)
        col_max = torch.where(all_nan, _scalar(float("nan"), col_max), col_max)

        # the dominant cost
        quantiles = torch.nanquantile(x_finite.to(quantile_dtype), qs, dim=0).to(
            x.dtype
        )
        return col_min, col_max, quantiles

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

        # only the output has to exist in full
        out = torch.empty_like(x)
        nan = _scalar(float("nan"), x)
        block = _block_size(x, dim=0, budget_bytes=_TRANSFORM_BLOCK_BYTES)
        for start in range(0, x.shape[0], block):
            stop = start + block
            # rows are safe to split on because every step in
            # `_transform_block` is elementwise
            self._transform_block(
                x[start:stop], out[start:stop], center, scale, zero_mask, nan
            )
        return out

    def _transform_block(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        zero_mask: torch.Tensor,
        nan: torch.Tensor,
    ) -> None:
        """Transform one block of rows in place, writing through into ``out``.

        ``out`` is a view into the full output.
        """
        b = self.max_absolute_value

        # Replace ±inf with NaN so the scale ops never produce 0 * inf = nan
        # for what was originally a finite outlier, and so soft-clip operates
        # on the centered/scaled finite distribution.
        torch.where(torch.isinf(x), nan, x, out=out)
        out.sub_(center).div_(scale)

        # Zero columns: finite entries become 0 while NaNs are left alone. One mask
        # buffer holds "is not NaN, and in a zero column".
        finite_in_zero_column = torch.isnan(x)
        finite_in_zero_column.logical_not_()
        finite_in_zero_column &= zero_mask
        out.masked_fill_(finite_in_zero_column, 0.0)
        del finite_in_zero_column

        # In-place form of `z / torch.sqrt(1.0 + (z / b) ** 2)`: the same kernels in
        # the same order, needing one temporary rather than a chain of them.
        denominator = out / b
        denominator.pow_(2)
        denominator.add_(1.0)
        denominator.sqrt_()
        out.div_(denominator)
        del denominator

        out.masked_fill_(torch.isposinf(x), b)
        out.masked_fill_(torch.isneginf(x), -b)

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
