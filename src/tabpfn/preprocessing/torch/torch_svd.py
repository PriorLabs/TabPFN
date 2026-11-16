#  Copyright (c) Prior Labs GmbH 2026.

"""Torch implementation of TruncatedSVD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


def _svd_flip_stable(v: torch.Tensor) -> torch.Tensor:
    """Sign correction for deterministic SVD output.

    Flips the sign of each component so the element with the largest absolute
    value in each row of *v* is positive (same convention as sklearn's
    ``svd_flip(u_based_decision=False)``), with leftmost-column tie-breaking.

    Note:
       This resolves sign ambiguity *within* a given decomposition, but cannot
       fix cross-platform differences where the eigensolver itself returns
       different vectors due to different LAPACK backends (MKL on Linux,
       Accelerate on macOS).  For fully deterministic SVD across platforms, use
       sklearn's ``TruncatedSVD(algorithm="arpack")``.
    """
    abs_v = torch.abs(v)
    max_vals = abs_v.max(dim=1, keepdim=True).values  # [n_components, 1]
    # Boolean mask: True where abs_v equals the row-max (handles ties)
    is_max = abs_v == max_vals
    # First True per row → leftmost max (argmax on bool returns first True)
    max_col_indices = is_max.to(torch.int8).argmax(dim=1)
    signs = torch.sign(v[torch.arange(v.shape[0], device=v.device), max_col_indices])
    # Avoid flipping by zero (if an entire row is zero, sign returns 0)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return v * signs.unsqueeze(1)


_GRAM_SPAN_ROWS = 4096
"""How many rows one float32 matrix product covers before it joins the float64 total.

The scaled values are not centred, so float32 rounding errors add up in the same
direction over a long sum. Shorter spans keep that error small, at the cost of more
float64 additions.
"""


def _sum_dtype(device: torch.device) -> torch.dtype:
    """The dtype sums are accumulated in: float64, except on MPS, which has none."""
    return torch.float32 if device.type == "mps" else torch.float64


def add_to_gram(gram: torch.Tensor | None, rows: torch.Tensor) -> torch.Tensor:
    """Returns `gram + rows.T @ rows`.

    `rows` has shape `[n, F]` and holds no NaN. The products are computed in the
    dtype of `rows`, `_GRAM_SPAN_ROWS` rows at a time, and summed in `_sum_dtype`.
    The spans are fixed, so the result does not depend on how the rows are split
    across calls, as long as each call starts at a multiple of `_GRAM_SPAN_ROWS`.
    """
    for start in range(0, rows.shape[0], _GRAM_SPAN_ROWS):
        span = rows[start : start + _GRAM_SPAN_ROWS]
        part = (span.T @ span).to(_sum_dtype(span.device))
        gram = part if gram is None else gram.add_(part)
    if gram is None:
        width = rows.shape[1]
        return torch.zeros(
            width, width, dtype=_sum_dtype(rows.device), device=rows.device
        )
    return gram


class TorchTruncatedSVD:
    """Truncated SVD for PyTorch tensors.

    Similar to sklearn's TruncatedSVD but without any implicit state.
    The state is returned explicitly. The fit is exact: it takes the top
    eigenvectors of `X^T X`. That matrix can be built a block of rows at a time with
    `add_to_gram`, so the fit's memory does not grow with the number of rows.

    Note: Unlike sklearn's TruncatedSVD, this does not center the data.
    If centering is needed, apply it before calling fit.
    """

    def __init__(self, n_components: int) -> None:
        """Initialize the truncated SVD.

        Args:
            n_components: Number of components to keep.
        """
        self.n_components = n_components

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the truncated SVD on the training data.

        Args:
            x: Input tensor with shape [n_samples, n_features].

        Returns:
            Cache dictionary with:
                - "components": The right singular vectors V^T
                    [n_components, n_features]
                - "singular_values": The singular values [n_components]
        """
        compute_dtype = x.dtype if x.dtype == torch.float64 else torch.float32
        x_filled = torch.nan_to_num(x.to(compute_dtype), nan=0.0)
        n_samples, n_features = x.shape
        if n_samples >= n_features:
            return self.fit_gram(
                add_to_gram(None, x_filled), n_samples=n_samples, dtype=compute_dtype
            )
        # With fewer rows than columns, use the smaller matrix `X X^T`. Its
        # eigenvectors U give the components as `U^T X / s`.
        row_gram = add_to_gram(None, x_filled.T)
        n_components = max(1, min(self.n_components, n_samples, n_features))
        if row_gram.device.type == "mps":
            # MPS has no float64 or eigh.
            row_gram = row_gram.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(row_gram.to(torch.float64))
        top = torch.arange(n_samples - 1, n_samples - 1 - n_components, -1)
        singular_values = eigenvalues[top].clamp(min=0).sqrt()
        singular_values = singular_values.to(device=x.device, dtype=compute_dtype)
        u = eigenvectors[:, top].to(device=x.device, dtype=compute_dtype)
        # A zero singular value has no direction in X, so its component stays zero.
        scale = torch.where(singular_values > 0, 1 / singular_values, 0.0)
        components = (u.T @ x_filled) * scale.unsqueeze(1)
        return {
            "components": _svd_flip_stable(components),
            "singular_values": singular_values,
        }

    def fit_gram(
        self, gram: torch.Tensor, *, n_samples: int, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Fits from `gram`, the matrix `X^T X` of `n_samples` training rows.

        Returns the same cache as `fit`, with the components in `dtype`.
        """
        n_features = gram.shape[0]
        n_components = max(1, min(self.n_components, n_samples, n_features))
        orig_device = gram.device
        if gram.device.type == "mps":
            # MPS has no float64 or eigh.
            gram = gram.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(gram.to(torch.float64))
        # eigh sorts ascending; the top components are the last columns.
        top = torch.arange(n_features - 1, n_features - 1 - n_components, -1)
        top = top.to(gram.device)
        singular_values = eigenvalues[top].clamp(min=0).sqrt()
        components = _svd_flip_stable(eigenvectors[:, top].T)
        return {
            "components": components.to(device=orig_device, dtype=dtype),
            "singular_values": singular_values.to(device=orig_device, dtype=dtype),
        }

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Project the data onto the SVD components.

        Automatically processes in row-chunks when the data is large to keep
        peak intermediate memory bounded.

        Args:
            x: Input tensor to transform [n_samples, n_features].
            fitted_cache: Cache returned by fit.

        Returns:
            Transformed tensor [n_samples, n_components].
        """
        if "components" not in fitted_cache:
            raise ValueError("Invalid fitted cache. Must contain 'components'.")

        components = fitted_cache["components"]
        orig_dtype = x.dtype
        compute_dtype = components.dtype
        chunk_size = self._get_transform_chunk_size(x, components)

        def _per_row(row: torch.Tensor) -> torch.Tensor:
            x_c = row.to(compute_dtype)
            nan_mask = torch.isnan(x_c)
            x_filled = torch.where(nan_mask, torch.zeros_like(x_c), x_c)
            out = x_filled @ components.T
            return torch.where(nan_mask.any(), float("nan"), out)

        result = torch.vmap(_per_row, chunk_size=chunk_size)(x)
        return result.to(orig_dtype) if orig_dtype != compute_dtype else result

    def _get_transform_chunk_size(
        self, x: torch.Tensor, components: torch.Tensor
    ) -> int:
        """Compute a row-chunk size that keeps intermediate memory bounded.

        Transform creates ~3x N*F intermediates (dtype cast, nan_mask,
        x_filled) plus the N*C result.  Target ~2 GB of temporaries.
        """
        n_features = x.shape[-1]
        n_components = components.shape[0]
        element_size = max(x.element_size(), components.element_size())
        # x_compute + nan_mask(~1B) + x_filled + result
        bytes_per_row = (3 * n_features + n_components) * element_size
        target_bytes = 2 * 1024**3  # 2 GB
        return max(1_000, target_bytes // max(bytes_per_row, 1))

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
    ) -> torch.Tensor:
        """Apply truncated SVD with optional train/test splitting.

        This is a convenience method similar to `fit_transform` but with
        train/test split handled automatically and no state being kept.

        Args:
            x: Input tensor of shape [n_samples, n_features].
            num_train_rows: Position to split train and test data. If provided,
                SVD is computed only from x[:num_train_rows]. If None,
                SVD is computed from all data.

        Returns:
            Transformed tensor [n_samples, n_components].
        """
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)


class TorchSafeStandardScaler:
    """Standard scaler that only scales (no mean centering) with NaN/inf handling.

    This is designed to be used before SVD, similar to sklearn's
    StandardScaler(with_mean=False) wrapped in make_standard_scaler_safe.
    """

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the standard deviation over the first dimension.

        Args:
            x: Input tensor with shape [n_samples, n_features].

        Returns:
            Cache dictionary with the std for the transform step.
        """
        return self.fit_blocks([x])

    def fit_blocks(self, blocks: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Same as `fit` on all the rows of `blocks`, reading one block at a time.

        Each column's mean is taken over its finite values. The sum of squared
        deviations is divided by the number of all rows, as if missing values were
        first replaced by the mean. Block statistics are merged in float64 (Chan et
        al.), so a different split of the rows changes the result only by rounding.
        """
        rows = 0
        count = mean = m2 = None
        out_dtype = torch.float32
        for block in blocks:
            if block.dtype == torch.float64:
                out_dtype = torch.float64
            x = block.to(out_dtype)
            sum_dtype = _sum_dtype(x.device)
            valid = torch.isfinite(x)
            x_valid = torch.where(valid, x, 0.0)
            block_count = valid.sum(dim=0, dtype=sum_dtype)
            block_mean = x_valid.sum(dim=0, dtype=sum_dtype) / block_count.clamp(
                min=1.0
            )
            deviation = torch.where(valid, x - block_mean.to(x.dtype), 0.0)
            block_m2 = (deviation * deviation).sum(dim=0, dtype=sum_dtype)
            rows += x.shape[0]
            if count is None:
                count, mean, m2 = block_count, block_mean, block_m2
                continue
            total = count + block_count
            delta = block_mean - mean
            weight = torch.where(
                total > 0, block_count / total.clamp(min=1.0), torch.zeros_like(total)
            )
            mean = mean + delta * weight
            m2 = m2 + block_m2 + delta * delta * count * weight
            count = total
        assert mean is not None
        assert m2 is not None
        std = torch.sqrt(m2 / max(rows, 1))
        # Constant or all-missing columns are not scaled.
        std = torch.where((std == 0) | torch.isnan(std), torch.ones_like(std), std)
        if rows == 1:
            std = torch.ones_like(std)
        return {"std": std.to(out_dtype), "mean": mean.to(out_dtype)}

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the fitted scaling to the data (no mean centering).

        Automatically processes in row-chunks when the data is large to keep
        peak intermediate memory bounded.

        Args:
            x: Input tensor to transform.
            fitted_cache: Cache returned by fit.

        Returns:
            Scaled tensor (divided by std only).
        """
        if "std" not in fitted_cache:
            raise ValueError("Invalid fitted cache. Must contain 'std'.")

        std = fitted_cache["std"]
        orig_dtype = x.dtype
        compute_dtype = std.dtype
        chunk_size = self._get_transform_chunk_size(
            x, compute_element_size=max(x.element_size(), std.element_size())
        )
        col_means = (
            fitted_cache["mean"].to(device=x.device, dtype=compute_dtype)
            if "mean" in fitted_cache
            else None
        )

        def _per_row(row: torch.Tensor) -> torch.Tensor:
            x_c = row.to(compute_dtype)
            x_safe = torch.where(torch.isinf(x_c), float("nan"), x_c)
            # Impute NaN with column means (matching CPU make_scaler_safe).
            if col_means is not None:
                nan_mask = torch.isnan(x_safe)
                x_safe = torch.where(nan_mask, col_means, x_safe)
            x_scaled = x_safe / (std + torch.finfo(compute_dtype).eps)
            x_scaled = torch.clip(x_scaled, min=-100, max=100)
            return torch.where(torch.isfinite(x_scaled), x_scaled, 0)

        result = torch.vmap(_per_row, chunk_size=chunk_size)(x)
        return result.to(orig_dtype) if orig_dtype != compute_dtype else result

    def _get_transform_chunk_size(
        self, x: torch.Tensor, compute_element_size: int
    ) -> int:
        """Compute a row-chunk size that keeps intermediate memory bounded.

        Transform creates ~5x N*F intermediates (dtype cast, isinf check,
        nan_mask, imputed, scaled, clipped, finite-check).  Target ~2 GB.
        """
        n_features = x.shape[-1] if x.ndim > 1 else 1
        bytes_per_row = n_features * compute_element_size * 5
        target_bytes = 2 * 1024**3  # 2 GB
        return max(1_000, target_bytes // max(bytes_per_row, 1))

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
    ) -> torch.Tensor:
        """Apply scaling with optional train/test splitting.

        Args:
            x: Input tensor of shape [n_samples, ...].
            num_train_rows: Position to split train and test data. If provided,
                statistics are computed only from x[:num_train_rows]. If None,
                statistics are computed from all data.

        Returns:
            Scaled tensor (divided by std, no mean centering).
        """
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)
