"""Torch implementation of TruncatedSVD."""

from __future__ import annotations

import warnings

import torch


def _svd_flip_stable(
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign correction for deterministic SVD output.

    Flips the sign of each component so the element with the largest absolute
    value in each row of *v* is positive (same convention as sklearn's
    ``svd_flip(u_based_decision=False)``), with leftmost-column tie-breaking.

    Note:
       This resolves sign ambiguity *within* a given SVD decomposition, but
       cannot fix cross-platform differences where ``torch.linalg.svd`` itself
       returns different singular vectors due to different LAPACK backends
       (MKL on Linux, Accelerate on macOS).  For fully deterministic SVD
       across platforms, use sklearn's ``TruncatedSVD(algorithm="arpack")``.
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
    u = u * signs
    v = v * signs.unsqueeze(1)
    return u, v


class TorchTruncatedSVD:
    """Truncated SVD for PyTorch tensors.

    Similar to sklearn's TruncatedSVD but without any implicit state.
    The state is returned explicitly. Uses full SVD and truncates to
    n_components (efficient for typical TabPFN dimensions).

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
        n_samples, n_features = x.shape

        # Handle NaN values by replacing with 0 for SVD computation
        nan_mask = torch.isnan(x)
        x_filled = torch.where(nan_mask, torch.zeros_like(x), x)

        # Clamp n_components to valid range
        n_components = min(self.n_components, n_samples, n_features)
        n_components = max(1, n_components)

        # torch.linalg.svd requires float32 or float64; cast up if needed
        compute_dtype = x_filled.dtype
        if compute_dtype not in (torch.float32, torch.float64):
            compute_dtype = torch.float32
            x_filled = x_filled.to(compute_dtype)

        # Compute full SVD: X = U @ diag(S) @ V^T
        # torch.linalg.svd returns V^T directly (not V)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*linalg_svd.*not currently supported on the MPS backend.*",
            )
            u, s, vh = torch.linalg.svd(x_filled, full_matrices=False)

        # Truncate to n_components
        u = u[:, :n_components]
        s = s[:n_components]
        vh = vh[:n_components, :]

        # Apply sign flip for deterministic output.
        # We use the same convention as sklearn (u_based_decision=False:
        # flip based on V rows) but use a tie-breaking rule that is stable
        # across different SVD algorithms / platforms: for each row of V,
        # pick the sign so that the element with the largest absolute value
        # is positive; break ties by choosing the leftmost column.
        u, vh = _svd_flip_stable(u, vh)

        return {
            "components": vh,
            "singular_values": s,
        }

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Project the data onto the SVD components.

        Args:
            x: Input tensor to transform [n_samples, n_features].
            fitted_cache: Cache returned by fit.

        Returns:
            Transformed tensor [n_samples, n_components].
        """
        if "components" not in fitted_cache:
            raise ValueError("Invalid fitted cache. Must contain 'components'.")

        components = fitted_cache["components"]

        # Components may be in float32 (from fit) while input is float16.
        # Compute in the components dtype, then cast back.
        orig_dtype = x.dtype
        compute_dtype = components.dtype

        # Handle NaN values: preserve them in output
        x_compute = x.to(compute_dtype) if x.dtype != compute_dtype else x
        nan_mask = torch.isnan(x_compute)
        x_filled = torch.where(nan_mask, torch.zeros_like(x_compute), x_compute)

        # Project: X @ V (V = components.T)
        result = x_filled @ components.T

        # If any input feature was NaN, the corresponding output should be NaN
        # Since projection is a linear combination, any NaN input affects all outputs
        any_nan_per_row = nan_mask.any(dim=-1, keepdim=True)
        nan_fill = torch.tensor(float("nan"), device=x.device, dtype=compute_dtype)
        result = torch.where(any_nan_per_row.expand_as(result), nan_fill, result)
        return result.to(orig_dtype) if orig_dtype != compute_dtype else result

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
        # Ensure float32+ for numerical stability (float16 has poor precision
        # for statistics and is not supported by some ops on all devices).
        if x.dtype not in (torch.float32, torch.float64):
            x = x.to(torch.float32)

        # Replace inf with nan for std computation
        x_safe = torch.where(
            torch.isinf(x),
            torch.tensor(float("nan"), device=x.device, dtype=x.dtype),
            x,
        )

        # Compute column means ignoring NaN (matching SimpleImputer(strategy="mean"))
        nan_mask = torch.isnan(x_safe)
        num_valid = (~nan_mask).float().sum(dim=0)
        x_filled = torch.where(nan_mask, torch.zeros_like(x_safe), x_safe)
        mean = x_filled.sum(dim=0) / num_valid.clamp(min=1.0)

        # Compute population std (ddof=0) matching the CPU path where NaN
        # values are imputed with column means BEFORE StandardScaler fits.
        # Imputed values contribute 0 variance, so we sum squared deviations
        # of valid values only but divide by N (total samples, not just valid).
        n_samples = max(x_safe.shape[0], 1)
        sq_diff = torch.where(
            nan_mask,
            torch.zeros_like(x_safe),
            (x_safe - mean.unsqueeze(0)) ** 2,
        ).sum(dim=0)
        std = torch.sqrt(sq_diff / n_samples)

        # Handle constant features (std=0) by setting std to 1
        std = torch.where(std == 0, torch.ones_like(std), std)
        std = torch.where(torch.isnan(std), torch.ones_like(std), std)

        if x.shape[0] == 1:
            std = torch.ones_like(std)

        return {"std": std, "mean": mean}

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the fitted scaling to the data (no mean centering).

        Args:
            x: Input tensor to transform.
            fitted_cache: Cache returned by fit.

        Returns:
            Scaled tensor (divided by std only).
        """
        if "std" not in fitted_cache:
            raise ValueError("Invalid fitted cache. Must contain 'std'.")

        std = fitted_cache["std"]

        # Align dtype: std is in float32+ from fit, input may be float16
        orig_dtype = x.dtype
        compute_dtype = std.dtype
        x_compute = x.to(compute_dtype) if x.dtype != compute_dtype else x

        # Replace inf with nan before scaling
        x_safe = torch.where(
            torch.isinf(x_compute),
            torch.tensor(float("nan"), device=x.device, dtype=compute_dtype),
            x_compute,
        )

        # Impute NaN with column means (matching CPU make_scaler_safe which
        # wraps the scaler with SimpleImputer(strategy="mean") pre/post).
        if "mean" in fitted_cache:
            nan_mask = torch.isnan(x_safe)
            if nan_mask.any():
                col_means = fitted_cache["mean"].to(
                    device=x_safe.device, dtype=x_safe.dtype
                )
                x_safe = torch.where(nan_mask, col_means.unsqueeze(0), x_safe)

        x_scaled = x_safe / (std + torch.finfo(std.dtype).eps)

        # Clip very large values
        x_scaled = torch.clip(x_scaled, min=-100, max=100)

        # Replace any inf that might have been created with nan, then impute
        # remaining non-finite values (matching CPU post-imputation safety net)
        result = torch.where(
            torch.isfinite(x_scaled), x_scaled, torch.zeros_like(x_scaled)
        )
        return result.to(orig_dtype) if orig_dtype != compute_dtype else result

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
