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
    The state is returned explicitly. Uses randomized SVD
    (``torch.svd_lowrank``) for large matrices and exact
    ``torch.linalg.svd`` for small ones.

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

        Uses randomized SVD (``torch.svd_lowrank``) when ``n_components`` is
        much smaller than the matrix dimensions.  This reduces memory from
        O(N * min(N, F)) to O(N * n_components) — a large saving when
        n_components << min(N, F) (e.g. 128 vs 500).

        Args:
            x: Input tensor with shape [n_samples, n_features].

        Returns:
            Cache dictionary with:
                - "components": The right singular vectors V^T
                    [n_components, n_features]
                - "singular_values": The singular values [n_components]
        """
        orig_device = x.device

        if x.device.type == "mps":
            warnings.warn(
                "SVD operators ('aten::linalg_svd', 'aten::linalg_qr') are not "
                "currently supported on the MPS backend and will fall back to "
                "run on the CPU. This may have performance implications.",
                stacklevel=2,
            )
            x = x.cpu()

        n_samples, n_features = x.shape

        # Handle NaN values by replacing with 0 for SVD computation
        nan_mask = torch.isnan(x)
        x_filled = torch.where(nan_mask, torch.zeros_like(x), x)

        # Clamp n_components to valid range
        n_components = min(self.n_components, n_samples, n_features)
        n_components = max(1, n_components)

        # torch SVD ops require float32 or float64; cast up if needed
        compute_dtype = x_filled.dtype
        if compute_dtype not in (torch.float32, torch.float64):
            compute_dtype = torch.float32
            x_filled = x_filled.to(compute_dtype)

        # Use randomized SVD only when it is both (a) accurate — the matrix
        # is large enough that the top components are well-separated — and
        # (b) faster — the projected rank q is well below the matrix rank.
        # Benchmark shows svd_lowrank becomes favorable once
        # min(N, F) >= 2*q; below that, the random-projection overhead
        # exceeds the savings from avoiding the full decomposition.
        oversampling = 10  # matches sklearn's TruncatedSVD n_oversamples default
        q = n_components + oversampling
        use_lowrank = (
            n_samples * n_features > 1_000_000 and min(n_samples, n_features) >= 2 * q
        )

        if use_lowrank:
            # torch.svd_lowrank returns (U, S, V) with A ≈ U diag(S) V^T
            u, s, v = torch.svd_lowrank(x_filled, q=q, niter=2)
            # Truncate oversampling dimensions
            u = u[:, :n_components]
            s = s[:n_components]
            vh = v[:, :n_components].T  # V [n_features, q] → V^T [n_comp, n_features]
        else:
            # Fall back to full SVD for small matrices or when n_components
            # is close to min(n_samples, n_features).
            with warnings.catch_warnings():  # warning thrown above already
                warnings.filterwarnings(
                    "ignore",
                    message=".*linalg_svd.*not currently supported on the MPS backend.*",  # noqa: E501
                )
                u, s, vh = torch.linalg.svd(x_filled, full_matrices=False)
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
            "components": vh.to(orig_device),
            "singular_values": s.to(orig_device),
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
