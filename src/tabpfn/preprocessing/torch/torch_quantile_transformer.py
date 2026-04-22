"""Torch implementation of QuantileTransformer with NaN handling."""

from __future__ import annotations

import torch


class TorchQuantileTransformer:
    """Quantile transformer for PyTorch tensors with NaN handling.

    Similar to sklearn's QuantileTransformer but without any implicit state.
    The state is returned explicitly. Only supports uniform output distribution.

    This transformer maps the data to a uniform distribution in [0, 1] using
    quantile information from the training data.
    """

    def __init__(self, n_quantiles: int = 1_000) -> None:
        """Initialize the quantile transformer.

        Args:
            n_quantiles: Number of quantiles to compute. More quantiles give
                a better approximation of the CDF but use more memory.
        """
        super().__init__()
        self.n_quantiles = n_quantiles

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the quantiles from the training data.

        Args:
            x: Input tensor with shape [T, ...] where T is the number of rows.

        Returns:
            Cache dictionary with:
                - "quantiles": The computed quantiles [n_quantiles, ...]
                - "references": The reference quantile positions [n_quantiles]
        """
        n_samples = x.shape[0]

        if n_samples <= 1:
            # Not enough data to compute meaningful quantiles.
            # Return a degenerate cache that makes transform return 0.5
            # for all non-NaN values (identity-like behaviour).
            references = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
            if n_samples == 0:
                val = torch.zeros(x.shape[1:], device=x.device, dtype=x.dtype)
            else:
                val = x[0]
            quantiles = torch.stack([val, val], dim=0)
            return {"quantiles": quantiles, "references": references}

        n_quantiles_effective = min(self.n_quantiles, n_samples)

        # torch.nanquantile requires float32 or float64; cast up if needed
        # (e.g. when inference_precision is float16).
        compute_dtype = x.dtype
        if x.dtype not in (torch.float32, torch.float64):
            compute_dtype = torch.float32

        references = torch.linspace(
            0, 1, n_quantiles_effective, device=x.device, dtype=compute_dtype
        )

        x_compute = x.to(compute_dtype) if x.dtype != compute_dtype else x
        quantiles = self._nanquantile_chunked(x_compute, references)

        # Ensure monotonicity (handle floating point issues)
        # Use cumulative maximum along the quantile dimension
        quantiles = torch.cummax(quantiles, dim=0).values

        return {"quantiles": quantiles, "references": references}

    def _nanquantile_chunked(
        self,
        x: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Compute nanquantile in column-chunks to bound peak memory.

        Each column is independent so we process subsets of columns to keep
        peak memory at O(N * chunk_cols) instead of O(N * F).  The chunk size
        is determined by ``_get_fit_chunk_cols``.
        """
        original_shape = x.shape  # [N, ...]
        n_samples = x.shape[0]

        # Flatten trailing dimensions to [N, F]
        x_flat = x.reshape(n_samples, -1) if x.ndim > 2 else x

        n_features = x_flat.shape[1] if x_flat.ndim > 1 else 1
        chunk_cols = self._get_fit_chunk_cols(x_flat)
        process_all_at_once = n_features <= chunk_cols

        if process_all_at_once:
            quantiles = torch.nanquantile(x, references, dim=0)
        else:
            chunks = []
            for col_start in range(0, n_features, chunk_cols):
                col_end = min(col_start + chunk_cols, n_features)
                q_chunk = torch.nanquantile(
                    x_flat[:, col_start:col_end], references, dim=0
                )
                chunks.append(q_chunk)
            quantiles = torch.cat(chunks, dim=-1)

            if x.ndim > 2:
                quantiles = quantiles.reshape(len(references), *original_shape[1:])

        return quantiles

    def _get_fit_chunk_cols(self, x_flat: torch.Tensor) -> int:
        """Compute a column-chunk size that keeps nanquantile peak memory bounded.

        ``torch.nanquantile`` internally sorts the data along dim=0, requiring
        roughly 10x the input size as temporary memory (sort buffer + indexing).
        We target ~2 GB of intermediates so that 100k x 500 fits in one shot
        while 1M-row datasets stay bounded at ~2 GB of workspace per chunk.
        """
        n_samples = x_flat.shape[0]
        element_size = x_flat.element_size()
        overhead_factor = 10
        target_bytes = 2 * 1024**3  # 2 GB
        bytes_per_col = n_samples * element_size * overhead_factor
        return max(1, target_bytes // max(bytes_per_col, 1))

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Transform the data to uniform distribution using fitted quantiles.

        Automatically processes in row-chunks when the data is large to keep
        peak intermediate memory bounded (~2 GB of temporaries).

        Args:
            x: Input tensor to transform.
            fitted_cache: Cache returned by fit.

        Returns:
            Transformed tensor with values in [0, 1].
        """
        if "quantiles" not in fitted_cache or "references" not in fitted_cache:
            raise ValueError(
                "Invalid fitted cache. Must contain 'quantiles' and 'references'."
            )

        quantiles = fitted_cache["quantiles"]
        references = fitted_cache["references"]

        # The cache may be in float32 (from fit) while input is float16.
        # Compute in the cache dtype, then cast back.
        orig_dtype = x.dtype
        compute_dtype = quantiles.dtype
        x_compute = x.to(compute_dtype) if x.dtype != compute_dtype else x

        chunk_size = self._get_transform_chunk_size(x_compute)
        n_samples = x_compute.shape[0]

        if n_samples <= chunk_size:
            result = self._transform_chunk(x_compute, quantiles, references)
        else:
            chunks = []
            for start in range(0, n_samples, chunk_size):
                chunk = x_compute[start : start + chunk_size]
                chunks.append(self._transform_chunk(chunk, quantiles, references))
            result = torch.cat(chunks, dim=0)

        return result.to(orig_dtype) if orig_dtype != compute_dtype else result

    def _transform_chunk(
        self,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Transform a single chunk, preserving NaN positions."""
        nan_mask = torch.isnan(x)
        result = self._interpolate(x, quantiles, references)
        return torch.where(nan_mask, float("nan"), result)

    def _get_transform_chunk_size(self, x: torch.Tensor) -> int:
        """Compute a row-chunk size that keeps intermediate memory bounded.

        ``_interpolate`` creates ~15x intermediate memory per input element
        (forward + backward searchsorted, gather, slope tensors).  We target
        ~2 GB of intermediates so that even very large datasets stay within
        reasonable GPU memory.
        """
        n_features = x.shape[-1] if x.ndim > 1 else 1
        element_size = x.element_size()
        overhead_factor = 15
        target_bytes = 2 * 1024**3  # 2 GB
        bytes_per_row = n_features * element_size * overhead_factor
        return max(1_000, target_bytes // max(bytes_per_row, 1))

    def _interpolate(
        self,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate values to get their quantile positions.

        Uses the same approach as sklearn: average of forward and backward
        interpolation to handle repeated quantile values.

        sklearn formula:
            0.5 * (interp(x, quantiles, references)
                   - interp(-x, -quantiles[::-1], -references[::-1]))

        Args:
            x: Input tensor [T, ...] where T is the number of rows.
            quantiles: Computed quantiles [n_quantiles, ...].
            references: Reference positions [n_quantiles].

        Returns:
            Interpolated values in [0, 1].
        """
        original_shape = x.shape
        n_quantiles = quantiles.shape[0]

        # Flatten all dimensions except the first (samples)
        if x.ndim > 1:
            x_flat = x.reshape(x.shape[0], -1).contiguous()
            quantiles_flat = quantiles.reshape(n_quantiles, -1).contiguous()
        else:
            x_flat = x.unsqueeze(-1).contiguous()
            quantiles_flat = quantiles.unsqueeze(-1).contiguous()

        references = references.contiguous()

        # Transpose to [n_features, ...] for batched searchsorted
        x_t = x_flat.t().contiguous()  # [n_features, n_samples]
        q_t = quantiles_flat.t().contiguous()  # [n_features, n_quantiles]

        # Forward interpolation: interp(x, quantiles, references)
        refs = references.unsqueeze(0).expand_as(q_t)  # [n_features, n_quantiles]
        interp_forward = self._interp_batched(x_t, q_t, refs)

        # Backward interpolation: interp(-x, -quantiles[::-1], -references[::-1])
        q_rev = torch.flip(-q_t, dims=[1]).contiguous()
        refs_rev = torch.flip(-refs, dims=[1]).contiguous()
        interp_backward = self._interp_batched(-x_t, q_rev, refs_rev)

        # sklearn formula: 0.5 * (forward - backward)
        # Since backward is in [-1, 0], subtracting it adds a positive value
        result = 0.5 * (interp_forward - interp_backward)  # [n_features, n_samples]
        result = result.t()  # [n_samples, n_features]

        result = torch.clamp(result, 0.0, 1.0)

        if len(original_shape) > 1:
            result = result.reshape(original_shape)
        else:
            result = result.squeeze(-1)

        return result

    def _interp_batched(
        self,
        x: torch.Tensor,
        xp: torch.Tensor,
        fp: torch.Tensor,
    ) -> torch.Tensor:
        """Batched 1D linear interpolation across all features.

        Vectorized version of numpy.interp that operates on all feature
        columns simultaneously using batched searchsorted.

        Args:
            x: Values to interpolate, shape [n_features, n_samples].
            xp: Sorted breakpoints per feature, shape [n_features, n_quantiles].
            fp: Function values at breakpoints, shape [n_features, n_quantiles].
            clamp_output: Whether to clamp output to [fp.min(), fp.max()] per feature.

        Returns:
            Interpolated values, shape [n_features, n_samples].
        """
        n_quantiles = xp.shape[1]

        # Detect constant-quantile features (all breakpoints identical)
        is_constant = xp[:, 0] == xp[:, -1]  # [n_features]
        mid_val = 0.5 * (fp[:, 0] + fp[:, -1])  # [n_features]

        # Batched binary search: [n_features, n_samples]
        indices = torch.searchsorted(xp.contiguous(), x.contiguous(), right=False)
        indices = torch.clamp(indices, 1, n_quantiles - 1)

        # Gather surrounding breakpoints and function values
        x_low = torch.gather(xp, 1, indices - 1)
        x_high = torch.gather(xp, 1, indices)
        f_low = torch.gather(fp, 1, indices - 1)
        f_high = torch.gather(fp, 1, indices)

        # Linear interpolation, avoiding division by zero
        dx = x_high - x_low
        dx = torch.where(dx == 0, torch.ones_like(dx), dx)
        slope = (f_high - f_low) / dx
        result = f_low + slope * (x - x_low)

        # Match np.interp boundary behaviour: values at or beyond the
        # breakpoint range are clamped to the first/last function value,
        # NOT extrapolated.
        fp_first = fp[:, :1]  # [n_features, 1]
        fp_last = fp[:, -1:]  # [n_features, 1]
        xp_first = xp[:, :1]
        xp_last = xp[:, -1:]
        result = torch.where(x <= xp_first, fp_first, result)
        result = torch.where(x >= xp_last, fp_last, result)

        # For constant features, return the midpoint value
        return torch.where(is_constant.unsqueeze(1), mid_val.unsqueeze(1), result)

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
    ) -> torch.Tensor:
        """Apply quantile transformation with optional train/test splitting.

        This is a convenience method similar to `fit_transform` but with
        train/test split handled automatically and no state being kept.
        This can be used in the forward pass of the model during training.

        Args:
            x: Input tensor of shape [T, ...] where T is the number of samples.
            num_train_rows: Position to split train and test data. If provided,
                quantiles are computed only from x[:num_train_rows]. If None,
                quantiles are computed from all data.

        Returns:
            Transformed tensor with values in [0, 1].
        """
        # Determine which data to use for fitting
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)
