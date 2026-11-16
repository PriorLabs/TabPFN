#  Copyright (c) Prior Labs GmbH 2026.

"""Torch implementation of QuantileTransformer with NaN handling."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

# Same as sklearn's QuantileTransformer: inputs this close to the fitted range
# boundaries map to the clipped tails of the normal output.
_BOUNDS_THRESHOLD = 1e-7
_NORMAL_CLIP_MAX = float(
    torch.special.ndtri(
        torch.tensor(1 - (_BOUNDS_THRESHOLD - np.spacing(1)), dtype=torch.float64)
    )
)
_NORMAL_CLIP_MIN = float(
    torch.special.ndtri(
        torch.tensor(_BOUNDS_THRESHOLD - np.spacing(1), dtype=torch.float64)
    )
)
# sklearn tests ``x - 1e-7 < q_min`` and ``x + 1e-7 > q_max`` in float64. From
# |x| = 2**30 on, 1e-7 is below half an ulp, so ``x +- 1e-7`` rounds back to x.
_TIE_LIMIT = 2.0**30


class TorchQuantileTransformer:
    """Quantile transformer for PyTorch tensors with NaN handling.

    Similar to sklearn's QuantileTransformer but without any implicit state.
    The state is returned explicitly.
    """

    def __init__(
        self,
        n_quantiles: int = 1_000,
        extrapolate_ratio: float | None = None,
        output_distribution: Literal["uniform", "normal"] = "uniform",
    ) -> None:
        """Initialize the quantile transformer.

        Args:
            n_quantiles: Number of quantiles to compute. More quantiles give
                a better approximation of the CDF but use more memory.
            extrapolate_ratio: When set, inputs outside the fitted range are
                linearly extrapolated past ``[0, 1]`` and clipped at
                ``-ratio`` / ``1 + ratio`` instead of being clamped at the
                boundary quantile. Mirrors the
                :class:`AdaptiveQuantileTransformer` mode-1 behaviour.
            output_distribution: Marginal distribution of the output, as in
                sklearn's ``QuantileTransformer``.
        """
        super().__init__()
        if extrapolate_ratio is not None and extrapolate_ratio < 0:
            raise ValueError("extrapolate_ratio must be non-negative.")
        if extrapolate_ratio is not None and output_distribution != "uniform":
            raise ValueError(
                "extrapolate_ratio is only supported for output_distribution='uniform'."
            )
        self.n_quantiles = n_quantiles
        self.extrapolate_ratio = extrapolate_ratio
        self.output_distribution = output_distribution

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

        # torch.nanquantile and ndtri need float32 or float64; cast up if needed
        # (e.g. when inference_precision is float16).
        compute_dtype = x.dtype
        if x.dtype not in (torch.float32, torch.float64):
            compute_dtype = torch.float32

        # The normal output assumes references == linspace(0, 1, n) and does
        # not read them.
        if n_samples <= 1:
            # Not enough data to compute meaningful quantiles. With the
            # degenerate cache, uniform output is 0.5 for every non-NaN value
            # and normal output is a clip bound.
            references = torch.tensor([0.0, 1.0], device=x.device, dtype=compute_dtype)
            if n_samples == 0:
                val = torch.zeros(x.shape[1:], device=x.device, dtype=compute_dtype)
            else:
                val = x[0].to(compute_dtype)
            quantiles = torch.stack([val, val], dim=0)
            return {"quantiles": quantiles, "references": references}

        n_quantiles_effective = min(self.n_quantiles, n_samples)
        references = torch.linspace(
            0, 1, n_quantiles_effective, device=x.device, dtype=compute_dtype
        )

        quantiles = self._nanquantile_chunked(x, references)

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
            quantiles = torch.nanquantile(x.to(references.dtype), references, dim=0)
        else:
            chunks = []
            for col_start in range(0, n_features, chunk_cols):
                x_chunk = x_flat[:, col_start : col_start + chunk_cols]
                q_chunk = torch.nanquantile(
                    x_chunk.to(references.dtype), references, dim=0
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
        # Computed in at least float32.
        element_size = max(x_flat.element_size(), 4)
        overhead_factor = 10
        target_bytes = 2 * 1024**3  # 2 GB
        bytes_per_col = n_samples * element_size * overhead_factor
        return max(1, target_bytes // max(bytes_per_col, 1))

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform the data to the output distribution using fitted quantiles.

        Automatically processes in row-chunks when the data is large to keep
        peak intermediate memory bounded (~2 GB of temporaries).

        Args:
            x: Input tensor to transform.
            fitted_cache: Cache returned by fit.
            out: Tensor of ``x``'s shape and device that receives the result,
                cast to its dtype. It may be ``x`` itself, which is then
                overwritten. If None, the result is a new tensor of ``x``'s
                dtype. Inputs that require grad need ``out=None``.

        Returns:
            Transformed tensor: ``out`` when given.

        Raises:
            ValueError: If the cache is incomplete, or ``out`` differs from
                ``x`` in shape or device.
        """
        if "quantiles" not in fitted_cache or "references" not in fitted_cache:
            raise ValueError(
                "Invalid fitted cache. Must contain 'quantiles' and 'references'."
            )
        if out is not None and (out.shape != x.shape or out.device != x.device):
            raise ValueError(
                f"out must match x in shape and device, got {tuple(out.shape)} on "
                f"{out.device} for x of {tuple(x.shape)} on {x.device}."
            )

        quantiles = fitted_cache["quantiles"]
        references = fitted_cache["references"]

        # The cache may be in float32 (from fit) while input is float16.
        # Compute each chunk in the cache dtype and cast it as it is written.
        chunk_size = self._get_transform_chunk_size(x)
        # The ``out=`` form of ``torch.where`` neither casts nor supports
        # autograd, so only a caller's ``out`` of the cache dtype goes through it.
        write_into_out = out is not None and out.dtype == quantiles.dtype
        if out is None:
            if x.shape[0] <= chunk_size:
                return self._transform_chunk(
                    x.to(quantiles.dtype), quantiles, references
                ).to(x.dtype)
            out = torch.empty_like(x)
        for start in range(0, x.shape[0], chunk_size):
            rows = slice(start, start + chunk_size)
            # A chunk is read in full before its rows of ``out`` are written,
            # so ``out`` may be ``x``.
            chunk = x[rows].to(quantiles.dtype)
            if write_into_out:
                self._transform_chunk(chunk, quantiles, references, out=out[rows])
            else:
                out[rows] = self._transform_chunk(chunk, quantiles, references)
        return out

    def _transform_chunk(
        self,
        x: torch.Tensor,
        quantiles: torch.Tensor,
        references: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform a single chunk, preserving NaN positions."""
        nan_mask = torch.isnan(x)
        result = self._interpolate(x, quantiles, references)
        # The ``out=`` form takes no Python scalar.
        nan = torch.full((), float("nan"), dtype=result.dtype, device=result.device)
        return torch.where(nan_mask, nan, result, out=out)

    def _get_transform_chunk_size(self, x: torch.Tensor) -> int:
        """Compute a row-chunk size that keeps intermediate memory bounded.

        ``_interpolate`` creates ~15x intermediate memory per input element
        (forward + backward searchsorted, gather, slope tensors).  We target
        ~2 GB of intermediates so that even very large datasets stay within
        reasonable GPU memory.
        """
        n_features = x[0].numel() if x.shape[0] else 1
        # Computed in at least float32.
        element_size = max(x.element_size(), 4)
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
            Values in the output distribution.
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

        if self.output_distribution == "normal":
            result = self._to_normal(x_t, q_t).t()
            if len(original_shape) > 1:
                return result.reshape(original_shape)
            return result.squeeze(-1)

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

        if self.extrapolate_ratio is None:
            result = torch.clamp(result, 0.0, 1.0)
        else:
            result = self._apply_extrapolation(result, x_flat, quantiles_flat)

        if len(original_shape) > 1:
            result = result.reshape(original_shape)
        else:
            result = result.squeeze(-1)

        return result

    def _to_normal(self, x: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        """Map ``x`` [n_features, n_samples] to sklearn's clipped normal output.

        ``references`` is always ``linspace(0, 1, n)``, so grid positions replace
        it. ``u`` and ``1 - u`` are each built from small terms, which keeps
        float32 precise in both tails. Inputs within ``_BOUNDS_THRESHOLD`` of the
        fitted min/max go to the clip bounds; the min side wins for constant
        features.
        """
        n = xp.shape[1]
        p_fwd, c_fwd = _interp_positions(x, xp)
        p_bwd, c_bwd = _interp_positions(-x, torch.flip(-xp, dims=[1]))
        # sklearn averages forward and backward np.interp; ties get the midpoint.
        lower = p_fwd.add_(c_bwd)  # 2 * (n - 1) * u
        upper = c_fwd.add_(p_bwd)  # 2 * (n - 1) * (1 - u)
        del c_bwd, p_bwd
        # ndtri of the smaller tail is <= 0; flip it for the upper half.
        result = torch.special.ndtri(torch.minimum(lower, upper).div_(2 * (n - 1)))
        result.copysign_(lower.sub_(upper))
        del lower, upper

        q_min, q_max = xp[:, :1], xp[:, -1:]
        if x.dtype == torch.float64:
            # sklearn's own test, in the dtype it runs in.
            above = (x + _BOUNDS_THRESHOLD) > q_max
            below = (x - _BOUNDS_THRESHOLD) < q_min
        else:
            # Per-feature thresholds; 0 where sklearn's float64 test misses ties.
            thr_min = torch.where(
                (q_min > -_TIE_LIMIT) & (q_min <= _TIE_LIMIT), _BOUNDS_THRESHOLD, 0.0
            )
            thr_max = torch.where(
                (q_max >= -_TIE_LIMIT) & (q_max < _TIE_LIMIT), _BOUNDS_THRESHOLD, 0.0
            )
            above = (q_max - x) < thr_max
            below = (x - q_min) < thr_min
        result = torch.where(above, _NORMAL_CLIP_MAX, result)
        result = torch.where(below, _NORMAL_CLIP_MIN, result)
        return result.clamp_(_NORMAL_CLIP_MIN, _NORMAL_CLIP_MAX)

    def _apply_extrapolation(
        self,
        result: torch.Tensor,
        x: torch.Tensor,
        quantiles: torch.Tensor,
    ) -> torch.Tensor:
        """Replace boundary-clamped values with linear extrapolation.

        ``_interp_batched`` clamps out-of-range inputs at ``fp_first``/``fp_last``
        (i.e. 0 and 1). When ``extrapolate_ratio`` is set, those values are
        instead linearly extrapolated past ``[0, 1]`` using the training-data
        range (``quantiles[0]`` and ``quantiles[-1]`` per feature) and clipped
        at ``-ratio`` / ``1 + ratio``.

        Args:
            result: Output of the standard transform, shape [n_samples, n_features].
            x: Original input, shape [n_samples, n_features].
            quantiles: Per-feature breakpoints, shape [n_quantiles, n_features].
                ``quantiles[0]`` and ``quantiles[-1]`` are the training min/max.
        """
        ratio = self.extrapolate_ratio
        x_min = quantiles[0]  # [n_features]
        x_max = quantiles[-1]  # [n_features]
        x_range = x_max - x_min
        safe_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)

        # (x - x_max) / range + 1 simplifies to (x - x_min) / range, so a
        # single normalised tensor serves both the below and above branches.
        norm = (x - x_min) / safe_range

        below_mask = x < x_min
        above_mask = x > x_max

        # Constant features (x_range == 0) have no meaningful extrapolation;
        # leave the standard 0.5 midpoint untouched.
        constant_mask = x_range == 0
        below_mask = below_mask & ~constant_mask
        above_mask = above_mask & ~constant_mask

        result = torch.where(below_mask, torch.clamp(norm, -ratio, 0.0), result)
        return torch.where(above_mask, torch.clamp(norm, 1.0, 1.0 + ratio), result)

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
            Transformed tensor.
        """
        # Determine which data to use for fitting
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)


def _interp_positions(
    x: torch.Tensor, xp: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``np.interp`` position ``p`` of ``x`` in each row of ``xp``, and ``n - 1 - p``.

    Both are a whole number plus a fraction in [0, 1], so neither cancels.
    """
    n = xp.shape[1]
    idx = torch.searchsorted(xp, x, right=True).clamp_(1, n - 1)
    x_high = torch.gather(xp, 1, idx)
    idx.sub_(1)
    x_low = torch.gather(xp, 1, idx)
    # xp[idx] <= x < xp[idx + 1] inside the range. Below it, p < 0 and
    # c > n - 1, so clamping fixes them. At or above it, set p and c directly:
    # a tie at the top gives 0 / 0.
    dx = x_high - x_low
    p = (x - x_low).div_(dx).add_(idx)
    c = (x_high - x).div_(dx).add_(torch.rsub(idx, n - 2))
    del idx, x_low, x_high, dx
    above = x >= xp[:, -1:]
    p = torch.where(above, n - 1.0, p).clamp_(min=0.0)
    c = torch.where(above, 0.0, c).clamp_(max=n - 1.0)
    return p, c
