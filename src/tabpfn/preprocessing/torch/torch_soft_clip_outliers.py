"""Torch implementation of outlier clipping with NaN handling."""

from __future__ import annotations

import torch

from tabpfn.preprocessing.torch.ops import torch_nanmean, torch_nanstd


class TorchSoftClipOutliers:
    """Softly clip outliers from PyTorch tensors based on standard deviation.

    Values outside the range [mean - n_sigma * std, mean + n_sigma * std] are
    softly clamped using a logarithmic function.

    The outlier detection is performed twice:
    1. First pass: Compute mean and std, identify outliers
    2. Second pass: Recompute mean and std excluding first-pass outliers
    """

    def __init__(self, n_sigma: float = 4.0) -> None:
        """Init.

        Args:
            n_sigma: Number of standard deviations to use for outlier threshold.
                Values outside [mean - n_sigma * std, mean + n_sigma * std] are
                considered outliers.
        """
        super().__init__()
        self.n_sigma = n_sigma

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the outlier bounds based on the training data.

        Uses a two-pass approach:
        1. First compute initial bounds based on mean and std
        2. Mask outliers with NaN and recompute bounds for more robust statistics

        Args:
            x: Input tensor with shape [T, ...] where T is the number of rows.

        Returns:
            Cache dictionary with the cache for the transform step.
                - "lower": Lower bound for each feature.
                - "upper": Upper bound for each feature.
        """
        if x.shape[0] <= 1:
            lower = torch.full(x.shape[1:], -torch.inf, dtype=x.dtype, device=x.device)
            upper = torch.full(x.shape[1:], torch.inf, dtype=x.dtype, device=x.device)
            return {"lower": lower, "upper": upper}

        # First pass: compute initial statistics
        data_mean = torch_nanmean(x, axis=0)
        data_std = torch_nanstd(x, axis=0)
        cut_off = data_std * self.n_sigma
        lower_initial = data_mean - cut_off
        upper_initial = data_mean + cut_off

        # Create a clean copy with outliers masked as NaN
        data_clean = x.clone()
        outlier_mask = torch.logical_or(
            data_clean > upper_initial, data_clean < lower_initial
        )
        data_clean = torch.where(
            outlier_mask, torch.full_like(data_clean, torch.nan), data_clean
        )

        # Second pass: recompute statistics without outliers
        data_mean = torch_nanmean(data_clean, axis=0)
        data_std = torch_nanstd(data_clean, axis=0)
        cut_off = data_std * self.n_sigma
        lower = data_mean - cut_off
        upper = data_mean + cut_off

        return {"lower": lower, "upper": upper}

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply softly clipping outliers using the fitted bounds.

        Values below the lower bound are softly clamped using:
            max(-log(1 + |x|) + lower, x)
        Values above the upper bound are softly clamped using:
            min(log(1 + |x|) + upper, x)

        Args:
            x: Input tensor to transform.
            fitted_cache: Cache returned by fit.

        Returns:
            Tensor with outliers softly clamped.
        """
        if "lower" not in fitted_cache or "upper" not in fitted_cache:
            raise ValueError("Invalid fitted cache. Must contain 'lower' and 'upper'.")

        lower = fitted_cache["lower"]
        upper = fitted_cache["upper"]

        if x.shape[0] == 1:
            return x

        clamped_lower = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
        return torch.minimum(
            torch.log(1 + torch.abs(clamped_lower)) + upper, clamped_lower
        )

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
    ) -> torch.Tensor:
        """Apply softly clipping outliers with optional train/test splitting.

        This is a convenience method similar to `fit_transform` but with
        train/test split handled automatically and no state being kept.
        This can be used in the forward pass of the model during training.

        Args:
            x: Input tensor of shape [T, ...] where T is the number of rows.
            num_train_rows: Position to split train and test data. If provided,
                bounds are computed only from x[:num_train_rows]. If None,
                bounds are computed from all data.

        Returns:
            Tensor with outliers softly clamped.
        """
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)
