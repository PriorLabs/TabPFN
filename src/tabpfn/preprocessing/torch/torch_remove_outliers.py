"""Torch implementation of outlier removal with NaN handling."""

from __future__ import annotations

import torch

from .ops import torch_nanmean, torch_nanstd


class TorchRemoveOutliers:
    """Remove outliers from PyTorch tensors based on standard deviation.

    Values outside the range [mean - n_sigma * std, mean + n_sigma * std] are
    softly clamped using a logarithmic function.

    The outlier detection is performed twice:
    1. First pass: Compute mean and std, identify outliers
    2. Second pass: Recompute mean and std excluding first-pass outliers
    """

    def __init__(self, n_sigma: float = 4.0) -> None:
        """Initialize the outlier remover.

        Args:
            n_sigma: Number of standard deviations to use for outlier threshold.
                Values outside [mean - n_sigma * std, mean + n_sigma * std] are
                considered outliers.
        """
        super().__init__()
        self.n_sigma = n_sigma
        self.lower_: torch.Tensor | None = None
        self.upper_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> TorchRemoveOutliers:
        """Compute the outlier bounds based on the training data.

        Uses a two-pass approach:
        1. First compute initial bounds based on mean and std
        2. Mask outliers with NaN and recompute bounds for more robust statistics

        Args:
            x: Input tensor with shape [T, ...] where T is the number of rows.

        Returns:
            Self for method chaining.
        """
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
            outlier_mask, torch.full_like(data_clean, float("nan")), data_clean
        )

        # Second pass: recompute statistics without outliers
        data_mean = torch_nanmean(data_clean, axis=0)
        data_std = torch_nanstd(data_clean, axis=0)
        cut_off = data_std * self.n_sigma
        self.lower_ = data_mean - cut_off
        self.upper_ = data_mean + cut_off

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply outlier removal using the fitted bounds.

        Values below the lower bound are softly clamped using:
            max(-log(1 + |x|) + lower, x)
        Values above the upper bound are softly clamped using:
            min(log(1 + |x|) + upper, x)

        Args:
            x: Input tensor to transform.

        Returns:
            Tensor with outliers softly clamped.
        """
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("Outlier remover has not been fitted. Call fit() first.")

        clamped_lower = torch.maximum(-torch.log(1 + torch.abs(x)) + self.lower_, x)
        return torch.minimum(
            torch.log(1 + torch.abs(clamped_lower)) + self.upper_, clamped_lower
        )

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply outlier removal with optional train/test splitting.

        This method supports two modes:
        1. Pre-computed bounds: If lower and upper are provided, they are used
           directly without fitting.
        2. Fit-then-transform: If bounds are not provided, they are computed
           from x[:num_train_rows] (or all of x if num_train_rows is None)
           and then applied to all of x.

        Args:
            x: Input tensor of shape [T, ...] where T is the number of rows.
            num_train_rows: Position to split train and test data. If provided,
                bounds are computed only from x[:num_train_rows]. If None,
                bounds are computed from all data.
            lower: Pre-computed lower bound to use. If provided, upper must also
                be provided.
            upper: Pre-computed upper bound to use. If provided, lower must also
                be provided.

        Returns:
            Tensor with outliers softly clamped.
        """
        if (lower is None) != (upper is None):
            raise ValueError(
                "Either both or neither of lower and upper must be provided."
            )

        if lower is not None and upper is not None:
            result = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
            return torch.minimum(torch.log(1 + torch.abs(result)) + upper, result)

        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        self.fit(fit_data)
        return self.transform(x)
