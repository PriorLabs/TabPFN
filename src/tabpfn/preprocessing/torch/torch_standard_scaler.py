"""Torch implementation of StandardScaler with NaN handling."""

from __future__ import annotations

import torch

from tabpfn.preprocessing.torch.ops import torch_nanmean, torch_nanstd


class TorchStandardScaler:
    """Standard scaler for PyTorch tensors with NaN handling.

    Similar to sklearn's StandardScaler but operates on PyTorch tensors and
    properly handles NaN values by ignoring them when computing statistics.

    Attributes:
        mean_: The computed mean per feature. None if not fitted.
        std_: The computed standard deviation per feature. None if not fitted.
    """

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> TorchStandardScaler:
        """Compute the mean and standard deviation over the first dimension.

        Args:
            x: Input tensor with shape [T, ...] where T is the number of rows.

        Returns:
            Self for method chaining.
        """
        self.mean_ = torch_nanmean(x, axis=0)
        self.std_ = torch_nanstd(x, axis=0)

        # Handle constant features (std=0) by setting std to 1
        self.std_ = torch.where(
            self.std_ == 0,
            torch.ones_like(self.std_),
            self.std_,
        )

        if x.shape[0] == 1:
            self.std_ = torch.ones_like(self.std_)

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fitted scaling to the data.

        Args:
            x: Input tensor to transform.

        Returns:
            Scaled tensor with mean 0 and std 1.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")

        x = (x - self.mean_) / (self.std_ + 1e-16)

        # Clip very large values
        return torch.clip(x, min=-100, max=100)

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply standard scaling with optional train/test splitting.

        This method supports two modes:
        1. Pre-computed statistics: If mean and std are provided, they are used
           directly without fitting.
        2. Fit-then-transform: If mean and std are not provided, statistics are
           computed from x[:num_train_rows] (or all of x if num_train_rows is
           None) and then applied to all of x.

        Args:
            x: Input tensor of shape [T, ...] where T is the number of samples.
            num_train_rows: Position to split train and test data. If provided,
                statistics are computed only from x[:num_train_rows]. If None,
                statistics are computed from all data.
            mean: Pre-computed mean to use. If provided, std must also be provided.
            std: Pre-computed std to use. If provided, mean must also be provided.

        Returns:
            Scaled tensor with mean 0 and std 1.
        """
        if (mean is None) != (std is None):
            raise ValueError("Either both or neither of mean and std must be provided.")

        if mean is not None and std is not None:
            # Use pre-computed statistics directly
            return (x - mean) / (std + 1e-16)

        # Determine which data to use for fitting
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        self.fit(fit_data)
        return self.transform(x)
