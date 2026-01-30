"""Torch implementation of StandardScaler with NaN handling."""

from __future__ import annotations

import torch

from tabpfn.preprocessing.torch.ops import torch_nanmean, torch_nanstd


class TorchStandardScaler:
    """Standard scaler for PyTorch tensors with NaN handling.

    Similar to sklearn's StandardScaler but without any implicit state.
    The state is returned explicitly.
    """

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the mean and standard deviation over the first dimension.

        Args:
            x: Input tensor with shape [T, ...] where T is the number of rows.

        Returns:
            Cache dictionary with the cache for the transform step.
        """
        mean = torch_nanmean(x, axis=0)
        std = torch_nanstd(x, axis=0)

        # Handle constant features (std=0) by setting std to 1
        std = torch.where(std == 0, torch.ones_like(std), std)

        if x.shape[0] == 1:
            std = torch.ones_like(std)

        return {"mean": mean, "std": std}

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the fitted scaling to the data.

        Args:
            x: Input tensor to transform.
            fitted_cache: Cache returned by fit.

        Returns:
            Scaled tensor with mean 0 and std 1.
        """
        if "mean" not in fitted_cache or "std" not in fitted_cache:
            raise ValueError("Invalid fitted cache. Must contain 'mean' and 'std'.")

        mean = fitted_cache["mean"]
        std = fitted_cache["std"]
        x = (x - mean) / (std + torch.finfo(std.dtype).eps)

        # Clip very large values
        return torch.clip(x, min=-100, max=100)

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
    ) -> torch.Tensor:
        """Apply standard scaling with optional train/test splitting.

        This is a convenience method similar to `fit_transform` but with
        train/test split handled automatically and no state being kept.
        This can be used in the forward pass of the model during training.

        Args:
            x: Input tensor of shape [T, ...] where T is the number of samples.
            num_train_rows: Position to split train and test data. If provided,
                statistics are computed only from x[:num_train_rows]. If None,
                statistics are computed from all data.

        Returns:
            Scaled tensor with mean 0 and std 1.
        """
        # Determine which data to use for fitting
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)
