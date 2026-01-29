"""Custom implementations of PyTorch functions. Needed for ONNX export."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import torch


def torch_nansum(
    x: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes the sum of a tensor, treating NaNs as zero.

    Args:
        x: The input tensor.
        axis: The dimension or dimensions to reduce.
        keepdim: Whether the output tensor has `axis` retained or not.
        dtype: The desired data type of the returned tensor.

    Returns:
        The sum of the tensor with NaNs treated as zero.
    """
    nan_mask = torch.isnan(x)
    masked_input = torch.where(
        nan_mask,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        x,
    )
    return torch.sum(masked_input, axis=axis, keepdim=keepdim, dtype=dtype)  # type: ignore


@overload
def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: Literal[False] = False,
    include_inf: bool = False,
) -> torch.Tensor: ...


@overload
def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: Literal[True],
    include_inf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: bool = False,
    include_inf: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean of a tensor over a given dimension, ignoring NaNs.

    Designed for stability: If all inputs are NaN, the mean will be 0.0.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.
        return_nanshare: If True, also return the proportion of NaNs.
        include_inf: If True, treat infinity as NaN for the purpose of the calculation.

    Returns:
        The mean of the input tensor, ignoring NaNs. If `return_nanshare` is True,
        returns a tuple containing the mean and the share of NaNs.
    """
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        dim=axis,
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    if return_nanshare:
        return value / num, 1.0 - (num / x.shape[axis])
    return value / num.clip(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Computes standard deviation of a tensor over a given dimension, ignoring NaNs.

    This implementation is designed for stability. It clips the denominator `(num - 1)`
    at a minimum of 1. This prevents division-by-zero errors that would produce `NaN`
    or `inf` when calculating the standard deviation of a single data point.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.

    Returns:
        The standard deviation of the input tensor, ignoring NaNs.
    """
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    mean = value / num.clip(min=1.0)
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis),
        x.shape[axis],
        dim=axis,
    )
    # Clip the denominator to avoid division by zero when num=1
    var = torch_nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1).clip(
        min=1.0
    )
    return torch.sqrt(var)


@overload
def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: Literal[False] = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor: ...


@overload
def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: Literal[True],
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: bool = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Normalize data to mean 0 and std 1 with high numerical stability.

    This function is designed to be robust against several edge cases:
    1.  **Constant Features**: If a feature is constant, its standard deviation (`std`)
        will be 0. This is handled by replacing `std=0` with `1` to prevent `0/0`
        division, effectively mapping constant features to a normalized value of 0.
    2.  **Single-Sample Normalization**: If the normalization is based on a single
        data point, `std` is explicitly set to `1` to prevent undefined behavior.
    3.  **Low-Precision Dtypes**: During the final division, a small epsilon (`1e-16`)
        is added to the denominator. This prevents division by a near-zero `std`,
        which could cause the value to overflow to infinity (`inf`), especially when
        using low-precision dtypes like `torch.float16`.
    4.  **Autograd Compatibility**: All inplace operations (`x[:] = ...`) have been
        replaced with their functional equivalents (`torch.where`, `torch.ones_like`)
        to ensure the computation graph is not corrupted, allowing gradients to be
        computed correctly.

    Args:
        data: The data to normalize. (T, B, H)
        normalize_positions: If > 0, only use the first `normalize_positions`
            positions for normalization.
        return_scaling: If True, return the scaling parameters as well (mean, std).
        std_only: If True, only divide by std.
        clip: If True, clip the data to [-100, 100].
        mean: If given, use this value instead of computing it.
        std: If given, use this value instead of computing it.

    Returns:
        The normalized data tensor, or a tuple containing the data and scaling factors.
    """
    assert (mean is None) == (std is None), (
        "Either both or none of mean and std must be given"
    )
    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)  # type: ignore
            std = torch_nanstd(data[:normalize_positions], axis=0)
        else:
            mean = torch_nanmean(data, axis=0)  # type: ignore
            std = torch_nanstd(data, axis=0)

        # Inplace assignments with functional equivalents to support autograd.
        std = torch.where(std == 0, torch.ones_like(std), std)

        if len(data) == 1 or normalize_positions == 1:
            std = torch.ones_like(std)

        if std_only:
            mean = torch.zeros_like(mean)

    assert mean is not None
    assert std is not None

    # Add epsilon for numerical stability
    data = (data - mean) / (std + 1e-16)

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def select_features(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Select features from the input tensor based on the selection mask,
    and arrange them contiguously in the last dimension.
    If batch size is bigger than 1, we pad the features with zeros to make the number of
    features fixed.

    Args:
        x: The input tensor of shape (sequence_length, batch_size, total_features)
        sel: The boolean selection mask indicating which features to keep of shape
        (batch_size, total_features)

    Returns:
        The tensor with selected features.
        The shape is (sequence_length, batch_size, number_of_selected_features) if
        batch_size is 1.
        The shape is (sequence_length, batch_size, total_features) if batch_size is
        greater than 1.
    """
    B, total_features = sel.shape

    # Do nothing if we need to select all of the features
    if torch.all(sel):
        return x

    # If B == 1, we don't need to append zeros, as the number of features don't need to
    # be fixed.
    if B == 1:
        return x[:, :, sel[0]]

    new_x = x.detach().clone()

    # For each batch, compute the number of selected features.
    sel_counts = sel.sum(dim=-1)  # shape: (B,)

    for b in range(B):
        s = int(sel_counts[b])
        if s != total_features:
            if s > 0:
                new_x[:, b, :s] = x[:, b, sel[b]]
            new_x[:, b, s:] = 0

    return new_x


def remove_outliers(
    X: torch.Tensor,
    n_sigma: float = 4,
    normalize_positions: int = -1,
    lower: None | torch.Tensor = None,
    upper: None | torch.Tensor = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # Expects T, B, H
    assert (lower is None) == (upper is None), "Either both or none of lower and upper"
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    if lower is None:
        data = X if normalize_positions == -1 else X[:normalize_positions]
        data_clean = data[:].clone()
        data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

        data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
        data_mean, data_std = (
            torch_nanmean(data_clean, axis=0),
            torch_nanstd(data_clean, axis=0),
        )
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

    assert lower is not None
    assert upper is not None

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    return X, (lower, upper)
