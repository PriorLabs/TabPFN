"""Torch operations for preprocessing with NaN handling."""

from __future__ import annotations

import torch


def torch_nansum(
    x: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute the sum of a tensor, treating NaNs as zero.

    Args:
        x: The input tensor.
        axis: The dimension or dimensions to reduce.
        keepdim: Whether the output tensor has `dims` retained or not.
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
    return torch.sum(masked_input, dim=axis, keepdim=keepdim, dtype=dtype)


def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    include_inf: bool = False,
) -> torch.Tensor:
    """Compute the mean of a tensor over a given dimension, ignoring NaNs.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.
        include_inf: If True, treat infinity as NaN for the purpose of the calculation.

    Returns:
        The mean of the input tensor, ignoring NaNs.
    """
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num_valid = torch.where(
        nan_mask,
        torch.zeros_like(x),
        torch.ones_like(x),
    ).sum(dim=axis)
    value_sum = torch.where(nan_mask, torch.zeros_like(x), x).sum(dim=axis)

    return value_sum / num_valid.clamp(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute standard deviation of a tensor over a given dimension, ignoring NaNs.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.

    Returns:
        The standard deviation of the input tensor, ignoring NaNs.
    """
    nan_mask = torch.isnan(x)
    num_valid = torch.where(
        nan_mask,
        torch.zeros_like(x),
        torch.ones_like(x),
    ).sum(dim=axis)
    value_sum = torch.where(nan_mask, torch.zeros_like(x), x).sum(dim=axis)

    mean = value_sum / num_valid.clamp(min=1.0)

    # Broadcast mean back to original shape for subtraction
    mean_broadcast = mean.unsqueeze(axis).expand_as(x)

    # Compute sum of squared differences, ignoring NaNs
    sq_diff = torch.where(
        nan_mask,
        torch.zeros_like(x),
        torch.square(x - mean_broadcast),
    ).sum(dim=axis)

    # Use correction (N-1) to match sklearn's behavior
    variance = sq_diff / (num_valid - 1).clamp(min=1.0)

    return torch.sqrt(variance)
