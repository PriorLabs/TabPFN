#  Copyright (c) Prior Labs GmbH 2026.

"""Torch operations for preprocessing with NaN handling."""

from __future__ import annotations

import torch


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
    nan_mask = ~x.isfinite() if include_inf else x.isnan()

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


def columnwise_nanmean_nanstd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation over dimension 0, ignoring NaNs.

    Equal to `torch_nanmean` and `torch_nanstd` up to rounding. Each column's values
    depend only on that column, whichever other columns `x` holds: its sums run in
    an order set by the number of rows alone.

    Args:
        x: The input tensor.

    Returns:
        The mean and the standard deviation (with N-1 correction) of each column.
    """
    nan_mask = torch.isnan(x)
    num_valid = (~nan_mask).sum(dim=0).to(x.dtype)
    mean = _pairwise_sum(torch.where(nan_mask, 0, x)) / num_valid.clamp(min=1.0)
    sq_diff = _pairwise_sum((x - mean).square_().masked_fill_(nan_mask, 0))
    variance = sq_diff / (num_valid - 1).clamp(min=1.0)
    return mean, torch.sqrt(variance)


def _pairwise_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum over dimension 0 by a pairwise tree of elementwise additions.

    Each column's sum reads only that column, in an order set by `x.shape[0]`.
    """
    leftovers = []
    while x.shape[0] > 1:
        if x.shape[0] % 2:
            leftovers.append(x[-1])
            x = x[:-1]
        half = x.shape[0] // 2
        x = x[:half] + x[half:]
    total = x.sum(dim=0)
    for leftover in leftovers:
        total = total + leftover
    return total


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

    num_rows = x.shape[0]

    # Compute destination indices using cumsum
    # (It would be easier to do argsort but that's not ONNX compatible).
    # Selected features go to positions [0, num_selected), unselected go to
    # [num_selected, total_features).
    sel_cumsum_BF = sel.cumsum(dim=-1)
    not_sel_cumsum_BF = (~sel).cumsum(dim=-1)
    num_selected_B1 = sel.sum(dim=-1, keepdim=True)

    # For selected features: destination = cumsum - 1
    # For unselected features: destination = num_selected + not_sel_cumsum - 1
    dest_indices_BF = torch.where(
        sel,
        sel_cumsum_BF - 1,
        num_selected_B1 + not_sel_cumsum_BF - 1,
    )

    # Compute source indices (inverse permutation) using scatter.
    # For each destination position, this tells us which source position it comes from.
    source_positions_BF = torch.arange(total_features, device=x.device).expand(B, -1)
    src_indices_BF = torch.zeros(B, total_features, dtype=torch.long, device=x.device)
    src_indices_BF.scatter_(dim=-1, index=dest_indices_BF, src=source_positions_BF)

    # Use gather to reorder features
    src_indices_RBF = src_indices_BF.unsqueeze(0).expand(num_rows, -1, -1)
    new_x_RBF = torch.gather(x, dim=2, index=src_indices_RBF)

    # Create a mask to zero out the padding positions.
    position_indices_F = torch.arange(total_features, device=x.device)
    padding_mask_BF = position_indices_F >= num_selected_B1

    return new_x_RBF.masked_fill(padding_mask_BF.unsqueeze(0), 0)
