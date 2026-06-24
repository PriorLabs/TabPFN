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


def _mode_col(x: torch.Tensor) -> float:
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return torch.nan
    # torch.unique returns values sorted ascending; break ties toward the
    # smallest value by taking the first index that reaches the max count
    # (matching np.argmax in the numpy ``mode`` in steps/utils.py).
    values, counts = torch.unique(finite, return_counts=True)
    return values[torch.nonzero(counts == counts.max())[0]].item()


def mode(x: torch.Tensor) -> torch.Tensor | float:
    """Compute the mode of each column along the row dimension (axis 0).

    For every column, returns the most frequently occurring value, ignoring
    non-finite entries (NaN, +inf, -inf) so the result is unaffected by missing
    data. Ties are broken deterministically by choosing the smallest value.
    Columns with no finite values yield NaN.

    This is the categorical counterpart to ``torch_nanmean``: it
    produces per-feature imputation values that respect the discrete nature of
    categorical data instead of averaging category codes.

    Args:
        x: Tensor of shape ``(n_samples, n_features)``.

    Returns:
        Tensor of shape ``(n_features,)`` with the per-column mode.
    """
    if x.ndim == 1:
        return _mode_col(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D Tensor, got {x.ndim}D.")

    return x.new_tensor([_mode_col(x[:, col]) for col in range(x.shape[1])])


def categorical_modes(
    x_RBC: torch.Tensor,
    num_train_rows: int,
    categorical_inds: list[list[int]],
) -> torch.Tensor:
    """Per-(batch, column) train mode for categorical columns, NaN elsewhere.

    ponytail: non-differentiable (uses ``torch.unique``) and graph-breaks under
    ``torch.compile`` — only invoked on the opt-in mode-imputation path.

    Args:
        x_RBC: Tensor of shape ``(R, B, C)``; first ``num_train_rows`` are train.
        num_train_rows: Number of leading train rows used to compute the mode.
        categorical_inds: Per-batch lists of categorical column indices.

    Returns:
        A ``(B, C)`` tensor with the per-column mode for categorical columns and
        NaN for non-categorical columns (and categorical columns with no finite
        training value).
    """
    modes_BC = x_RBC.new_full(x_RBC.shape[1:], float("nan"))
    x_train = x_RBC[:num_train_rows]
    for b, cats in enumerate(categorical_inds):
        if not cats:
            continue
        m = mode(x_train[:, b][:, cats])
        valid = torch.isfinite(m)
        if valid.any():
            cols = torch.tensor(cats, device=x_RBC.device)[valid]
            modes_BC[b, cols] = m[valid]
    return modes_BC


def categorical_mode_fill(
    x_RBC: torch.Tensor,
    num_train_rows: int,
    categorical_inds: list[list[int]],
    mean_fill_BC: torch.Tensor,
) -> torch.Tensor:
    """Return ``mean_fill_BC`` with categorical columns overwritten by their mode.

    Categorical columns with no finite training value keep the mean.

    Args:
        x_RBC: Tensor of shape ``(R, B, C)``; first ``num_train_rows`` are train.
        num_train_rows: Number of leading train rows used to compute the mode.
        categorical_inds: Per-batch lists of categorical column indices.
        mean_fill_BC: Tensor of shape ``(B, C)`` with mean values.

    Returns:
        A ``(B, C)`` fill tensor (mode for categoricals, ``mean_fill_BC`` elsewhere).
    """
    modes_BC = categorical_modes(x_RBC, num_train_rows, categorical_inds)
    return torch.where(torch.isnan(modes_BC), mean_fill_BC, modes_BC)


def impute_categorical_mode(
    x_RBC: torch.Tensor,
    num_train_rows: int,
    categorical_inds: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill NaN/inf in categorical columns with the per-column train mode.

    Non-categorical columns are left untouched (their NaN/inf are preserved for
    the caller's mean handling), as are categorical columns with no finite value.

    Returns:
        ``(x_imputed, modes_BC)`` where ``modes_BC`` is the ``(B, C)`` mode tensor
        from :func:`categorical_modes` (NaN for non-categorical columns).
    """
    modes_BC = categorical_modes(x_RBC, num_train_rows, categorical_inds)
    has_mode = ~torch.isnan(modes_BC)
    use_mode = has_mode.unsqueeze(0) & ~torch.isfinite(x_RBC)
    fill = torch.nan_to_num(modes_BC).unsqueeze(0).expand_as(x_RBC)
    return torch.where(use_mode, fill, x_RBC), modes_BC


def categorical_mask_from_inds(
    categorical_inds: list[list[int]],
    batch_size: int,
    num_columns: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    """Build a ``(1, B, C)`` one-hot mask marking categorical columns.

    Architectures v2/v2.5/v2.6 impute on the constant-pruned, padded, grouped tensor,
    where original column indices no longer align. Rather than re-deriving the layout,
    callers route this mask through the *same* ``_remove_constant_features`` and
    ``_pad_and_reshape_feature_groups`` the data goes through, then read the grouped
    categorical positions back with :func:`grouped_inds_from_mask`.

    Args:
        categorical_inds: Per-batch lists of original categorical column indices.
        batch_size: Original batch size ``B``.
        num_columns: Number of columns ``C`` before grouping.
        ref: Tensor whose device/dtype the mask should match.

    Returns:
        A ``(1, B, C)`` mask (``1.0`` at categorical columns, ``0.0`` elsewhere).
    """
    mask = ref.new_zeros(1, batch_size, num_columns)
    for b, cats in enumerate(categorical_inds):
        if cats:
            mask[0, b, cats] = 1.0
    return mask


def grouped_inds_from_mask(grouped_mask: torch.Tensor) -> list[list[int]]:
    """Read per-(folded-batch) categorical feature indices from a grouped mask.

    The input is a categorical mask (see :func:`categorical_mask_from_inds`) after it
    has been routed through the data's constant-removal and grouping transforms, i.e.
    of shape ``(1, B*G, F)``.

    ponytail: uses ``.tolist()`` (host sync, graph-breaks under ``torch.compile``);
    only invoked on the opt-in mode-imputation path.
    """
    return [row.nonzero(as_tuple=True)[0].tolist() for row in grouped_mask[0] > 0.5]
