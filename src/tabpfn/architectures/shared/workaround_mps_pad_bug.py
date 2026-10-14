#  Copyright (c) Prior Labs GmbH 2026.
"""Workaround for a silent-correctness bug in PyTorch's MPS constant padding.

On MPS with torch <= 2.14, ``F.pad`` with a non-zero fill value (such as NaN)
corrupts its input once the tensor has 2**16 = 65,536 rows or more, even when the
pad width is zero. The output then holds NaN or other rows' values, and no error is
raised. ``F.pad`` is correct on other devices.

See https://github.com/pytorch/pytorch/issues/194922. The bug is fixed in torch
nightly builds. This workaround can be removed once the minimum supported torch
version includes the fix (expected to be torch 2.15).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812


def pad_rows_with_nan(x: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Append NaN rows to ``x`` along its first dimension.

    Args:
        x: Tensor whose first dimension indexes rows.
        num_rows: Number of rows in the output. Must be at least ``x.shape[0]``.

    Returns:
        ``x`` followed by ``num_rows - x.shape[0]`` rows filled with NaN.
    """
    num_pad_rows = num_rows - x.shape[0]
    if x.device.type == "mps":
        # Avoid F.pad on MPS, which corrupts inputs with >= 2**16 rows (see the
        # module docstring). Building the NaN rows separately gives the same result.
        nan_rows = x.new_full((num_pad_rows, *x.shape[1:]), float("nan"))
        return torch.cat([x, nan_rows])
    # F.pad lists pad widths from the last dimension backwards, so only the final
    # pair (the first dimension) is non-zero.
    pad = (0, 0) * (x.ndim - 1) + (0, num_pad_rows)
    return F.pad(x, pad, value=float("nan"))
