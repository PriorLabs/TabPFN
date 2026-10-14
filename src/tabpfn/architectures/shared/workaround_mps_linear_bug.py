#  Copyright (c) Prior Labs GmbH 2026.
"""Workarounds for silent-correctness bugs in PyTorch's MPS backend.

nn.Linear with bias:

On macOS 26 + Apple M1, nn.Linear with bias=True returns silently wrong results when the
input is more than 2D with a unary (size-1) dimension. The workaround for this,
mirroring what PyTorch did upstream, is to decompose the nn.Linear(bias=True) into
F.linear() without bias followed by adding on the bias.

See https://github.com/pytorch/pytorch/issues/188438.

This workaround can be removed once
https://github.com/pytorch/pytorch/commit/6394b2e9723f87bc6a85d12e52a725abd5240471
is included in the minimum PyTorch version (torch>=2.13 should have it), or M1 is no
longer supported.

F.pad with a non-zero fill value:

On MPS with torch <= 2.14, F.pad with a non-zero fill value (such as NaN) corrupts its
input once the tensor has 2**16 = 65,536 rows or more, even when the pad width is
zero. The output then holds NaN or other rows' values, and no error is raised. F.pad is
correct on other devices. The workaround builds the fill rows separately and
concatenates them on MPS.

See https://github.com/pytorch/pytorch/issues/194922.

This workaround can be removed once the minimum PyTorch version includes the fix
(expected to be torch 2.15).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

logger = logging.getLogger(__name__)


@functools.cache
def _mps_linear_bias_is_buggy() -> bool:
    """Return True if this machine suffers from the nn.Linear MPS bug."""
    if not torch.backends.mps.is_available():
        return False
    with torch.no_grad():
        # Use a local generator so we don't disturb the global RNG state.
        gen = torch.Generator().manual_seed(0)
        weight = torch.randn(192, 2, generator=gen)
        bias = torch.randn(192, generator=gen)
        x = torch.randn(80, 1, 2, generator=gen)
        cpu = F.linear(x, weight, bias)
        mps = F.linear(x.to("mps"), weight.to("mps"), bias.to("mps")).cpu()
        rel = ((cpu - mps).abs().max() / cpu.abs().max()).item()
    return rel > 1e-3


class MpsSafeLinear(nn.Linear):
    """A Linear layer that decomposes the bias add on MPS, else identical.

    Only changes forward(), so it can load the nn.Linear() state dict.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer, decomposing the bias add on MPS."""
        if input.is_mps and self.bias is not None:
            return F.linear(input, self.weight, None) + self.bias
        return super().forward(input)


def maybe_replace_linears_on_mps(
    model: nn.Module, devices: Sequence[torch.device]
) -> nn.Module:
    """Retype bias-enabled Linear layers in-place if MPS is buggy on this machine.

    If devices does not contain an MPS device, does nothing. Otherwise, checks if this
    device suffers from the MPS bug (see module docstring), and applies the workaround.
    """
    if not any(device.type == "mps" for device in devices):
        return model
    if not _mps_linear_bias_is_buggy():
        return model
    logger.debug("Detected MPS nn.Linear bug. Applying workaround")
    for module in model.modules():
        if type(module) is nn.Linear and module.bias is not None:
            # Same memory layout and parameters, so re-typing in place is safe and
            # preserves weights, buffers, hooks, and state_dict compatibility.
            module.__class__ = MpsSafeLinear
    return model


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
