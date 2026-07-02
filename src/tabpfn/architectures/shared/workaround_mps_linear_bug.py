#  Copyright (c) Prior Labs GmbH 2026.
"""Workaround for a silent-correctness bug in PyTorch's MPS Linear layer.


On macOS 26 + Apple M1, nn.Linear with bias=True returns silently wrong results when the
input is more than 2D with a unary (size-1) dimension. The workaround for this,
mirroring what PyTorch did upstream, is to decompose the nn.Linear(bias=True) into
F.linear() without bias followed by adding on the bias.

See https://github.com/pytorch/pytorch/issues/188438.

This workaround can be removed once
https://github.com/pytorch/pytorch/commit/6394b2e9723f87bc6a85d12e52a725abd5240471
is included in the minimum PyTorch version (torch>=2.13 should have it), or M1 is no
longer supported.
"""

from __future__ import annotations

import functools

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


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


def maybe_replace_linears_on_mps(model: nn.Module) -> nn.Module:
    """Retype bias-enabled Linear layers in-place if MPS is buggy on this machine.

    No-op unless the MPS self-test finds the bug. Only vanilla bias-enabled
    nn.Linear modules are affected; bias=False layers are left untouched (they are
    not affected by the bug). Mutates and returns model.
    """
    if not _mps_linear_bias_is_buggy():
        return model
    for module in model.modules():
        if type(module) is nn.Linear and module.bias is not None:
            # Same memory layout and parameters, so re-typing in place is safe and
            # preserves weights, buffers, hooks, and state_dict compatibility.
            module.__class__ = MpsSafeLinear
    return model
