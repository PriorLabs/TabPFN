#  Copyright (c) Prior Labs GmbH 2026.

"""Check for the enable_gqa parameter of PyTorch scaled_dot_product_attention."""

from __future__ import annotations

import functools

import torch
from torch.torch_version import TorchVersion


@functools.cache
def gqa_is_supported(device: torch.device) -> bool:
    """Check if scaled_dot_product_attention supports enable_gqa on ``device``.

    This checks whether the installed PyTorch has the enable_gqa kwarg and whether
    the device has kernels that run it efficiently. CUDA needs PyTorch 2.5+ and
    compute capability 8.0+. CPU needs PyTorch 2.9+. Other devices return False.
    """
    if device.type == "cpu":
        return torch.__version__ >= TorchVersion("2.9")
    if device.type == "cuda":
        return (
            # We need compute capability >= 8.0 for efficient GQA
            torch.__version__ >= TorchVersion("2.5")
            and torch.cuda.get_device_capability(device)[0] >= 8
        )
    return False
