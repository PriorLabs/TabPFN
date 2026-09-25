#  Copyright (c) Prior Labs GmbH 2026.

"""Check for the enable_gqa parameter of PyTorch scaled_dot_product_attention."""

from __future__ import annotations

import functools

import torch
from torch.torch_version import TorchVersion


@functools.cache
def gqa_is_supported() -> bool:
    """Check if PyTorch's scaled_dot_product_attention supports enable_gqa parameter.

    This checks whether torch.nn.functional.scaled_dot_product_attention has a
    kwarg enable_gqa and if we have sufficient NVIDIA compute capability.
    PyTorch 2.5+ includes enable_gqa support.
    """
    if not torch.cuda.is_available():
        return False

    has_enable_gqa = torch.__version__ >= TorchVersion("2.5")
    if not has_enable_gqa:
        return False

    # Check compute capability only if CUDA is available
    # We need compute capability >= 8.0 for efficient GQA
    device = torch.cuda.current_device()
    nvidia_compute_capability = torch.cuda.get_device_capability(device)
    return nvidia_compute_capability[0] >= 8
