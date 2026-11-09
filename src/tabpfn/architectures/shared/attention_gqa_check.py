#  Copyright (c) Prior Labs GmbH 2026.

"""Check for the enable_gqa parameter of PyTorch scaled_dot_product_attention."""

from __future__ import annotations

import functools

import torch
from torch.torch_version import TorchVersion


@functools.cache
def gqa_is_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Check if scaled_dot_product_attention supports enable_gqa on ``device``.

    This checks whether the installed PyTorch has the enable_gqa kwarg and whether
    the device has kernels that run it efficiently for the given ``dtype``.
    CUDA needs PyTorch >=2.5, compute capability >=8.0 and fp16 or bf16.
    CPU needs PyTorch >=2.9 and works with any dtype. Other devices return False.
    """
    if device.type == "cpu":
        return torch.__version__ >= TorchVersion("2.9")
    if device.type == "cuda":
        return (
            torch.__version__ >= TorchVersion("2.5")
            and dtype in {torch.float16, torch.bfloat16}
            # We need compute capability >= 8.0 for efficient GQA
            and torch.cuda.get_device_capability(device)[0] >= 8
        )
    return False
