#  Copyright (c) Prior Labs GmbH 2026.

"""Check for the enable_gqa parameter of PyTorch scaled_dot_product_attention."""

from __future__ import annotations

import functools

import torch


@functools.cache
def gqa_is_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Check if scaled_dot_product_attention supports enable_gqa on ``device``.

    This checks whether the device has kernels that run it efficiently for the given
    ``dtype``. CUDA needs compute capability >=8.0 and fp16 or bf16. CPU works with
    any dtype. Other devices return False.
    """
    if device.type == "cpu":
        return True
    if device.type == "cuda":
        return (
            dtype in {torch.float16, torch.bfloat16}
            # We need compute capability >= 8.0 for efficient GQA
            and torch.cuda.get_device_capability(device)[0] >= 8
        )
    return False
