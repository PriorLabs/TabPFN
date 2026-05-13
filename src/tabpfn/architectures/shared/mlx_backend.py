#  Copyright (c) Prior Labs GmbH 2026.
"""MLX backend for scaled dot product attention on MPS devices.

Pytorch SDPA has only very limited support for flash attention on MPS, which is
incompatible with our model. To reduce memory usage, we route to MLX during inference.
"""

from __future__ import annotations

import os

import numpy as np
import torch

try:
    import mlx.core as mx
except ImportError:
    mx = None


# Float32 on M5 chips runs with lower precision by default. Disabling TF32 circumvents
# this. See https://github.com/ml-explore/mlx/issues/3534.
os.environ["MLX_ENABLE_TF32"] = "0"


def is_eligible_for_mlx(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """True iff MLX can serve this attention call (capability gate, not perf)."""
    head_dim = q.shape[-1]
    _DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    return (
        mx is not None
        and (q.is_mps and k.is_mps and v.is_mps)
        and not (q.requires_grad or k.requires_grad or v.requires_grad)
        and (q.dtype in _DTYPES and k.dtype in _DTYPES and v.dtype in _DTYPES)
        and head_dim <= 128
    )


def is_mlx_preferred(
    q_BHSD: torch.Tensor, k_BJSD: torch.Tensor, v_BJSD: torch.Tensor
) -> bool:
    """True iff MLX is preferred for this attention call."""
    # Note: Inference speed of MLX is worse for seq_len < 1500 but the memory savings
    # are often required to not run out of memory.
    return is_eligible_for_mlx(q_BHSD, k_BJSD, v_BJSD) and k_BJSD.shape[2] >= 128


def _torch_to_mlx(torch_tensor: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to an MLX array, handling bfloat16 via int16."""
    if torch_tensor.dtype == torch.bfloat16:
        return mx.array(torch_tensor.view(torch.int16).cpu().numpy()).view(mx.bfloat16)
    return mx.array(torch_tensor.cpu().numpy())


def _mlx_to_torch(
    mlx_array: mx.array, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if dtype == torch.bfloat16:
        return (
            torch.from_numpy(np.array(mlx_array.view(mx.int16)))
            .view(torch.bfloat16)
            .to(device)
        )
    return torch.from_numpy(np.array(mlx_array)).to(device)


def flash_attention_mlx(
    q_BHSD: torch.Tensor, k_BHJD: torch.Tensor, v_BHJD: torch.Tensor
) -> torch.Tensor:
    """Scaled dot product attention using MLX on MPS devices, with padding for head_dim.

    Pytorch SDPA has only very limited support for flash attention on MPS, which is
    incompatible with shapes encountered in our model. To reduce memory usage, we route
    to MLX during inference.
    Zero-padding does not change the dot product, and we scale by the original head_dim.
    """
    D = q_BHSD.shape[-1]
    if D > 128:
        raise ValueError(f"MLX only runs flash SDPA for head_dim <= 128; got {D}")
    target_dim = 64 if D <= 64 else 128  # Flash path runs only at 64 or 128.
    pad_width = target_dim - D

    if pad_width > 0:
        q_BHSD = torch.nn.functional.pad(q_BHSD, (0, pad_width))
        k_BHJD = torch.nn.functional.pad(k_BHJD, (0, pad_width))
        v_BHJD = torch.nn.functional.pad(v_BHJD, (0, pad_width))

    out = mx.fast.scaled_dot_product_attention(
        _torch_to_mlx(q_BHSD),
        _torch_to_mlx(k_BHJD),
        _torch_to_mlx(v_BHJD),
        scale=D**-0.5,  # scale uses original D, not padded dim
    )
    if pad_width > 0:
        out = out[..., :D]
    mx.eval(out)
    return _mlx_to_torch(out, q_BHSD.device, q_BHSD.dtype)
