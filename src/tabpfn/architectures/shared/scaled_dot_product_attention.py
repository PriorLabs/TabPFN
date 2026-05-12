#  Copyright (c) Prior Labs GmbH 2026.
"""Scaled Dot Product Attention (SDPA) with additional backends."""

from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.torch_version import TorchVersion

from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.fa3_backend import fa3_attn_func, is_fa3_preferred
from tabpfn.architectures.shared.mlx_backend import (
    flash_attention_mlx,
    is_mlx_preferred,
)

# MATH is the reference implementation. Keeping it as a final fallback
# avoids "No available kernel" errors on GPUs where none of the fast
# backends are eligible — e.g. FlashAttention requires sm80+, so on a
# Turing card (sm75, T4) all three faster backends bail and SDPA crashes
# without this entry.
_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.MATH,
]


def is_torch_mps_preferred(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """True iff PyTorch's MPS SDPA is preferred for this attention call."""
    _DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    return (
        # Torch added support for flash attention after 2.13.0.dev20260510. Note that
        torch.__version__ >= TorchVersion("2.13.0.dev20260510")
        and (q.is_mps and k.is_mps and v.is_mps)
        and (q.dtype in _DTYPES and k.dtype in _DTYPES and v.dtype in _DTYPES)
    )


def torch_mps_sdpa(
    q_BSHD: torch.Tensor,
    k_BJSD: torch.Tensor,
    v_BJSD: torch.Tensor,
    *,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """SDPA with head dim padded to the nearest MPS-supported dim."""
    supported_dims = [32, 64, 96, 128, 256]
    head_dim = q_BSHD.shape[-1]
    target_dim = min((d for d in supported_dims if d >= head_dim), default=head_dim)
    pad_width = target_dim - head_dim
    if pad_width > 0:
        q_BSHD = torch.nn.functional.pad(q_BSHD, (0, pad_width))
        k_BJSD = torch.nn.functional.pad(k_BJSD, (0, pad_width))
        v_BJSD = torch.nn.functional.pad(v_BJSD, (0, pad_width))
    out = torch.nn.functional.scaled_dot_product_attention(
        q_BSHD,
        k_BJSD,
        v_BJSD,
        attn_mask=None,
        enable_gqa=enable_gqa,
        scale=head_dim**-0.5,
    )
    if pad_width > 0:
        out = out[..., :head_dim]
    return out


def scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor,
    v_BSJD: torch.Tensor,
    _backends_override: list[SDPBackend] | None = None,
) -> torch.Tensor:
    """Scaled dot-product, optimized for various scenarios.

    This is a more robust and potentially faster version of
    torch.nn.functional.scaled_dot_product_attention.

    Specifically, it
    - works around very large batch size errors
    - supports and auto selects the following additional backends if present:
        - FA3 (Hopper GPUs, fp16/bf16)
        - MLX flash attention (MPS)
    """
    msg = "SDPA expects (B, S, H, D); got tensor of shape"
    assert q_BSHD.dim() == 4, f"{msg} q:{tuple(q_BSHD.shape)}"
    assert k_BSJD.dim() == 4, f"{msg} k:{tuple(k_BSJD.shape)}"
    assert v_BSJD.dim() == 4, f"{msg} v:{tuple(v_BSJD.shape)}"

    if is_fa3_preferred(q_BSHD, k_BSJD):
        return fa3_attn_func(
            q_BSHD.contiguous(), k_BSJD.contiguous(), v_BSJD.contiguous()
        )

    q_BHSD = q_BSHD.permute(0, 2, 1, 3)
    k_BJSD = k_BSJD.permute(0, 2, 1, 3)
    v_BJSD = v_BSJD.permute(0, 2, 1, 3)
    num_q_heads = q_BHSD.shape[-3]
    num_kv_heads = k_BJSD.shape[-3]

    if is_torch_mps_preferred(q_BHSD, k_BJSD, v_BJSD):
        return torch_mps_sdpa(
            q_BHSD, k_BJSD, v_BJSD, enable_gqa=(num_q_heads == num_kv_heads)
        ).permute(0, 2, 1, 3)
    if is_mlx_preferred(q_BHSD, k_BJSD, v_BJSD):
        # Note: MLX supports GQA and doesn't seem to have a max grid batch size issue.
        return flash_attention_mlx(q_BHSD, k_BJSD, v_BJSD).permute(0, 2, 1, 3)

    dtype_supports_gqa = q_BHSD.dtype in {torch.float16, torch.bfloat16}
    if num_q_heads == num_kv_heads:
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {}
    elif gqa_is_supported() and dtype_supports_gqa:
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {"enable_gqa": True}
    else:
        repeat = num_q_heads // num_kv_heads
        keys = k_BJSD.repeat_interleave(repeat, dim=-3)
        values = v_BJSD.repeat_interleave(repeat, dim=-3)
        enable_gqa = {}

    backends = _backends_override if _backends_override is not None else _SDPA_BACKENDS

    num_parallel_calls = q_BHSD.shape[:2].numel()
    torch._check(num_parallel_calls >= 1)  # These checks help torch.compile.
    torch._check(q_BHSD.shape[0] >= 1)
    CUDA_MAX_GRID = 65536
    num_iterations = (num_parallel_calls + CUDA_MAX_GRID - 1) // CUDA_MAX_GRID
    sub_batch = (q_BHSD.shape[0] + num_iterations - 1) // num_iterations

    with sdpa_kernel(backends=backends):
        outputs = []
        for i in range(num_iterations):
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_BHSD[i * sub_batch : (i + 1) * sub_batch].contiguous(),
                    keys[i * sub_batch : (i + 1) * sub_batch].contiguous(),
                    values[i * sub_batch : (i + 1) * sub_batch].contiguous(),
                    attn_mask=None,
                    **enable_gqa,
                )
            )
    output_BHSD = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output_BHSD.permute(0, 2, 1, 3)
