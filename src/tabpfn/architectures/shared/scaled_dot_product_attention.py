#  Copyright (c) Prior Labs GmbH 2026.
"""Scaled Dot Product Attention (SDPA) with additional backends."""

from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.fa3_backend import fa3_attn_func, is_fa3_preferred
from tabpfn.architectures.shared.mlx_backend import (
    flash_attention_mlx,
    is_mlx_preferred,
)

_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]
_SDPA_BACKENDS_CPU = [*_SDPA_BACKENDS, SDPBackend.MATH]


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

    if is_mlx_preferred(q_BHSD, k_BJSD, v_BJSD):
        # Note: MLX supports GQA and doesn't seem to have a max grid batch size issue.
        return flash_attention_mlx(q_BHSD, k_BJSD, v_BJSD).permute(0, 2, 1, 3)

    num_q_heads = q_BHSD.shape[-3]
    num_kv_heads = k_BJSD.shape[-3]
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

    if _backends_override is not None:
        backends = _backends_override
    else:
        backends = (
            _SDPA_BACKENDS_CPU if not torch.cuda.is_available() else _SDPA_BACKENDS
        )

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
