#  Copyright (c) Prior Labs GmbH 2026.
"""Scaled Dot Product Attention (SDPA) with additional backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabpfn.architectures.shared.attention_backends import (
    find_attention_backend,
    register_attention_backend,
)
from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.fa3_backend import FA3_BACKEND
from tabpfn.architectures.shared.mlx_backend import MLX_BACKEND
from tabpfn.architectures.shared.torch_mps_backend import TORCH_MPS_BACKEND

if TYPE_CHECKING:
    from tabpfn.architectures.kv_cache import QuantizedKVCacheEntry
    from tabpfn.architectures.shared.attention_backends import AttentionBackend

register_attention_backend(
    # Listed in consult order: the first one that prefers a call takes it.
    # Backends whose dependency is missing are left out entirely.
    *(
        backend
        for backend in (FA3_BACKEND, TORCH_MPS_BACKEND, MLX_BACKEND)
        if backend.is_available()
    )
)


def scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor | None,
    v_BSJD: torch.Tensor | None,
    _backends_override: list[SDPBackend] | None = None,
    *,
    quantized_kv: QuantizedKVCacheEntry | None = None,
    backend: AttentionBackend | None | Literal["auto"] = "auto",
) -> torch.Tensor:
    """Attention dispatch: run a registered backend, or the torch SDPA path.

    The backend is resolved from the shapes of this call (the default
    ``backend="auto"``); pass a backend to run it without consulting the
    registry, or ``None`` to force the standard SDPA path.

    ``quantized_kv`` supplies the keys/values as a quantized cache entry
    instead of ``k_BSJD``/``v_BSJD``.

    Dequantization happens exactly once, here: whoever takes the call
    receives dense ``k``/``v`` — unless it is a backend declaring
    ``consumes_quantized_kv = True``, which receives the entry as stored.
    """
    if isinstance(backend, str):  # "auto"
        backend = find_attention_backend(
            q_BSHD, k_BSJD, v_BSJD, quantized_kv=quantized_kv
        )

    if quantized_kv is not None and not (
        backend is not None and getattr(backend, "consumes_quantized_kv", False)
    ):
        entry = quantized_kv.dequantize(q_BSHD.dtype)
        k_BSJD, v_BSJD = entry.key, entry.value
        quantized_kv = None

    if backend is not None:
        return backend.run(q_BSHD, k_BSJD, v_BSJD, quantized_kv=quantized_kv)
    return _torch_sdpa(q_BSHD, k_BSJD, v_BSJD, _backends_override)


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


def _torch_sdpa(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor | None,
    v_BSJD: torch.Tensor | None,
    _backends_override: list[SDPBackend] | None = None,
) -> torch.Tensor:
    """The standard path: torch SDPA, hardened for tabpfn's shapes.

    A more robust version of torch.nn.functional.scaled_dot_product_attention:
    enables GQA where needed, and works around the
    CUDA maximum grid size for very large ``batch * heads`` (pytorch #142228)
    by chunking the batch.
    """
    msg = "SDPA expects (B, S, H, D); got tensor of shape"
    assert q_BSHD.dim() == 4, f"{msg} q:{tuple(q_BSHD.shape)}"
    assert k_BSJD is not None
    assert v_BSJD is not None
    assert k_BSJD.dim() == 4, f"{msg} k:{tuple(k_BSJD.shape)}"
    assert v_BSJD.dim() == 4, f"{msg} v:{tuple(v_BSJD.shape)}"

    q_BHSD = q_BSHD.permute(0, 2, 1, 3)
    k_BJSD = k_BSJD.permute(0, 2, 1, 3)
    v_BJSD = v_BSJD.permute(0, 2, 1, 3)
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

    backends = _backends_override if _backends_override is not None else _SDPA_BACKENDS

    num_parallel_calls = q_BHSD.shape[:2].numel()
    if torch.compiler.is_compiling():
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
