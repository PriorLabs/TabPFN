#  Copyright (c) Prior Labs GmbH 2026.
"""MLX backend for scaled dot product attention on MPS devices.

Pytorch SDPA has only very limited support for flash attention on MPS, which is
incompatible with our model. To reduce memory usage, we route to MLX during inference.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tabpfn.architectures.shared.attention_backends import AttentionBackend

if TYPE_CHECKING:
    from tabpfn.architectures.shared.attention_backends import AttentionSpec

try:
    import mlx.core as mx
except ImportError:
    mx = None


# Float32 on M5 chips runs with lower precision by default. Disabling TF32 circumvents
# this. See https://github.com/ml-explore/mlx/issues/3534.
os.environ["MLX_ENABLE_TF32"] = "0"

# Below this KV sequence length MLX is slower than torch SDPA, but the memory
# savings of its flash path are often required to not run out of memory.
_MLX_MIN_KV_SEQLEN = 128


def is_eligible_for_mlx(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    *,
    is_grad_enabled: bool,
) -> bool:
    """True iff MLX can serve such an attention call (capability gate, not perf).

    Forward-only: the torch->mlx->torch round-trip detaches from autograd,
    so any call that may need a backward pass is ineligible.
    """
    _DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    return (
        device.type == "mps"
        and not is_grad_enabled
        and dtype in _DTYPES
        and head_dim <= 128
    )


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


class MLXBackend(AttentionBackend):
    """MLX flash attention as an :class:`~.attention_backends.AttentionBackend`.

    Wraps the eligibility/crossover logic above unchanged; registered at
    import (below) so MLX keeps its historical auto-dispatch on MPS.
    """

    name = "mlx"

    @staticmethod
    def is_available() -> bool:
        """Whether the mlx package is installed (checked at registration)."""
        return mx is not None

    def is_preferred(self, spec: AttentionSpec) -> bool:
        """Eligibility plus the KV-length crossover (also a memory guard).

        Unknown (None) KV lengths never argue for MLX.
        """
        return (
            is_eligible_for_mlx(
                spec.device,
                spec.dtype,
                spec.head_dim,
                is_grad_enabled=spec.is_grad_enabled,
            )
            and (spec.seq_len_kv or 0) >= _MLX_MIN_KV_SEQLEN
        )

    def run(
        self,
        q_BSHD: torch.Tensor,
        k_BSJD: torch.Tensor | None,
        v_BSJD: torch.Tensor | None,
        **_informational: Any,  # forward-compat context; safe to ignore
    ) -> torch.Tensor:
        """Run MLX attention (k/v arrive dense; the SDPA wrapper dequantizes)."""
        assert k_BSJD is not None
        assert v_BSJD is not None
        out_BHSD = flash_attention_mlx(
            q_BSHD.permute(0, 2, 1, 3),
            k_BSJD.permute(0, 2, 1, 3),
            v_BSJD.permute(0, 2, 1, 3),
        )
        return out_BHSD.permute(0, 2, 1, 3)


# Registered by the shared SDPA module, which owns the consult order.
MLX_BACKEND = MLXBackend()
