#  Copyright (c) Prior Labs GmbH 2026.
"""Torch-native MPS flash attention backend.

Recent torch builds (2.13 nightlies onward) ship a flash-attention SDPA
kernel for MPS, but only for a fixed set of head dims — ``torch_mps_sdpa``
pads the head dimension to the nearest supported size around the native op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.torch_version import TorchVersion

from tabpfn.architectures.shared.attention_backends import AttentionBackend

if TYPE_CHECKING:
    from tabpfn.architectures.shared.attention_backends import AttentionSpec


# Torch added flash attention for MPS after 2.13.0.dev20260510.
_MIN_TORCH_FOR_MPS_FLASH = "2.13.0.dev20260510"


def is_torch_mps_preferred(device: torch.device, dtype: torch.dtype) -> bool:
    """True iff PyTorch's MPS SDPA is preferred for such an attention call.

    Assumes the torch build has the kernel — see
    :meth:`TorchMPSBackend.is_available`.
    """
    _DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    return device.type == "mps" and dtype in _DTYPES


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


class TorchMPSBackend(AttentionBackend):
    """Torch-native MPS flash attention as an ``AttentionBackend``.

    Wraps ``torch_mps_sdpa`` (head-dim padding around the native op).
    Registered after the MLX backend (see the import order at the shared
    SDPA module), so it is consulted first: on a torch build with MPS flash
    attention the native path wins, preserving the historical probe order.
    """

    name = "torch-mps"

    @staticmethod
    def is_available() -> bool:
        """Whether this torch build has the MPS flash kernel."""
        return torch.__version__ >= TorchVersion(_MIN_TORCH_FOR_MPS_FLASH)

    def is_preferred(self, spec: AttentionSpec) -> bool:
        """Torch-version + device/dtype gate; no shape constraints."""
        return is_torch_mps_preferred(spec.device, spec.dtype)

    def run(
        self,
        q_BSHD: torch.Tensor,
        k_BSJD: torch.Tensor | None,
        v_BSJD: torch.Tensor | None,
        **_informational: Any,  # forward-compat context; safe to ignore
    ) -> torch.Tensor:
        """Run MPS SDPA (k/v arrive dense; the SDPA wrapper dequantizes)."""
        assert k_BSJD is not None
        assert v_BSJD is not None
        out_BHSD = torch_mps_sdpa(
            q_BSHD.permute(0, 2, 1, 3),
            k_BSJD.permute(0, 2, 1, 3),
            v_BSJD.permute(0, 2, 1, 3),
            enable_gqa=q_BSHD.shape[2] != k_BSJD.shape[2],
        )
        return out_BHSD.permute(0, 2, 1, 3)


# Registered by the shared SDPA module, which owns the consult order.
TORCH_MPS_BACKEND = TorchMPSBackend()
