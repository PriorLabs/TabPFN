"""PyTorch compatibility helpers.

This module contains small shims to keep TabPFN compatible across multiple
PyTorch versions (notably 2.1+ and 2.4+), without cluttering higher-level
entry points (e.g. finetuning wrappers).
"""

from __future__ import annotations

from functools import partial

import torch

# Handle torch.amp deprecation (torch.cuda.amp deprecated in PyTorch 2.4+).
try:
    from torch.amp import (
        GradScaler as _amp_GradScaler,
        autocast as _amp_autocast,
    )

    autocast = partial(_amp_autocast, device_type="cuda")
    GradScaler = partial(_amp_GradScaler, "cuda")
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# Handle torch.nn.attention API introduced in PyTorch 2.4+.
# For older versions (2.1-2.3), use torch.backends.cuda.sdp_kernel.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # This excludes running the cuDNN backend. Otherwise we started to
    # get stride warnings/errors running the cuDNN backend, after
    # changing the loss calculation to be "per_estimator".
    # Since the cuDNN backend was found to be slower anyway, this
    # fix is good enough for now!
    _SDPA_BACKENDS = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    sdpa_kernel_context = partial(sdpa_kernel, _SDPA_BACKENDS)
except ImportError:
    sdpa_kernel_context = partial(  # type: ignore[misc]
        torch.backends.cuda.sdp_kernel,
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True,
    )


__all__ = [
    "GradScaler",
    "autocast",
    "sdpa_kernel_context",
]
