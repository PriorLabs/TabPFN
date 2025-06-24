from __future__ import annotations

import torch


def check_cpu_float16_support() -> bool:
    """Checks if CPU float16 operations are supported by attempting a minimal operation.
    Returns True if supported, False otherwise.
    """
    try:
        # Attempt a minimal operation that fails on older PyTorch versions on CPU
        torch.randn(2, 2, dtype=torch.float16, device="cpu") @ torch.randn(
            2, 2, dtype=torch.float16, device="cpu"
        )
        return True
    except RuntimeError as e:
        if "addmm_impl_cpu_" in str(e) or "not implemented for 'Half'" in str(e):
            return False
        raise  # Re-raise unexpected exceptions
