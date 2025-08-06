from __future__ import annotations

import os

import torch


def get_pytest_devices() -> list[str]:
    exclude_devices = {
        d.strip()
        for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",")
        if d.strip()
    }

    devices = []
    if "cpu" not in exclude_devices:
        devices.append("cpu")
    if torch.cuda.is_available() and "cuda" not in exclude_devices:
        devices.append("cuda")
    if torch.backends.mps.is_available() and "mps" not in exclude_devices:
        devices.append("mps")

    if len(devices) == 0:
        raise RuntimeError("No devices available for testing.")

    return devices


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
