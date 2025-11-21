from __future__ import annotations

import functools
import os
from collections.abc import Generator, Iterable

import pytest
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


@functools.cache
def is_cpu_float16_supported() -> bool:
    """Check if this version of PyTorch supports CPU float16 operations."""
    try:
        # Attempt a minimal operation that fails on older PyTorch versions on CPU
        torch.randn(2, 2, dtype=torch.float16, device="cpu") @ torch.randn(
            2, 2, dtype=torch.float16, device="cpu"
        )
        return True
    except RuntimeError as e:
        if "addmm_impl_cpu_" in str(e) or "not implemented for 'Half'" in str(e):
            return False
        raise e


def mark_mps_configs_as_slow(configs: Iterable[tuple]) -> Generator[tuple]:
    """Add a pytest "slow" mark to any configurations that run on MPS.

    This is useful to disable MPS tests in PRs, which we have found can be very slow.
    It assumes that the device is given by the first element of the config tuple.

    Use it like follows:
    ```
    @pytest.mark.parametrize(
        ("device", "config_option"),
        mark_mps_configs_as_slow(
            itertools.product(get_pytest_devices(), ["value_a", "value_b"])
        )
    )
    def test___my_function(device: str, config_option: str) -> None: ...
    ```
    """
    for config in configs:
        if config[0] == "mps":
            yield pytest.param(*config, marks=pytest.mark.slow)
        else:
            yield config
