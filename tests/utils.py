from __future__ import annotations

import functools
import os
from collections.abc import Generator, Iterable

import pytest
import torch
from torch import nn


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


def patch_layernorm_no_affine(model: nn.Module) -> None:
    """Workaround for ONNX export issue with LayerNorm(affine=False) in torch<=2.1.3.

    This patch function was necessary to enable successful ONNX export
    of the TabPFN model when using PyTorch version 2.1.3. The issue arose
    because the ONNX exporter in that version (and potentially earlier ones)
    failed to correctly handle `nn.LayerNorm` layers initialized with
    `affine=False`, which means they lack the learnable 'weight' (gamma) and
    'bias' (beta) parameters.

    However, testing indicated that this issue is resolved in later PyTorch
    versions; specifically, the ONNX export runs without errors on
    PyTorch 2.6.0 even without this patch.

    This function circumvents the problem by iterating through the model's
    modules and, for any `nn.LayerNorm` layer where `layer.weight` is None
    (indicating `affine=False`), it manually adds non-learnable
    (`requires_grad=False`) parameters for 'weight' (initialized to ones) and
    'bias' (initialized to zeros). This addition satisfies the requirements
    of the older ONNX exporter without changing the model's functional
    behavior, as these added parameters represent an identity affine
    transformation.
    """
    for layer in model.modules():
        if isinstance(layer, nn.LayerNorm) and layer.weight is None:
            # Build tensors on the same device/dtype as the layer's buffer
            device = next(layer.parameters(), torch.tensor([], device="cpu")).device
            dtype = getattr(layer, "weight_dtype", torch.float32)

            gamma = torch.ones(layer.normalized_shape, dtype=dtype, device=device)
            beta = torch.zeros_like(gamma)

            layer.weight = nn.Parameter(gamma, requires_grad=False)
            layer.bias = nn.Parameter(beta, requires_grad=False)

            # Optional: mark that we changed it (useful for logging)
            layer._patched_for_onnx = True
