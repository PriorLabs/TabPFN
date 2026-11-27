#  Copyright (c) Prior Labs GmbH 2026.
"""Heuristics for when to save peak memory during inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tabpfn.constants import MemorySavingMode


def should_save_peak_mem(
    memory_saving_mode: MemorySavingMode,
    X_train_shape: tuple[int, int],
    X_test_shape: tuple[int, int],
    devices: Sequence[torch.device],
    dtype_byte_size: int,
) -> bool:
    """Uses heuristics to determine whether to save peak memory.

    The aim is not only to avoid running out of memory for larger datasets, but also to
    make inference faster. Enabling/disabling memory saving optimally can have a big
    impact on fit+predict speed, sometimes greater than 2x.

    See details in https://github.com/PriorLabs/TabPFN/pull/605.
    """
    if isinstance(memory_saving_mode, bool):
        return memory_saving_mode

    if all(device.type == "mps" for device in devices):
        # - Memory saving usually seems to be faster even for small datasets on MPS
        # - Running out of memory is quite bad because it locks up the whole MacOS UI
        return True

    if all(device.type == "cpu" for device in devices):
        return _should_save_peak_mem_cpu(X_train_shape, X_test_shape)

    if all(device.type == "cuda" for device in devices):
        return _should_save_peak_mem_cuda(
            X_train_shape, X_test_shape, devices, dtype_byte_size
        )

    # For an unrecognised device, enable memory saving to be safe.
    return True


def _should_save_peak_mem_cpu(
    X_train_shape: tuple[int, int], X_test_shape: tuple[int, int]
) -> bool:
    # TODO: Refine the CPU heuristic.
    return _get_num_cells(X_train_shape, X_test_shape) > 200_000


def _should_save_peak_mem_cuda(
    X_train_shape: tuple[int, int],
    X_test_shape: tuple[int, int],
    devices: Sequence[torch.device],
    dtype_byte_size: int,
) -> bool:
    free_memory_bytes = min(_get_free_cuda_memory_bytes(device) for device in devices)

    # Our baseline is 2 byte floats on an 80GB H100.
    # We observe that the threshold shifts roughly linearly with GPU memory size, so we
    # make that adjustment.
    baseline_cell_threshold = 6_000_000
    baseline_dtype_byte_size = 2
    baseline_gpu_memory_bytes = 80e9
    cell_threshold = baseline_cell_threshold * (
        baseline_dtype_byte_size / dtype_byte_size
    )
    cell_threshold = cell_threshold * (free_memory_bytes / baseline_gpu_memory_bytes)

    # If we have multiple GPUs, we reduce the threshold a bit, based on empirical
    # results.
    if len(devices) > 1:
        cell_threshold *= 0.8

    return _get_num_cells(X_train_shape, X_test_shape) > cell_threshold


def _get_free_cuda_memory_bytes(device: torch.device) -> float:
    system_free_memory, _ = torch.cuda.mem_get_info(device)
    pytorch_cache_free_memory = torch.cuda.memory_reserved(
        device
    ) - torch.cuda.memory_allocated(device)
    return system_free_memory + pytorch_cache_free_memory


def _get_num_cells(
    X_train_shape: tuple[int, int], X_test_shape: tuple[int, int]
) -> int:
    n_train, n_features = X_train_shape
    n_test, _ = X_test_shape
    return (n_train + n_test) * n_features
