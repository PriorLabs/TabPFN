#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from types import MethodType
from typing import Any, Literal, Union

import torch

from tabpfn.settings import settings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings.pytorch.pytorch_cuda_alloc_conf
DEFAULT_SAVE_PEAK_MEMORY_FACTOR = 8


def support_save_peak_mem_factor(method: MethodType) -> Callable:
    """Can be applied to a method acting on a tensor 'x' whose first dimension is a
    flat batch dimension
    (i.e. the operation is trivially parallel over the first dimension).

    For additional tensor arguments, it is assumed that the first dimension is again
    the batch dimension, and that non-tensor arguments can be passed as-is
    to splits when parallelizing over the batch dimension.

    The decorator adds options 'add_input' to add the principal input 'x' to the
    result of the method and 'allow_inplace'.
    By setting 'allow_inplace', the caller indicates that 'x'
    is not used after the call and its buffer can be reused for the output.

    Setting 'allow_inplace' does not ensure that the operation will be inplace,
    and the return value should be used for clarity and simplicity.

    Moreover, it adds an optional int parameter 'save_peak_mem_factor' that is
    only supported in combination with 'allow_inplace' during inference and subdivides
    the operation into the specified number of chunks to reduce peak memory consumption.
    """

    def method_(
        self: torch.nn.Module,
        x: torch.Tensor,
        *args: torch.Tensor,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert isinstance(self, torch.nn.Module)
        assert save_peak_mem_factor is None or allow_inplace, (
            "The parameter save_peak_mem_factor only supported with 'allow_inplace' set"
        )
        assert isinstance(x, torch.Tensor)

        tensor_inputs = list(tuple(self.parameters()) + tuple(args))

        assert (
            save_peak_mem_factor is None
            or not any(t.requires_grad for t in tensor_inputs)
            or not torch.is_grad_enabled()
        ), "The parameter save_peak_mem_factor is only supported during inference."

        if save_peak_mem_factor is not None:
            assert isinstance(save_peak_mem_factor, int)
            assert save_peak_mem_factor > 1
            split_size = (x.size(0) + save_peak_mem_factor - 1) // save_peak_mem_factor

            split_args = zip(
                *[
                    torch.split(arg, split_size)
                    if isinstance(arg, torch.Tensor)
                    else [arg] * save_peak_mem_factor
                    for arg in (x, *args)
                ],
            )

            for x_, *args_ in split_args:
                if add_input:
                    x_[:] += method(self, x_, *args_, **kwargs)
                else:
                    x_[:] = method(self, x_, *args_, **kwargs)
            return x

        if add_input:
            return x + method(self, x, *args, **kwargs)

        return method(self, x, *args, **kwargs)

    return method_


MemorySavingMode = Union[bool, Literal["auto"], float, int]


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
