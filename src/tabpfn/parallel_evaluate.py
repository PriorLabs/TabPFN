"""Parallel evaluation of a set of functions across multiple PyTorch devices."""

from __future__ import annotations

import queue
import time
from collections.abc import Generator, Iterable, Sequence
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from typing import Callable, TypeVar, cast

import torch
from torch import multiprocessing

R = TypeVar("R")


def parallel_evaluate(
    devices: Sequence[torch.device],
    functions: Iterable[Callable[[torch.device], R]],
) -> Generator[R]:
    """Evaluate the given functions in parallel across `devices`.

    The function evaluations are parallelised using Python threads, so this will only
    result in a speed-up if the functions do not hold the global interpreter lock. It
    works well for functions that spend most of their time executing GPU kernels.

    Args:
        devices: The devices to use for evaluation.
        functions: The functions to evaluate. Each function should accept a single
            `device` argument, and perform all its computation on that device. Any
            Tensors in the return value may be on any device.
            TODO: Note about function sometimes being deepcopied.

    Returns:
        A generator consisting of the return values of the functions, in the same order
        as `functions`.
    """
    if len(devices) == 1:
        # If we only have one device then just use the current thread to avoid overhead.
        yield from _evaluate_in_current_thread(devices[0], functions)
    else:
        yield from _evaluate_with_multithreading(devices, functions)
        # yield from _evaluate_with_multiprocessing(devices, functions)


def _evaluate_in_current_thread(
    device: torch.device, functions: Iterable[Callable[[torch.device], R]]
) -> Generator[R]:
    for function in functions:
        yield function(device)


def _evaluate_with_multithreading(
    devices: Sequence[torch.device],
    functions: Iterable[Callable[[torch.device], R]],
) -> Generator[R]:
    free_devices: queue.Queue[int] = queue.Queue(maxsize=len(devices))
    for device_index, _ in enumerate(devices):
        free_devices.put(device_index, block=False)

    with ThreadPool(processes=len(devices)) as pool:
        async_results = [
            pool.apply_async(
                # We deepcopy the input function to avoid threads mutating each others
                # state. For example, if the function uses an nn.Module this allows
                # different threads to move the module to different devices.
                _execute_function_in_thread,
                (devices, free_devices, deepcopy(func)),
            )
            for func in functions
        ]
        for async_result in async_results:
            yield async_result.get()


def _execute_function_in_thread(
    all_devices: Sequence[torch.device],
    free_devices: queue.Queue[int],
    function: Callable[[torch.device], R],
) -> R:
    device_index = free_devices.get(block=True)
    try:
        device = all_devices[device_index]
        # We use a separate stream per thread so that threads can execute kernels in
        # parallel.
        with torch.cuda.stream(torch.cuda.Stream(device=device)):
            return function(device)
    finally:
        free_devices.put(device_index)


def _evaluate_with_multiprocessing(
    devices: Sequence[torch.device],
    functions: Iterable[Callable[[torch.device], R]],
) -> Generator[R]:
    # Create our own context to avoid conflicts with the caller's.
    mp_context = multiprocessing.get_context(method="spawn")

    time.time()
    with mp_context.Manager() as manager:
        time.time()
        free_devices = manager.Queue(maxsize=len(devices))
        for device_index, _ in enumerate(devices):
            free_devices.put(device_index, block=False)

        with mp_context.Pool(processes=len(devices)) as pool:
            async_results = [
                pool.apply_async(
                    _execute_function_in_process, (devices, free_devices, func)
                )
                for func in functions
            ]
            for async_result in async_results:
                yield async_result.get()


def _execute_function_in_process(
    all_devices: Sequence[torch.device],
    free_devices: multiprocessing.Queue,
    function: Callable[[torch.device], R],
) -> R:
    device_index = cast(int, free_devices.get(block=True))
    try:
        device = all_devices[device_index]
        torch.cuda.set_device(device)
        output = function(device)
        # Passing cuda tensors between processes is apparently expensive, and it fails
        # entirely for mps tensors. As this output is probably quite small, just move it
        # back to CPU for now.
        if isinstance(output, dict):
            return {k: v.cpu() for k, v in output.items()}
        return output.cpu()
    finally:
        free_devices.put(device_index)
