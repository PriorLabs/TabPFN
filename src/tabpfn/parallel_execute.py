"""Parallel evaluation of a set of functions across multiple PyTorch devices."""

from __future__ import annotations

import queue
from collections.abc import Generator, Iterable, Sequence
from multiprocessing.pool import ThreadPool
from typing import Generic, Protocol, TypeVar

import torch

R_co = TypeVar("R_co", covariant=True)


class ParallelFunction(Protocol, Generic[R_co]):
    """Interface that functions submitted to `parallel_execute()` should implement."""

    def __call__(self, *, device: torch.device, is_parallel: bool) -> R_co:
        """Execute the function.

        Args:
            device: PyTorch device that all computation should be performed on.
            is_parallel: Indicates whether this function is being executed in parallel
                with other functions. If True, then the function should take care to
                copy any state shared with other functions before mutating it. For
                example, any nn.Modules should be deep copied before moving them to
                `device`. If False, then copying can be avoided to reduce overhead.

        Returns:
            Any desired value. Any Tensors in the returned value can be on any device.
        """
        ...


def parallel_execute(
    devices: Sequence[torch.device],
    functions: Iterable[ParallelFunction[R_co]],
) -> Generator[R_co]:
    """Evaluate the given functions in parallel across `devices`.

    The function evaluations are parallelised using Python threads, so this will only
    result in a speed-up if the functions do not hold the global interpreter lock. It
    works well for functions that spend most of their time executing GPU kernels.

    If only one device is provided, then the functions are executed in the current
    thread to reduce overhead.

    Args:
        devices: The devices to use for evaluation.
        functions: The functions to evaluate following the `ParallelFunction` protocol.

    Returns:
        A generator consisting of the return values of the functions, in the same order
        as `functions`.
    """
    if len(devices) == 1:
        # If we only have one device then just use the current thread to avoid overhead.
        yield from _execute_in_current_thread(devices[0], functions)
    else:
        yield from _execute_with_multithreading(devices, functions)


def _execute_in_current_thread(
    device: torch.device, functions: Iterable[ParallelFunction[R_co]]
) -> Generator[R_co]:
    for function in functions:
        yield function(device=device, is_parallel=False)


def _execute_with_multithreading(
    devices: Sequence[torch.device],
    functions: Iterable[ParallelFunction[R_co]],
) -> Generator[R_co]:
    free_devices: queue.Queue[int] = queue.Queue(maxsize=len(devices))
    for device_index, _ in enumerate(devices):
        free_devices.put(device_index, block=False)

    with ThreadPool(processes=len(devices)) as pool:
        async_results = [
            pool.apply_async(_execute_function_in_thread, (devices, free_devices, func))
            for func in functions
        ]
        for async_result in async_results:
            yield async_result.get()


def _execute_function_in_thread(
    all_devices: Sequence[torch.device],
    free_devices: queue.Queue[int],
    function: ParallelFunction[R_co],
) -> R_co:
    device_index = free_devices.get(block=True)
    try:
        device = all_devices[device_index]
        if device.type == "cuda":
            # We use a separate stream per thread so that threads can execute kernels in
            # parallel.
            stream = torch.cuda.Stream(device)
            with torch.cuda.stream(stream), torch.cuda.device(device):
                output = function(device=device, is_parallel=True)
                # The returned output will be consumed on a different CUDA stream, hence
                # we synchronize before returning so that the output is ready for the
                # consumer. It would be more efficient for the consumer to wait, so this
                # thread can start with the next function, but this approach is simpler.
                stream.synchronize()
                return output
        # Theoretically it is possible to parallelise over classes of device other than
        # GPUs, but mainly this is useful for unit testing with multiple CPU devices.
        return function(device=device, is_parallel=True)
    finally:
        free_devices.put(device_index)
