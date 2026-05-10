#  Copyright (c) Prior Labs GmbH 2026.

"""Parallel evaluation of a set of functions across multiple PyTorch devices."""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator, Iterable, Sequence
from multiprocessing.pool import ThreadPool
from typing import Callable, Generic, Protocol, TypeVar

import torch

R_co = TypeVar("R_co", covariant=True)

_LAPACK_PREWARM_LOCK = threading.Lock()
_LAPACK_PREWARMED_DEVICES: set[torch.device] = set()


def _prewarm_lapack_lazy_init(devices: Sequence[torch.device]) -> None:
    """Touch ``torch.linalg.qr`` once per cuda device on the main thread.

    PyTorch's LAPACK lazy wrapper is not thread-safe on first use: when
    several threads first-call ``torch.linalg.qr`` (transitively, via
    ``torch.svd_lowrank`` in the GPU SVD preprocessing step) at the same
    time, PyTorch raises ``RuntimeError: lazy wrapper should be called at
    most once`` and the process aborts.  We avoid the race by warming the
    binding on each cuda device on the main thread before the thread pool
    spawns.  Memoised so the cost is paid at most once per device per
    process.
    """
    with _LAPACK_PREWARM_LOCK:
        for device in devices:
            if device.type != "cuda" or device in _LAPACK_PREWARMED_DEVICES:
                continue
            with torch.cuda.device(device):
                torch.linalg.qr(torch.empty(2, 2, device=device).normal_())
            _LAPACK_PREWARMED_DEVICES.add(device)


class ParallelFunction(Protocol, Generic[R_co]):
    """Interface that functions submitted to `parallel_execute()` should implement."""

    def __call__(self, *, device: torch.device) -> R_co:
        """Execute the function.

        Args:
            device: PyTorch device that all computation should be performed on.

        Returns:
            Any desired value. Any Tensors in the returned value should be on `device`.
        """
        ...


def parallel_execute(
    devices: Sequence[torch.device],
    functions: Iterable[ParallelFunction[R_co]],
    *,
    prewarm_lapack: bool = False,
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
        prewarm_lapack: If True, pre-initialise ``torch.linalg.qr`` on each cuda
            device on the main thread before the thread pool spawns.  Required when
            the parallel functions use ``torch.svd_lowrank`` (or any other path
            that hits PyTorch's racy LAPACK lazy wrapper).

    Returns:
        A generator consisting of the return values of the functions, in the same order
        as `functions`.
    """
    if len(devices) == 1:
        # If we only have one device then just use the current thread to avoid overhead.
        yield from _execute_in_current_thread(devices[0], functions)
    else:
        yield from _execute_with_multithreading(
            devices, functions, prewarm_lapack=prewarm_lapack
        )


def _execute_in_current_thread(
    device: torch.device, functions: Iterable[ParallelFunction[R_co]]
) -> Generator[R_co]:
    for function in functions:
        yield function(device=device)


def _execute_with_multithreading(
    devices: Sequence[torch.device],
    functions: Iterable[ParallelFunction[R_co]],
    *,
    prewarm_lapack: bool = False,
) -> Generator[R_co]:
    if prewarm_lapack:
        _prewarm_lapack_lazy_init(devices)
    free_devices: queue.Queue[int] = queue.Queue(maxsize=len(devices))
    for device_index, _ in enumerate(devices):
        free_devices.put(device_index, block=False)

    with ThreadPool(processes=len(devices)) as pool:
        async_results = [
            pool.apply_async(_execute_function_in_thread, (devices, free_devices, func))
            for func in functions
        ]
        for async_result in async_results:
            sync_and_get_output = async_result.get()
            yield sync_and_get_output()


def _execute_function_in_thread(
    all_devices: Sequence[torch.device],
    free_devices: queue.Queue[int],
    function: ParallelFunction[R_co],
) -> Callable[[], R_co]:
    device_index = free_devices.get(block=True)
    try:
        device = all_devices[device_index]
        if device.type == "cuda":
            with torch.cuda.device(device):
                output = function(device=device)

                # The output will be consumed on a different cuda stream, which needs to
                # wait for the computation on this stream to be complete. Thus we insert
                # "ready" event after the model evaluation, and return a function to the
                # consumer that waits on this event.
                output_ready_event = torch.cuda.Event()
                output_ready_event.record()

                def sync_stream_and_get_output() -> R_co:
                    output_ready_event.synchronize()
                    return output

                return sync_stream_and_get_output

        # Theoretically it is possible to parallelise over classes of device other than
        # GPUs, but mainly this is useful for unit testing with multiple CPU devices.
        output = function(device=device)
        return lambda: output
    finally:
        free_devices.put(device_index)
