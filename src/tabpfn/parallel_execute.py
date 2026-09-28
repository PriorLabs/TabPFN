#  Copyright (c) Prior Labs GmbH 2026.

"""Parallel evaluation of a set of functions across multiple PyTorch devices."""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Generic, Protocol, TypeVar

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


def _execute_with_multithreading(  # noqa: C901
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

    # Callers create a function's inputs as it is taken, so functions are taken
    # only while fewer than one per device, plus one ready to start, are
    # unfinished. They are taken on this thread while it waits for outputs, run on
    # one worker thread per device, finish in any order, and are returned in order.
    work: queue.SimpleQueue[tuple[int, ParallelFunction[R_co]] | None] = (
        queue.SimpleQueue()
    )
    finished: dict[int, tuple[Callable[[], R_co] | None, BaseException | None]] = {}
    finished_changed = threading.Condition()

    def run() -> None:
        while (item := work.get()) is not None:
            index, function = item
            try:
                output = _execute_function_in_thread(devices, free_devices, function)
                outcome = (output, None)
            except BaseException as e:  # noqa: BLE001  re-raised on the consumer
                outcome = (None, e)
            with finished_changed:
                finished[index] = outcome
                finished_changed.notify()

    # One worker per device, so each always finds a free device in the queue.
    workers = [threading.Thread(target=run) for _ in devices]
    for worker in workers:
        worker.start()

    functions_iter = iter(functions)
    max_unfinished = len(devices) + 1
    n_taken = 0
    exhausted = False

    def has_room(n_returned: int) -> bool:
        return not exhausted and n_taken - n_returned - len(finished) < max_unfinished

    def take_one() -> None:
        nonlocal n_taken, exhausted
        function = next(functions_iter, None)  # creates the function's inputs
        if function is None:
            exhausted = True
        else:
            work.put((n_taken, function))
            n_taken += 1

    def next_output(
        n_returned: int,
    ) -> tuple[Callable[[], R_co] | None, BaseException | None] | None:
        """Take functions while there is room, then return output `n_returned`."""
        while True:
            with finished_changed:
                finished_changed.wait_for(
                    lambda: (
                        has_room(n_returned)
                        or n_returned in finished
                        or (exhausted and n_returned >= n_taken)
                    )
                )
                if not has_room(n_returned):
                    return finished.pop(n_returned, None)
            take_one()  # outside the lock, so workers can record their outputs

    try:
        for n_returned in itertools.count():
            outcome = next_output(n_returned)
            if outcome is None:
                return
            sync_and_get_output, error = outcome
            if error is not None:
                raise error
            assert sync_and_get_output is not None
            yield sync_and_get_output()
    finally:
        for _ in workers:
            work.put(None)
        for worker in workers:
            worker.join()


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
