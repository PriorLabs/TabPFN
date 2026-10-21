#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.parallel_execute."""

from __future__ import annotations

import threading
import time
from functools import partial

import torch

from tabpfn.parallel_execute import parallel_execute


def test__parallel_execute__single_device__executes_in_current_thread() -> None:
    def test_function(device: torch.device) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu")], functions=[test_function, test_function]
    )

    current_thread_id = threading.get_ident()
    assert list(thread_ids) == [current_thread_id, current_thread_id]


def test__parallel_execute__single_device__results_in_same_order_as_functions() -> None:
    def a(device: torch.device) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(devices=[torch.device("cpu")], functions=[a, b, c])

    assert list(results) == ["a", "b", "c"]


def test__parallel_execute__multiple_devices__executes_in_worker_threads() -> None:
    def test_function(device: torch.device) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu"), torch.device("meta")],
        functions=[test_function, test_function],
    )

    current_thread_id = threading.get_ident()
    for thread_id in thread_ids:
        assert thread_id != current_thread_id


def test__parallel_execute__multiple_devices__results_in_same_order_as_functions() -> (
    None
):
    def a(device: torch.device) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(
        devices=[torch.device("meta"), torch.device("meta")], functions=[a, b, c]
    )

    assert list(results) == ["a", "b", "c"]


def test__parallel_execute__multiple_devices__bounds_unfinished_functions() -> None:
    """At most one function per device, plus one ready to start, is unfinished.

    Callers create a function's inputs as it is taken, so this bounds them.
    """
    devices = [torch.device("meta"), torch.device("meta")]
    lock = threading.Lock()
    unfinished = 0
    most_unfinished = 0

    def run(device: torch.device, i: int) -> int:  # noqa: ARG001
        nonlocal unfinished
        time.sleep(0.01)
        with lock:
            unfinished -= 1
        return i

    def functions():  # noqa: ANN202
        nonlocal unfinished, most_unfinished
        for i in range(8):
            with lock:
                unfinished += 1
                most_unfinished = max(most_unfinished, unfinished)
            yield partial(run, i=i)

    results = list(parallel_execute(devices=devices, functions=functions()))

    assert results == list(range(8))
    assert most_unfinished <= len(devices) + 1


def test__parallel_execute__multiple_devices__slow_function_does_not_stall_others() -> (
    None
):
    """Later functions keep running on free devices while an earlier one is slow."""
    devices = [torch.device("meta"), torch.device("meta")]
    last_started = threading.Event()

    def slow_first(device: torch.device) -> str:  # noqa: ARG001
        # Only finishes once the last function has started on the other device.
        assert last_started.wait(timeout=10)
        return "slow"

    def quick(device: torch.device, name: str) -> str:  # noqa: ARG001
        if name == "last":
            last_started.set()
        return name

    functions = [
        slow_first,
        partial(quick, name="b"),
        partial(quick, name="c"),
        partial(quick, name="last"),
    ]
    results = list(parallel_execute(devices=devices, functions=functions))

    assert results == ["slow", "b", "c", "last"]
