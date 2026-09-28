#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.parallel_execute."""

from __future__ import annotations

import threading

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


def test__parallel_execute__multiple_devices__takes_functions_as_devices_free_up() -> (
    None
):
    """At most one function per device is taken ahead of the consumer."""
    devices = [torch.device("meta"), torch.device("meta")]
    taken: list[int] = []

    def functions():  # noqa: ANN202
        for i in range(6):
            taken.append(i)
            yield lambda device, i=i: i  # noqa: ARG005

    results = parallel_execute(devices=devices, functions=functions())
    for consumed, result in enumerate(results):
        assert result == consumed
        assert len(taken) <= consumed + 1 + len(devices)

    assert taken == list(range(6))
