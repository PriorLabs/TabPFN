"""Tests for tabpfn.parallel_execute."""

from __future__ import annotations

import threading

import torch

from tabpfn.parallel_execute import parallel_execute


def test__parallel_execute__single_device__executes_in_current_thread() -> None:
    def test_function(device: torch.device, is_parallel: bool) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu")], functions=[test_function, test_function]
    )

    current_thread_id = threading.get_ident()
    assert list(thread_ids) == [current_thread_id, current_thread_id]


def test__parallel_execute__single_device__sets_is_parallel_to_False() -> None:
    def test_function(device: torch.device, is_parallel: bool) -> bool:  # noqa: ARG001
        return is_parallel

    is_parallels = parallel_execute(
        devices=[torch.device("cpu")], functions=[test_function, test_function]
    )

    assert list(is_parallels) == [False, False]


def test__parallel_execute__single_device__results_in_same_order_as_functions() -> None:
    def a(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(devices=[torch.device("cpu")], functions=[a, b, c])

    assert list(results) == ["a", "b", "c"]


def test__parallel_execute__multiple_devices__executes_in_worker_threads() -> None:
    def test_function(device: torch.device, is_parallel: bool) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu"), torch.device("meta")],
        functions=[test_function, test_function],
    )

    current_thread_id = threading.get_ident()
    for thread_id in thread_ids:
        assert thread_id != current_thread_id


def test__parallel_execute__multiple_devices__sets_is_parallel_to_True() -> None:
    def test_function(device: torch.device, is_parallel: bool) -> bool:  # noqa: ARG001
        return is_parallel

    is_parallels = parallel_execute(
        devices=[torch.device("cpu"), torch.device("meta")],
        functions=[test_function, test_function],
    )

    assert list(is_parallels) == [True, True]


def test__parallel_execute__multiple_devices__results_in_same_order_as_functions() -> (
    None
):
    def a(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(
        devices=[torch.device("meta"), torch.device("meta")], functions=[a, b, c]
    )

    assert list(results) == ["a", "b", "c"]
