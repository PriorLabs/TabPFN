#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for delivering usage events: tabpfn.telemetry.collector."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from typing_extensions import override

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from tabpfn.browser_auth import check_telemetry_enabled
from tabpfn.settings import settings
from tabpfn.telemetry import collector, decorator, log_usage
from tabpfn.telemetry.collector import Collector, start, stop

Event = dict[str, Any]

# A fake API key, which only the fake API below sees.
_TOKEN = "tabpfn_sk_test"  # noqa: S105


class _Api:
    """A telemetry API on localhost, which records the events it accepts."""

    def __init__(self) -> None:
        super().__init__()
        # The answer to whether usage telemetry is enabled; None fails the check.
        self.enabled: bool | None = True
        self.check_delay = 0.0
        self.checks = 0
        self.status = 204
        self.delay = 0.0
        self.hang_up = False
        self.batches: list[list[Event]] = []
        self.requests = 0
        self._lock = threading.Lock()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                with api._lock:
                    api.checks += 1
                time.sleep(api.check_delay)
                if self.path != "/account/telemetry" or api.enabled is None:
                    self.send_response(404 if api.enabled is not None else 500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"enabled": api.enabled}).encode())

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                try:
                    events: list[Event] = json.loads(self.rfile.read(length))["events"]
                except ValueError:
                    events = []
                with api._lock:
                    api.requests += 1
                time.sleep(api.delay)
                if api.hang_up:
                    # As if the network went down: no response at all.
                    self.close_connection = True
                    return
                status = api.status if events else 422
                if status == 204:
                    with api._lock:
                        api.batches.append(events)
                self.send_response(status)
                self.end_headers()

            @override
            def log_message(self, format: str, *args: Any) -> None:
                # Keeps the test output quiet.
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def event_ids(self) -> list[str]:
        with self._lock:
            return [e["event_id"] for batch in self.batches for e in batch]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def api() -> Generator[_Api]:
    server = _Api()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _quick(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setattr(collector, "FLUSH_INTERVAL", 0.05)
    monkeypatch.setattr(collector, "RETRY_DELAY", 0.05)
    # Each test asks its own fake API, which may reuse an earlier one's port.
    check_telemetry_enabled.cache_clear()
    yield
    check_telemetry_enabled.cache_clear()


def _events(count: int) -> list[Event]:
    return [{"event_id": str(uuid.uuid4())} for _ in range(count)]


def _body(events: list[Event]) -> bytes:
    return collector._body([json.dumps(e) for e in events])


def _ids(events: list[Event]) -> list[str]:
    return sorted(e["event_id"] for e in events)


def _saved_ids(directory: Path) -> list[str]:
    """The ids of the events saved to disk, oldest batch first."""
    return [
        e["event_id"]
        for path in sorted(directory.glob("*.json"))
        for e in json.loads(path.read_text())["events"]
    ]


def _started(api_url: str, directory: Path) -> Collector:
    started = Collector(_TOKEN, api_url, directory)
    started.start()
    return started


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for the condition.")
        time.sleep(0.01)


@pytest.mark.parametrize("status", [204, 401, 403, 422, 503])
def test__post__returns_the_status_of_the_response(api: _Api, status: int) -> None:
    api.status = status

    assert collector._post(_body(_events(1)), token=_TOKEN, api_url=api.url) == status


@pytest.mark.parametrize(
    "api_url",
    [
        # Nothing listens on port 9 of localhost, so connecting is refused.
        "http://127.0.0.1:9",
        "api.priorlabs.ai",
    ],
)
def test__post__no_response__none(api_url: str) -> None:
    assert collector._post(_body(_events(1)), token=_TOKEN, api_url=api_url) is None


def test__post__api_too_slow__none(api: _Api, monkeypatch: pytest.MonkeyPatch) -> None:
    api.delay = 0.5
    monkeypatch.setattr(collector, "REQUEST_TIMEOUT", 0.1)

    assert collector._post(_body(_events(1)), token=_TOKEN, api_url=api.url) is None


def test__collector__sends_events_in_batches_of_at_most_100(
    api: _Api, tmp_path: Path
) -> None:
    events = _events(250)
    started = _started(api.url, tmp_path)
    for event in events:
        started.collect(event)
    started.stop()

    assert sorted(api.event_ids) == _ids(events)
    assert max(len(batch) for batch in api.batches) <= 100
    assert not any(tmp_path.iterdir())


def test__collector__sends_once_10_events_wait(
    api: _Api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "FLUSH_INTERVAL", 60.0)
    events = _events(10)
    started = _started(api.url, tmp_path)
    for event in events[:9]:
        started.collect(event)
    time.sleep(0.3)
    assert api.requests == 0

    started.collect(events[9])
    _wait_until(lambda: len(api.event_ids) == 10)
    started.stop()

    assert [_ids(batch) for batch in api.batches] == [_ids(events)]


def test__collector__stop__sends_without_waiting_for_the_flush_interval(
    api: _Api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "FLUSH_INTERVAL", 60.0)
    events = _events(3)
    started = _started(api.url, tmp_path)
    for event in events:
        started.collect(event)

    began = time.monotonic()
    started.stop()

    assert time.monotonic() - began < 1.0
    assert sorted(api.event_ids) == _ids(events)


def test__collector__telemetry_not_enabled__events_and_saved_batches_dropped(
    api: _Api, tmp_path: Path
) -> None:
    collector._save(tmp_path, _body(_events(5)))
    api.enabled = False
    decorator.set_sink(lambda _: None)
    try:
        started = _started(api.url, tmp_path)
        started.collect(_events(1)[0])
        _wait_until(lambda: not started.is_alive())
        sink = decorator._sink
    finally:
        decorator.set_sink(None)
    started.stop()

    assert sink is None
    assert api.requests == 0
    assert not any(tmp_path.iterdir())


def test__collector__check_fails__events_dropped_and_saved_batches_kept(
    api: _Api, tmp_path: Path
) -> None:
    saved = _events(5)
    collector._save(tmp_path, _body(saved))
    api.enabled = None
    started = _started(api.url, tmp_path)
    started.collect(_events(1)[0])
    _wait_until(lambda: not started.is_alive())
    started.stop()

    assert api.requests == 0
    assert _saved_ids(tmp_path) == [e["event_id"] for e in saved]


def test__collector__check_raises__events_dropped_quietly(
    api: _Api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: Any) -> bool:
        raise RuntimeError("A bug in the check.")

    monkeypatch.setattr(collector, "check_telemetry_enabled", fail)
    failures: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", failures.append)
    decorator.set_sink(lambda _: None)
    try:
        started = _started(api.url, tmp_path)
        started.collect(_events(1)[0])
        _wait_until(lambda: not started.is_alive())
        sink = decorator._sink
    finally:
        decorator.set_sink(None)
    started.stop()

    assert failures == []
    assert sink is None
    assert api.requests == 0


def test__stop__collector_never_started__returns(
    api: _Api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: Collector) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(Collector, "start", fail)
    with pytest.raises(RuntimeError):
        start(_TOKEN, api.url, root=tmp_path)
    stop()

    assert collector._collector is None
    assert decorator._sink is None


def test__collector__stopped_before_the_check_answers__nothing_kept(
    api: _Api, tmp_path: Path
) -> None:
    api.check_delay = 1.0
    started = _started(api.url, tmp_path)
    for event in _events(5):
        started.collect(event)
    started.stop(timeout=0.2)

    assert api.requests == 0
    assert not any(tmp_path.iterdir())


def test__collector__no_response__saved_and_sent_by_the_next_one(
    api: _Api, tmp_path: Path
) -> None:
    api.hang_up = True
    offline_events = _events(30)
    offline = _started(api.url, tmp_path)
    for event in offline_events:
        offline.collect(event)
    offline.stop()

    assert sorted(_saved_ids(tmp_path)) == _ids(offline_events)

    # The saved batches go once a batch of the next collector gets through.
    api.hang_up = False
    online_events = _events(1)
    online = _started(api.url, tmp_path)
    online.collect(online_events[0])
    online.stop()

    assert sorted(api.event_ids) == _ids(offline_events + online_events)
    assert not any(tmp_path.iterdir())


def test__collector__send_fails__saved_and_sent_once_the_api_recovers(
    api: _Api, tmp_path: Path
) -> None:
    api.status = 503
    events = _events(101)
    started = _started(api.url, tmp_path)
    for event in events[:100]:
        started.collect(event)
    _wait_until(lambda: len(_saved_ids(tmp_path)) == 100)

    api.status = 204
    time.sleep(0.1)
    started.collect(events[100])
    _wait_until(lambda: len(api.event_ids) == len(events))
    started.stop()

    assert sorted(api.event_ids) == _ids(events)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("status", [307, 408, 429, 500, 503])
def test__collector__send_fails__batch_saved(
    api: _Api, tmp_path: Path, status: int
) -> None:
    api.status = status
    events = _events(10)
    started = _started(api.url, tmp_path)
    for event in events:
        started.collect(event)
    started.stop()

    assert sorted(_saved_ids(tmp_path)) == _ids(events)


@pytest.mark.parametrize("status", [400, 401, 422])
def test__collector__batch_refused__dropped(
    api: _Api, tmp_path: Path, status: int
) -> None:
    api.status = status
    started = _started(api.url, tmp_path)
    for event in _events(10):
        started.collect(event)
    started.stop()

    assert api.requests == 1
    assert not any(tmp_path.iterdir())


def test__collector__telemetry_disabled_while_sending__stops_and_forgets_the_events(
    api: _Api, tmp_path: Path
) -> None:
    collector._save(tmp_path, _body(_events(5)))
    api.status = 403
    decorator.set_sink(lambda _: None)
    try:
        started = _started(api.url, tmp_path)
        started.collect(_events(1)[0])
        _wait_until(lambda: not started.is_alive())
        sink = decorator._sink
    finally:
        decorator.set_sink(None)
    # Collected after usage telemetry was found disabled, by a call in flight.
    started.collect(_events(1)[0])
    started.stop()

    assert sink is None
    assert api.event_ids == []
    assert not any(tmp_path.iterdir())


def test__collector__exits_while_sending__the_batch_is_saved(
    api: _Api, tmp_path: Path
) -> None:
    api.delay = 3.0
    events = _events(100)
    started = _started(api.url, tmp_path)
    for event in events:
        started.collect(event)
    _wait_until(lambda: api.requests == 1)
    started.stop(timeout=0.2)

    assert sorted(_saved_ids(tmp_path)) == _ids(events)


def test__collector__queue_full__collect_drops_rather_than_waits(
    api: _Api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "MAX_QUEUE_SIZE", 3)
    api.delay = 0.3
    events = _events(11)
    started = _started(api.url, tmp_path)
    started.collect(events[0])
    _wait_until(lambda: api.requests == 1)

    began = time.monotonic()
    for event in events[1:]:
        started.collect(event)
    assert time.monotonic() - began < 0.1
    started.stop(timeout=5.0)

    assert sorted(api.event_ids) == _ids(events[:4])


def test__collector__event_not_json__dropped_alone(api: _Api, tmp_path: Path) -> None:
    events = _events(4)
    started = _started(api.url, tmp_path)
    started.collect(events[0])
    started.collect({"event_id": "numpy", "num_rows": np.int64(3)})
    started.collect(events[1])
    started.collect({"event_id": "nan", "duration_ms": float("nan")})
    started.collect(events[2])
    started.collect(events[3])
    started.stop()

    assert sorted(api.event_ids) == _ids(events)


def test__save__keeps_the_newest_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "MAX_SAVED_BATCHES", 3)
    batches = [_events(2) for _ in range(5)]
    for batch in batches:
        collector._save(tmp_path, _body(batch))

    assert _saved_ids(tmp_path) == [e["event_id"] for b in batches[2:] for e in b]


def test__collector__saved_batch_corrupt__refused_and_deleted(
    api: _Api, tmp_path: Path
) -> None:
    (tmp_path / f"{time.time_ns()}-x.json").write_text("{not json")
    started = _started(api.url, tmp_path)
    started.collect(_events(1)[0])
    started.stop()

    assert not any(tmp_path.iterdir())


def test__collector__saved_batch_unreadable__later_ones_still_sent(
    api: _Api, tmp_path: Path
) -> None:
    (tmp_path / "1-a.json").mkdir()
    later = _events(2)
    collector._save(tmp_path, _body(later))
    events = _events(1)
    started = _started(api.url, tmp_path)
    started.collect(events[0])
    started.stop()

    assert sorted(api.event_ids) == _ids(later + events)


class _Estimator(ClassifierMixin, BaseEstimator):
    @log_usage("fit")
    def fit(self, X: np.ndarray, y: np.ndarray) -> _Estimator:  # noqa: ARG002
        self.n_train_samples_ = len(X)
        return self


def _fit() -> None:
    _Estimator().fit(np.zeros((3, 2)), np.array([0, 1, 0]))


@pytest.fixture
def first_event_starts(
    api: _Api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None]:
    """Log usage as a fresh process does, against the fake API."""
    monkeypatch.setattr(settings.tabpfn, "auth_api_url", api.url)
    monkeypatch.setattr(collector, "SAVED_BATCHES_ROOT", tmp_path)
    decorator.set_sink(collector.collect)
    yield
    stop()
    decorator.set_sink(None)


@pytest.mark.usefixtures("first_event_starts")
def test__collect__api_key__first_event_starts_the_collector(
    api: _Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "get_cached_token", lambda: _TOKEN)
    _fit()
    _fit()
    stop()

    assert api.checks == 1
    assert [e["event"] for b in api.batches for e in b] == ["fit_called"] * 2


@pytest.mark.usefixtures("first_event_starts")
def test__collect__no_api_key__nothing_more_logged(
    api: _Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "get_cached_token", lambda: None)
    _fit()

    assert decorator._sink is None
    assert collector._collector is None
    assert api.checks == 0


@pytest.mark.usefixtures("first_event_starts")
def test__collect__check_is_slow__the_call_does_not_wait(
    api: _Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "get_cached_token", lambda: _TOKEN)
    api.check_delay = 1.0

    began = time.monotonic()
    _fit()

    assert time.monotonic() - began < 0.5


def test__start__delivers_the_usage_of_logged_methods_until_stop(
    api: _Api, tmp_path: Path
) -> None:
    start(_TOKEN, api.url, root=tmp_path)
    try:
        _fit()
    finally:
        stop()

    assert decorator._sink is None
    (batch,) = api.batches
    assert [(e["event"], e["method"], e["num_rows"]) for e in batch] == [
        ("fit_called", "_Estimator.fit", 3)
    ]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Needs os.fork.")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test__start__forked_child__logs_nothing(api: _Api, tmp_path: Path) -> None:
    start(_TOKEN, api.url, root=tmp_path)
    try:
        # Still queued in the parent when it forks.
        _fit()
        pid = os.fork()
        if pid == 0:
            try:
                _fit()
                stop()
            finally:
                os._exit(0 if decorator._sink is None else 1)
        _, status = os.waitpid(pid, 0)
    finally:
        stop()

    assert os.waitstatus_to_exitcode(status) == 0
    # Only the parent's event, sent by the parent alone.
    assert len(api.event_ids) == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Needs os.fork.")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
@pytest.mark.usefixtures("first_event_starts")
def test__collect__forked_before_the_first_event__child_logs_nothing(
    api: _Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "get_cached_token", lambda: _TOKEN)
    pid = os.fork()
    if pid == 0:
        try:
            _fit()
        finally:
            os._exit(0 if collector._collector is None else 1)
    _, status = os.waitpid(pid, 0)

    # Starting to collect would have the child make its first network request,
    # which on macOS can crash a forked process.
    assert os.waitstatus_to_exitcode(status) == 0
    assert api.checks == 0
