#  Copyright (c) Prior Labs GmbH 2026.

"""Delivery of usage events to the Prior Labs API.

Nothing is sent unless the account opted in. An account is opted in only manually
by the Prior Labs team. By default, no usage events are sent.

The first usage event of a process looks for an API key, and starts a collector
with it; without one, nothing more is logged. The collector asks the API, from a
background thread, whether usage telemetry is enabled for the account. Until it
is told so, it sends and saves nothing, and if it is not, or the API cannot say,
it drops the events and nothing more is logged. Nothing a call does waits for
the API.

Modelled on the queue of PostHog's Python client. Usage events are put on a
bounded queue in memory, and the collector's thread takes them off in batches
and sends them. Where PostHog drops a batch it cannot send, this saves it to
disk, and sends it once a later batch gets through, in this process or a later
one that uses the same API key. A batch may be sent twice, by two processes or
after an exit while it was being sent; the API passes each event's id on to
PostHog, which keeps one of them.

Events are dropped only if they are not JSON, if the API refuses them, if usage
telemetry is not enabled for the account, once the most batches are saved, or
while the queue is full because events come faster than they can be sent or
saved. A forked process, such as a multiprocessing worker started by forking,
logs nothing.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import http.client
import itertools
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any
from typing_extensions import override

from tabpfn.browser_auth import CACHE_DIR, check_telemetry_enabled, get_cached_token
from tabpfn.settings import settings
from tabpfn.telemetry.decorator import set_sink

logger = logging.getLogger(__name__)

FLUSH_AT = 10
"""Events are sent once this many are waiting, or after `FLUSH_INTERVAL`."""

FLUSH_INTERVAL = 5.0
"""Seconds to wait for `FLUSH_AT` events before sending fewer."""

BATCH_SIZE = 100
"""The most events sent in one request, which is what the API accepts."""

MAX_QUEUE_SIZE = 10_000
"""The most events waiting in memory."""

MAX_SAVED_BATCHES = 500
"""The most batches saved to disk, of about 100KB each; the oldest go first."""

REQUEST_TIMEOUT = 10.0
"""Seconds to wait for the API to answer."""

RETRY_DELAY = 60.0
"""Seconds after a failed request in which batches are saved without sending them."""

SHUTDOWN_TIMEOUT = 2.0
"""Seconds that sending the queued events may delay the exit of the process."""

SAVED_BATCHES_ROOT = CACHE_DIR / "telemetry"
"""Where batches are saved: next to the cached API key."""

# Put on the queue to stop the collector once it has delivered the events before.
_STOP = object()

# Numbers the batches this process saves, in order.
_saves = itertools.count()


class Collector(threading.Thread):
    """Queues usage events, and delivers them in batches from a background thread."""

    def __init__(self, token: str, api_url: str, directory: Path) -> None:
        """Create a collector, which `start` sets going.

        Args:
            token: The API key of the account.
            api_url: The base URL of the telemetry API.
            directory: Where to save the batches that cannot be sent.
        """
        super().__init__(name="tabpfn-telemetry", daemon=True)
        self.token = token
        self.api_url = api_url
        self.directory = directory
        self.queue: Queue[Any] = Queue(MAX_QUEUE_SIZE)
        self.running = True
        # Set once the API confirms that usage telemetry is enabled for the account.
        self.opted_in = False
        # The encoded events being delivered, saved by `stop` if still being sent.
        self.batch: list[str] = []
        self._retry_at = 0.0

    def collect(self, event: dict[str, Any]) -> None:
        """Queue a usage event.

        The event is encoded right away, so that one that is not JSON is dropped
        on its own rather than with the batch it would be sent in.
        """
        try:
            self.queue.put(json.dumps(event, allow_nan=False), block=False)
        except (TypeError, ValueError, Full):
            logger.debug("A usage event is dropped: not JSON, or the queue is full.")

    def stop(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        """Deliver the queued events, and save those not delivered within `timeout`."""
        if not self.running or self.ident is None:
            # Stopped already, never started, or usage telemetry is not enabled
            # for the account.
            return

        deadline = time.monotonic() + timeout
        with contextlib.suppress(Full):
            self.queue.put(_STOP, timeout=timeout)

        self.join(max(0.0, deadline - time.monotonic()))
        self.running = False
        if not self.opted_in:
            # Still waiting to hear whether the account opted in: nothing is kept.
            return

        # The thread is still sending if it is alive, and ends with the process.
        unsent = (list(self.batch) if self.is_alive() else []) + _drain(self.queue)
        with contextlib.suppress(OSError):
            for start in range(0, len(unsent), BATCH_SIZE):
                _save(self.directory, _body(unsent[start : start + BATCH_SIZE]))

    @override
    def run(self) -> None:
        try:
            enabled = check_telemetry_enabled(self.token, self.api_url)
        except Exception:
            # As if the API could not say: the events are dropped.
            logger.debug("Could not check for usage telemetry.", exc_info=True)
            enabled = None
        if not enabled:
            # The saved batches are kept if the API could not say, to be sent
            # by a process that hears that the account opted in.
            self._disable(keep_saved=enabled is None)
            return

        self.opted_in = True
        while self.running:
            self.batch = self.next()
            if self.batch:
                try:
                    self._deliver(_body(self.batch))
                except Exception:
                    logger.debug("Could not deliver usage events.", exc_info=True)
            self.batch = []

    def next(self) -> list[str]:
        """The waiting events, once `FLUSH_AT` wait or `FLUSH_INTERVAL` passed."""
        batch: list[str] = []
        deadline = time.monotonic() + FLUSH_INTERVAL
        while len(batch) < BATCH_SIZE:
            # Past `FLUSH_AT`, only the events already waiting are taken.
            wait = deadline - time.monotonic() if len(batch) < FLUSH_AT else 0.0
            try:
                event = self.queue.get(timeout=max(0.0, wait))
            except Empty:
                break

            if event is _STOP:
                self.running = False
                break
            batch.append(event)

        return batch

    def _deliver(self, body: bytes) -> None:
        """Send a batch, then the saved ones; or save it if it cannot be sent."""
        if self._send(body):
            self._send_saved()
        else:
            _save(self.directory, body)

    def _send_saved(self) -> None:
        """Send the saved batches, oldest first, for as long as the API takes them."""
        for path in sorted(self.directory.glob("*.json")):
            try:
                body = path.read_bytes()
            except OSError:
                # Sent by another process meanwhile, or not a batch.
                continue

            if not self._send(body):
                return

            path.unlink(missing_ok=True)

    def _send(self, body: bytes) -> bool:
        """Send a batch, unless a request failed shortly before.

        Returns:
            Whether the batch is done with, rather than to be sent again later.
        """
        if time.monotonic() < self._retry_at:
            return False

        status = _post(body, token=self.token, api_url=self.api_url)
        if status == 403:
            self._disable()
        elif status is not None and 400 <= status < 500 and status not in (408, 429):
            logger.debug("The telemetry API refused usage events: HTTP %d.", status)
        elif status is None or not 200 <= status < 300:
            self._retry_at = time.monotonic() + RETRY_DELAY
            return False

        return True

    def _disable(self, *, keep_saved: bool = False) -> None:
        """Stop, and forget the events: the account does not have usage telemetry."""
        logger.debug("Usage telemetry is not enabled for this account.")
        set_sink(None)
        self.running = False
        _drain(self.queue)
        if keep_saved:
            return
        with contextlib.suppress(OSError):
            for path in self.directory.glob("*"):
                path.unlink(missing_ok=True)


def _body(events: list[str]) -> bytes:
    """The request body that sends the encoded events."""
    return ('{"events":[' + ",".join(events) + "]}").encode()


def _post(body: bytes, *, token: str, api_url: str) -> int | None:
    """Send a batch to the telemetry API.

    Returns:
        The status of the response, or None if there was none.
    """
    try:
        request = urllib.request.Request(  # noqa: S310
            f"{api_url.rstrip('/')}/telemetry",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:  # noqa: S310
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, ValueError, http.client.HTTPException):
        # No network, an unreachable host, a timeout, a broken connection or a
        # malformed URL.
        return None


def _save(directory: Path, body: bytes) -> None:
    """Save a batch to disk, then delete the oldest beyond `MAX_SAVED_BATCHES`."""
    directory.mkdir(parents=True, exist_ok=True)
    # Named after the time, then the order of saving in this process, so that
    # the names sort oldest first even within one tick of a coarse clock, as on
    # Windows.
    path = directory / f"{time.time_ns()}-{next(_saves):09d}-{uuid.uuid4().hex}.json"
    # Written whole, then renamed, so that a batch is never read half-written.
    writing = path.with_suffix(".tmp")
    writing.write_bytes(body)
    writing.replace(path)
    for old in sorted(directory.glob("*.json"))[:-MAX_SAVED_BATCHES]:
        old.unlink(missing_ok=True)


def _drain(queue: Queue[Any]) -> list[str]:
    """Take every event off the queue."""
    events: list[str] = []
    with contextlib.suppress(Empty):
        while True:
            event = queue.get(block=False)
            if event is not _STOP:
                events.append(event)
    return events


_collector: Collector | None = None
_starting = threading.Lock()


def collect(event: dict[str, Any]) -> None:
    """Start collecting with the first usage event, if there is an API key and
    if user was enrolled in usage analytics by the Prior Labs team.

    The sink `log_usage` hands events to until then. It only reads the API key
    from the environment or a file, and leaves asking the API to the collector.
    """
    with _starting:
        if _collector is None:
            # User not authenticated, do not collect any events.
            token = get_cached_token()
            if token is None:
                set_sink(None)
                return

            # Start background thread to collect events. The underlying
            # collector will ask the API to confirm if usage analytics is enabled
            # for the account. If not, the events are not collected and the
            # background thread is killed.
            start(token, settings.tabpfn.auth_api_url)
        collector = _collector

    if collector is not None:
        collector.collect(event)


def start(token: str, api_url: str, *, root: Path | None = None) -> None:
    """Start collecting usage events with an API key.

    Batches are only ever sent with the API key they were collected with, so each
    key has a directory of its own to save them in.

    Args:
        token: The API key of the account.
        api_url: The base URL of the telemetry API.
        root: The directory to save batches in, `SAVED_BATCHES_ROOT` by default.
    """
    global _collector  # noqa: PLW0603
    if _collector is not None:
        return

    key = hashlib.sha256(token.encode()).hexdigest()[:16]
    _collector = Collector(token, api_url, (root or SAVED_BATCHES_ROOT) / key)
    # Installed first, so that the thread can uninstall it if the account did
    # not opt in, however quickly it hears so.
    set_sink(_collector.collect)
    _collector.start()


def stop() -> None:
    """Stop collecting, and deliver or save the events collected so far."""
    global _collector  # noqa: PLW0603
    if _collector is None:
        return
    set_sink(None)
    _collector.stop()
    _collector = None


def _after_fork_in_child() -> None:
    # A forked process logs nothing. The collector's thread does not survive the
    # fork, and on macOS the first network request of a forked process can crash
    # it, as a multiprocessing worker would. The parent delivers the events
    # queued before the fork.
    global _collector  # noqa: PLW0603
    _collector = None
    set_sink(None)


atexit.register(stop)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
