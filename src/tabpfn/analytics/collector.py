#  Copyright (c) Prior Labs GmbH 2026.

"""Delivery of usage events to the Prior Labs API.

Nothing is sent unless the account opted in. An account is opted in only manually
by the Prior Labs team. By default, no usage events are sent.

The first usage event of a process looks for an API key, and starts a collector
with it; without one, nothing more is logged. The collector asks the API, from a
background thread, whether usage analytics is enabled for the account. Until it
is told so, it sends nothing, and if it is not, or the API cannot say, it drops
the events and nothing more is logged. Nothing a call does waits for the API.

Modelled on the queue of PostHog's Python client. Usage events are put on a
bounded queue in memory, and the collector's thread takes them off in batches
and sends them. A batch the API does not take is kept in memory and sent again,
while later events wait in the queue; nothing is written to disk. A batch may
be sent twice, if a request reached the API but its answer did not arrive; the
API passes each event's id on to PostHog, which keeps one of them.

Events are dropped if they are not JSON, if the API refuses them, if usage
analytics is not enabled for the account or the API cannot say, while the queue
is full, and if the process ends before they are sent. A forked process, such
as a multiprocessing worker started by forking, logs nothing.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from queue import Empty, Full, Queue
from typing import Any
from typing_extensions import override

from tabpfn.analytics.decorator import set_sink
from tabpfn.browser_auth import check_telemetry_enabled, get_cached_token
from tabpfn.settings import settings

logger = logging.getLogger(__name__)

FLUSH_AT = 10
"""Events are sent once this many are waiting, or after `FLUSH_INTERVAL`."""

FLUSH_INTERVAL = 5.0
"""Seconds to wait for `FLUSH_AT` events before sending fewer."""

BATCH_SIZE = 100
"""The most events sent in one request, which is what the API accepts."""

MAX_QUEUE_SIZE = 10_000
"""The most events waiting in memory."""

REQUEST_TIMEOUT = 10.0
"""Seconds to wait for the API to answer."""

RETRY_DELAY = 60.0
"""Seconds to wait before sending a batch again that the API did not take."""

SHUTDOWN_TIMEOUT = 2.0
"""Seconds that sending the queued events may delay the exit of the process."""

# Put on the queue to stop the collector once it has delivered the events before.
_STOP = object()


class Collector(threading.Thread):
    """Queues usage events, and delivers them in batches from a background thread."""

    def __init__(self, token: str, api_url: str) -> None:
        """Create a collector, which `start` sets going.

        Args:
            token: The API key of the account.
            api_url: The base URL of the analytics API.
        """
        super().__init__(name="tabpfn-analytics", daemon=True)
        self.token = token
        self.api_url = api_url
        self.queue: Queue[Any] = Queue(MAX_QUEUE_SIZE)
        self.running = True
        # The encoded events being sent, kept until the API takes them.
        self.batch: list[str] = []
        # Set by `stop`, to end a wait to send a batch again.
        self._stopping = threading.Event()

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
        """Send the queued events; those not sent within `timeout` are dropped."""
        if not self.running or self.ident is None:
            # Stopped already, never started, or usage analytics is not enabled
            # for the account.
            return

        deadline = time.monotonic() + timeout
        self._stopping.set()
        with contextlib.suppress(Full):
            self.queue.put(_STOP, timeout=timeout)

        self.join(max(0.0, deadline - time.monotonic()))
        self.running = False

    @override
    def run(self) -> None:
        try:
            enabled = check_telemetry_enabled(self.token, self.api_url)
        except Exception:
            logger.debug("Could not check for usage analytics.", exc_info=True)
            enabled = None
        if not enabled:
            # Not enabled for the account, or the API could not say: nothing is
            # recorded.
            self._disable()
            return

        while self.running:
            self.batch = self.batch or self.next()
            if self.batch and not self._send(self.batch):
                # Kept to be sent again, while later events wait in the queue.
                # Once stopping, the API has just failed, so the events are dropped.
                if self._stopping.wait(RETRY_DELAY):
                    return
                continue

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

    def _send(self, events: list[str]) -> bool:
        """Send a batch.

        Returns:
            Whether the batch is done with, rather than to be sent again.
        """
        status = _post(_body(events), token=self.token, api_url=self.api_url)
        if status == 403:
            self._disable()
        elif status is not None and 400 <= status < 500 and status not in (408, 429):
            logger.debug("The analytics API refused usage events: HTTP %d.", status)
        elif status is None or not 200 <= status < 300:
            return False

        return True

    def _disable(self) -> None:
        """Stop, and drop the events: usage analytics is not enabled for the account."""
        logger.debug("Usage analytics is not enabled for this account.")
        set_sink(None)
        self.running = False
        with contextlib.suppress(Empty):
            while True:
                self.queue.get(block=False)


def _body(events: list[str]) -> bytes:
    """The request body that sends the encoded events."""
    return ('{"events":[' + ",".join(events) + "]}").encode()


def _post(body: bytes, *, token: str, api_url: str) -> int | None:
    """Send a batch to the analytics API.

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
    except Exception:
        # No network, an unreachable host, a timeout, a broken connection or a
        # malformed URL.
        logger.debug("Could not reach the analytics API.", exc_info=True)
        return None


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
            # background thread ends.
            start(token, settings.tabpfn.auth_api_url)
        collector = _collector

    if collector is not None:
        collector.collect(event)


def start(token: str, api_url: str) -> None:
    """Start collecting usage events with an API key.

    Args:
        token: The API key of the account.
        api_url: The base URL of the analytics API.
    """
    global _collector  # noqa: PLW0603
    if _collector is not None:
        return

    _collector = Collector(token, api_url)
    # Installed first, so that the thread can uninstall it if the account did
    # not opt in, however quickly it hears so.
    set_sink(_collector.collect)
    _collector.start()


def stop() -> None:
    """Stop collecting, and send the events collected so far."""
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
