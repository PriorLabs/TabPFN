"""Telemetry helpers for TabPFN-owned call paths."""

from __future__ import annotations

import contextlib
import contextvars
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from filelock import FileLock, Timeout
from tabpfn_common_utils.telemetry.interactive import capture_session, ping

_user_config_dir: Callable[[str, str], str] | None
try:
    from platformdirs import user_config_dir as _platformdirs_user_config_dir
except ImportError:  # pragma: no cover - platformdirs is a telemetry dependency
    _user_config_dir = None
else:
    _user_config_dir = _platformdirs_user_config_dir

_SUPPRESS_TELEMETRY = contextvars.ContextVar(
    "tabpfn_suppress_telemetry",
    default=False,
)
_INITIALIZE_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_SESSION_CAPTURED = False


def _telemetry_state_path() -> Path | None:
    """Return the common telemetry state path when it can be resolved."""
    if p := os.getenv("TABPFN_STATE_PATH"):
        return Path(p).expanduser()
    if d := os.getenv("TABPFN_STATE_DIR"):
        return Path(d).expanduser() / "state.json"
    if _user_config_dir is None:
        return None
    return Path(_user_config_dir(".tabpfn", "priorlabs")) / "state.json"


def _ping_with_lock() -> None:
    """Run ping under a cross-process lock when the telemetry state path exists."""
    state_path = _telemetry_state_path()
    if state_path is None:
        ping()
        return

    lock = FileLock(state_path.with_suffix(".ping.lock"), timeout=10)
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        lock.acquire()
    except (OSError, Timeout):
        ping()
        return

    try:
        ping()
    finally:
        with contextlib.suppress(Exception):
            lock.release()


def is_telemetry_suppressed() -> bool:
    """Return whether telemetry is suppressed for the current context."""
    return _SUPPRESS_TELEMETRY.get()


@contextlib.contextmanager
def suppress_telemetry() -> Iterator[None]:
    """Suppress telemetry emitted by TabPFN-owned internal calls."""
    token = _SUPPRESS_TELEMETRY.set(True)
    try:
        yield
    finally:
        _SUPPRESS_TELEMETRY.reset(token)


def initialize_telemetry() -> None:
    """Initialize telemetry and acknowledge an anonymous process session.

    If user opted out of telemetry using `TABPFN_DISABLE_TELEMETRY`, the
    downstream telemetry package does not emit events.
    """
    if is_telemetry_suppressed():
        return

    global _TELEMETRY_SESSION_CAPTURED  # noqa: PLW0603
    with _INITIALIZE_TELEMETRY_LOCK:
        _ping_with_lock()
        if not _TELEMETRY_SESSION_CAPTURED:
            capture_session()
            _TELEMETRY_SESSION_CAPTURED = True
