"""Telemetry for usage analytics.

Telemetry is opt-out and only collects aggregate usage statistics — never
any data passed to or returned from the model (inputs, outputs, or features).

To disable, set the environment variable::

    TABPFN_DISABLE_TELEMETRY=1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import jwt
from posthog import Posthog
from tabpfn_common_utils.telemetry.core import (
    decorators as base_decorators,
    service as telemetry_service,
)
from tabpfn_common_utils.telemetry.core.config import download_config
from tabpfn_common_utils.telemetry.core.decorators import (
    set_init_params,
    set_model_config,
    track_model_call,
)
from tabpfn_common_utils.telemetry.core.events import (
    BaseTelemetryEvent,
    ModelLoadEvent,
    PingEvent,
    SessionEvent,
)
from tabpfn_common_utils.telemetry.core.service import ProductTelemetry
from tabpfn_common_utils.telemetry.interactive import (
    capture_session,
    flows as base_flows,
    ping,
)

from tabpfn.browser_auth import get_cached_token as get_cached_auth_token

__all__ = ["init", "set_init_params", "set_model_config", "track_model_call"]

_cached_user_id: str | None = None


@dataclass
class _TelemetryConfig:
    project_token: str
    api_host: str


def init() -> None:
    """Initialize telemetry and acknowledge anonymous session."""
    for func in (
        _patch_client,
        ping,
        capture_session,
    ):
        func()


def _patch_client() -> None:
    """Patch the telemetry client with the custom configuration."""
    config = _get_telemetry_config()
    instance = telemetry_service.ProductTelemetry()

    if config is None:
        # This will block the telemetry service from sending events
        instance._posthog_client = None
        return

    # Patch the telemetry client with the custom configuration
    instance._posthog_client = Posthog(
        project_api_key=config.project_token,
        host=config.api_host,
        disable_geoip=True,
        enable_exception_autocapture=False,
        max_queue_size=10,
        flush_at=10,
    )


def _get_telemetry_config() -> _TelemetryConfig | None:
    """Load the telemetry configuration. Information we fetch include
    the public authentication token and the API host.

    We do not cache the configuration in memory because download_config()
    is already a cached function with a TTL of 1 hour.

    Returns:
        The telemetry configuration.
    """
    config = download_config()
    if config is None:
        return None

    project_token = config.get("project_token")
    api_host = config.get("api_host")

    # Silently ignore if the configuration is not complete
    if any(v is None for v in [project_token, api_host]):
        return None

    return _TelemetryConfig(
        project_token=cast("str", project_token), api_host=cast("str", api_host)
    )


def _get_user_id() -> str | None:
    global _cached_user_id  # noqa: PLW0603
    if _cached_user_id is not None:
        return _cached_user_id

    token = get_cached_auth_token()
    if token is None:
        return None

    try:
        payload = jwt.decode(token, options={"verify_signature": False})
    except Exception:  # noqa: BLE001
        return None

    user = payload.get("user")
    if user is not None:
        _cached_user_id = user

    return user


def _capture_event_with_user_id(
    event: BaseTelemetryEvent, properties: dict[str, Any] | None = None
) -> None:
    """Capture an event with the user ID.

    Args:
        event: The event to capture.
        properties: The properties to capture with the event.
    """
    config = _get_telemetry_config()
    if config is None:
        return

    user_id = _get_user_id()

    # We passthrough the session and ping events anonymously.
    # These events still contain anonymous and valuable runtime metadata.
    passthrough_events = (
        SessionEvent,
        PingEvent,
        ModelLoadEvent,
    )
    if user_id is None and not isinstance(event, passthrough_events):
        return

    kwargs: dict[str, Any] = {
        "properties": properties,
    }

    # Unless the user is authenticated, we capture the event anonymously
    if user_id is not None:
        kwargs["distinct_id"] = user_id

    service = ProductTelemetry(api_key=config.project_token)
    service.capture(event, **kwargs)


# Replace the capture_event reference that _send_model_called_event holds.
# The capture_event_with_user_id function is a wrapper around the
# base_decorators.capture_event function so that we can capture the event with
# the user ID if user is authenticated. Otherwise, the event is captured anonymously.
base_decorators.capture_event = _capture_event_with_user_id
base_flows.capture_event = _capture_event_with_user_id
