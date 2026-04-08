"""Telemetry events for the license acceptance flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tabpfn_common_utils.telemetry import capture_event
from tabpfn_common_utils.telemetry.core.events import (
    BaseTelemetryEvent,
    _get_install_id,
)

logger = logging.getLogger(__name__)


@dataclass
class LicenseFlowEvent(BaseTelemetryEvent):
    """Event emitted at key points of the license acceptance flow.

    Used to build a funnel: started -> success vs error, broken down by
    environment and failure reason.
    """

    outcome: str = ""

    environment: str | None = None

    method: str | None = None

    reason: str | None = None

    install_id: str = field(default_factory=_get_install_id, init=False)

    @property
    def name(self) -> str:  # noqa: D102
        return "license_flow"


def track_license_event(
    outcome: str,
    *,
    environment: str | None = None,
    method: str | None = None,
    reason: str | None = None,
) -> None:
    """Fire a license flow telemetry event, silently ignoring errors."""
    try:
        capture_event(
            LicenseFlowEvent(
                outcome=outcome,
                environment=environment,
                method=method,
                reason=reason,
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to capture license flow event", exc_info=True)
