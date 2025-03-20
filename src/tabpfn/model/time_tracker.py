#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import threading
import time
import warnings

from tabpfn.constants import TIME_WARNING_CPU


class TimeUsageTracker:
    """Tracks time usage for code segments.

    Supports starting, stopping, and resetting segments. It will
    warn if any segment runs longer than the threshold defined by
    `TIME_WARNING_CPU` (default: 10 minutes). Note that resetting an
    active segment issues a warning.

    Attributes:
        active_segments (dict[str, float]):
            Maps segment labels to their start times.

        completed_segments (list[tuple[str, float]]):
            Records completed segments with their elapsed times.

        _timers (dict[str, threading.Timer]):
            Timers that trigger warnings when a segment exceeds the threshold.

    Raises:
        RuntimeError: If attempting to start a segment that's
                      already active or stop one that isn't active.
    """

    def __init__(self) -> None:
        self.active_segments: dict[str, float] = {}
        self.completed_segments: list[tuple[str, float]] = []
        self._timers: dict[str, threading.Timer] = {}

    def _warn_if_over_limit(self, label: str, start_time: float) -> None:
        # Verify the segment is still active and its start time hasn't changed.
        if label in self.active_segments and self.active_segments[label] == start_time:
            warnings.warn(
                f"{label} is taking > 10 minutes to run. "
                "Use GPU for faster processing, or if unavailable, "
                "try tabpfn-client API https://github.com/PriorLabs/tabpfn-client",
                UserWarning,
                stacklevel=2,
            )

    def _start_timer(self, label: str, start_time: float) -> None:
        # Schedule a timer to check the segment after the limit
        timer = threading.Timer(
            TIME_WARNING_CPU, self._warn_if_over_limit, args=(label, start_time)
        )
        timer.daemon = True  # Ensure timer doesn't block program exit
        timer.start()
        self._timers[label] = timer

    def start(self, label: str) -> None:
        """Start a new segment with the given label.

        Raises:
            RuntimeError: If a segment with this label is already active.
        """
        if label in self.active_segments:
            raise RuntimeError(f"Segment '{label}' is already active.")
        start_time = time.time()
        self.active_segments[label] = start_time
        self._start_timer(label, start_time)

    def stop(self, label: str) -> None:
        """Stop the segment with the given label.

        Raises:
            RuntimeError: If no active segment exists with the given label.
        """
        if label not in self.active_segments:
            raise RuntimeError(f"No active segment found with label '{label}'.")

        start_time = self.active_segments.pop(label)
        # Cancel the timer for this segment, if it exists
        if label in self._timers:
            timer = self._timers.pop(label)
            timer.cancel()
        self.completed_segments.append((label, time.time() - start_time))

    def reset(self, label: str) -> None:
        """Reset an active segment with the given label.

        Issues a warning if resetting an active segment.
        """
        if label in self.active_segments:
            warnings.warn(f"Resetting {label} time segment.", stacklevel=2)
            # Cancel the current timer, if it exists
            if label in self._timers:
                self._timers[label].cancel()

            # Restart the timer with the new start time
            start_time = time.time()
            self.active_segments[label] = start_time
            self._start_timer(label, start_time)

    def total_time(self, labels: list[str] | None = None) -> float:
        """Return the total elapsed time for segments with the given labels.
        If no labels are provided, returns the total for all completed segments.
        """
        if labels is None:
            return sum(duration for _, duration in self.completed_segments)
        return sum(
            duration for seg, duration in self.completed_segments if seg in labels
        )
