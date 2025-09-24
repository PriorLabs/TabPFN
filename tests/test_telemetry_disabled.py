"""Test to verify that telemetry is disabled during test execution."""

import os


def test_telemetry_disabled():
    """Verify that TABPFN_DISABLE_TELEMETRY is set to 1 during tests."""
    assert os.environ.get("TABPFN_DISABLE_TELEMETRY") == "1", (
        "TABPFN_DISABLE_TELEMETRY should be set to '1' during test execution. "
        "This ensures telemetry is disabled for all tests."
    )
