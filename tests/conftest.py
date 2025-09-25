"""Pytest configuration for TabPFN tests.

This module sets up global test configuration, including disabling telemetry
for all tests to ensure consistent behavior and avoid external dependencies
during testing.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Configure pytest with global settings."""
    # Disable telemetry for all tests to ensure consistent behavior
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
