"""Pytest configuration for TabPFN tests.

This module sets up global test configuration, including disabling telemetry
for all tests to ensure consistent behavior and avoid external dependencies
during testing.
"""

from __future__ import annotations

import os
import random

import pytest

import numpy as np
import torch


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Configure pytest with global settings."""
    # Disable telemetry for all tests to ensure consistent behavior
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


@pytest.fixture(autouse=True, scope="function")  # noqa: PT003
def set_global_seed() -> None:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
