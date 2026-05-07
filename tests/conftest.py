"""Pytest configuration for TabPFN tests.

This module sets up global test configuration, including disabling telemetry
for all tests to ensure consistent behavior and avoid external dependencies
during testing.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from tabpfn.model_loading import ModelSource, get_cache_dir


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


def _is_v3_classifier_in_cache() -> bool:
    cache_dir = get_cache_dir()
    return (cache_dir / ModelSource.get_classifier_v3().default_filename).exists()


def _is_v3_regressor_in_cache() -> bool:
    cache_dir = get_cache_dir()
    return (cache_dir / ModelSource.get_regressor_v3().default_filename).exists()


@pytest.fixture
def skip_if_v3_classifier_unavailable() -> None:
    """Skip the test when the V3 classifier model is not in the local cache."""
    if not _is_v3_classifier_in_cache():
        pytest.skip("V3 classifier model not in cache; skipping V3-specific test.")


@pytest.fixture
def skip_if_v3_regressor_unavailable() -> None:
    """Skip the test when the V3 regressor model is not in the local cache."""
    if not _is_v3_regressor_in_cache():
        pytest.skip("V3 regressor model not in cache; skipping V3-specific test.")
