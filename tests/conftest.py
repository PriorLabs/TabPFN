#  Copyright (c) Prior Labs GmbH 2026.

"""Pytest configuration for TabPFN tests."""

from __future__ import annotations

import gc
import os
import random
from collections.abc import Generator

import numpy as np
import pytest
import torch

from tabpfn.constants import ModelVersion
from tabpfn.settings import settings


@pytest.fixture(autouse=True, scope="session")
def default_to_fast_model() -> None:
    """Run tests that do not pin a version on the fast TabPFN-3.5 checkpoint.

    The full 3.5 checkpoint is roughly four times the size of TabPFN-3 and made a
    default fit-and-predict cycle 3 to 6 times slower in CI. Most tests only need
    *a* model; the ones that check the full model pin ``ModelVersion.V3_5`` and are
    unaffected. Set ``TABPFN_MODEL_VERSION`` explicitly to run everything on
    another version, e.g. ``TABPFN_MODEL_VERSION=v3.5 pytest``.
    """
    if "TABPFN_MODEL_VERSION" not in os.environ:
        settings.tabpfn.model_version = ModelVersion.V3_5_FAST


@pytest.fixture(autouse=True, scope="session")
def freeze_import_time_objects() -> None:
    """Move everything alive after collection out of the garbage collector's reach.

    ``release_mps_memory`` runs a full ``gc.collect()`` after every test. By the
    time the first test runs, tabpfn, torch and sklearn have put a few hundred
    thousand long-lived objects on the collector's lists, and scanning them on
    every test adds up to minutes on the macOS CI runners. Freezing them once
    makes each later collection scan only the objects that tests created.
    """
    gc.collect()
    gc.freeze()


@pytest.fixture(autouse=True, scope="function")  # noqa: PT003
def set_global_seed() -> None:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)


@pytest.fixture(autouse=True)
def release_mps_memory() -> Generator[None]:
    """Release cached MPS memory after each test.

    PyTorch's MPS caching allocator holds freed memory for the lifetime of the
    process. On the ~7GB macos-latest CI runners the cache accumulates across
    tests until the ~3.3 GiB MPS limit is hit and unrelated tests OOM.
    gc.collect() first so tensors kept alive by reference cycles (e.g.
    unittest.mock call records) are actually freed before emptying the cache.
    """
    yield
    if torch.backends.mps.is_available():
        gc.collect()
        torch.mps.empty_cache()
