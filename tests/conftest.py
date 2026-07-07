#  Copyright (c) Prior Labs GmbH 2026.

"""Pytest configuration for TabPFN tests."""

from __future__ import annotations

import gc
import random
from collections.abc import Generator

import numpy as np
import pytest
import torch


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
