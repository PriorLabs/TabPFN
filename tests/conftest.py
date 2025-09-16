# conftest.py
from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def set_global_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
    # Optional: force deterministic behavior where applicable
    torch.use_deterministic_algorithms(mode=True)
