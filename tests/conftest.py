from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True, scope="function")  # noqa: PT003
def set_global_seed() -> None:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
