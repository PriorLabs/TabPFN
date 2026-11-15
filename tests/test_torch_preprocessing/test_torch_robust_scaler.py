#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for TorchRobustScaler."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import tabpfn.preprocessing.torch.torch_robust_scaler as module
from tabpfn.preprocessing.steps.reshape_feature_distribution_step import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.torch.torch_robust_scaler import TorchRobustScaler


def _make_columns(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.column_stack(
        [
            rng.standard_normal(n),
            rng.lognormal(0, 1, n),
            1e4 * rng.standard_normal(n),
            rng.integers(0, 3, n).astype(np.float64),
            np.full(n, 3.0),
            np.zeros(n),
        ]
    )


@pytest.mark.parametrize("n_samples", [1, 5, 1000])
@pytest.mark.parametrize("nan_fraction", [0.0, 0.1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test__fit_transform__matches_cpu_robust(
    n_samples: int, nan_fraction: float, dtype: torch.dtype
):
    """Centers, scales and outputs match the CPU ``robust`` preset."""
    rng = np.random.default_rng(0)
    x_train = _make_columns(rng, n_samples)
    x_train[rng.random(x_train.shape) < nan_fraction] = np.nan
    x_test = np.concatenate([x_train, 5 * _make_columns(rng, 50)])
    x_test[rng.random(x_test.shape) < nan_fraction] = np.nan

    cpu = get_all_reshape_feature_distribution_preprocessors(n_samples)["robust"]
    cpu.fit(x_train)
    expected = cpu.transform(x_test.copy())

    scaler = TorchRobustScaler()
    cache = scaler.fit(torch.from_numpy(x_train).to(dtype))
    result = scaler.transform(torch.from_numpy(x_test).to(dtype), fitted_cache=cache)

    rtol = 1e-12 if dtype == torch.float64 else 1e-5
    np.testing.assert_allclose(cache["center"].double(), cpu.center_, rtol=rtol)
    np.testing.assert_allclose(cache["scale"].double(), cpu.scale_, rtol=rtol)
    np.testing.assert_allclose(
        result.double(), expected, rtol=rtol, atol=rtol, equal_nan=True
    )


def test__fit__column_blocks__match_one_pass(monkeypatch: pytest.MonkeyPatch):
    """Fitting in column blocks gives the statistics of a single pass."""
    x = torch.from_numpy(_make_columns(np.random.default_rng(1), 500)).float()
    one_pass = TorchRobustScaler().fit(x)
    monkeypatch.setattr(module, "_FIT_BLOCK_BYTES", x.shape[0] * x.element_size())
    blocked = TorchRobustScaler().fit(x)
    for key in one_pass:
        assert torch.equal(one_pass[key], blocked[key])
