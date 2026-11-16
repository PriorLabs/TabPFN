#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for TorchRobustScaler."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import RobustScaler

from tabpfn.preprocessing.steps.utils import subsample_row_indices
from tabpfn.preprocessing.torch import TorchRobustScaler


def _make_columns(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            rng.lognormal(0, 1, n),
            rng.standard_t(3, n),
            rng.integers(0, 3, n).astype(np.float64),
            np.full(n, 2.0),
        ]
    )
    x[rng.random(x.shape) < 0.1] = np.nan
    return x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test__fit_transform__matches_sklearn_robust_preset(dtype: torch.dtype):
    x_train = _make_columns(500)
    x_test = _make_columns(50, seed=1) * 3
    cpu = RobustScaler(unit_variance=True).fit(x_train)

    scaler = TorchRobustScaler(subsample=None)
    cache = scaler.fit(torch.from_numpy(x_train).to(dtype))
    result = scaler.transform(torch.from_numpy(x_test).to(dtype), cache)

    tol = 1e-5 if dtype == torch.float32 else 1e-10
    np.testing.assert_allclose(cache["center"].numpy(), cpu.center_, rtol=tol)
    np.testing.assert_allclose(cache["scale"].numpy(), cpu.scale_, rtol=tol)
    np.testing.assert_allclose(
        result.numpy(), cpu.transform(x_test), rtol=tol, atol=tol
    )
    assert result.dtype == dtype


def test__fit__subsample__equals_fit_on_selected_rows_and_is_deterministic():
    x = torch.from_numpy(_make_columns(2000)).unsqueeze(1)
    rows = subsample_row_indices(2000, 300, 4)
    assert rows is not None

    cache = TorchRobustScaler(subsample=300, random_state=4).fit(x)
    again = TorchRobustScaler(subsample=300, random_state=4).fit(x)
    expected = TorchRobustScaler(subsample=None).fit(x[torch.from_numpy(rows)])

    for key, value in expected.items():
        torch.testing.assert_close(cache[key], value, rtol=0, atol=0)
        torch.testing.assert_close(again[key], value, rtol=0, atol=0)


def test__fit__subsampled__close_to_full_fit():
    x = torch.from_numpy(_make_columns(50_000))
    full = TorchRobustScaler(subsample=None).fit(x)
    sub = TorchRobustScaler(subsample=10_000, random_state=0).fit(x)
    # Center shift in units of the scale, and relative scale change.
    assert ((sub["center"] - full["center"]) / full["scale"]).abs().max() < 0.05
    assert (sub["scale"] / full["scale"] - 1).abs().max() < 0.05


def test__infinities__ignored_in_fit_and_kept_in_transform():
    x_np = _make_columns(200)
    x_np[[3, 7], 0] = [np.inf, -np.inf]
    x = torch.from_numpy(x_np)
    finite = x.masked_fill(torch.isinf(x), float("nan"))

    scaler = TorchRobustScaler(subsample=None)
    cache = scaler.fit(x)
    result = scaler.transform(x, cache)

    for key, value in scaler.fit(finite).items():
        torch.testing.assert_close(cache[key], value, rtol=0, atol=0)
    assert result[3, 0] == float("inf")
    assert result[7, 0] == float("-inf")
