#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for TorchSafePowerTransformer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.preprocessing.steps.reshape_feature_distribution_step import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.torch.torch_safe_power_transformer import (
    TorchSafePowerTransformer,
)


def _make_columns(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.column_stack(
        [
            rng.standard_normal(n),
            rng.lognormal(0, 1, n),
            -rng.lognormal(0, 1.5, n),
            100 * rng.exponential(1, n),
            1e4 * rng.standard_normal(n),
            rng.uniform(-1, 1, n) ** 3,
            rng.integers(0, 3, n).astype(np.float64),
            np.full(n, 3.0),
            np.zeros(n),
        ]
    )


@pytest.mark.parametrize("n_samples", [5, 100, 1000])
@pytest.mark.parametrize("nan_fraction", [0.0, 0.1])
def test__fit_transform__matches_cpu_safepower(n_samples: int, nan_fraction: float):
    """Lambdas and outputs match the CPU ``safepower`` preset."""
    rng = np.random.default_rng(0)
    x_train = _make_columns(rng, n_samples)
    # NaNs in a constant column leave the CPU lambda search on a flat,
    # rounding-noise objective, which no implementation reproduces.
    nan_mask = rng.random(x_train.shape) < nan_fraction
    nan_mask[:, 7:] = False
    x_train[nan_mask] = np.nan
    x_test = np.concatenate([x_train, 5 * _make_columns(rng, 50)])

    cpu = get_all_reshape_feature_distribution_preprocessors(n_samples)["safepower"]
    cpu.fit(x_train)
    expected = cpu.transform(x_test.copy())

    torch_pt = TorchSafePowerTransformer()
    cache = torch_pt.fit(torch.from_numpy(x_train))
    result = torch_pt.transform(torch.from_numpy(x_test), fitted_cache=cache)

    np.testing.assert_allclose(
        cache["lambdas"].numpy(), cpu.steps[0][1].lambdas_, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)


def test__transform__keeps_shape_and_dtype_of_3d_input():
    x = torch.randn(40, 2, 3, dtype=torch.float32)
    torch_pt = TorchSafePowerTransformer()
    result = torch_pt.transform(x, fitted_cache=torch_pt.fit(x))
    assert result.shape == x.shape
    assert result.dtype == torch.float32


def test__fit_transform__chunking_matches_single_chunk(monkeypatch):
    """Column-chunked fit and row-chunked transform match one chunk."""
    rng = np.random.default_rng(0)
    x_np = _make_columns(rng, 300)
    x_np[:, :7][rng.random((300, 7)) < 0.1] = np.nan
    x = torch.from_numpy(x_np)

    torch_pt = TorchSafePowerTransformer()
    cache = torch_pt.fit(x)
    expected = torch_pt.transform(x, fitted_cache=cache)

    monkeypatch.setattr(
        "tabpfn.preprocessing.torch.torch_safe_power_transformer._CHUNK_TARGET_BYTES",
        1,
    )
    chunked_cache = torch_pt.fit(x)
    chunked = torch_pt.transform(x, fitted_cache=chunked_cache)

    # Reductions over chunks of different widths sum in a different order, so
    # the lambda search may stop at a different point within its tolerance.
    for key, value in cache.items():
        torch.testing.assert_close(chunked_cache[key], value, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(chunked, expected, rtol=1e-6, atol=1e-6)
