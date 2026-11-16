#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for TorchSafePowerTransformer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.reshape_feature_distribution_step import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.steps.utils import subsample_row_indices
from tabpfn.preprocessing.torch.pipeline_interface import TorchPreprocessingPipeline
from tabpfn.preprocessing.torch.steps import TorchSelectiveSafePowerStep
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
    x_train[rng.random(x_train.shape) < nan_fraction] = np.nan
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
    x_np[rng.random(x_np.shape) < 0.1] = np.nan
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test__fit_transform__constant_observed_values__lambda_one_and_finite(
    dtype: torch.dtype,
):
    """Near-constant observed values get lambda 1 on CPU and torch."""
    nan, one_up = np.nan, np.nextafter(1.0, 2.0)
    x_train = np.array(
        [
            [1.0, -3.0, nan, 1.0],
            [1.0, -3.0, 5.0, one_up],
            [nan, nan, nan, nan],
            [1.0, -3.0, nan, 1.0],
        ]
    )
    x_test = np.array(
        [[1.0, -3.0, 5.0, 1.0], [2.0, -1.0, 7.0, 2.0], [nan, nan, nan, nan]]
    )

    cpu = get_all_reshape_feature_distribution_preprocessors(4)["safepower"]
    cpu.fit(x_train)
    expected = cpu.transform(x_test.copy())

    torch_pt = TorchSafePowerTransformer()
    cache = torch_pt.fit(torch.from_numpy(x_train).to(dtype))
    result = torch_pt.transform(torch.from_numpy(x_test).to(dtype), fitted_cache=cache)

    np.testing.assert_array_equal(cpu.steps[0][1].lambdas_, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(cache["lambdas"].numpy(), [1.0, 1.0, 1.0, 1.0])
    assert torch.isfinite(result).all()
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test__pipeline__mps_fitted_cache__matches_uncached():
    """The float64 cache survives reuse on MPS, which has no float64."""
    rng = np.random.default_rng(0)
    x = torch.from_numpy(_make_columns(rng, 120)).float().unsqueeze(1).to("mps")
    n_train = 100
    n_cols = x.shape[-1]
    schema = FeatureSchema(
        features=[
            Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL)
            for i in range(n_cols)
        ]
    )
    pipeline = TorchPreprocessingPipeline(
        steps=[(TorchSelectiveSafePowerStep(list(range(n_cols)), 0), None)],
        keep_fitted_cache=True,
    )

    uncached = pipeline(x, schema, num_train_rows=n_train).x
    cached = pipeline(x[n_train:], schema, use_fitted_cache=True).x

    assert cached.device.type == "mps"
    torch.testing.assert_close(cached, uncached[n_train:])


@pytest.mark.parametrize("seed", [0, 7])
def test__fit_transform__subsampled__matches_cpu_safepower(seed: int):
    """With a lambda subsample, CPU and torch fit on the same rows and agree."""
    rng = np.random.default_rng(1)
    x_train = _make_columns(rng, 3000)
    x_train[rng.random(x_train.shape) < 0.1] = np.nan
    x_test = np.concatenate([x_train, 5 * _make_columns(rng, 50)])

    cpu = get_all_reshape_feature_distribution_preprocessors(3000, random_state=seed)[
        "safepower"
    ]
    cpu.set_params(input_transformer__subsample=500)
    cpu.fit(x_train)
    expected = cpu.transform(x_test.copy())

    torch_pt = TorchSafePowerTransformer(subsample=500, random_state=seed)
    cache = torch_pt.fit(torch.from_numpy(x_train))
    result = torch_pt.transform(torch.from_numpy(x_test), fitted_cache=cache)

    np.testing.assert_allclose(
        cache["lambdas"].numpy(), cpu.steps[0][1].lambdas_, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)


def test__fit__subsample__deterministic_and_close_to_full_fit():
    """The subsampled fit depends only on the seed and stays near the full fit."""
    rng = np.random.default_rng(2)
    x_np = _make_columns(rng, 20_000)
    x_np[rng.random(x_np.shape) < 0.05] = np.nan
    x = torch.from_numpy(x_np)

    full_pt = TorchSafePowerTransformer(subsample=None)
    full = full_pt.transform(x, fitted_cache=full_pt.fit(x))
    sub_pt = TorchSafePowerTransformer(subsample=5000, random_state=3)
    cache = sub_pt.fit(x)
    sub = sub_pt.transform(x, fitted_cache=cache)

    for key, value in sub_pt.fit(x).items():
        torch.testing.assert_close(value, cache[key], rtol=0, atol=0)
    other = TorchSafePowerTransformer(subsample=5000, random_state=4).fit(x)
    assert not torch.equal(other["lambdas"], cache["lambdas"])
    # Standardised outputs, so the tolerance is in units of the column's std.
    assert (sub - full).abs().quantile(0.99) < 0.05


def test__fit__subsample_at_least_n_rows__same_as_full_fit():
    rng = np.random.default_rng(3)
    x = torch.from_numpy(_make_columns(rng, 400))
    full = TorchSafePowerTransformer(subsample=None).fit(x)
    same = TorchSafePowerTransformer(subsample=400, random_state=5).fit(x)
    for key, value in full.items():
        torch.testing.assert_close(same[key], value, rtol=0, atol=0)


def test__fit__no_observed_value_in_subsample__lambda_one_and_finite():
    """A column observed only outside the lambda subsample keeps lambda 1."""
    n, k = 1000, 10
    rows = subsample_row_indices(n, k, 0)
    assert rows is not None
    outside = np.setdiff1d(np.arange(n), rows)[:3]
    x_np = np.full((n, 2), np.nan)
    x_np[:, 0] = np.random.default_rng(0).lognormal(size=n)
    x_np[outside, 1] = [1.0, 2.0, 10.0]

    cpu = get_all_reshape_feature_distribution_preprocessors(n, random_state=0)[
        "safepower"
    ]
    cpu.set_params(input_transformer__subsample=k)
    expected = cpu.fit_transform(x_np.copy())
    torch_pt = TorchSafePowerTransformer(subsample=k, random_state=0)
    cache = torch_pt.fit(torch.from_numpy(x_np))
    result = torch_pt.transform(torch.from_numpy(x_np), fitted_cache=cache)

    assert cpu.steps[0][1].lambdas_[1] == 1.0
    assert cache["lambdas"][1] == 1.0
    assert np.isfinite(expected).all()
    assert torch.isfinite(result).all()
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)
