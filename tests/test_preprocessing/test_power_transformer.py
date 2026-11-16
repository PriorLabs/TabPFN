#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import warnings

import numpy as np
from sklearn.preprocessing import PowerTransformer

from tabpfn.preprocessing.steps import SafePowerTransformer
from tabpfn.preprocessing.steps.safe_power_transformer import (
    _yeojohnson_inverse_transform,
)


def test__safe_power_transformer__normal_cases__same_results_as_power_transformer():
    """Test SafePowerTransformer returns the same results as PowerTransformer for normal data."""  # noqa: E501
    rng = np.random.default_rng(42)

    # Test cases with normal, well-behaved data
    test_cases = [
        # Normal distribution
        rng.normal(0, 1, (100, 3)),
        # Slightly skewed data
        rng.exponential(1, (100, 2)),
        # Uniform data
        rng.uniform(-2, 2, (100, 4)),
        # Mixed positive/negative
        np.concatenate([rng.normal(5, 2, (50, 2)), rng.normal(-3, 1, (50, 2))]),
    ]

    for i, X in enumerate(test_cases):
        # Fit both transformers
        sklearn_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        safe_transformer = SafePowerTransformer(method="yeo-johnson", standardize=False)

        sklearn_result = sklearn_transformer.fit_transform(X)
        safe_result = safe_transformer.fit_transform(X)

        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_allclose(
            sklearn_result,
            safe_result,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"Results differ for test case {i}",
        )

        # Lambdas should also be close
        np.testing.assert_allclose(
            sklearn_transformer.lambdas_,
            safe_transformer.lambdas_,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"Lambdas differ for test case {i}",
        )


def test__safe_power_transformer__power_transformer_fails__no_error():
    """Test that SafePowerTransformer handles cases where PowerTransformer overflows."""
    # this input produces scipy.optimize._optimize.BracketError
    # with sklearn's PowerTransformer with sklearn==1.6.1
    # and scipy==1.15.2
    X = np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0]).reshape(-1, 1)

    safe_transformer = SafePowerTransformer(method="yeo-johnson", standardize=False)

    # Check SafePowerTransformer produces no warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        safe_result = safe_transformer.fit_transform(X)

    # Assert no warnings from SafePowerTransformer
    assert len(w) == 0, (
        f"SafePowerTransformer produced {len(w)} warning(s): "
        f"{[str(warning.message) for warning in w]}"
    )

    # check if result contains nan or inf
    assert np.all(np.isfinite(safe_result)), (
        "SafePowerTransformer produced non-finite values"
    )


def test__safe_power_transformer__inverse_transform_float32__no_overflow_warning():
    """Test that inverse_transform on float32 data does not emit overflow warnings."""
    # Use a small lambda to amplify the inverse, pushing float64 intermediates
    # beyond float32 range — the exact scenario that previously produced
    # "RuntimeWarning: overflow encountered in cast".
    lmbda = 0.01
    transformer = SafePowerTransformer(method="yeo-johnson", standardize=False)
    transformer.lambdas_ = np.array([lmbda])
    transformer._scaler = None
    transformer.n_features_in_ = 1

    # Forward-transform a large value so the inverse must reconstruct it
    x_large = np.array([[1e30]], dtype=np.float32)
    x_transformed = transformer.transform(x_large)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x_inv = transformer.inverse_transform(x_transformed)

    overflow_warnings = [x for x in w if "overflow" in str(x.message).lower()]
    assert len(overflow_warnings) == 0, (
        f"Unexpected overflow warning(s): {[str(x.message) for x in overflow_warnings]}"
    )
    assert x_inv.dtype == np.float32
    assert np.all(np.isfinite(x_inv))


def test__yeojohnson_inverse_transform__float32_boundary__finite_no_overflow():
    """Float32 boundary inputs must clip safely without overflow."""
    float32_max = np.finfo(np.float32).max

    cases = [
        (float32_max, 0.0),  # x >= 0, lambda == 0
        (float32_max, 0.5),  # x >= 0, lambda != 0
        (-float32_max, 1.5),  # x < 0, lambda != 2
        (-float32_max, 2.0),  # x < 0, lambda == 2
    ]

    for value, lmbda in cases:
        x = np.array([value], dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = _yeojohnson_inverse_transform(x, lmbda)

        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))


def test__safe_power_transformer__transform_then_inverse_transform__returns_original():
    """Test that SafePowerTransformer inverse_transform returns data to original scale."""  # noqa: E501
    rng = np.random.default_rng(42)

    # Test cases with different data distributions
    test_cases = [
        # Normal distribution
        rng.normal(0, 1, (100, 3)),
        # Slightly skewed data
        rng.exponential(1, (100, 2)),
        # Uniform data
        rng.uniform(-2, 2, (100, 4)),
        # Mixed positive/negative
        np.concatenate([rng.normal(5, 2, (50, 2)), rng.normal(-3, 1, (50, 2))]),
    ]

    for i, X_original in enumerate(test_cases):
        # Create and fit transformer
        transformer = SafePowerTransformer(method="yeo-johnson", standardize=False)

        # Transform data
        X_transformed = transformer.fit_transform(X_original)

        # Inverse transform back to original scale
        X_inverse = transformer.inverse_transform(X_transformed)

        # Assert that inverse transform returns data close to original
        np.testing.assert_allclose(
            X_original,
            X_inverse,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"Inverse transform failed for test case {i}",
        )


def _skewed_columns(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.column_stack(
        [
            rng.lognormal(0, 1, n),
            -rng.gamma(2, 2, n),
            rng.poisson(3, n).astype(np.float64),
            rng.normal(5, 2, n),
        ]
    )
    x[rng.random(x.shape) < 0.05] = np.nan
    return x


def test__safe_power_transformer__subsampled_lambdas__deterministic_and_close():
    """Lambdas from a seeded subsample repeat exactly and stay near the full fit."""
    x = _skewed_columns(20_000)

    full = SafePowerTransformer(standardize=True, subsample=None).fit(x)
    sub = SafePowerTransformer(standardize=True, subsample=5000, random_state=1)
    sub_out = sub.fit_transform(x.copy())
    again = SafePowerTransformer(standardize=True, subsample=5000, random_state=1)
    other = SafePowerTransformer(standardize=True, subsample=5000, random_state=2)

    np.testing.assert_array_equal(again.fit(x).lambdas_, sub.lambdas_)
    assert not np.array_equal(other.fit(x).lambdas_, sub.lambdas_)
    np.testing.assert_allclose(sub.lambdas_, full.lambdas_, atol=0.1)
    delta = np.abs(sub_out - full.transform(x.copy()))
    assert np.nanquantile(delta, 0.99) < 0.05


def test__safe_power_transformer__subsample_at_least_n_rows__same_as_full_fit():
    x = _skewed_columns(500)
    full = SafePowerTransformer(subsample=None).fit(x)
    same = SafePowerTransformer(subsample=500, random_state=3).fit(x)
    np.testing.assert_array_equal(same.lambdas_, full.lambdas_)


def test__safe_power_transformer__all_nan_column__lambda_one():
    x = _skewed_columns(200)
    x[:, 1] = np.nan
    transformer = SafePowerTransformer(standardize=False).fit(x)
    assert transformer.lambdas_[1] == 1.0
    assert np.isnan(transformer.transform(x.copy())[:, 1]).all()
