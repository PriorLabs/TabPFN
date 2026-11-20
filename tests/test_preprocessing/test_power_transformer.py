#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.preprocessing import PowerTransformer

from tabpfn.preprocessing.steps import SafePowerTransformer
from tabpfn.preprocessing.steps.safe_power_transformer import (
    _yeojohnson_centered_inverse_transform,
    _yeojohnson_centered_transform,
    _yeojohnson_inverse_transform,
    _yeojohnson_transform,
)

FAR_FROM_ZERO_COLUMNS = {
    "years": lambda rng, n: rng.integers(1990, 2021, n).astype(np.float64),
    "offset_1e3_spread_1": lambda rng, n: 1000 + rng.standard_normal(n),
    "offset_1e4_spread_10": lambda rng, n: 1e4 + 10 * rng.standard_normal(n),
    "negative_offset_1e3_spread_1": lambda rng, n: -1000 + rng.standard_normal(n),
}
"""Columns that are far from zero compared with their spread, such as years."""


def make_far_from_zero_columns(
    kind: str, n_samples: int, n_seeds: int = 50
) -> np.ndarray:
    """Draw one column of the given kind per seed, as an [n_samples, n_seeds] array."""
    draw = FAR_FROM_ZERO_COLUMNS[kind]
    return np.column_stack(
        [draw(np.random.default_rng(seed), n_samples) for seed in range(n_seeds)]
    )


def make_offset_1e12_columns(n_samples: int) -> np.ndarray:
    """A positive and a negative column with spread 1 at an offset of 1e12.

    Neighbouring values differ in the last few digits that float64 can hold.
    """
    values = 1e12 + np.random.default_rng(0).standard_normal(n_samples)
    return np.column_stack([values, -values])


def make_wide_range_columns(n_samples: int) -> np.ndarray:
    """A positive and a negative column that span many orders of magnitude.

    A few small values next to values around 1e18, such as nanosecond timestamps
    in a column that also holds zeros.
    """
    values = 1e18 * np.random.default_rng(0).lognormal(0, 2, n_samples)
    values[:3] = [0.0, 0.5, 2.0]
    return np.column_stack([values, -values])


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


def _standardized_yeo_johnson_in_exact_arithmetic(
    x: np.ndarray, lmbda: float
) -> np.ndarray:
    """Standardized Yeo-Johnson transform of a column, in 200-digit arithmetic."""
    with localcontext() as context:
        context.prec = 200
        lam = Decimal(float(lmbda))
        y = [
            ((1 + v) ** lam - 1) / lam
            if v >= 0
            else -((1 - v) ** (2 - lam) - 1) / (2 - lam)
            for v in map(Decimal, x.tolist())
        ]
        mean = sum(y, Decimal(0)) / len(y)
        std = (sum(((v - mean) ** 2 for v in y), Decimal(0)) / len(y)).sqrt()
        return np.array([float((v - mean) / std) for v in y])


def _standardized(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std()


def _make_ordinary_columns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.column_stack(
        [
            rng.normal(0, 1, 200),
            rng.exponential(1, 200),
            rng.uniform(-2, 2, 200),
            -rng.lognormal(0, 1, 200),
            np.concatenate([rng.normal(5, 2, 100), rng.normal(-3, 1, 100)]),
        ]
    )


@pytest.mark.parametrize("kind", list(FAR_FROM_ZERO_COLUMNS))
@pytest.mark.parametrize("n_samples", [100, 1000])
def test__safe_power_transformer__far_from_zero__equals_exact_yeo_johnson(
    kind: str, n_samples: int
):
    """Year-like columns get the Yeo-Johnson transform that exact arithmetic gives.

    The classic computation loses the values of such a column to rounding, and
    its lambda search optimises the rounding noise.
    """
    X = make_far_from_zero_columns(kind, n_samples, n_seeds=5)

    transformer = SafePowerTransformer(relative_to_geometric_mean=True)
    result = transformer.fit_transform(X.copy())

    expected = np.column_stack(
        [
            _standardized_yeo_johnson_in_exact_arithmetic(X[:, i], lmbda)
            for i, lmbda in enumerate(transformer.lambdas_)
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test__safe_power_transformer__offset_1e12__equals_exact_yeo_johnson():
    """Values that differ only in their last digits keep full accuracy."""
    X = make_offset_1e12_columns(200)

    transformer = SafePowerTransformer(relative_to_geometric_mean=True)
    result = transformer.fit_transform(X.copy())

    expected = np.column_stack(
        [
            _standardized_yeo_johnson_in_exact_arithmetic(X[:, i], lmbda)
            for i, lmbda in enumerate(transformer.lambdas_)
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test__safe_power_transformer__relative_to_geometric_mean__same_as_sklearn():
    """The option changes neither the lambdas nor the standardized output."""
    X = _make_ordinary_columns()

    sklearn_transformer = PowerTransformer(standardize=True)
    transformer = SafePowerTransformer(relative_to_geometric_mean=True)
    expected = sklearn_transformer.fit_transform(X)
    result = transformer.fit_transform(X)

    np.testing.assert_allclose(
        transformer.lambdas_, sklearn_transformer.lambdas_, rtol=1e-6
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        transformer.transform(X + 0.5),
        sklearn_transformer.transform(X + 0.5),
        rtol=1e-6,
        atol=1e-7,
    )


def test__safe_power_transformer__wide_range_columns__same_as_classic():
    """Values far below the geometric mean keep the accuracy of the classic form."""
    X = make_wide_range_columns(200)

    classic = SafePowerTransformer()
    transformer = SafePowerTransformer(relative_to_geometric_mean=True)
    expected = classic.fit_transform(X.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = transformer.fit_transform(X.copy())

    np.testing.assert_allclose(transformer.lambdas_, classic.lambdas_, rtol=1e-5)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("standardize", [True, False])
def test__safe_power_transformer__relative_to_geometric_mean__fit_then_transform(
    dtype: type[np.floating], standardize: bool
):
    """``fit`` leaves its input alone and agrees with ``fit_transform``."""
    X = _make_ordinary_columns().astype(dtype)
    X_before = X.copy()

    transformer = SafePowerTransformer(
        standardize=standardize, copy=False, relative_to_geometric_mean=True
    ).fit(X)
    expected = clone(transformer).fit_transform(X.copy())
    result = transformer.transform(X.copy())

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == dtype


def test__safe_power_transformer__constant_and_empty_columns__left_unchanged():
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [np.full(50, 2000.0), np.full(50, np.nan), 2000 + rng.standard_normal(50)]
    )

    transformer = SafePowerTransformer(
        standardize=False, relative_to_geometric_mean=True
    )
    result = transformer.fit_transform(X.copy())

    assert transformer.lambdas_[0] == 1.0
    assert np.isnan(transformer.lambdas_[1])
    np.testing.assert_array_equal(result[:, :2], X[:, :2])
    assert np.isfinite(result[:, 2]).all()


@pytest.mark.parametrize("standardize", [True, False])
def test__safe_power_transformer__relative_to_geometric_mean__inverse_transform(
    standardize: bool,
):
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [
            rng.integers(1990, 2021, 200).astype(np.float64),
            -119.5 + 2 * rng.standard_normal(200),
            15 + rng.lognormal(0, 1, 200),
            rng.lognormal(0, 1, 200),
            rng.standard_normal(200),
            np.where(rng.random(200) < 0.2, np.nan, 2000 + rng.standard_normal(200)),
        ]
    )
    transformer = SafePowerTransformer(
        standardize=standardize, relative_to_geometric_mean=True
    )

    X_transformed = transformer.fit_transform(X.copy())
    X_test = 1.01 * X
    X_inverse = transformer.inverse_transform(transformer.transform(X_test.copy()))

    np.testing.assert_allclose(
        transformer.inverse_transform(X_transformed), X, rtol=1e-9
    )
    np.testing.assert_allclose(X_inverse, X_test, rtol=1e-9)


@pytest.mark.parametrize("log_mean", [0.7, -0.7])
@pytest.mark.parametrize(
    ("value", "lmbda"),
    [
        (np.finfo(np.float32).max, 0.0),
        (np.finfo(np.float32).max, 0.5),
        (-np.finfo(np.float32).max, 1.5),
        (-np.finfo(np.float32).max, 2.0),
    ],
)
def test__safe_power_transformer__relative_to_geometric_mean__inverse_float32_boundary(
    value: float, lmbda: float, log_mean: float
):
    """Extreme float32 values invert to finite values without an overflow warning.

    The same cases as for the classic inverse above.
    """
    transformer = SafePowerTransformer(
        standardize=False, relative_to_geometric_mean=True
    )
    transformer.lambdas_ = np.array([lmbda])
    transformer.log_means_ = np.array([log_mean])
    transformer.n_features_in_ = 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = transformer.inverse_transform(np.array([[value]], dtype=np.float32))

    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("lmbda", [0.0, 2.0, 0.5])
@pytest.mark.parametrize("log_mean", [0.7, -0.7])
def test__yeojohnson_centered_transform__same_as_classic_once_standardized(
    lmbda: float, log_mean: float
):
    """The transform is the classic one up to a shift and a scale.

    Checked on both sides of zero, and in the log limits lambda = 0 and lambda = 2.
    """
    x = np.linspace(-3.0, 3.0, 25)

    result = _yeojohnson_centered_transform(x, lmbda, log_mean)

    np.testing.assert_allclose(
        _standardized(result),
        _standardized(_yeojohnson_transform(x, lmbda)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("lmbda", [0.0, 2.0, 0.5])
@pytest.mark.parametrize("log_mean", [0.7, -0.7])
def test__yeojohnson_centered_inverse_transform__returns_input(
    lmbda: float, log_mean: float
):
    x = np.linspace(-3.0, 3.0, 25)
    transformed = _yeojohnson_centered_transform(x, lmbda, log_mean)

    result = _yeojohnson_centered_inverse_transform(transformed, lmbda, log_mean)

    np.testing.assert_allclose(result, x, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("lmbda", [0.0, 2.0, 0.5])
def test__yeojohnson_centered_transform__zero_log_mean__is_the_classic_transform(
    lmbda: float,
):
    x = np.linspace(-3.0, 3.0, 25)

    result = _yeojohnson_centered_transform(x, lmbda, 0.0)

    np.testing.assert_allclose(result, _yeojohnson_transform(x, lmbda), rtol=1e-14)


def test__safe_power_transformer__state_without_new_option__works_as_before():
    """An instance pickled before ``relative_to_geometric_mean`` existed still works."""
    X = np.random.default_rng(0).lognormal(0, 1, (100, 3))
    transformer = SafePowerTransformer().fit(X)
    expected = transformer.transform(X)
    old_state = transformer.__getstate__()
    del old_state["relative_to_geometric_mean"]

    restored = SafePowerTransformer.__new__(SafePowerTransformer)
    restored.__setstate__(old_state)

    np.testing.assert_array_equal(restored.transform(X), expected)
    assert "SafePowerTransformer" in repr(restored)
    assert clone(restored).relative_to_geometric_mean is False
