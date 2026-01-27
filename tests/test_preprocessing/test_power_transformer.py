from __future__ import annotations

import warnings

import numpy as np
from sklearn.preprocessing import PowerTransformer

from tabpfn.preprocessing.steps import SafePowerTransformer


def test__safe_power_transformer__normal_cases__same_results_as_power_transformer():
    """Test SafePowerTransformer gives same results as
    PowerTransformer for normal data.
    """
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


def test__safe_power_transformer__transform_then_inverse_transform__returns_original():
    """Test that SafePowerTransformer inverse_transform returns data to original scale.
    We haven't changed the.
    """
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
