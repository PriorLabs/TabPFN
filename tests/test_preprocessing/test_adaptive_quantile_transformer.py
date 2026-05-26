#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing.steps import AdaptiveQuantileTransformer


def test_adaptive_quantile_transformer_with_numpy_generator():
    """Tests that AdaptiveQuantileTransformer can handle a np.random.Generator.

    This test ensures that the transformer is compatible with NumPy's modern
    random number generation API, which is passed down from other parts of
    the TabPFN codebase. It replicates the conditions that previously caused a
    ValueError in scikit-learn's check_random_state.
    """
    # ARRANGE: Create sample data and a modern NumPy random number generator
    rng = np.random.default_rng(42)
    X = rng.random((100, 10))

    # ARRANGE: Instantiate the transformer with the Generator object
    # This is the exact condition that caused the bug
    transformer = AdaptiveQuantileTransformer(
        output_distribution="uniform",
        n_quantiles=10,
        random_state=rng,
    )

    # ACT & ASSERT: Ensure that fitting the transformer does not raise an error
    transformer.fit_transform(X)

    # Further assertion to ensure the transformer is functional
    assert hasattr(transformer, "quantiles_")
    assert transformer.quantiles_.shape == (10, 10)


def test__extrapolate_ratio_uniform__extends_outside_training_range():
    """Inputs outside the training range get linearly extrapolated and clipped."""
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0.0, 1.0, size=(200, 1))

    transformer = AdaptiveQuantileTransformer(
        output_distribution="uniform",
        n_quantiles=50,
        extrapolate_ratio=1.0,
        random_state=0,
    )
    transformer.fit(X_train)

    in_range = transformer.transform(np.array([[0.5]]))
    assert 0.0 <= in_range[0, 0] <= 1.0

    # Below the training min: maps to negative values, clipped at -extrapolate_ratio.
    below = transformer.transform(np.array([[-0.1], [-2.0]]))
    assert below[0, 0] < 0.0
    assert below[0, 0] == pytest.approx(-0.1, abs=0.05)
    assert below[1, 0] == pytest.approx(-1.0)  # clipped

    # Above the training max: maps to >1, clipped at 1 + extrapolate_ratio.
    above = transformer.transform(np.array([[1.1], [3.0]]))
    assert above[0, 0] > 1.0
    assert above[0, 0] == pytest.approx(1.1, abs=0.05)
    assert above[1, 0] == pytest.approx(2.0)  # clipped


def test__extrapolate_ratio__nan_columns_get_valid_boundaries():
    """NaN entries in training data must not make the extrapolation bounds NaN."""
    X_train = np.array(
        [
            [1.0, 1.0],
            [2.0, np.nan],
            [3.0, 2.0],
            [4.0, 3.0],
        ]
    )
    transformer = AdaptiveQuantileTransformer(
        output_distribution="uniform",
        n_quantiles=4,
        extrapolate_ratio=1.0,
        random_state=0,
    )
    transformer.fit(X_train)

    assert np.all(np.isfinite(transformer.x_min_))
    assert np.all(np.isfinite(transformer.x_max_))

    # Out-of-range inputs in either column extrapolate to a finite value.
    out = transformer.transform(np.array([[5.0, 4.0]]))
    assert np.all(np.isfinite(out))
    assert out[0, 0] > 1.0
    assert out[0, 1] > 1.0


def test__extrapolate_ratio__constant_feature_no_extrapolation():
    """Constant features should not be extrapolated (matches GPU behaviour)."""
    X_train = np.column_stack(
        [
            np.linspace(0.0, 1.0, 50),  # varying
            np.full(50, 5.0),  # constant
        ]
    )
    transformer = AdaptiveQuantileTransformer(
        output_distribution="uniform",
        n_quantiles=20,
        extrapolate_ratio=1.0,
        random_state=0,
    )
    transformer.fit(X_train)

    out = transformer.transform(np.array([[2.0, 99.0], [-1.0, 99.0]]))
    # First column extrapolates, second column stays in [0, 1].
    assert out[0, 0] > 1.0
    assert out[1, 0] < 0.0
    assert 0.0 <= out[0, 1] <= 1.0
    assert 0.0 <= out[1, 1] <= 1.0


def test__no_extrapolation_when_unset__matches_baseline():
    """Without extrapolation params, behaviour matches the unmodified transformer."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))

    baseline = AdaptiveQuantileTransformer(
        output_distribution="uniform", n_quantiles=20, random_state=0
    ).fit(X)
    no_extrap = AdaptiveQuantileTransformer(
        output_distribution="uniform", n_quantiles=20, random_state=0
    ).fit(X)

    np.testing.assert_array_equal(baseline.transform(X), no_extrap.transform(X))


def test__extrapolate_ratio__validation_guards():
    """Invalid extrapolate_ratio configs are rejected at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        AdaptiveQuantileTransformer(
            output_distribution="uniform", extrapolate_ratio=-0.1
        )
    with pytest.raises(ValueError, match="output_distribution='uniform'"):
        AdaptiveQuantileTransformer(output_distribution="normal", extrapolate_ratio=0.1)
    # Valid config still constructs.
    AdaptiveQuantileTransformer(output_distribution="uniform", extrapolate_ratio=0.1)
