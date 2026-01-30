from __future__ import annotations

import numpy as np

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
    transformer.fit(X)

    # Further assertion to ensure the transformer is functional
    assert hasattr(transformer, "quantiles_")
    assert transformer.quantiles_.shape == (10, 10)
