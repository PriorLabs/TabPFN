"""Adaptive Quantile Transformer."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class AdaptiveQuantileTransformer(QuantileTransformer):
    """A QuantileTransformer that automatically adapts the 'n_quantiles' parameter
    based on the number of samples provided during the 'fit' method.

    This fixes an issue in older versions of scikit-learn where the 'n_quantiles'
    parameter could not exceed the number of samples in the input data.

    This code prevents errors that occur when the requested 'n_quantiles' is
    greater than the number of available samples in the input data (X).
    This situation can arises because we first initialize the transformer
    based on total samples and then subsample.
    """

    def __init__(
        self,
        *,
        n_quantiles: int = 1_000,
        subsample: int = 100_000,  # default in sklearn is 10k
        **kwargs: Any,
    ) -> None:
        # Store the user's desired n_quantiles to use as an upper bound
        self._user_n_quantiles = n_quantiles
        # Initialize parent with this, but it will be adapted in fit
        super().__init__(n_quantiles=n_quantiles, subsample=subsample, **kwargs)

    @override
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> AdaptiveQuantileTransformer:
        n_samples = X.shape[0]

        # Adapt n_quantiles for this fit: min of user's preference and available samples
        # Ensure n_quantiles is at least 1.
        # We allow the number of quantiles to be a maximum of 20% of the subsample size
        # because we found that the `np.nanpercentile()` function inside sklearn's
        # QuantileTransformer takes a long time to compute when the ratio
        # of `quantiles / subsample` is too high (roughly higher than 0.25).
        effective_n_quantiles = max(
            1,
            min(
                self._user_n_quantiles,
                n_samples,
                int(self.subsample * 0.2),
            ),
        )

        # Set self.n_quantiles to the effective value BEFORE calling super().fit()
        # This ensures the parent class uses the adapted value for fitting
        # and self.n_quantiles will reflect the value used for the fit.
        self.n_quantiles = effective_n_quantiles

        # Convert Generator to RandomState if needed for sklearn compatibility
        if isinstance(self.random_state, np.random.Generator):
            seed = int(self.random_state.integers(0, 2**32))
            self.random_state = np.random.RandomState(seed)
        elif hasattr(self.random_state, "bit_generator"):
            raise ValueError(
                f"Unsupported random state type: {type(self.random_state)}. "
                "Please provide an integer seed or np.random.RandomState object."
            )

        return super().fit(X, y)


__all__ = [
    "AdaptiveQuantileTransformer",
]
