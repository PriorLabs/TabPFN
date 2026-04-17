"""Adaptive Quantile Transformer."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import numpy as np
from sklearn.preprocessing import QuantileTransformer

_DEFAULT_SUBSAMPLE = 100_000


def compute_effective_n_quantiles(
    user_n_quantiles: int,
    n_samples: int,
    subsample: int = _DEFAULT_SUBSAMPLE,
) -> int:
    """Compute the effective number of quantiles.

    Adapt n_quantiles for this fit: min of user's preference and available samples
    Ensure n_quantiles is at least 1.
    We allow the number of quantiles to be a maximum of 20% of the subsample size
    because we found that the `np.nanpercentile()` function inside sklearn's
    QuantileTransformer takes a long time to compute when the ratio
    of `quantiles / subsample` is too high (roughly higher than 0.25).

    TODO: This could be revisited for GPU-based quantile transformer.
    """
    return max(1, min(user_n_quantiles, n_samples, int(subsample * 0.2)))


def get_user_n_quantiles_for_preset(transform_name: str, n_samples: int) -> int:
    """Return the ``user_n_quantiles`` for a named quantile preset.

    Args:
        transform_name: One of the ``quantile_*`` preset names.
        n_samples: Number of training samples (used in the formula).

    Raises:
        ValueError: If *transform_name* is not a known quantile preset.
    """
    if transform_name in ("quantile_uni", "quantile_norm"):
        return max(n_samples // 5, 2)
    if transform_name in ("quantile_uni_coarse", "quantile_norm_coarse"):
        return max(n_samples // 10, 2)
    if transform_name in ("quantile_uni_fine", "quantile_norm_fine"):
        return n_samples
    raise ValueError(f"Unknown quantile preset: {transform_name}")


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
        subsample: int = _DEFAULT_SUBSAMPLE,
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

        self.n_quantiles = compute_effective_n_quantiles(
            self._user_n_quantiles, n_samples, self.subsample
        )

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
