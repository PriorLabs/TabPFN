#  Copyright (c) Prior Labs GmbH 2026.

"""Adaptive Quantile Transformer."""

from __future__ import annotations

from typing import Literal
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
    if transform_name in (
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_extrapolate",
    ):
        return max(n_samples // 5, 2)
    if transform_name in ("quantile_uni_coarse", "quantile_norm_coarse"):
        return max(n_samples // 10, 2)
    if transform_name in ("quantile_uni_fine", "quantile_norm_fine"):
        return n_samples
    raise ValueError(f"Unknown quantile preset: {transform_name}")


def get_extrapolate_ratio_for_preset(transform_name: str) -> float | None:
    """Return the default ``extrapolate_ratio`` for a named quantile preset."""
    if transform_name == "quantile_uni_extrapolate":
        return 1.0
    return None


class AdaptiveQuantileTransformer(QuantileTransformer):
    """A QuantileTransformer that automatically adapts the 'n_quantiles' parameter
    based on the number of samples provided during the 'fit' method.

    This fixes an issue in older versions of scikit-learn where the 'n_quantiles'
    parameter could not exceed the number of samples in the input data.

    This code prevents errors that occur when the requested 'n_quantiles' is
    greater than the number of available samples in the input data (X).
    This situation can arises because we first initialize the transformer
    based on total samples and then subsample.

    When ``extrapolate_ratio`` is set, inputs outside the training range
    are linearly extrapolated beyond ``[0, 1]`` at ``transform`` time and
    clipped at ``-extrapolate_ratio`` / ``1 + extrapolate_ratio``. This
    preserves OOD information that would otherwise be flattened by the
    boundary clamp. Intended for ``output_distribution="uniform"``.
    """

    def __init__(
        self,
        *,
        n_quantiles: int = 1_000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        ignore_implicit_zeros: bool = False,
        subsample: int = _DEFAULT_SUBSAMPLE,
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
        copy: bool = True,
        extrapolate_ratio: float | None = None,
    ) -> None:
        # All parent parameters are declared explicitly (no **kwargs): sklearn's
        # get_params() only discovers named constructor parameters, and clone()
        # rebuilds from get_params(). Parameters hidden behind **kwargs would be
        # silently dropped whenever sklearn clones this transformer (e.g. inside
        # ColumnTransformer.fit), resetting output_distribution/random_state to
        # their defaults.
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )
        if extrapolate_ratio is not None:
            if extrapolate_ratio < 0:
                raise ValueError("extrapolate_ratio must be non-negative.")
            if output_distribution != "uniform":
                raise ValueError(
                    "extrapolate_ratio is only supported for "
                    "output_distribution='uniform'."
                )
        self.extrapolate_ratio = extrapolate_ratio

    @override
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> AdaptiveQuantileTransformer:
        if self.extrapolate_ratio is not None and X.shape[0] > 0:
            self.x_min_ = np.nanmin(X, axis=0, keepdims=True)
            self.x_max_ = np.nanmax(X, axis=0, keepdims=True)

        n_samples = X.shape[0]

        # Convert Generator to RandomState if needed for sklearn compatibility
        random_state = self.random_state
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, 2**32))
            random_state = np.random.RandomState(seed)
        elif hasattr(random_state, "bit_generator"):
            raise ValueError(
                f"Unsupported random state type: {type(random_state)}. "
                "Please provide an integer seed or np.random.RandomState object."
            )

        # The parent reads n_quantiles/random_state from self, so set the
        # adapted values only for the duration of the fit and restore the
        # user-provided hyperparameters afterwards: fit() must not modify
        # constructor parameters, or clones/refits of this transformer would
        # be configured with the adapted values instead of the originals.
        user_n_quantiles = self.n_quantiles
        user_random_state = self.random_state
        self.n_quantiles = compute_effective_n_quantiles(
            user_n_quantiles, n_samples, self.subsample
        )
        self.random_state = random_state
        try:
            return super().fit(X, y)
        finally:
            self.n_quantiles = user_n_quantiles
            self.random_state = user_random_state

    @override
    def transform(self, X: np.ndarray) -> np.ndarray:
        out = super().transform(X)

        if self.extrapolate_ratio is not None and X.shape[0] > 0:
            x_range = self.x_max_ - self.x_min_
            # Skip constant features (matches the GPU path); also avoids a
            # divide-by-zero in the normalisation below.
            extrap_mask = x_range > 0
            min_idcs = (self.x_min_ > X) & extrap_mask
            max_idcs = (self.x_max_ < X) & extrap_mask
            if np.any(min_idcs) or np.any(max_idcs):
                # (X - x_max)/range + 1 simplifies to (X - x_min)/range, so
                # one normalised array suffices for both branches.
                with np.errstate(divide="ignore", invalid="ignore"):
                    norm = (X - self.x_min_) / x_range
                if np.any(min_idcs):
                    out[min_idcs] = np.clip(
                        norm[min_idcs], -self.extrapolate_ratio, 0.0
                    )
                if np.any(max_idcs):
                    out[max_idcs] = np.clip(
                        norm[max_idcs], 1.0, 1.0 + self.extrapolate_ratio
                    )

        return out


__all__ = [
    "AdaptiveQuantileTransformer",
]
