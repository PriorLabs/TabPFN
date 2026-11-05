#  Copyright (c) Prior Labs GmbH 2026.

"""Invertible regression target pipelines.

Each member maps raw targets to model inputs and inversely maps borders to raw
units. The estimator maps those borders into a shared standardized space for
aggregation, avoiding float32 precision loss for targets with large offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin as Transformer

STANDARDIZE_STEP = "standardize_target"
TARGET_TRANSFORM_STEP = "target_transform"


class StandardizeTarget(TransformerMixin, BaseEstimator):
    """Standardize using fitted ``mean_`` and ``std_`` (population std + epsilon).

    Keep NumPy arithmetic rather than StandardScaler to preserve exact model
    inputs when the member sees the full training target.
    """

    EPSILON = 1e-20
    """Prevent division by zero for constant targets."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> StandardizeTarget:
        """Learn the mean and standard deviation of the target ``X``."""
        del y
        self.mean_ = float(np.mean(X))
        self.std_ = float(np.std(X)) + self.EPSILON
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the standardized target."""
        return (np.asarray(X) - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` in the original units of the target."""
        return np.asarray(X) * self.std_ + self.mean_


class QuantileGapTransformer(TransformerMixin, BaseEstimator):
    """log(y) for y > 0, and a gap below the positive logs for y = 0.

    The zeros sit at q - sd, with q the 1% quantile and sd the standard
    deviation of the positive logs. The inverse decodes values below
    q - sd / 2 to 0. A member with this transform is clamped at 0 before
    aggregation, see `transform_borders_one`.
    """

    QUANTILE = 0.01

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> QuantileGapTransformer:
        """Place the zeros below the positive logs of the target ``X``."""
        del y
        values = np.asarray(X, dtype=np.float64)
        if (values < 0).any():
            raise ValueError(
                "The 'quantile_gap' regression target transform requires "
                "nonnegative raw targets (y >= 0)."
            )
        logs = np.log(values[values > 0])
        if np.unique(logs).size < 2:
            raise ValueError(
                "The 'quantile_gap' regression target transform requires at least "
                "two distinct positive targets."
            )
        sd = float(np.std(logs))
        self.zero_ = float(np.quantile(logs, self.QUANTILE)) - sd
        self.threshold_ = self.zero_ + sd / 2
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return log(y), with the zeros at the bottom of the gap."""
        values = np.asarray(X, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.where(values > 0, np.log(values), self.zero_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return 0 below the middle of the gap, else exp(X)."""
        t = np.asarray(X, dtype=np.float64)
        # The top borders overflow to inf; border repair cancels them.
        with np.errstate(over="ignore"):
            return np.where(t < self.threshold_, 0.0, np.exp(t))


def uses_quantile_gap(target_transform: Transformer | Pipeline) -> bool:
    """Whether a member's target pipeline applies `QuantileGapTransformer`."""
    return isinstance(target_transform, Pipeline) and isinstance(
        target_transform.named_steps.get(TARGET_TRANSFORM_STEP),
        QuantileGapTransformer,
    )


def make_target_transform(transform: Transformer | Pipeline | None) -> Pipeline:
    """Apply the optional preset to raw targets, then standardize.

    The inverse returns values in the target's original units.
    """
    if transform is None:
        return Pipeline(steps=[(STANDARDIZE_STEP, StandardizeTarget())])
    return Pipeline(
        steps=[
            # Reshape the raw target before standardizing it for the model.
            (TARGET_TRANSFORM_STEP, transform),
            (STANDARDIZE_STEP, StandardizeTarget()),
        ],
    )


__all__ = [
    "STANDARDIZE_STEP",
    "TARGET_TRANSFORM_STEP",
    "QuantileGapTransformer",
    "StandardizeTarget",
    "make_target_transform",
    "uses_quantile_gap",
]
