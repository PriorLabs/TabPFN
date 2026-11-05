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
NEGATE_STEP = "negate_target"


class NegateTarget(TransformerMixin, BaseEstimator):
    """Flip the sign of the target. Its own inverse."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NegateTarget:
        """Nothing to learn."""
        del X, y
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the negated target."""
        return -np.asarray(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` with its sign flipped back."""
        return -np.asarray(X)


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


def negate_target(transform: Transformer | Pipeline) -> Pipeline:
    """Prepend a sign flip to a target pipeline, so it reads the negated raw target.

    The inverse maps back to the original sign. The returned pipeline shares its
    other steps with ``transform``.
    """
    steps = (
        transform.steps
        if isinstance(transform, Pipeline)
        else [(TARGET_TRANSFORM_STEP, transform)]
    )
    return Pipeline(steps=[(NEGATE_STEP, NegateTarget()), *steps])


__all__ = [
    "NEGATE_STEP",
    "STANDARDIZE_STEP",
    "TARGET_TRANSFORM_STEP",
    "NegateTarget",
    "StandardizeTarget",
    "make_target_transform",
    "negate_target",
]
