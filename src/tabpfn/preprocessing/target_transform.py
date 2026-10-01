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
    "StandardizeTarget",
    "make_target_transform",
]
