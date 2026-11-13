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

    The zeros sit at q - sd, with q the 1% quantile and sd the standard deviation
    of the positive logs. Values below q - sd / 2 decode to 0, so the zeros come
    back as a point mass.
    """

    QUANTILE = 0.01

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> QuantileGapTransformer:
        """Place the zeros below the positive logs of the target ``X``."""
        del y
        logs = _positive_logs(X, "raw_log_qgap")
        sd = float(np.std(logs))
        self.zero_ = float(np.quantile(logs, self.QUANTILE)) - sd
        self.cut_ = self.zero_ + sd / 2
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
            return np.where(t < self.cut_, 0.0, np.exp(t))


class FixedAtomTransformer(TransformerMixin, BaseEstimator):
    """log(y) standardized on the positive targets, with the zeros at a fixed atom.

    The zeros sit at ``ATOM``. Positives below the cut ``ATOM + 1/2`` are raised to
    it, and values below the cut decode to 0. The output is standardized already,
    so `make_target_transform` adds no standardization after it.
    """

    ATOM = -4.0
    CUT = ATOM + 0.5

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> FixedAtomTransformer:
        """Learn the mean and standard deviation of the positive logs of ``X``."""
        del y
        logs = _positive_logs(X, "raw_log_fixed_atom")
        self.mean_ = float(np.mean(logs))
        self.scale_ = float(np.std(logs))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the standardized log(y), with the zeros at ``ATOM``."""
        values = np.asarray(X, dtype=np.float64)
        positive = values > 0
        t = (np.log(np.where(positive, values, 1.0)) - self.mean_) / self.scale_
        return np.where(positive, np.maximum(t, self.CUT), self.ATOM)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return 0 below the cut, else exp of the unstandardized ``X``."""
        t = np.asarray(X, dtype=np.float64)
        with np.errstate(over="ignore"):
            return np.where(t < self.CUT, 0.0, np.exp(self.mean_ + self.scale_ * t))


ZERO_INFLATED_TARGET_TRANSFORMS: dict[str, type[TransformerMixin]] = {
    "raw_log_qgap": QuantileGapTransformer,
    "raw_log_fixed_atom": FixedAtomTransformer,
}
"""Presets of `REGRESSION_Y_PREPROCESS_TRANSFORMS` that decode a range to exactly 0."""


def _positive_logs(X: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if (values < 0).any():
        raise ValueError(
            f"The '{name}' regression target transform requires nonnegative raw "
            "targets (y >= 0)."
        )
    logs = np.log(values[values > 0])
    if np.unique(logs).size < 2:
        raise ValueError(
            f"The '{name}' regression target transform requires at least two "
            "distinct positive targets."
        )
    return logs


def decodes_zeros(target_transform: Transformer | Pipeline) -> bool:
    """Whether a member's target pipeline decodes a range of values to exactly 0."""
    return isinstance(target_transform, Pipeline) and isinstance(
        target_transform.named_steps.get(TARGET_TRANSFORM_STEP),
        tuple(ZERO_INFLATED_TARGET_TRANSFORMS.values()),
    )


def make_target_transform(transform: Transformer | Pipeline | None) -> Pipeline:
    """Apply the optional preset to raw targets, then standardize.

    `FixedAtomTransformer` standardizes itself and is not standardized again. The
    inverse returns values in the target's original units.
    """
    if transform is None:
        return Pipeline(steps=[(STANDARDIZE_STEP, StandardizeTarget())])
    if isinstance(transform, FixedAtomTransformer):
        # Standardizing again would move the zeros away from the fixed atom.
        return Pipeline(steps=[(TARGET_TRANSFORM_STEP, transform)])
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
    "ZERO_INFLATED_TARGET_TRANSFORMS",
    "FixedAtomTransformer",
    "QuantileGapTransformer",
    "StandardizeTarget",
    "decodes_zeros",
    "make_target_transform",
]
