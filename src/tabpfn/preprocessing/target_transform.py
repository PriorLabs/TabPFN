#  Copyright (c) Prior Labs GmbH 2026.

"""Target (y) transformation pipelines for regression.

Every regression ensemble member owns one invertible map from the target, in
its original units, to the target the model is fitted on:

* ``transform`` standardizes the target -- optionally after reshaping it with
  one of the `REGRESSION_Y_PREPROCESS_TRANSFORMS` presets -- because the
  checkpoint's bar distribution is defined for a standardized target.
* ``inverse_transform`` maps the model's bar-distribution borders back into
  the original units of the target.

The estimator owns the affine frame the ensemble is aggregated in --
``y_train_mean_`` and ``y_train_std_``, the same frame ``raw_space_bardist_``
decodes from -- and applies it to the borders after the inverse. That split
keeps these pipelines self-contained: the statistics of a member's own target
are learned in ``fit``, so nothing has to be baked into a pipeline when it is
built, and nothing has to be rebound when it is refitted on a different split.

Aggregating in the frame of the checkpoint rather than in the original units
of the target is deliberate: the bar distribution has thousands of borders,
and `translate_probs_across_borders` resolves positions within a bucket in
float32. For a target with a large offset (say 1e8 with a standard deviation
of 1e2) the borders in original units are spaced far below float32 resolution
at that magnitude, and would collapse into duplicates.
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
    """Z-normalise the target, learning the statistics in ``fit``.

    `sklearn.preprocessing.StandardScaler` would do, but this keeps the
    arithmetic of the estimator's own z-normalisation -- `np.mean` and `np.std`
    plus an epsilon -- so that a member without a target transform is fitted on
    exactly the target the regressor used to compute before preprocessing.

    Attributes:
        mean_: Mean of the target seen in `fit`.
        std_: Standard deviation of that target, plus `EPSILON`.
    """

    EPSILON = 1e-20
    """Guards against a division by zero for a (near-)constant target, which
    `TabPFNRegressor.fit` rejects before it reaches the model anyway."""

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
    """Build the target pipeline of one ensemble member.

    Args:
        transform: The preset to reshape the target with, e.g. one of
            :func:`get_all_reshape_feature_distribution_preprocessors`, or
            None to only standardize the target.

    Returns:
        A pipeline mapping the target in its original units to the target the
        model is fitted on, whose ``inverse_transform`` maps back.
    """
    if transform is None:
        return Pipeline(steps=[(STANDARDIZE_STEP, StandardizeTarget())])
    return Pipeline(
        steps=[
            # The preset reshapes the standardized target, as it always has;
            # the ordering of these two steps is what RES-2639 changes.
            (STANDARDIZE_STEP, StandardizeTarget()),
            (TARGET_TRANSFORM_STEP, transform),
        ],
    )


__all__ = [
    "STANDARDIZE_STEP",
    "TARGET_TRANSFORM_STEP",
    "StandardizeTarget",
    "make_target_transform",
]
