#  Copyright (c) Prior Labs GmbH 2026.

"""Target (y) transformation pipelines for regression.

The regressor z-normalises the target before handing it to the ensemble
preprocessing, because the shared bar-distribution space of the checkpoint is
defined in that z-normalised space. A target transform such as ``1_plus_log``
is however only meaningful on the target in its *original* units: applied to
z-normalised values it operates on a shifted, rescaled target, which is not
what one would expect (and, for the log-like transforms, mostly produces NaNs
because roughly half of a z-normalised target is negative).

:func:`wrap_target_transform` therefore composes each transform into a
three-step pipeline that

1. undoes the regressor's z-normalisation, so the transform sees the target in
   its original units,
2. applies the transform, and
3. standardises the result again, which is the scale the model expects.

Keeping the pipeline's input and output in the z-normalised space means the
rest of the regressor -- in particular the bar-distribution borders and their
sanity limits, which are expressed in z-units -- is unaffected:
``pipeline.inverse_transform`` maps the model's borders straight back into the
z-normalised space, exactly as an unwrapped transform did.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn.preprocessing.steps.utils import make_scaler_safe

if TYPE_CHECKING:
    from collections.abc import Iterable

UNSTANDARDIZE_STEP = "unstandardize_target"
TARGET_TRANSFORM_STEP = "target_transform"
STANDARDIZE_STEP = "standardize_target"


class UnstandardizeTarget(TransformerMixin, BaseEstimator):
    """Map a z-normalised target back to its original units.

    ``transform`` undoes a z-normalisation with the given statistics and
    ``inverse_transform`` re-applies it, so this transformer is the first step
    of the pipelines built by :func:`wrap_target_transform`.

    Args:
        mean: Mean that was subtracted by the z-normalisation.
        std: Standard deviation the z-normalisation divided by.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> UnstandardizeTarget:
        """Stateless, the statistics are given at construction time."""
        del X, y
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` in the original units of the target."""
        return np.asarray(X) * self.std + self.mean

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` z-normalised again."""
        return (np.asarray(X) - self.mean) / self.std


def wrap_target_transform(
    transform: TransformerMixin | Pipeline,
    *,
    mean: float,
    std: float,
) -> Pipeline:
    """Compose a target transform so that it acts on the unnormalised target.

    Args:
        transform: The target transform to wrap, e.g. one of the presets of
            :func:`get_all_reshape_feature_distribution_preprocessors`.
        mean: Mean of the training target, used to undo its z-normalisation.
        std: Standard deviation of the training target, used to undo its
            z-normalisation.

    Returns:
        A pipeline mapping the z-normalised target to the transformed and
        re-standardised target, whose ``inverse_transform`` maps back into the
        z-normalised space.
    """
    return Pipeline(
        steps=[
            (UNSTANDARDIZE_STEP, UnstandardizeTarget(mean=mean, std=std)),
            (TARGET_TRANSFORM_STEP, transform),
            # The transform may leave the target on an arbitrary scale (e.g. a
            # log target), while the model expects a standardised one. The safe
            # wrapper also keeps non-finite outputs, such as the log of a
            # non-positive target, from reaching the model.
            (STANDARDIZE_STEP, make_scaler_safe("standard", StandardScaler())),
        ],
    )


def rebind_target_transform_statistics(
    transforms: Iterable[TransformerMixin | Pipeline | None],
    *,
    mean: float,
    std: float,
) -> None:
    """Point wrapped target transforms at another z-normalisation, in place.

    Needed when the transforms were built for one z-normalisation but are
    (re-)fitted on a target normalised with different statistics, as in the
    fine-tuning data pipeline, which re-splits the dataset and z-normalises
    with the statistics of every new training split.

    Transforms that are not pipelines from :func:`wrap_target_transform`, such
    as the ``None`` entries of an unwrapped target transform, are ignored.

    Args:
        transforms: The target transforms to update.
        mean: Mean of the z-normalisation the transforms will be fitted on.
        std: Standard deviation of that z-normalisation.
    """
    for transform in transforms:
        if not isinstance(transform, Pipeline):
            continue
        step = transform.named_steps.get(UNSTANDARDIZE_STEP)
        if isinstance(step, UnstandardizeTarget):
            step.mean = mean
            step.std = std


__all__ = [
    "STANDARDIZE_STEP",
    "TARGET_TRANSFORM_STEP",
    "UNSTANDARDIZE_STEP",
    "UnstandardizeTarget",
    "rebind_target_transform_statistics",
    "wrap_target_transform",
]
