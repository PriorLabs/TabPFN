#  Copyright (c) Prior Labs GmbH 2026.
"""Post-hoc rescaling of predictions toward a reference target distribution.

TabPFN predicts in-context, so the prior it expresses is the one it sees in its
context rows. Two things pull that away from the training data: class balancing
requested by the user, and row subsampling that changes the class mix per
estimator (``SAMPLE_SUBSAMPLING_METHOD="majority_downsample"``). Every mode in
this module is one multiplicative correction of the predicted distribution; they
differ only in where the factors come from.

Classification: a per-class weight vector applied to the averaged probabilities
and renormalized. Regression: a per-bucket weight on the bar distribution for the
sampler correction, and an affine map of the raw-space borders for the holdout
level correction.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from tabpfn.architectures.shared.bar_distribution import (
        FullSupportBarDistribution,
    )

PredictionScaling = Literal["auto", "none", "balanced", "sampler", "holdout"]


class PredictionScalingMode(str, Enum):
    """Source of the prediction scaling factors."""

    AUTO = "auto"
    NONE = "none"
    BALANCED = "balanced"
    SAMPLER = "sampler"
    HOLDOUT = "holdout"


HOLDOUT_WEIGHT_FIT_ITERATIONS = 200
"""Fixed-point iterations for matching mean predicted class probabilities to the
holdout class frequencies. The map contracts quickly; this is a generous cap."""

HOLDOUT_WEIGHT_FIT_TOLERANCE = 1e-8


def resolve_prediction_scaling(
    prediction_scaling: PredictionScaling | PredictionScalingMode,
    *,
    task_type: Literal["classifier", "regressor"],
    sampler_shifted_prior: bool,
) -> PredictionScalingMode:
    """Resolve ``"auto"`` and validate the mode against the task.

    ``"auto"`` becomes ``"sampler"`` when the row sampler changed the target prior
    of the context, and ``"none"`` otherwise. ``"balanced"`` has no meaning for a
    continuous target and is rejected for regressors.
    """
    mode = PredictionScalingMode(prediction_scaling)
    if mode == PredictionScalingMode.AUTO:
        return (
            PredictionScalingMode.SAMPLER
            if sampler_shifted_prior
            else PredictionScalingMode.NONE
        )
    if task_type == "regressor" and mode == PredictionScalingMode.BALANCED:
        raise ValueError(
            "prediction_scaling='balanced' is only defined for classification. "
            "Use 'none', 'sampler', 'holdout', or 'auto' for regression."
        )
    return mode


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def context_class_prior(
    y_encoded: np.ndarray,
    row_indices: list[np.ndarray] | None,
    n_classes: int,
) -> np.ndarray:
    """Class prior the estimators see in their context, averaged over estimators.

    With no row subsampling every estimator sees the full training set, so the
    context prior is the training prior.
    """
    y_encoded = np.asarray(y_encoded).astype(np.int64, copy=False)
    if row_indices is None:
        counts = np.bincount(y_encoded, minlength=n_classes).astype(np.float64)
        return counts / counts.sum()
    priors = []
    for idx in row_indices:
        counts = np.bincount(y_encoded[idx], minlength=n_classes).astype(np.float64)
        priors.append(counts / counts.sum())
    return np.mean(priors, axis=0)


def sampler_class_weights(
    train_class_counts: np.ndarray,
    context_prior: np.ndarray,
) -> np.ndarray:
    """Weights that undo a label shift between context and training data.

    Under label shift ``p_train(c | x) ∝ p_context(c | x) * π_train(c) / π_context(c)``.
    Classes absent from the context cannot be corrected and keep weight one.
    """
    train_counts = np.asarray(train_class_counts, dtype=np.float64)
    train_prior = train_counts / train_counts.sum()
    context_prior = np.asarray(context_prior, dtype=np.float64)
    weights = np.ones_like(train_prior)
    present = context_prior > 0
    weights[present] = train_prior[present] / context_prior[present]
    return weights


def balanced_class_weights(
    train_class_counts: np.ndarray,
    context_prior: np.ndarray | None,
) -> np.ndarray:
    """Weights that move the predicted prior toward uniform.

    Balancing divides by the prior the model actually expresses. Without row
    subsampling that is the training prior, which reproduces the historical
    ``balance_probabilities`` behavior exactly. When the sampler shifted the
    context prior, dividing by the context prior balances relative to what the
    model saw, which is the sampler correction composed with plain balancing.
    """
    train_counts = np.asarray(train_class_counts, dtype=np.float64)
    prior = (
        train_counts / train_counts.sum()
        if context_prior is None
        else np.asarray(context_prior, dtype=np.float64)
    )
    weights = np.ones_like(prior)
    present = prior > 0
    weights[present] = 1.0 / prior[present]
    return weights


def apply_class_weights(
    probas: torch.Tensor,
    weights: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Multiply class probabilities by per-class weights and renormalize."""
    w = torch.as_tensor(np.asarray(weights), dtype=probas.dtype, device=probas.device)
    scaled = probas * w
    return scaled / scaled.sum(dim=-1, keepdim=True)


def fit_holdout_class_weights(
    holdout_probas: np.ndarray,
    holdout_y_true: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Weights that make the mean predicted probability per class match the
    holdout class frequency.

    Solves ``mean_i normalize(p_i * w)_c = freq_c`` for ``w`` by fixed-point
    iteration ``w_c <- w_c * freq_c / mean_i normalize(p_i * w)_c``. The solution
    is unique up to a common scale; the result is normalized so its
    prediction-weighted mean is one. Classes without holdout rows keep weight one.
    """
    probas = np.asarray(holdout_probas, dtype=np.float64)
    y_true = np.asarray(holdout_y_true).astype(np.int64, copy=False)
    freq = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    freq /= freq.sum()
    present = freq > 0

    weights = np.ones(n_classes, dtype=np.float64)
    for _ in range(HOLDOUT_WEIGHT_FIT_ITERATIONS):
        scaled = probas * weights
        scaled /= scaled.sum(axis=1, keepdims=True)
        mean_pred = scaled.mean(axis=0)
        update = np.ones_like(weights)
        update[present] = freq[present] / np.maximum(mean_pred[present], 1e-12)
        weights *= update
        if np.max(np.abs(update - 1.0)) < HOLDOUT_WEIGHT_FIT_TOLERANCE:
            break
    return weights / weights[present].mean()


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def majority_value_shares(
    y: np.ndarray,
    row_indices: list[np.ndarray] | None,
) -> tuple[float, float, float]:
    """The most frequent target value and its share in training and in context.

    Returns ``(value, train_share, context_share)``; the context share is averaged
    over estimators and equals the training share without row subsampling.
    """
    y = np.asarray(y, dtype=np.float64)
    values, counts = np.unique(y, return_counts=True)
    value = float(values[np.argmax(counts)])
    train_share = float(counts.max() / len(y))
    if row_indices is None:
        return value, train_share, train_share
    context_share = float(np.mean([(y[idx] == value).mean() for idx in row_indices]))
    return value, train_share, context_share


def sampler_bucket_log_weights(
    bardist: FullSupportBarDistribution,
    *,
    majority_value: float,
    train_share: float,
    context_share: float,
) -> torch.Tensor:
    """Per-bucket log weights that undo a label shift on one target value.

    The bucket holding the majority value is reweighted by
    ``train_share / context_share`` and every other bucket by the ratio of the
    complements, so a predicted distribution that was calibrated to the context
    is moved back to the training prior. Adding the result to the aggregated
    log-probabilities before the bar distribution's ``log_softmax`` applies it.
    """
    if not (0 < context_share < 1) or not (0 < train_share < 1):
        raise ValueError(
            "Sampler prediction scaling needs the majority target value to cover "
            f"part of both the training data and the context, got "
            f"train_share={train_share}, context_share={context_share}."
        )
    borders = bardist.borders.detach()
    idx = int(
        bardist.map_to_bucket_idx(
            torch.tensor([majority_value], dtype=borders.dtype, device=borders.device)
        ).item()
    )
    log_weights = torch.full(
        (bardist.num_bars,),
        float(np.log((1.0 - train_share) / (1.0 - context_share))),
        dtype=torch.float32,
    )
    log_weights[idx] = float(np.log(train_share / context_share))
    return log_weights


def fit_holdout_level_scaling(
    holdout_pred_mean: np.ndarray,
    holdout_y_true: np.ndarray,
) -> tuple[float, float]:
    """Affine map ``y -> scale * y + shift`` aligning the predicted level to the
    holdout target mean.

    Multiplicative when both means are positive, which is the natural correction
    for nonnegative targets such as claim amounts; an additive shift otherwise.
    """
    pred_mean = float(np.mean(holdout_pred_mean))
    true_mean = float(np.mean(holdout_y_true))
    if pred_mean > 0 and true_mean > 0:
        return true_mean / pred_mean, 0.0
    return 1.0, true_mean - pred_mean
