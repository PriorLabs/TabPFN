#  Copyright (c) Prior Labs GmbH 2026.
"""Undo the target prior shift introduced by ``majority_downsample`` row sampling.

TabPFN predicts in-context, so the prior it expresses is the one it sees in its
context rows. ``SAMPLE_SUBSAMPLING_METHOD="majority_downsample"`` keeps every
row outside the most frequent target value and subsamples that majority value,
so every estimator's context over-represents the rare targets relative to the
training data. Under the label-shift assumption the fix is exact and free:
multiply the predicted distribution by the training-to-context prior ratio and
renormalize.

Classification: a per-class weight vector applied to the averaged probabilities.
Regression: a per-bar log weight added to the aggregated log-probabilities of
the raw-space bar distribution, derived from the sampler's known per-row
inclusion probabilities rather than from realized context counts, so it does not
depend on how many context rows happened to land in a bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from tabpfn.architectures.shared.bar_distribution import (
        FullSupportBarDistribution,
    )


def context_class_prior(
    y_encoded: np.ndarray,
    row_indices: list[np.ndarray],
    n_classes: int,
) -> np.ndarray:
    """Class prior the estimators see in their context, averaged over estimators."""
    y_encoded = np.asarray(y_encoded).astype(np.int64, copy=False)
    priors = []
    for idx in row_indices:
        counts = np.bincount(y_encoded[idx], minlength=n_classes).astype(np.float64)
        priors.append(counts / counts.sum())
    return np.mean(priors, axis=0)


def downsample_class_weights(
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


def apply_class_weights(
    probas: torch.Tensor,
    weights: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Multiply class probabilities by per-class weights and renormalize."""
    w = torch.as_tensor(np.asarray(weights), dtype=probas.dtype, device=probas.device)
    scaled = probas * w
    return scaled / scaled.sum(dim=-1, keepdim=True)


def downsample_bucket_log_weights(
    bardist: FullSupportBarDistribution,
    *,
    y_raw: np.ndarray,
    row_inclusion_probabilities: np.ndarray,
    unobserved_target_probability: float,
) -> torch.Tensor:
    """Log ratios of training to expected-context bar probabilities.

    Histogram all training targets on the raw-space bars twice: once uniformly,
    once weighted by the sampler's per-row inclusion probabilities. For a
    populated bar the ratio is ``mean(inclusion) / mean(inclusion | bar)``. No
    realized context counts enter, so the estimate is as stable as the design.

    Empty training bars use the design's inclusion probability for unobserved
    target values. For majority downsampling that is one, so empty bars get the
    same correction as the other non-majority bars. Uniform inclusion gives
    identity weights. Targets outside the outer borders map to the tail bars.
    When inclusion varies within a bar, its average rate is an approximation to
    the continuous-target correction.
    """
    y_raw = np.asarray(y_raw, dtype=np.float64)
    probabilities = np.asarray(row_inclusion_probabilities, dtype=np.float64)
    if y_raw.ndim != 1 or probabilities.shape != y_raw.shape or not len(y_raw):
        raise ValueError(
            "Targets and inclusion probabilities must be nonempty 1D arrays "
            "of equal length."
        )
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities > 0) & (probabilities <= 1)).all()
        or not 0 < unobserved_target_probability <= 1
    ):
        raise ValueError(
            "Sampling inclusion probabilities must be finite and in (0, 1]."
        )
    if np.all(probabilities == unobserved_target_probability):
        return torch.zeros(bardist.num_bars, dtype=torch.float32)

    borders = bardist.borders.detach()
    bins = (
        bardist.map_to_bucket_idx(
            torch.as_tensor(y_raw, dtype=borders.dtype, device=borders.device)
        )
        .clamp(0, bardist.num_bars - 1)
        .cpu()
        .numpy()
    )
    counts = np.bincount(bins, minlength=bardist.num_bars)
    expected_counts = np.bincount(
        bins, weights=probabilities, minlength=bardist.num_bars
    )
    mean_inclusion = probabilities.mean()
    weights = np.full(bardist.num_bars, mean_inclusion / unobserved_target_probability)
    present = counts > 0
    weights[present] = mean_inclusion * counts[present] / expected_counts[present]
    return torch.from_numpy(np.log(weights)).float()


def temper_and_correct_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    log_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the ensemble temperature, then the per-bar correction.

    The order matters: the temperature reshapes the model's aggregated
    distribution, and the correction then undoes the sampler's prior shift on
    the result. Prediction and temperature tuning must both go through here so
    the two never compose differently.
    """
    if temperature != 1.0:
        logits = logits / temperature
    if log_weights is not None:
        logits = logits + log_weights.to(device=logits.device, dtype=logits.dtype)
    return logits
