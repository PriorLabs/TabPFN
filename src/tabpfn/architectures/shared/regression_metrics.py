#  Copyright (c) Prior Labs GmbH 2026.

"""Regression metrics over binned target distributions.

Collects the regression metrics that scikit-learn does not provide, so that the
finetuning losses (`tabpfn.finetuning.finetuned_regressor`) and the inference-time
temperature calibration (`tabpfn.inference_tuning`) score predictions with one shared
implementation and cannot drift apart. Metrics that scikit-learn already implements
(MSE, MAE, R^2) are used from there instead.

Metrics here return one loss per query rather than a reduced scalar, leaving the
reduction to the caller: the finetuning losses average over the batch, while the
calibration search pools folds weighted by their holdout size.

The negative log-likelihood of a bar distribution is deliberately absent; it is
`BarDistribution.forward` in `tabpfn.architectures.shared.bar_distribution`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from tabpfn.architectures.shared.bar_distribution import BarDistribution


def ranked_probability_score_loss_from_bar_logits(
    *,
    logits_BQL: torch.Tensor,
    targets_BQ: torch.Tensor,
    bardist_loss_fn: BarDistribution,
    loss_type: Literal["crps", "crls"] = "crps",
) -> torch.Tensor:
    """Compute a ranked-probability loss from bar distribution logits.

    This implements scoring rules for *ordered categorical* outcomes via the
    cumulative distribution function (CDF), using the bar bins as ordered
    categories. Definitions taken from https://scoringrules.readthedocs.io/en/latest/theory.html

        CRPS (squared): sum_k=1^K w_k (CDF(k) - y_k)^2
        CRLS (log):     -sum_k=1^K w_k log(|CDF(k) + y_k - 1|)

    where K is the number of bins, w_k is the width of bin k, CDF(k) is the predicted
    cumulative probability up to bin k, and y_k is the target cumulative
    probability up to bin k.
    The weighting uses bar/bin widths of the bar distribution.

    Note that this is the RPS/RLS loss weighted by the bar/bin widths, so we refer
    to it as 'Continuous' RPS/CRLS loss, i.e. CRPS/CRLS loss.

    Shapes suffixes:
        B=batch * estimators, L=logits, Q=n_queries.

    Args:
        logits_BQL: Bar distribution logits of shape (B, Q, L).
        targets_BQ: Targets of shape (B, Q)
        bardist_loss_fn: The BarDistribution instance used for bin mapping and
            bucket index mapping.
        loss_type: Which variant to compute. "crps" uses squared CDF differences,
            "crls" uses a log score applied to the cumulative probabilities.

    Returns:
        The loss of every query, of shape (B, Q), left unreduced so that callers
        choose the reduction. Queries whose target is NaN contribute 0.0, so that a
        mean over the result matches the convention of `BarDistribution.forward` and
        counts them as if they were perfectly predicted.
    """
    bucket_widths_L = bardist_loss_fn.bucket_widths.to(logits_BQL.device)
    assert bucket_widths_L.shape == (logits_BQL.shape[-1],), (
        f"bucket_widths_L.shape: {bucket_widths_L.shape} "
        f"logits_BQL.shape: {logits_BQL.shape}"
    )
    probs_BQL = torch.softmax(logits_BQL, dim=-1)
    pred_cdf_BQL = torch.cumsum(probs_BQL, dim=-1)

    ignore_loss_mask_BQ = torch.isnan(targets_BQ)
    # Filled with zeros if the target is NaN.
    # Will be ignored in final loss below.
    filled_targets_BQ = torch.where(
        ignore_loss_mask_BQ, torch.zeros_like(targets_BQ), targets_BQ
    )

    target_bins_BQ = bardist_loss_fn.map_to_bucket_idx(filled_targets_BQ).clamp(
        0, bardist_loss_fn.num_bars - 1
    )
    # The target CDF is a step function: 0 for bins < target_bin, 1 for bins >=
    # target_bin.
    bin_indices_L = torch.arange(probs_BQL.shape[-1], device=probs_BQL.device)
    target_cdf_BQL = (bin_indices_L.view(1, 1, -1) >= target_bins_BQ.unsqueeze(-1)).to(
        probs_BQL.dtype
    )

    if loss_type == "crps":
        cdf_diff_BQL = pred_cdf_BQL - target_cdf_BQL
        cdf_term_losses_BQL = cdf_diff_BQL.square()
    else:
        eps = torch.finfo(pred_cdf_BQL.dtype).eps
        cdf = pred_cdf_BQL.clamp(eps, 1 - eps)
        # target_cdf_BQL is binary, so we can expand the log(|CDF(k) + y_k - 1|) term
        # into two separate terms and use log1p. This is numerically more stable.
        cdf_term_losses_BQL = target_cdf_BQL * (-torch.log(cdf)) + (
            1 - target_cdf_BQL
        ) * (-torch.log1p(-cdf))

    weighted_term_losses_BQL = cdf_term_losses_BQL * bucket_widths_L.view(1, 1, -1)
    losses_BQ = weighted_term_losses_BQL.sum(dim=-1)

    if ignore_loss_mask_BQ.any():
        losses_BQ = torch.where(
            ignore_loss_mask_BQ, torch.zeros_like(losses_BQ), losses_BQ
        )

    return losses_BQ
