#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.regression_metrics import (
    ranked_probability_score_loss_from_bar_logits,
)


def _bardist() -> FullSupportBarDistribution:
    return FullSupportBarDistribution(
        borders=torch.tensor([0.0, 1.0, 2.0, 2.5, 4.0, 6.0])
    )


@pytest.mark.parametrize("loss_type", ["crps", "crls"])
def test__rps_loss__returns_one_loss_per_query(loss_type: str) -> None:
    # The loss is returned unreduced so that callers choose the reduction: the
    # finetuning losses average over the batch, the calibration search pools folds
    # weighted by their holdout size.
    bardist = _bardist()
    torch.manual_seed(0)
    logits_BQL = torch.randn(2, 3, bardist.num_bars)
    targets_BQ = torch.tensor([[0.5, 3.0, 5.0], [1.5, 2.2, 4.4]])

    losses_BQ = ranked_probability_score_loss_from_bar_logits(
        logits_BQL=logits_BQL,
        targets_BQ=targets_BQ,
        bardist_loss_fn=bardist,
        loss_type=loss_type,  # type: ignore[arg-type]
    )

    assert losses_BQ.shape == (2, 3)
    assert torch.isfinite(losses_BQ).all()
    assert (losses_BQ >= 0.0).all()


@pytest.mark.parametrize("loss_type", ["crps", "crls"])
def test__rps_loss__scores_nan_targets_as_zero(loss_type: str) -> None:
    # Zero rather than NaN, so that a mean over the result stays finite and matches
    # the convention of `BarDistribution.forward`.
    bardist = _bardist()
    torch.manual_seed(0)
    logits_BQL = torch.randn(1, 3, bardist.num_bars)
    targets_BQ = torch.tensor([[0.5, float("nan"), 5.0]])

    losses_BQ = ranked_probability_score_loss_from_bar_logits(
        logits_BQL=logits_BQL,
        targets_BQ=targets_BQ,
        bardist_loss_fn=bardist,
        loss_type=loss_type,  # type: ignore[arg-type]
    )

    assert losses_BQ[0, 1] == 0.0
    assert (losses_BQ[0, [0, 2]] > 0.0).all()


def test__rps_loss__is_minimised_by_the_confident_correct_prediction() -> None:
    # A sanity check on orientation: both variants are losses, so putting all mass on
    # the target's bin must score below spreading it over every bin.
    bardist = _bardist()
    targets_BQ = torch.tensor([[2.2]])
    # 2.2 falls in the bin spanning [2.0, 2.5), i.e. index 2.
    confident_BQL = torch.tensor([[[-10.0, -10.0, 10.0, -10.0, -10.0]]])
    uniform_BQL = torch.zeros(1, 1, bardist.num_bars)

    for loss_type in ("crps", "crls"):
        confident = ranked_probability_score_loss_from_bar_logits(
            logits_BQL=confident_BQL,
            targets_BQ=targets_BQ,
            bardist_loss_fn=bardist,
            loss_type=loss_type,  # type: ignore[arg-type]
        )
        uniform = ranked_probability_score_loss_from_bar_logits(
            logits_BQL=uniform_BQL,
            targets_BQ=targets_BQ,
            bardist_loss_fn=bardist,
            loss_type=loss_type,  # type: ignore[arg-type]
        )
        assert float(confident) < float(uniform)
