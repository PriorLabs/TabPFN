"""Encoder step to remove duplicate features. Note, this is a No-op currently."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tabpfn.architectures.encoders import SeqEncStep

if TYPE_CHECKING:
    import torch


class RemoveDuplicateFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove duplicate features."""

    def __init__(self, *, normalize_on_train_only: bool = True, **kwargs: Any):
        """Initialize the RemoveDuplicateFeaturesEncoderStep.

        Args:
            normalize_on_train_only: Whether to normalize only on the training set.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Currently does nothing. Fit functionality not implemented."""

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor]:
        """Remove duplicate features from the input tensor.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the input tensor (removal not implemented).
        """
        # TODO: This uses a lot of memory, as it computes the covariance matrix for
        # each batch
        #   This could be done more efficiently, models go OOM with this
        return (x,)
        # normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        # x_norm = normalize_data(x[:, :normalize_position])
        # sel = torch.zeros(x.shape[1], x.shape[2], dtype=torch.bool, device=x.device)
        # for B in range(x_norm.shape[1]):
        #     cov_mat = (torch.cov(x_norm[:, B].transpose(1, 0)) > 0.999).float()
        #     cov_mat_sum_below_trace = torch.triu(cov_mat).sum(dim=0)
        #     sel[B] = cov_mat_sum_below_trace == 1.0

        # new_x = select_features(x, sel)

        # return (new_x,)
