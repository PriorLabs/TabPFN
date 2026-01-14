"""Encoder step to handle variable number of features."""

from __future__ import annotations

from typing import Any

import torch

from tabpfn.architectures.encoders import SeqEncStep


class VariableNumFeaturesEncoderStep(SeqEncStep):
    """Encoder step to handle variable number of features.

    Transforms the input to a fixed number of features by appending zeros.
    Also normalizes the input by the number of used features to keep the variance
    of the input constant, even when zeros are appended.
    """

    def __init__(
        self,
        num_features: int,
        *,
        normalize_by_used_features: bool = True,
        normalize_by_sqrt: bool = True,
        **kwargs: Any,
    ):
        """Initialize the VariableNumFeaturesEncoderStep.

        Args:
            num_features: The number of features to transform the input to.
            normalize_by_used_features: Whether to normalize by the number of used
            features.
            normalize_by_sqrt: Legacy option to normalize by sqrt instead of the number
            of used features.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_by_used_features = normalize_by_used_features
        self.num_features = num_features
        self.normalize_by_sqrt = normalize_by_sqrt
        self.number_of_used_features_ = None

    def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:  # noqa: ARG002
        """Compute the number of used features on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)
        self.number_of_used_features_ = torch.clip(
            sel.sum(-1).unsqueeze(-1),
            min=1,
        ).cpu()

    def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:  # noqa: ARG002
        """Transform the input tensor to have a fixed number of features.

        Args:
            x: The input tensor of shape (seq_len, batch_size, num_features_old).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor of shape
            (seq_len, batch_size, num_features).
        """
        if x.shape[2] == 0:
            return (
                torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    self.num_features,
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
        if self.normalize_by_used_features:
            if self.normalize_by_sqrt:
                # Verified that this gives indeed unit variance with appended zeros
                x = x * torch.sqrt(
                    self.num_features / self.number_of_used_features_.to(x.device),
                )
            else:
                x = x * (self.num_features / self.number_of_used_features_.to(x.device))

        zeros_appended = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros_appended], -1)
        return (x,)
