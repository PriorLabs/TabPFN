"""Encoder step to remove empty (constant) features."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from typing import Any

import torch

from tabpfn.architectures.encoders import SeqEncStep

from ._ops import select_features


class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove empty (constant) features."""

    def __init__(self, **kwargs: Any):
        """Initialize the RemoveEmptyFeaturesEncoderStep.

        Args:
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.register_buffer("column_selection_mask", None, persistent=False)

    def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:  # noqa: ARG002
        """Compute the non-empty feature selection mask on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        self.column_selection_mask = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)

    def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:  # noqa: ARG002
        """Remove empty features from the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor with empty features removed.
        """
        # Ensure that the mask is a bool, because the buffer may get converted to a
        # a float if .to() is called on the containing module.
        return (select_features(x, self.column_selection_mask.type(torch.bool)),)
