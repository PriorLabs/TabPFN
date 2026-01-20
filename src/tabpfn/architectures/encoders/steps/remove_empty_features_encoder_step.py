"""Encoder step to remove empty (constant) features."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import TorchPreprocessingStep

from ._ops import select_features


class RemoveEmptyFeaturesEncoderStep(TorchPreprocessingStep):
    """Encoder step to remove empty (constant) features."""

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the RemoveEmptyFeaturesEncoderStep.

        Args:
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == len(out_keys) == 1, (
            f"{self.__class__.__name__} expects a single input and output key."
        )
        super().__init__(in_keys, out_keys)
        self.register_buffer("column_selection_mask", None, persistent=False)

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Compute the non-empty feature selection mask on the training set.

        Args:
            state: The state dict containing the input tensor under `in_keys[0]`.
            **kwargs: Additional keyword arguments (unused).
        """
        del kwargs
        x = state[self.in_keys[0]]
        self.column_selection_mask = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Remove empty features from the input tensor.

        Args:
            state: The state dict containing the input tensor under `in_keys[0]`.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A dict mapping `out_keys[0]` to the transformed tensor with empty features
            removed.
        """
        del kwargs
        x = state[self.in_keys[0]]

        orig_last_dim = x.shape[-1]
        # Ensure that the mask is a bool, because the buffer may get converted to a
        # a float if .to() is called on the containing module.
        x = select_features(x, self.column_selection_mask.type(torch.bool))

        potential_padding = -x.shape[-1] % orig_last_dim
        x = torch.nn.functional.pad(x, pad=(0, potential_padding), value=0)

        return {self.out_keys[0]: x}
