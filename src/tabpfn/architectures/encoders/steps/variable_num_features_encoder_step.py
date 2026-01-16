"""Encoder step to handle variable number of features."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import GPUPreprocessingStep


class VariableNumFeaturesEncoderStep(GPUPreprocessingStep):
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
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the VariableNumFeaturesEncoderStep.

        Args:
            num_features: The number of features to transform the input to.
            normalize_by_used_features: Whether to normalize by the number of used
                features. No-op if this is False.
            normalize_by_sqrt: Legacy option to normalize by sqrt instead of the number
                of used features.
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == len(out_keys) == 1, (
            f"{self.__class__.__name__} expects a single input and output key."
        )

        super().__init__(in_keys, out_keys)
        self.normalize_by_used_features = normalize_by_used_features
        self.num_features_per_group = num_features
        self.normalize_by_sqrt = normalize_by_sqrt
        self.number_of_used_features_: torch.Tensor | None = None

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Compute the number of used features on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        del kwargs
        x = state[self.in_keys[0]]

        # Checks for constant features to scale features in group that
        # have constant features. Constant features could have been added
        # from padding to feature group size.
        sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)
        self.number_of_used_features_ = torch.clip(
            sel.sum(-1).unsqueeze(-1),
            min=1,
        )
        self.padding_features_ = -x.shape[-1] % self.num_features_per_group

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Transform the input tensor to have a fixed number of features.

        Args:
            state: The dictionary containing the input tensors of shape
            [..., F]
            where
            - F = number of features per group

        Returns:
            A dict mapping `out_keys[0]` to the transformed tensor.
            The output tensor has shape [..., F], where
            F = number of features per group.
        """
        del kwargs
        x = state[self.in_keys[0]]

        assert self.number_of_used_features_ is not None, (
            "number_of_used_features_ is not set. This step must be fitted before "
            "calling _transform."
        )

        if self.padding_features_ > 0:
            x = torch.nn.functional.pad(x, pad=(0, self.padding_features_), value=0)

        if self.normalize_by_used_features:
            scale = self.num_features_per_group / self.number_of_used_features_
            x = x * torch.sqrt(scale) if self.normalize_by_sqrt else x * scale

        return {self.out_keys[0]: x}
