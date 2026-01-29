"""Encoder step to handle variable number of features."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import TorchPreprocessingStep


class NormalizeFeatureGroupsEncoderStep(TorchPreprocessingStep):
    """Encoder step to scale feature groups that have been padded.

    Scales the input by the number of used features to keep the variance
    of the input constant, even when zeros are appended.
    """

    def __init__(
        self,
        num_features_per_group: int,
        *,
        normalize_by_sqrt: bool = True,
        normalize_by_used_features: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the NormalizeFeatureGroupsEncoderStep.

        Args:
            num_features_per_group: The number of features to transform the input to.
            normalize_by_sqrt: Legacy option to normalize by sqrt instead of the number
                of used features.
            normalize_by_used_features: Whether to normalize by the number of used
                features. No-op if this is False. This flag is deprecated and will be
                removed in the future.
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == len(out_keys) == 1, (
            f"{self.__class__.__name__} expects a single input and output key."
        )

        super().__init__(in_keys, out_keys)
        self.num_features_per_group = num_features_per_group
        self.normalize_by_sqrt = normalize_by_sqrt
        self.normalize_by_used_features = normalize_by_used_features
        self.number_of_used_features_: torch.Tensor | None = None

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Compute the number of used features on the training set.

        Args:
            state: The state dict containing the input tensors.
            **kwargs: Additional keyword arguments (unused).
        """
        del kwargs
        if not self.normalize_by_used_features:
            return

        x = state[self.in_keys[0]]

        if x.shape[-1] % self.num_features_per_group != 0:
            raise ValueError(
                f"The number of features per group must be a divisor of the number of "
                f"features in the input tensor. Got `{x.shape[-1]}` and "
                f"{self.num_features_per_group=}. This can be fixed by padding the "
                f"input tensor with zeros to make it divisible by "
                f"{self.num_features_per_group=}."
            )

        # Checks for constant features to scale features in group that
        # have constant features. Constant features could have been added
        # from padding to feature group size.
        self.non_constants_mask_ = ((x[1:] == x[0]).sum(0) != (x.shape[0] - 1)).cpu()
        # move to cpu to avoid dangling GPU memory, tested in
        # test__to__after_fit_and_predict__no_tensors_left_on_old_device
        self.number_of_used_features_ = torch.clip(
            self.non_constants_mask_.sum(-1).unsqueeze(-1),
            min=1,
        ).cpu()

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
        if not self.normalize_by_used_features:
            return {self.out_keys[0]: state[self.in_keys[0]]}

        if x.shape[-1] == 0:
            return {
                self.out_keys[0]: torch.zeros(
                    *x.shape[:-1],
                    self.num_features_per_group,
                    device=x.device,
                    dtype=x.dtype,
                )
            }

        assert self.number_of_used_features_ is not None, (
            "number_of_used_features_ is not set. This step must be fitted before "
            "calling _transform."
        )

        scale = self.num_features_per_group / self.number_of_used_features_.to(x.device)
        x = x * torch.sqrt(scale) if self.normalize_by_sqrt else x * scale

        # Set constant features to 0 to avoid scaling them
        x[:, ~self.non_constants_mask_] = 0

        return {self.out_keys[0]: x}
