"""Encoder step to pad features to a multiple of a group size."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import TorchPreprocessingStep


class FeatureGroupPaddingAndReshapeStep(TorchPreprocessingStep):
    """Encodes and reshapes feature dimension.

    Pads feature dimension with zeros so it is divisible by `num_features_per_group`.
    Reshapes so feature group size is the last dimension.

    This is needed for the VariableNumFeaturesEncoderStep to work correctly.
    """

    def __init__(
        self,
        num_features_per_group: int,
        *,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the FeatureGroupEncoderAndReshapeStep.

        The output tensor after this step has shape [Ri, B, G, F], where:
        - R = number of rows
        - B = batch size
        - G = number of feature groups
        - F = number of features per group

        Args:
            num_features_per_group: The number of features per group. If the input
                feature dimension is not divisible by this number, the step pads with
                zeros.
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == len(out_keys) == 1, (
            f"{self.__class__.__name__} expects a single input and output key."
        )

        super().__init__(in_keys, out_keys)
        if num_features_per_group <= 0:
            raise ValueError(
                f"num_features_per_group must be > 0, got {num_features_per_group}"
            )
        self.num_features_per_group = num_features_per_group

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Fit is a no-op as padding depends only on the runtime feature dimension."""
        del state, kwargs

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Pad the last dimension with zeros if needed.

        Args:
            state: The dictionary containing the input tensors.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A dict mapping each `out_key` to the padded tensor.
        """
        del kwargs

        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                f"{self.__class__.__name__} requires in_keys and out_keys to have the "
                f"same length, got {len(self.in_keys)} and {len(self.out_keys)}."
            )

        outputs: dict[str, torch.Tensor] = {}
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            x_RiBC = state[in_key]

            # Ri = number of input rows (train + test, before adding thinking rows)
            # B = batch size
            # C = original number of columns before padding and grouping
            Ri, B, C = x_RiBC.shape
            num_padding_features = (-C) % self.num_features_per_group
            if num_padding_features > 0:
                # C (columns) now padded
                x_RiBC = torch.nn.functional.pad(
                    x_RiBC,
                    pad=(0, num_padding_features),
                    value=0,
                )

            num_padded_columns = x_RiBC.shape[-1]
            assert num_padded_columns % self.num_features_per_group == 0
            num_feature_groups = num_padded_columns // self.num_features_per_group

            # Reshape so that the feature group dimension is last.
            outputs[out_key] = x_RiBC.reshape(
                Ri,
                B,
                num_feature_groups,
                self.num_features_per_group,
            )

        return outputs
