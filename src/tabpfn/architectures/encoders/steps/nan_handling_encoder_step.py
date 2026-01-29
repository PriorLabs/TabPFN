"""Encoder step to handle NaN and infinite values in the input."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import TorchPreprocessingStep

from ._ops import torch_nanmean


class NanHandlingEncoderStep(TorchPreprocessingStep):
    """Encoder step to handle NaN and infinite values in the input."""

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        *,
        keep_nans: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main", "nan_indicators"),
    ):
        """Initialize the NanHandlingEncoderStep.

        Args:
            keep_nans: Whether to keep NaN values as separate indicators. Defaults to
            True.
            in_keys: The keys of the input tensors. Must be a single key.
            out_keys: The keys to assign the output tensors to.
        """
        self._validate_keys(in_keys=in_keys, out_keys=out_keys, keep_nans=keep_nans)

        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans
        self.register_buffer("feature_means_", torch.tensor([]), persistent=False)

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Compute the feature means on the training set for replacing NaNs.

        Args:
            state: The dictionary containing the input tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        del kwargs
        if single_eval_pos is None:
            raise ValueError(
                f"single_eval_pos must be provided for {self.__class__.__name__}"
            )

        x = state[self.in_keys[0]]
        self.feature_means_ = torch_nanmean(
            x[:single_eval_pos],
            axis=0,
            include_inf=True,
        )

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Replace NaN and infinite values in the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor and optionally the NaN indicators.
        """
        del kwargs
        x = state[self.in_keys[0]]
        nans_indicator: torch.Tensor | None = None
        if self.keep_nans:
            # TODO: There is a bug here: The values arriving here are already mapped
            # to nan if they were inf before
            nans_indicator = (
                torch.isnan(x) * NanHandlingEncoderStep.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * NanHandlingEncoderStep.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * NanHandlingEncoderStep.neg_inf_indicator
            ).to(x.dtype)

        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # replace nans with the mean of the corresponding feature
        x = x.clone()  # clone to avoid inplace operations
        x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]

        outputs = {self.out_keys[0]: x}
        if nans_indicator is not None:
            outputs[self.out_keys[1]] = nans_indicator
        return outputs

    def _validate_keys(
        self,
        *,
        in_keys: tuple[str, ...],
        out_keys: tuple[str, ...],
        keep_nans: bool,
    ) -> None:
        if len(in_keys) != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a single input key, got "
                f"`{len(in_keys)}`."
            )
        if len(out_keys) > 1 and not keep_nans:
            raise ValueError(
                f"{self.__class__.__name__} expects a single output key if keep_nans is"
                f" False, got `{len(out_keys)}`."
            )
        if keep_nans and len(out_keys) < 2:
            raise ValueError(
                f"{self.__class__.__name__} expects at least two output keys if "
                f"keep_nans is True, got `{len(out_keys)}`."
            )
