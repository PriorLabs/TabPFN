#  Copyright (c) Prior Labs GmbH 2025.

"""Interfaces for encoders."""

from __future__ import annotations

import abc
from typing import Any
from typing_extensions import override

import torch
from torch import nn


# Note that inheriting from nn.Sequential is not strictly necessary, because
# we don't want to include learnable parameters in this pipeline.
# Because previous TabPFN checkpoints used nn.Sequential and contain
# keys like `encoder.5.layer.weight` in the state_dict, we keep the inheritance
# for now.
class TorchPreprocessingPipeline(torch.nn.Sequential):
    """Container for a sequence of GPU preprocessing steps.

    These transformations can alter feature sets or values or add
    additional "encoding" like nan indicators to the input.
    These steps must be non-learnable (i.e., no trainable weights or model
    parameters) and strictly used for data preparation.
    """

    def __init__(
        self,
        steps: list[TorchPreprocessingStep],
        # output_key is set for backwards compatibility
        # with encoders that still have the projections in this pipeline.
        output_key: str | None = None,
    ):
        super().__init__(*steps)
        self.output_key = output_key

        # If we are in backwards compatibility mode, we expect the last step to
        # be a LinearInputEncoderStep or MLPInputEncoderStep that combine
        # the entries of the state dict parsed through the encoder.
        if self.output_key is not None:
            assert any(
                s.__class__.__name__
                # Avoid circular import by checking the class name instead of importing
                in ("LinearInputEncoderStep", "MLPInputEncoderStep")
                for s in self
            ), "output_key is only supported for LinearInputEncoderStep"

    # For now, we disable compilation for the preprocessing pipeline because
    # there are multiple data-dependent control flows in the steps that break the graph.
    @torch.compiler.disable
    @override
    def forward(
        self,
        input: dict[str, torch.Tensor] | torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Fit and transform the preprocessing pipeline on input data.

        Args:
            input: The input tensor or dictionary of tensors in the case of
                multiple modalities. If a tensor is provided, it is wrapped in a
                dictionary with the key "main". The tensor must have the shape
                [R, B * G, F] with
                R = number of rows (train + test),
                B = batch size
                G = number of feature groups
                F = number of features per group
            single_eval_pos: The position to use to split train and test data.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            A dictionary of tensors with all keys created in the pipeline or
            (for backwards compatibility only) the output tensor specified by
            the `output_key` parameter of the last step.
        """
        x = {"main": input} if isinstance(input, torch.Tensor) else input
        for module in self:
            x = module(x, single_eval_pos=single_eval_pos, **kwargs)

        # For backwards compatibility
        if self.output_key is not None:
            return x[self.output_key]

        return x


class TorchPreprocessingStep(abc.ABC, nn.Module):
    """Abstract base class for sequential encoder steps.

    TorchPreprocessingStep is a wrapper around a module that defines the expected
    input keys and the produced output keys. The outputs are assigned to the output keys
    in the order specified by `out_keys`.

    Subclasses should create any state that depends on the train set in
    `_fit` and using it in `_transform`. This allows fitting on data first
    and doing inference later without refitting.
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Init.

        Args:
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    @abc.abstractmethod
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step on the training set.

        Args:
            state: The dictionary containing the input tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.
        """
        ...

    @abc.abstractmethod
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Transform the data using the fitted encoder step.

        Args:
            state: The dictionary containing the input tensors.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The transformed output dictionary.
        """
        ...

    def _validate_input_keys(self, state: dict) -> None:
        """Raise KeyError if expected input keys are missing from state."""
        missing = [k for k in self.in_keys if k not in state]
        if missing:
            raise KeyError(
                f"{self.__class__.__name__}: missing input tensor in dict `{missing}`. "
                f"Available keys in state: `{list(state.keys())}`"
            )

    def _validate_output_keys(self, outputs: dict) -> None:
        """Raise KeyError if unexpected output keys are present in outputs."""
        unexpected = set(outputs.keys()) - set(self.out_keys)
        if unexpected:
            raise KeyError(
                f"{self.__class__.__name__}: unexpected output tensor in dict "
                f"`{unexpected}`. Available keys in state: "
                f"`{list(outputs.keys())}`"
            )

        missing = [k for k in self.out_keys if k not in outputs]
        if missing:
            raise KeyError(
                f"{self.__class__.__name__}: missing output tensor in dict "
                f"`{missing}`. Available keys in state: `{list(outputs.keys())}`"
            )

    @override
    def forward(
        self,
        state: dict,
        *,
        single_eval_pos: int | None = None,
        cache_trainset_representation: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Perform the forward pass of the encoder step.

        Args:
            state: The input state dictionary containing the input tensors.
            cache_trainset_representation:
                Whether to cache the training set representation. Only supported for
                _fit and _transform (not _forward).
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The updated state dictionary with the output tensor or a tuple of
            output tensors.
        """
        self._validate_input_keys(state)

        do_fit = (
            single_eval_pos is not None and single_eval_pos > 0
        ) or not cache_trainset_representation
        if do_fit:
            self._fit(state, single_eval_pos=single_eval_pos, **kwargs)
        outputs = self._transform(state, single_eval_pos=single_eval_pos, **kwargs)

        self._validate_output_keys(outputs)

        state.update(outputs)
        return state
