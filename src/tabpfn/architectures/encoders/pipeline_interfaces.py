#  Copyright (c) Prior Labs GmbH 2025.

"""Interfaces for encoders."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class InputEncoder(nn.Module):
    """Base class for input encoders.

    All input encoders should subclass this class and implement the `forward` method.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        """Encode the input tensor.

        Args:
            x: The input tensor to encode.
            single_eval_pos: The position to use for single evaluation.

        Returns:
            The encoded tensor.
        """
        raise NotImplementedError


class SequentialEncoder(nn.Sequential, InputEncoder):
    """An encoder that applies a sequence of encoder steps.

    SequentialEncoder allows building an encoder from a sequence of EncoderSteps.
    The input is passed through each step in the provided order.
    """

    def __init__(self, *args: SeqEncStep, output_key: str = "output", **kwargs: Any):
        """Initialize the SequentialEncoder.

        Args:
            *args: A list of SeqEncStep instances to apply in order.
            output_key:
                The key to use for the output of the encoder in the state dict.
                Defaults to "output", i.e. `state["output"]` will be returned.
            **kwargs: Additional keyword arguments passed to `nn.Sequential`.
        """
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def forward(self, input: dict, **kwargs: Any) -> torch.Tensor:
        """Apply the sequence of encoder steps to the input.

        Args:
            input:
                The input state dictionary.
                If the input is not a dict and the first layer expects one input key,
                the input tensor is mapped to the key expected by the first layer.
            **kwargs: Additional keyword arguments passed to each encoder step.

        Returns:
            The output of the final encoder step.
        """
        # If the input is not a dict and the first layer expects one input, mapping the
        #   input to the first input key must be correct
        if not isinstance(input, dict) and len(self[0].in_keys) == 1:
            input = {self[0].in_keys[0]: input}  # noqa: A001

        for module in self:
            input = module(input, **kwargs)  # noqa: A001

        return input[self.output_key] if self.output_key is not None else input


class SeqEncStep(nn.Module):
    """Abstract base class for sequential encoder steps.

    SeqEncStep is a wrapper around a module that defines the expected input keys
    and the produced output keys. The outputs are assigned to the output keys
    in the order specified by `out_keys`.

    Subclasses should either implement `_forward` or `_fit` and `_transform`.
    Subclasses that transform `x` should always use `_fit` and `_transform`,
    creating any state that depends on the train set in `_fit` and using it in
    `_transform`. This allows fitting on data first and doing inference later
    without refitting.
    Subclasses that work with `y` can alternatively use `_forward` instead.
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the SeqEncStep.

        Args:
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    # Either implement _forward:

    def _forward(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Forward pass of the encoder step.

        Implement this if not implementing _fit and _transform.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The output tensor or a tuple of output tensors.
        """
        raise NotImplementedError()

    # Or implement _fit and _transform:

    def _fit(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step on the training set.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.
        """
        raise NotImplementedError

    def _transform(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, ...]:
        """Transform the data using the fitted encoder step.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The transformed output tensor or a tuple of output tensors.
        """
        raise NotImplementedError

    def forward(
        self,
        state: dict,
        *,
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
            The updated state dictionary with the output tensors assigned to the output
            keys.
        """
        args = [state[in_key] for in_key in self.in_keys]
        if hasattr(self, "_fit"):
            if kwargs["single_eval_pos"] or not cache_trainset_representation:
                self._fit(*args, **kwargs)
            out = self._transform(*args, **kwargs)
        else:
            assert not cache_trainset_representation
            out = self._forward(*args, **kwargs)
            # TODO: I think nothing is using _forward now

        assert isinstance(
            out,
            tuple,
        ), (
            f"out is not a tuple: {out}, type: {type(out)}, class: "
            f"{self.__class__.__name__}"
        )
        assert len(out) == len(self.out_keys)
        state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
        return state
