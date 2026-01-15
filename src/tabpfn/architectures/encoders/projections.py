"""Projections from cell-level tensors to embedding space."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tabpfn.architectures.encoders import SeqEncStep

# TODO: Remove the SeqEncStep inheritance in here and make this a
# regular nn.Module that get's a dict of inputs and projects them up
# according to some agreement specified in the model!


class LinearInputEncoderStep(SeqEncStep):
    """A simple linear input encoder step."""

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        """Initialize the LinearInputEncoderStep.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            replace_nan_by_zero: Whether to replace NaN values in the input by zero.
                Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
            in_keys: The keys of the input tensors. Defaults to ("main",).
            out_keys: The keys to assign the output tensors to. Defaults to ("output",).
        """
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _fit(self, *x: torch.Tensor, **kwargs: Any) -> None:
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:  # noqa: ARG002
        """Apply the linear transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore

        # Ensure input tensor dtype matches the layer's weight dtype
        # Since this layer gets input from the outside we verify the dtype
        x = x.to(self.layer.weight.dtype)

        return (self.layer(x),)


class MLPInputEncoderStep(SeqEncStep):
    """An MLP-based input encoder step."""

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        num_layers: int = 2,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        """Initialize the MLPInputEncoderStep.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            hidden_dim: The hidden dimension of the MLP. If None, defaults to emsize.
            activation: The activation function to use. Either "gelu" or "relu".
            num_layers: The number of layers in the MLP (minimum 2).
            replace_nan_by_zero: Whether to replace NaN values in the input by zero.
            Defaults to False.
            bias: Whether to use a bias term in the linear layers. Defaults to True.
            in_keys: The keys of the input tensors. Defaults to ("main",).
            out_keys: The keys to assign the output tensors to. Defaults to ("output",).
        """
        super().__init__(in_keys, out_keys)

        if hidden_dim is None:
            hidden_dim = emsize

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 for an MLP encoder")

        self.replace_nan_by_zero = replace_nan_by_zero

        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        # First layer: input -> hidden
        layers.append(nn.Linear(num_features, hidden_dim, bias=bias))
        layers.append(act_fn)

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(act_fn)

        # Output layer: hidden -> emsize
        layers.append(nn.Linear(hidden_dim, emsize, bias=bias))

        self.mlp = nn.Sequential(*layers)

    def _fit(self, *x: torch.Tensor, **kwargs: Any) -> None:
        """Fit the encoder step. Does nothing for MLPInputEncoderStep."""

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:  # noqa: ARG002
        """Apply the MLP transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore

        # Ensure input tensor dtype matches the first layer's weight dtype
        x = x.to(self.mlp[0].weight.dtype)

        return (self.mlp(x),)
