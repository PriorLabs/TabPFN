"""DEPRECATED: Projections from cell-level tensors to embedding space.

Please use the standalone modules like `nn.Linear` or an MLP module
to project the feature groups to the embedding space instead of using
this in the preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch
from torch import nn

from tabpfn.architectures.encoders import TorchPreprocessingStep


class LinearInputEncoderStep(TorchPreprocessingStep):
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

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""
        del state, kwargs

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Apply the linear transformation to the input.

        Args:
            state: Input state dictionary.
            single_eval_pos: Unused.
            **kwargs: Unused keyword arguments.

        Returns:
            Dictionary mapping `out_keys` to projected tensors.
        """
        del kwargs
        x = torch.cat([state[key] for key in self.in_keys], dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore

        # Ensure input tensor dtype matches the layer's weight dtype
        # Since this layer gets input from the outside we verify the dtype
        x = x.to(self.layer.weight.dtype)

        return {self.out_keys[0]: self.layer(x)}


class MLPInputEncoderStep(TorchPreprocessingStep):
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

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step. Does nothing for MLPInputEncoderStep."""
        del state, kwargs

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Apply the MLP transformation to the input.

        Args:
            state: Input state dictionary.
            **kwargs: Unused keyword arguments.

        Returns:
            Dictionary mapping `out_keys` to projected tensors.
        """
        del kwargs
        x = torch.cat([state[key] for key in self.in_keys], dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore

        # Ensure input tensor dtype matches the first layer's weight dtype
        x = x.to(self.mlp[0].weight.dtype)

        return {self.out_keys[0]: self.mlp(x)}
