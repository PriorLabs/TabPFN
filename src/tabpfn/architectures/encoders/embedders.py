"""Embedders from feature groups to embedding space."""

from __future__ import annotations

from typing import cast
from typing_extensions import override

import torch
from torch import nn


class LinearFeatureGroupEmbedder(nn.Module):
    """A simple linear projection from input feature group to embedding space."""

    def __init__(
        self,
        *,
        num_features_per_group_with_metadata: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
    ) -> None:
        """Initialize the projection.

        Args:
            num_features_per_group_with_metadata: The number of input features per group
                with additional metadata features like nan_indicators.
            emsize: The embedding size, i.e. the number of output features.
            replace_nan_by_zero: Whether to replace NaN values in the input by zero.
            bias: Whether to use a bias term in the linear layer.
        """
        super().__init__()
        self.layer = nn.Linear(num_features_per_group_with_metadata, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input tensor to embedding space.

        Args:
            x: Input tensor with last dimension size
            `num_features_per_group_with_metadata`.

        Returns:
            Projected tensor with last dimension size `emsize`.
        """
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore
        x = x.to(self.layer.weight.dtype)  # Why is this necessary?
        return self.layer(x)


class MLPFeatureGroupEmbedder(nn.Module):
    """An MLP projection from input feature group to embedding space."""

    def __init__(
        self,
        *,
        num_features_per_group_with_metadata: int,
        emsize: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        num_layers: int = 2,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
    ) -> None:
        """Initialize the projection.

        Args:
            num_features_per_group_with_metadata: The number of input features per group
                with additional metadata features like nan_indicators.
            emsize: The embedding size, i.e. the number of output features.
            hidden_dim: The hidden dimension of the MLP. If None, defaults to emsize.
            activation: The activation function to use. Either "gelu" or "relu".
            num_layers: The number of layers in the MLP (minimum 2).
            replace_nan_by_zero: Whether to replace NaN values in the input by zero.
            bias: Whether to use a bias term in the linear layers.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = emsize
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 for an MLP projection")

        self.replace_nan_by_zero = replace_nan_by_zero

        if activation == "gelu":
            act_fn: nn.Module = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list[nn.Module] = []
        layers.append(
            nn.Linear(num_features_per_group_with_metadata, hidden_dim, bias=bias)
        )
        layers.append(act_fn)

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, emsize, bias=bias))
        self.mlp = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input tensor to embedding space.

        Args:
            x: Input tensor with last dimension size
            `num_features_per_group_with_metadata`.

        Returns:
            Projected tensor with last dimension size `emsize`.
        """
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore
        first_layer = cast("nn.Linear", self.mlp[0])
        x = x.to(first_layer.weight.dtype)  # Why is this necessary?
        return self.mlp(x)
