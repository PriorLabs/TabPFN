"""Differentiable Z-Norm Step."""

from __future__ import annotations

from typing_extensions import override

import torch

from tabpfn.preprocessing.steps.preprocessing_helpers import (
    FeaturePreprocessingTransformerStep,
)


class DifferentiableZNormStep(FeaturePreprocessingTransformerStep):
    """Differentiable Z-Norm Step."""

    def __init__(self):
        super().__init__()

        self.means = torch.tensor([])
        self.stds = torch.tensor([])

    @override
    def _fit(self, X: torch.Tensor, categorical_features: list[int]) -> list[int]:  # type: ignore
        self.means = X.mean(dim=0, keepdim=True)
        self.stds = X.std(dim=0, keepdim=True)
        return categorical_features

    @override
    def _transform(self, X: torch.Tensor, *, is_test: bool = False) -> torch.Tensor:  # type: ignore
        assert X.shape[1] == self.means.shape[1]
        assert X.shape[1] == self.stds.shape[1]
        return (X - self.means) / self.stds


__all__ = [
    "DifferentiableZNormStep",
]
