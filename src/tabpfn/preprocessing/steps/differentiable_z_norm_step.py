"""Differentiable Z-Norm Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import torch

from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureSchema


class DifferentiableZNormStep(PreprocessingStep):
    """Differentiable Z-Norm Step."""

    def __init__(self):
        super().__init__()

        self.means = torch.tensor([])
        self.stds = torch.tensor([])

    @override
    def _fit(  # type: ignore
        self,
        X: torch.Tensor,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        self.means = X.mean(dim=0, keepdim=True)
        self.stds = X.std(dim=0, keepdim=True)
        return feature_schema

    @override
    def _transform(  # type: ignore
        self, X: torch.Tensor, *, is_test: bool = False
    ) -> tuple[torch.Tensor, None, None]:
        assert X.shape[1] == self.means.shape[1]
        assert X.shape[1] == self.stds.shape[1]
        return (X - self.means) / self.stds, None, None


__all__ = [
    "DifferentiableZNormStep",
]
