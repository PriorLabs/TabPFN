#  Copyright (c) Prior Labs GmbH 2025.

"""Factory for creating torch preprocessing pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
)
from tabpfn.preprocessing.torch.steps import (
    TorchSoftClipOutliersStep,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.configs import EnsembleConfig


def create_gpu_preprocessing_pipeline(
    config: EnsembleConfig,
    *,
    keep_fitted_cache: bool = False,
) -> TorchPreprocessingPipeline | None:
    """Create a GPU preprocessing pipeline based on configuration."""
    steps: list[tuple[TorchPreprocessingStep, set[FeatureModality]]] = []

    if config.outlier_removal_std is not None:
        steps.append(
            (
                TorchSoftClipOutliersStep(n_sigma=config.outlier_removal_std),
                {FeatureModality.NUMERICAL},
            )
        )

    if len(steps) > 0:
        return TorchPreprocessingPipeline(steps, keep_fitted_cache=keep_fitted_cache)

    return None
