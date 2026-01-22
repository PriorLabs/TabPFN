#  Copyright (c) Prior Labs GmbH 2025.

"""Factory for creating torch preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tabpfn.preprocessing.torch.datamodel import FeatureModality
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
)
from tabpfn.preprocessing.torch.steps import (
    TorchRemoveOutliersStep,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.configs import EnsembleConfig


@dataclass
class PipelineConfig:
    """Configuration for creating a preprocessing pipeline.

    Attributes:
        remove_outliers: Whether to apply outlier removal.
        outlier_n_sigma: Number of standard deviations for outlier threshold.
        standard_scale: Whether to apply standard scaling.
        scale_categorical: Whether to also scale categorical features.
    """

    remove_outliers: bool = True
    outlier_n_sigma: float = 4.0
    standard_scale: bool = True
    scale_categorical: bool = True


def create_gpu_preprocessing_pipeline(
    config: EnsembleConfig,
) -> TorchPreprocessingPipeline | None:
    """Create a GPU preprocessing pipeline based on configuration."""
    steps: list[tuple[TorchPreprocessingStep, set[FeatureModality]]] = []

    if config.outlier_removal_std is not None:
        steps.append(
            (
                TorchRemoveOutliersStep(n_sigma=config.outlier_removal_std),
                {FeatureModality.NUMERICAL},
            )
        )

    if len(steps) > 0:
        return TorchPreprocessingPipeline(steps)
    return None
