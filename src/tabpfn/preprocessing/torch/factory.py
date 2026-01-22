#  Copyright (c) Prior Labs GmbH 2025.

"""Factory for creating torch preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from .datamodel import FeatureModality
from .pipeline_interface import TorchPreprocessingPipeline, TorchPreprocessingStep
from .steps import TorchRemoveOutliersStep, TorchStandardScalerStep


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


def create_preprocessing_pipeline(
    config: PipelineConfig,
) -> TorchPreprocessingPipeline:
    """Create a preprocessing pipeline based on configuration.

    Args:
        config: Configuration specifying which preprocessing steps to include.

    Returns:
        A TorchPreprocessingPipeline with the configured steps.
    """
    steps: list[tuple[TorchPreprocessingStep, set[FeatureModality]]] = []

    # Outlier removal only for numerical features
    if config.remove_outliers:
        steps.append(
            (
                TorchRemoveOutliersStep(n_sigma=config.outlier_n_sigma),
                {FeatureModality.NUMERICAL},
            )
        )

    # Standard scaling can apply to multiple modalities
    if config.standard_scale:
        scale_modalities = {FeatureModality.NUMERICAL}
        if config.scale_categorical:
            scale_modalities.add(FeatureModality.CATEGORICAL)
        steps.append(
            (
                TorchStandardScalerStep(),
                scale_modalities,
            )
        )

    return TorchPreprocessingPipeline(steps)
