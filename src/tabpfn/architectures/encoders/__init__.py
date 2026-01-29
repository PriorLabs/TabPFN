"""DEPRECATION WARNING: Please note that this module will be deprecated in future
versions of TabPFN in favor of a new torch-based preprocessing pipeline.
"""

from .pipeline_interfaces import (
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
)
from .steps import (
    FeatureTransformEncoderStep,
    FrequencyFeatureEncoderStep,
    LinearInputEncoderStep,
    MLPInputEncoderStep,
    MulticlassClassificationTargetEncoderStep,
    NanHandlingEncoderStep,
    NormalizeFeatureGroupsEncoderStep,
    RemoveDuplicateFeaturesEncoderStep,
    RemoveEmptyFeaturesEncoderStep,
)

__all__ = (
    "FeatureTransformEncoderStep",
    "FrequencyFeatureEncoderStep",
    "LinearInputEncoderStep",
    "MLPInputEncoderStep",
    "MulticlassClassificationTargetEncoderStep",
    "NanHandlingEncoderStep",
    "NormalizeFeatureGroupsEncoderStep",
    "RemoveDuplicateFeaturesEncoderStep",
    "RemoveEmptyFeaturesEncoderStep",
    "TorchPreprocessingPipeline",
    "TorchPreprocessingStep",
)
