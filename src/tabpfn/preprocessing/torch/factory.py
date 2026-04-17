#  Copyright (c) Prior Labs GmbH 2025.

"""Factory for creating torch preprocessing pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.torch.gpu_preprocessing_metadata import (
    compute_effective_n_quantiles,
)
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
)
from tabpfn.preprocessing.torch.steps import (
    TorchAddFingerprintFeaturesStep,
    TorchAddSVDFeaturesStep,
    TorchSelectiveQuantileTransformerStep,
    TorchShuffleFeaturesStep,
    TorchSoftClipOutliersStep,
)

if TYPE_CHECKING:
    import numpy as np

    from tabpfn.preprocessing.configs import EnsembleConfig
    from tabpfn.preprocessing.datamodel import FeatureSchema


def create_gpu_preprocessing_pipeline(
    config: EnsembleConfig,
    *,
    keep_fitted_cache: bool = False,
    enable_gpu_preprocessing: bool = False,
    feature_schema: FeatureSchema | None = None,
    n_train_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> TorchPreprocessingPipeline | None:
    """Create a GPU preprocessing pipeline based on configuration.

    Args:
        config: Ensemble configuration.
        keep_fitted_cache: Whether to keep fitted state for cache reuse.
        enable_gpu_preprocessing: When True, adds quantile transform, SVD, and
            shuffle steps.  Requires ``feature_schema`` (to read GPU quantile
            target annotations) and ``n_train_samples``.
        feature_schema: Feature schema from the CPU preprocessing output.
            Used to read ``scheduled_gpu_transform`` column annotations.
        n_train_samples: Number of training samples (after subsampling).
        random_state: Random state for the shuffle step.
    """
    steps: list[tuple[TorchPreprocessingStep, set[FeatureModality] | None]] = []
    pconfig = config.preprocess_config

    if enable_gpu_preprocessing and feature_schema is not None:
        # Quantile transform — target columns are annotated in the schema
        # by ReshapeFeatureDistributionsStep(mark_quantile_for_gpu=True).
        quantile_target_indices = (
            feature_schema.get_indices_marked_for_gpu_quantile_transform()
        )
        quantile_on_gpu = len(quantile_target_indices) > 0
        if quantile_on_gpu and n_train_samples is not None:
            n_quantiles = compute_effective_n_quantiles(pconfig.name, n_train_samples)
            steps.append(
                (
                    TorchSelectiveQuantileTransformerStep(
                        n_quantiles=n_quantiles,
                        target_column_indices=quantile_target_indices,
                    ),
                    None,  # operates on explicit indices, receives full tensor
                )
            )

        # SVD features
        svd_on_gpu = (
            pconfig.global_transformer_name is not None
            and pconfig.global_transformer_name != "None"
            and not pconfig.differentiable
        )
        if svd_on_gpu and pconfig.global_transformer_name is not None:
            steps.append(
                (
                    TorchAddSVDFeaturesStep(
                        global_transformer_name=pconfig.global_transformer_name,
                    ),
                    None,
                )
            )

        # Fingerprint: after quantile + SVD, before shuffle — same relative
        # position as in the CPU-only path.  The hash values will differ from
        # CPU-only because the torch quantile implementation produces slightly
        # different boundary values, but the position is consistent.
        # Note that this will slow down the GPU pipeline unnecessarily because
        # of GPU<->CPU transfers.
        # TODO: Run fingerprint features on GPU natively.
        add_fingerprint = config.add_fingerprint_feature and (
            quantile_on_gpu or svd_on_gpu
        )
        if add_fingerprint:
            steps.append((TorchAddFingerprintFeaturesStep(), None))

        # Shuffle features (always when enable_gpu_preprocessing)
        steps.append(
            (
                TorchShuffleFeaturesStep(
                    shuffle_method=config.feature_shift_decoder,
                    shuffle_index=config.feature_shift_count,
                    random_state=random_state,
                ),
                None,  # operates on all columns
            )
        )

    # Outlier removal runs LAST to match the CPU-only execution order where
    # it is the only GPU step and runs after all CPU preprocessing.
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
