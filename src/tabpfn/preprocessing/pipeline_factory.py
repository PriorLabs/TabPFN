"""Methods to generate a preprocessing pipeline from ensemble configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingPipeline,
    PreprocessingStep,
    StepWithModalities,
)
from tabpfn.preprocessing.steps import (
    AddFingerprintFeaturesStep,
    AddSVDFeaturesStep,
    DifferentiableZNormStep,
    DuplicateImportantFeaturesStep,
    EncodeCategoricalFeaturesStep,
    NanHandlingPolynomialFeaturesStep,
    RemoveConstantFeaturesStep,
    ReshapeFeatureDistributionsStep,
    ShuffleFeaturesStep,
)
from tabpfn.preprocessing.torch.gpu_preprocessing_metadata import (
    is_gpu_quantile_eligible,
)

if TYPE_CHECKING:
    import numpy as np

    from tabpfn.preprocessing.configs import EnsembleConfig


def _polynomial_feature_settings(
    polynomial_features: Literal["no", "all"] | int,
) -> tuple[bool, int | None]:
    if isinstance(polynomial_features, int):
        assert polynomial_features > 0, "Poly. features to add must be >0!"
        return True, polynomial_features
    if polynomial_features == "all":
        return True, None
    if polynomial_features == "no":
        return False, None
    raise ValueError(f"Invalid polynomial_features value: {polynomial_features}")


def create_preprocessing_pipeline(
    config: EnsembleConfig,
    *,
    random_state: int | np.random.Generator | None,
    enable_gpu_preprocessing: bool = False,
    duplicate_local_indices: list[int] | None = None,
) -> PreprocessingPipeline:
    """Convert the ensemble configuration to a preprocessing pipeline.

    Args:
        config: Ensemble configuration.
        random_state: Random state for reproducibility.
        enable_gpu_preprocessing: When True, the quantile transform (if GPU-
            eligible), SVD, and shuffle steps are omitted from the CPU pipeline
            because they will run on GPU instead.
        duplicate_local_indices: Local column indices (in the post-subsample input
            space of this pipeline) whose values should be appended as duplicate
            columns at the very end of the pipeline. When ``None`` or empty, no
            duplication is performed.
    """
    steps: list[PreprocessingStep | StepWithModalities] = []

    pconfig = config.preprocess_config
    use_poly_features, max_poly_features = _polynomial_feature_settings(
        config.polynomial_features
    )
    if use_poly_features:
        steps.append(
            NanHandlingPolynomialFeaturesStep(
                max_features=max_poly_features,
                random_state=random_state,
            ),
        )

    # Insert duplicate step early so its appended columns flow through every
    # subsequent transform identically to their originals. ``local_indices``
    # reference positions in the post-subsample input (the values that flow
    # into RemoveConstantFeaturesStep next).
    if duplicate_local_indices:
        steps.append(DuplicateImportantFeaturesStep(duplicate_local_indices))

    steps.append(RemoveConstantFeaturesStep())

    # Decide whether the quantile transform moves to GPU.
    # The reshape step still runs on CPU (handling subsampling, categorical
    # reclassification, append_to_original) but uses "none" (identity) as the
    # transform so the actual quantile work happens on GPU.
    schedule_quantile_for_gpu = enable_gpu_preprocessing and is_gpu_quantile_eligible(
        pconfig.name
    )

    if pconfig.differentiable:
        steps.append(DifferentiableZNormStep())
    else:
        reshape_transform_name = "none" if schedule_quantile_for_gpu else pconfig.name
        steps.append(
            ReshapeFeatureDistributionsStep(
                transform_name=reshape_transform_name,
                append_to_original=pconfig.append_original,
                max_features_per_estimator=pconfig.max_features_per_estimator,
                apply_to_categorical=(pconfig.categorical_name == "numeric"),
                random_state=random_state,
                schedule_quantile_for_gpu=schedule_quantile_for_gpu,
            )
        )

        steps.append(
            EncodeCategoricalFeaturesStep(
                pconfig.categorical_name,
                random_state=random_state,
                max_onehot_cardinality=pconfig.max_onehot_cardinality,
            )
        )

        # SVD moves to GPU when enable_gpu_preprocessing is on.
        if not enable_gpu_preprocessing:
            use_global_transformer = (
                pconfig.global_transformer_name is not None
                and pconfig.global_transformer_name != "None"
            )
            if use_global_transformer:
                steps.append(
                    AddSVDFeaturesStep(
                        global_transformer_name=pconfig.global_transformer_name,  # type: ignore
                        random_state=random_state,
                    )
                )

    # Fingerprint moves to the GPU pipeline only when quantile or SVD is on
    # GPU (those steps precede fingerprint and change the data the hash sees).
    # When neither is on GPU, fingerprint stays on CPU at its original position.
    fingerprint_on_gpu = enable_gpu_preprocessing and (
        schedule_quantile_for_gpu
        or (
            not pconfig.differentiable
            and pconfig.global_transformer_name is not None
            and pconfig.global_transformer_name != "None"
        )
    )
    if config.add_fingerprint_feature and not fingerprint_on_gpu:
        steps.append(AddFingerprintFeaturesStep())

    # Shuffle moves to GPU when enable_gpu_preprocessing is on.
    if not enable_gpu_preprocessing:
        steps.append(
            ShuffleFeaturesStep(
                shuffle_method=config.feature_shift_decoder,
                shuffle_index=config.feature_shift_count,
                random_state=random_state,
            ),
        )

    return PreprocessingPipeline(steps)
