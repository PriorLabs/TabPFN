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
    EncodeCategoricalFeaturesStep,
    NanHandlingPolynomialFeaturesStep,
    RemoveConstantFeaturesStep,
    ReshapeFeatureDistributionsStep,
    ShuffleFeaturesStep,
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
) -> PreprocessingPipeline:
    """Convert the ensemble configuration to a preprocessing pipeline."""
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

    steps.append(RemoveConstantFeaturesStep())

    if pconfig.differentiable:
        steps.append(DifferentiableZNormStep())
    else:
        steps.append(
            ReshapeFeatureDistributionsStep(
                transform_name=pconfig.name,
                append_to_original=pconfig.append_original,
                max_features_per_estimator=pconfig.max_features_per_estimator,
                apply_to_categorical=(pconfig.categorical_name == "numeric"),
                random_state=random_state,
            )
        )

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

        steps.append(
            EncodeCategoricalFeaturesStep(
                pconfig.categorical_name,
                random_state=random_state,
            )
        )

    if config.add_fingerprint_feature:
        steps.append(AddFingerprintFeaturesStep())

    steps.append(
        ShuffleFeaturesStep(
            shuffle_method=config.feature_shift_decoder,
            shuffle_index=config.feature_shift_count,
            random_state=random_state,
        ),
    )
    return PreprocessingPipeline(steps)
