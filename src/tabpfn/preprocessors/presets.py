"""Predefined preprocessor configurations for different model versions.

This module provides factory functions that return preprocessor configurations
for different versions of the model (v2, v2.5, default).
"""

from __future__ import annotations

from .definitions import PreprocessorConfig

_V2_FEATURE_SUBSAMPLING_THRESHOLD = 1_000_000


def default_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
    """Get default preprocessor configurations for classification."""
    return [
        PreprocessorConfig(
            name="squashing_scaler_default",
            append_original=False,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            name="none",
            categorical_name="numeric",
            max_features_per_estimator=500,
        ),
    ]


def default_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
    """Default preprocessor configurations for regression.

    These are the defaults used when training new models, which will then be stored in
    the model checkpoint.

    See `v2_regressor_preprocessor_configs()`, `v2_5_regressor_preprocessor_configs()`
    for the preprocessing used earlier versions of the model.
    """
    return [
        PreprocessorConfig(
            name="quantile_uni_coarse",
            append_original="auto",
            categorical_name="numeric",
            global_transformer_name=None,
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            name="squashing_scaler_default",
            append_original=False,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            max_features_per_estimator=500,
        ),
    ]


def v2_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
    """Get the preprocessor configuration for classification in v2 of the model."""
    return [
        PreprocessorConfig(
            "quantile_uni_coarse",
            append_original="auto",
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
            max_features_per_estimator=_V2_FEATURE_SUBSAMPLING_THRESHOLD,
        ),
        PreprocessorConfig(
            "none",
            categorical_name="numeric",
            max_features_per_estimator=_V2_FEATURE_SUBSAMPLING_THRESHOLD,
        ),
    ]


def v2_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
    """Get the preprocessor configuration for regression in v2 of the model."""
    return [
        PreprocessorConfig(
            "quantile_uni",
            append_original=True,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
        ),
        PreprocessorConfig("safepower", categorical_name="onehot"),
    ]


def v2_5_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
    """Get the preprocessor configuration for classification in v2.5 of the model."""
    return [
        PreprocessorConfig(
            name="squashing_scaler_default",
            append_original=False,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            name="none",
            categorical_name="numeric",
            max_features_per_estimator=500,
        ),
    ]


def v2_5_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
    """Get the preprocessor configuration for regression in v2.5 of the model."""
    return [
        PreprocessorConfig(
            name="quantile_uni_coarse",
            append_original="auto",
            categorical_name="numeric",
            global_transformer_name=None,
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            name="squashing_scaler_default",
            append_original=False,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            max_features_per_estimator=500,
        ),
    ]


__all__ = [
    "_V2_FEATURE_SUBSAMPLING_THRESHOLD",
    "default_classifier_preprocessor_configs",
    "default_regressor_preprocessor_configs",
    "v2_5_classifier_preprocessor_configs",
    "v2_5_regressor_preprocessor_configs",
    "v2_classifier_preprocessor_configs",
    "v2_regressor_preprocessor_configs",
]
