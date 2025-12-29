"""Dataclasses and type definitions for preprocessing."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution


@dataclass
class BaseDatasetConfig:
    """Base configuration class for holding dataset specifics."""

    config: EnsembleConfig
    X_raw: object
    y_raw: object
    cat_ix: list[int]


@dataclass
class ClassifierDatasetConfig(BaseDatasetConfig):
    """Classification Dataset + Model Configuration class."""


@dataclass
class RegressorDatasetConfig(BaseDatasetConfig):
    """Regression Dataset + Model Configuration class."""

    znorm_space_bardist_: FullSupportBarDistribution | None = field(default=None)

    @property
    def bardist_(self) -> FullSupportBarDistribution:
        """DEPRECATED: Accessing `bardist_` is deprecated.
        Use `znorm_space_bardist_` instead.
        """
        warnings.warn(
            "`bardist_` is deprecated and will be removed in a future version. "
            "Please use `znorm_space_bardist_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.znorm_space_bardist_

    @bardist_.setter
    def bardist_(self, value: FullSupportBarDistribution) -> None:
        """DEPRECATED: Setting `bardist_` is deprecated.
        Use `znorm_space_bardist_`.
        """
        warnings.warn(
            "`bardist_` is deprecated and will be removed in a future version. "
            "Please use `znorm_space_bardist_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.znorm_space_bardist_ = value


@dataclass(frozen=True, eq=True)
class PreprocessorConfig:
    """Configuration for data preprocessors."""

    name: Literal[
        "per_feature",  # a different transformation for each feature
        "power",  # a standard sklearn power transformer
        "safepower",  # a power transformer that prevents some numerical issues
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",  # quantile transformations with few quantiles up to many
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "squashing_scaler_default",
        "squashing_scaler_max10",
        "robust",  # a standard sklearn robust scaler
        "kdi",
        "none",  # no transformation (only standardization in transformer)
        "kdi_random_alpha",
        "kdi_uni",
        "kdi_random_alpha_uni",
        "adaptive",
        "norm_and_kdi",
        # KDI with alpha collection
        "kdi_alpha_0.3_uni",
        "kdi_alpha_0.5_uni",
        "kdi_alpha_0.8_uni",
        "kdi_alpha_1.0_uni",
        "kdi_alpha_1.2_uni",
        "kdi_alpha_1.5_uni",
        "kdi_alpha_2.0_uni",
        "kdi_alpha_3.0_uni",
        "kdi_alpha_5.0_uni",
        "kdi_alpha_0.3",
        "kdi_alpha_0.5",
        "kdi_alpha_0.8",
        "kdi_alpha_1.0",
        "kdi_alpha_1.2",
        "kdi_alpha_1.5",
        "kdi_alpha_2.0",
        "kdi_alpha_3.0",
        "kdi_alpha_5.0",
    ]
    categorical_name: Literal[
        # categorical features are pretty much treated as ordinal, just not resorted
        "none",
        # categorical features are treated as numeric,
        # that means they are also power transformed for example
        "numeric",
        # "onehot": categorical features are onehot encoded
        "onehot",
        # "ordinal": categorical features are sorted and encoded as
        # integers from 0 to n_categories - 1
        "ordinal",
        # "ordinal_shuffled": categorical features are encoded as integers
        # from 0 to n_categories - 1 in a random order
        "ordinal_shuffled",
        "ordinal_very_common_categories_shuffled",
    ] = "none"
    append_original: bool | Literal["auto"] = False
    max_features_per_estimator: int = 500
    global_transformer_name: (
        Literal[
            "scaler",
            "svd",
            "svd_quarter_components",
        ]
        | None
    ) = None
    differentiable: bool = False

    def __str__(self) -> str:  # pragma: no cover - delegated functionality
        return (
            f"{self.name}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (f"_max_feats_per_est_{self.max_features_per_estimator}")
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
        )


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble member."""

    preprocess_config: PreprocessorConfig
    add_fingerprint_feature: bool
    polynomial_features: Literal["no", "all"] | int
    feature_shift_count: int
    feature_shift_decoder: Literal["shuffle", "rotate"] | None
    subsample_ix: object
    _model_index: int

    @classmethod
    def generate_for_classification(  # noqa: PLR0913
        cls,
        *,
        num_estimators: int,
        subsample_samples: int | float | list[object] | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        preprocessor_configs: list[PreprocessorConfig],
        class_shift_method: Literal["rotate", "shuffle"] | None,
        n_classes: int,
        random_state: int | object | None,
        num_models: int,
    ) -> list[ClassifierEnsembleConfig]:
        from .core import generate_classifier_ensemble_configs

        return generate_classifier_ensemble_configs(
            num_estimators=num_estimators,
            subsample_samples=subsample_samples,
            max_index=max_index,
            add_fingerprint_feature=add_fingerprint_feature,
            polynomial_features=polynomial_features,
            feature_shift_decoder=feature_shift_decoder,
            preprocessor_configs=preprocessor_configs,
            class_shift_method=class_shift_method,
            n_classes=n_classes,
            random_state=random_state,
            num_models=num_models,
        )

    @classmethod
    def generate_for_regression(  # noqa: PLR0913
        cls,
        *,
        num_estimators: int,
        subsample_samples: int | float | list[object] | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        preprocessor_configs: list[PreprocessorConfig],
        target_transforms: list[object],
        random_state: int | object | None,
        num_models: int,
    ) -> list[RegressorEnsembleConfig]:
        from .core import generate_regressor_ensemble_configs

        return generate_regressor_ensemble_configs(
            num_estimators=num_estimators,
            subsample_samples=subsample_samples,
            max_index=max_index,
            add_fingerprint_feature=add_fingerprint_feature,
            polynomial_features=polynomial_features,
            feature_shift_decoder=feature_shift_decoder,
            preprocessor_configs=preprocessor_configs,
            target_transforms=target_transforms,
            random_state=random_state,
            num_models=num_models,
        )

    def to_pipeline(self, *, random_state: int | object | None):
        from .core import build_preprocessing_pipeline

        return build_preprocessing_pipeline(self, random_state=random_state)


@dataclass
class ClassifierEnsembleConfig(EnsembleConfig):
    """Configuration for a classifier ensemble member."""

    class_permutation: object | None


@dataclass
class RegressorEnsembleConfig(EnsembleConfig):
    """Configuration for a regression ensemble member."""

    target_transform: object | None


__all__ = [
    "BaseDatasetConfig",
    "ClassifierDatasetConfig",
    "RegressorDatasetConfig",
    "PreprocessorConfig",
    "EnsembleConfig",
    "ClassifierEnsembleConfig",
    "RegressorEnsembleConfig",
]
