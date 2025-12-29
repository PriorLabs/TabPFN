from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence, TYPE_CHECKING

import numpy as np
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline
    import torch

    from tabpfn.preprocessors.preprocessing_helpers import SequentialFeatureTransformer


@dataclass
class BaseDatasetConfig:
    """Base configuration class for holding dataset specifics."""

    config: Sequence["EnsembleConfig"]
    X_raw: np.ndarray | "torch.Tensor"
    y_raw: np.ndarray | "torch.Tensor"
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
    """Configuration for data preprocessors.

    Attributes:
        name: Name of the preprocessor.
        categorical_name:
            Name of the categorical encoding method.
            Options: "none", "numeric", "onehot", "ordinal", "ordinal_shuffled", "none".
        append_to_original: If set to "auto", this is dynamically set to
            True if the number of features is less than 500, and False otherwise.
            Note that if set to "auto" and `max_features_per_estimator` is set as well,
            this flag will become False if the number of features is larger than
            `max_features_per_estimator / 2`. If True, the transformed features are
            appended to the original features, however both are capped at the
            max_features_per_estimator threshold, this should be used with caution as a
            given model might not be configured for it.
        max_features_per_estimator: Maximum number of features per estimator. In case
            the dataset has more features than this, the features are subsampled for
            each estimator independently. If append to original is set to True we can
            still have more features.
        global_transformer_name: Name of the global transformer to use.
    """

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

    def __str__(self) -> str:  # noqa: D401
        """Return a concise string identifier for this preprocessor configuration.

        The format is ``"{name}_cat:{categorical_name}[_and_none]_max_feats_per_est_{max_features_per_estimator}[_global_transformer_{global_transformer_name}]"``.
        The ``[_and_none]`` segment is included only when ``append_original`` is
        ``True`` and the ``[_global_transformer_{global_transformer_name}]``
        segment is included only when ``global_transformer_name`` is not ``None``.
        """

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


@dataclass(frozen=True, eq=True)
class EnsembleConfig:
    """Configuration for an ensemble member."""

    preprocess_config: PreprocessorConfig
    add_fingerprint_feature: bool
    polynomial_features: Literal["no", "all"] | int
    feature_shift_count: int
    feature_shift_decoder: Literal["shuffle", "rotate"] | None
    subsample_ix: "npt.NDArray[np.int64]" | None  # OPTIM: Could use uintp
    # Internal index specifying which model to use for this ensemble member.
    _model_index: int

    @classmethod
    def generate_for_classification(  # noqa: PLR0913
        cls,
        *,
        num_estimators: int,
        subsample_samples: int | float | list[np.ndarray] | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        preprocessor_configs: Sequence[PreprocessorConfig],
        class_shift_method: Literal["rotate", "shuffle"] | None,
        n_classes: int,
        random_state: int | np.random.Generator | None,
        num_models: int,
    ) -> list["ClassifierEnsembleConfig"]:
        from .core import generate_classification_ensemble_configs

        return generate_classification_ensemble_configs(
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
        subsample_samples: int | float | list[np.ndarray] | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        preprocessor_configs: Sequence[PreprocessorConfig],
        random_state: int | np.random.Generator | None,
        target_transforms: Sequence["TransformerMixin" | "Pipeline" | None],
        num_models: int,
    ) -> list["RegressorEnsembleConfig"]:
        from .core import generate_regression_ensemble_configs

        return generate_regression_ensemble_configs(
            num_estimators=num_estimators,
            subsample_samples=subsample_samples,
            max_index=max_index,
            add_fingerprint_feature=add_fingerprint_feature,
            polynomial_features=polynomial_features,
            feature_shift_decoder=feature_shift_decoder,
            preprocessor_configs=preprocessor_configs,
            random_state=random_state,
            target_transforms=target_transforms,
            num_models=num_models,
        )

    def to_pipeline(
        self, *, random_state: int | np.random.Generator | None
    ) -> "SequentialFeatureTransformer":
        from .core import build_pipeline

        return build_pipeline(self, random_state=random_state)


@dataclass
class ClassifierEnsembleConfig(EnsembleConfig):
    """Configuration for a classifier ensemble member."""

    class_permutation: np.ndarray | None


@dataclass
class RegressorEnsembleConfig(EnsembleConfig):
    """Configuration for a regression ensemble member."""

    target_transform: "TransformerMixin" | "Pipeline" | None


__all__ = [
    "BaseDatasetConfig",
    "ClassifierDatasetConfig",
    "ClassifierEnsembleConfig",
    "EnsembleConfig",
    "PreprocessorConfig",
    "RegressorDatasetConfig",
    "RegressorEnsembleConfig",
]
