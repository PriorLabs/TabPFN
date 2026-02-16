"""Preprocessor and ensemble config objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline


@dataclass(frozen=True, eq=True)
class PreprocessorConfig:
    """Configuration for data preprocessing.

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
            "svd",
            "svd_quarter_components",
        ]
        | None
    ) = None
    differentiable: bool = False

    @override
    def __str__(self) -> str:
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


# TODO: (Klemens)
# Make this frozen (frozen=True)
@dataclass
class EnsembleConfig:
    """Configuration for an ensemble member.

    Attributes:
        preprocess_config: Preprocessor configuration to use.
        add_fingerprint_feature: Whether to add fingerprint features.
        polynomial_features: Maximum number of polynomial features to add, if any.
        feature_shift_count: How much to shift the features columns.
        feature_shift_decoder: How to shift features.
        subsample_ix: Indices of samples to use for this ensemble member.
            If `None`, no subsampling is done.
        outlier_removal_std: Number of standard deviations from the mean to consider a
            sample an outlier. If `None`, no outliers are removed.
    """

    preprocess_config: PreprocessorConfig
    add_fingerprint_feature: bool
    polynomial_features: Literal["no", "all"] | int
    feature_shift_count: int
    feature_shift_decoder: Literal["shuffle", "rotate"] | None
    subsample_ix: npt.NDArray[np.int64] | None  # OPTIM: Could use uintp
    outlier_removal_std: float | None
    # Internal index specifying which model to use for this ensemble member.
    _model_index: int


@dataclass
class ClassifierEnsembleConfig(EnsembleConfig):
    """Configuration for a classifier ensemble member."""

    class_permutation: np.ndarray | None


@dataclass
class RegressorEnsembleConfig(EnsembleConfig):
    """Configuration for a regression ensemble member."""

    target_transform: TransformerMixin | Pipeline | None


__all__ = [
    "ClassifierEnsembleConfig",
    "EnsembleConfig",
    "PreprocessorConfig",
    "RegressorEnsembleConfig",
]
