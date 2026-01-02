"""Additional configuration options for inference."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import Literal

import pydantic

from tabpfn.constants import ModelVersion, TaskType
from tabpfn.preprocessing import (
    PreprocessorConfig,
    default_classifier_preprocessor_configs,
    default_regressor_preprocessor_configs,
    v2_5_classifier_preprocessor_configs,
    v2_5_regressor_preprocessor_configs,
    v2_classifier_preprocessor_configs,
    v2_regressor_preprocessor_configs,
)


# By default Pydantic dataclasses will ignore unrecognised config items, extra="forbid"
# will raise an exception instead.
@pydantic.dataclasses.dataclass(config=pydantic.ConfigDict(extra="forbid"))
class InferenceConfig:
    """Additional configuration options for inference.

    Several configuration options for inference are exposed in the `TabPFNClassifier`
    and `TabPFNRegressor` interfaces. The options in this class are more advanced and
    not expected to be changed by the (standard) user.

    Several of the preprocessing options are supported by our code for efficiency
    reasons (to avoid loading TabPFN multiple times). However, these can also be
    applied outside of the model interface.

    This class must be serializable as it is peristed in the model checkpoints.

    Do not edit the default values in this class, as this can affect the backwards
    compatibility of the model checkpoints. Instead, edit `get_default()`.
    """

    PREPROCESS_TRANSFORMS: list[PreprocessorConfig]
    """The preprocessing applied to the data before passing it to TabPFN. See
    `PreprocessorConfig` for options and more details. If multiple `PreprocessorConfig`
    are provided, they are (repeatedly) applied across different estimators.

    By default, for classification, two preprocessors are applied:
        1. Uses the original input data, all features transformed with a quantile
            scaler, and the first n-many components of SVD transformer (whereby
            n is a fract of on the number of features or samples). Categorical features
            are ordinal encoded but all categories with less than 10 features are
            ignored.
        2. Uses the original input data, with categorical features as ordinal encoded.

    By default, for regression, two preprocessor are applied:
        1. The same as for classification, with a minimal different quantile scaler.
        2. The original input data power transformed and categories onehot encoded.
    """

    MAX_UNIQUE_FOR_CATEGORICAL_FEATURES: int = 30
    """The maximum number of unique values for a feature to be considered
    categorical. Otherwise, it is considered numerical."""
    MIN_UNIQUE_FOR_NUMERICAL_FEATURES: int = 4
    """The minimum number of unique values for a feature to be considered numerical.
    Otherwise, it is considered categorical."""
    MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE: int = 100
    """The minimum number of samples in the data to run our infer which features might
    be categorical."""

    OUTLIER_REMOVAL_STD: float | None | Literal["auto"] = "auto"
    """The number of standard deviations from the mean to consider a sample an outlier.
        - If None, no outliers are removed.
        - If float, the number of standard deviations from the mean to consider a sample
            an outlier.
        - If "auto", the OUTLIER_REMOVAL_STD is automatically determined.
            -> 12.0 for classification and None for regression.
    """

    FEATURE_SHIFT_METHOD: Literal["shuffle", "rotate"] | None = "shuffle"
    """The method used to shift features during preprocessing for ensembling to emulate
     the effect of invariance to feature position. Without ensembling, TabPFN is not
     invariant to feature position due to using a transformer. Moreover, shifting
     features can have a positive effect on the model's performance. The options are:
        - If "shuffle", the features are shuffled.
        - If "rotate", the features are rotated (think of a ring).
        - If None, no feature shifting is done.
    """
    CLASS_SHIFT_METHOD: Literal["rotate", "shuffle"] | None = "shuffle"
    """The method used to shift classes during preprocessing for ensembling to emulate
    the effect of invariance to class order. Without ensembling, TabPFN is not
    invariant to class order due to using a transformer. Shifting classes can
    have a positive effect on the model's performance. The options are:
        - If "shuffle", the classes are shuffled.
        - If "rotate", the classes are rotated (think of a ring).
        - If None, no class shifting is done.
    """

    FINGERPRINT_FEATURE: bool = True
    """Whether to add a fingerprint feature to the data. The added feature is a hash of
    the row, counting up for duplicates. This helps TabPFN to distinguish between
    duplicated data points in the input data. Otherwise, duplicates would be less
    obvious during attention. This is expected to improve prediction performance and
    help with stability if the data has many sample duplicates."""
    POLYNOMIAL_FEATURES: Literal["no", "all"] | int = "no"
    """The number of 2 factor polynomial features to generate and add to the original
    data before passing the data to TabPFN. The polynomial features are generated by
    multiplying the original features together, e.g., this might add a feature `x1*x2`
    to the features, if `x1` and `x2` are features. In  total, this can add up O(n^2)
    many features. Adding polynomial features can  improve predictive performance by
    exploiting simple feature engineering.
        - If "no", no polynomial features are added.
        - If "all", all possible polynomial features are added.
        - If an int, determines the maximal number of polynomial features to add to the
         original data.
    """
    SUBSAMPLE_SAMPLES: int | float | list | None = None
    """Subsample the input data sample/row-wise before performing any preprocessing
    and the TabPFN forward pass.
        - If None, no subsampling is done.
        - If an int, the number of samples to subsample (or oversample if
            `SUBSAMPLE_SAMPLES` is larger than the number of samples).
        - If a float, the percentage of samples to subsample.
        - If a list arrays of indices, the indices to subsample for each estimator.
            If the length of the outer list is less than the number of estimators, the
            indices are repeated for the remaining estimators.
    """

    REGRESSION_Y_PREPROCESS_TRANSFORMS: tuple[str | None, ...] = (None, "safepower")
    """The preprocessing applied to the target variable before passing it to TabPFN for
    regression. This can be understood as scaling the target variable to better predict
    it. The preprocessors should be passed as a tuple/list and are then (repeatedly)
    used by the estimators in the ensembles.

    By default, we use no preprocessing and a power transformation (if we have
    more than one estimator).

    The options are:
        - None: no preprocessing is done.
        - One of the options from
          `tabpfn.preprocessing.get_all_reshape_feature_distribution_preprocessors()`
    """

    USE_SKLEARN_16_DECIMAL_PRECISION: bool = False
    """Whether to round the probabilities to float 16 to match the precision of
     scikit-learn. This can help with reproducibility and compatibility with
     scikit-learn but is not recommended for general use. This is not exposed to the
     user or as a hyperparameter.
     To improve reproducibility,set `._sklearn_16_decimal_precision = True` before
     calling `.predict()` or `.predict_proba()`."""

    MAX_NUMBER_OF_CLASSES: int = 10
    """The number of classes seen during pretraining for classification. If the
    number of classes is larger than this number, TabPFN requires an additional step
    to predict for more than classes."""
    MAX_NUMBER_OF_FEATURES: int = 500
    """The number of features that the pretraining was intended for. If the number of
    features is larger than this number, you may see degraded performance. Note, this
    is not the number of features seen by the model during pretraining but also accounts
    for expected generalization (i.e., length extrapolation)."""
    MAX_NUMBER_OF_SAMPLES: int = 10_000
    """The number of samples that the pretraining was intended for. If the number of
    samples is larger than this number, you may see degraded performance. Note, this
    is not the number of samples seen by the model during pretraining but also accounts
    for expected generalization (i.e., length extrapolation)."""

    FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM: bool = True
    """Whether to repair any borders of the bar distribution in regression that are NaN
     after the transformation. This can happen due to multiple reasons and should in
     general always be done."""

    _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD: None = None
    _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD: float = 12.0

    def override_with_user_input(
        self, user_config: dict | InferenceConfig | None
    ) -> InferenceConfig:
        """Return a new config with fields specified in `user_config` overwritten.

        Args:
            user_config: Config provided by the user at inference time.
                If a dictionary, then the keys must match attributes of
                    `InferenceConfig` and will be used to override these attributes.
                If an `InferenceConfig` object, then the whole config is overridden with
                    the values from the user config.
                If None, then a copy of this config is returned with no fields changed.
        """
        if user_config is None:
            return deepcopy(self)
        if isinstance(user_config, InferenceConfig):
            return deepcopy(user_config)
        if isinstance(user_config, dict):
            return dataclasses.replace(self, **user_config)
        raise ValueError(
            f"{user_config=}\nUnknown user config provided, see config above."
        )

    @classmethod
    def get_default(
        cls, task_type: TaskType, model_version: ModelVersion | Literal["latest"]
    ) -> InferenceConfig:
        """Return the default config for the given model version and task type.

        For model versions after v2.5, the inference config is generated by calling this
        function with `model_version=latest` and stored in the checkpoint. This stored
        config is then loaded and used for inference.

        For v2 and v2.5, the config is not stored in the checkpoint. Thus, for backwards
        compatiblity, we define the v2 and v2.5 configs here and use those during
        inference.
        """
        if model_version == ModelVersion.V2:
            if task_type == "multiclass":
                return _get_v2_config(v2_classifier_preprocessor_configs())
            if task_type == "regression":
                return _get_v2_config(v2_regressor_preprocessor_configs())
        if model_version == ModelVersion.V2_5:
            if task_type == "multiclass":
                return _get_v2_5_config(v2_5_classifier_preprocessor_configs())
            if task_type == "regression":
                return _get_v2_5_config(v2_5_regressor_preprocessor_configs())

        if task_type == "multiclass":
            return InferenceConfig(
                PREPROCESS_TRANSFORMS=default_classifier_preprocessor_configs(),
                MAX_NUMBER_OF_FEATURES=2000,
                MAX_NUMBER_OF_SAMPLES=50_000,
            )
        if task_type == "regression":
            return InferenceConfig(
                PREPROCESS_TRANSFORMS=default_regressor_preprocessor_configs(),
                MAX_NUMBER_OF_FEATURES=2000,
                MAX_NUMBER_OF_SAMPLES=50_000,
            )
        raise ValueError(f"Unknown {task_type=} {model_version=}")


def _get_v2_config(
    preprocessor_configs: list[PreprocessorConfig],
) -> InferenceConfig:
    return InferenceConfig(
        MAX_UNIQUE_FOR_CATEGORICAL_FEATURES=30,
        MIN_UNIQUE_FOR_NUMERICAL_FEATURES=4,
        MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE=100,
        OUTLIER_REMOVAL_STD="auto",
        FEATURE_SHIFT_METHOD="shuffle",
        CLASS_SHIFT_METHOD="shuffle",
        FINGERPRINT_FEATURE=True,
        POLYNOMIAL_FEATURES="no",
        SUBSAMPLE_SAMPLES=None,
        PREPROCESS_TRANSFORMS=preprocessor_configs,
        REGRESSION_Y_PREPROCESS_TRANSFORMS=(None, "safepower"),
        USE_SKLEARN_16_DECIMAL_PRECISION=False,
        MAX_NUMBER_OF_CLASSES=10,
        MAX_NUMBER_OF_FEATURES=500,
        MAX_NUMBER_OF_SAMPLES=10_000,
        FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM=True,
        _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD=None,
        _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD=12.0,
    )


def _get_v2_5_config(
    preprocessor_configs: list[PreprocessorConfig],
) -> InferenceConfig:
    return InferenceConfig(
        MAX_UNIQUE_FOR_CATEGORICAL_FEATURES=30,
        MIN_UNIQUE_FOR_NUMERICAL_FEATURES=4,
        MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE=100,
        OUTLIER_REMOVAL_STD="auto",
        FEATURE_SHIFT_METHOD="shuffle",
        CLASS_SHIFT_METHOD="shuffle",
        FINGERPRINT_FEATURE=True,
        POLYNOMIAL_FEATURES="no",
        SUBSAMPLE_SAMPLES=None,
        PREPROCESS_TRANSFORMS=preprocessor_configs,
        REGRESSION_Y_PREPROCESS_TRANSFORMS=(None, "safepower"),
        USE_SKLEARN_16_DECIMAL_PRECISION=False,
        MAX_NUMBER_OF_CLASSES=10,
        MAX_NUMBER_OF_FEATURES=2000,
        MAX_NUMBER_OF_SAMPLES=50_000,
        FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM=True,
        _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD=None,
        _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD=12.0,
    )
