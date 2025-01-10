"""Various constants used throughout the library."""

#  Copyright (c) Prior Labs GmbH 2025.

# TODO(eddiebergman): Should probably put these where they belong but
# for the time being, this just helps with typing and not the possible
# enumeration of things
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import TypeAlias

import joblib
import numpy as np
from packaging import version

if TYPE_CHECKING:
    from tabpfn.preprocessing import (
        PreprocessorConfig,
    )

TaskType: TypeAlias = Literal["multiclass", "regression"]
TaskTypeValues: tuple[TaskType, ...] = ("multiclass", "regression")

# TODO
XType: TypeAlias = Any
SampleWeightType: TypeAlias = Any
YType: TypeAlias = Any
TODO_TYPE1: TypeAlias = str


@dataclass
class ModelInterfaceConfig:
    """Constants used as default HPs in the model interfaces.

    These constants are not exposed to the models' init on purpose
    to reduce the complexity for users. Furthermore, most of these
    should not be optimized over by the (standard) user.

    Several of the preprocessing options are supported by our code for efficiency
    reasons (to avoid loading TabPFN multiple times). However, these can also be
    applied outside of the model interface.
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
    SUBSAMPLE_SAMPLES: (
        int | float | None  # (0,1) percentage, (1+) n samples
    ) = None
    """Subsample the input data sample/row-wise before performing any preprocessing
    and the TabPFN forward pass.
        - If None, no subsampling is done.
        - If an int, the number of samples to subsample (or oversample if
            `SUBSAMPLE_SAMPLES` is larger than the number of samples).
        - If a float, the percentage of samples to subsample.
    """

    PREPROCESS_TRANSFORMS: list[PreprocessorConfig] | None = None
    """The preprocessing applied to the data before passing it to TabPFN. See
    `PreprocessorConfig` for options and more details. If a list of `PreprocessorConfig`
    is provided, the preprocessors are (repeatedly) applied across different estimators.

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
    REGRESSION_Y_PREPROCESS_TRANSFORMS: tuple[
        Literal["safepower", "power", "quantile_norm", None],
        ...,
    ] = (None, "safepower")
    """The preprocessing applied to the target variable before passing it to TabPFN for
    regression. This can be understood as scaling the target variable to better predict
    it. The preprocessors should be passed as a tuple/list and are then (repeatedly)
    used by the estimators in the ensembles.

    By default, we use no preprocessing and a power transformation (if we have
    more than one estimator).

    The options are:
        - If None, no preprocessing is done.
        - If "power", a power transformation is applied.
        - If "safepower", a power transformation is applied with a safety factor to
            avoid numerical issues.
        - If "quantile_norm", a quantile normalization is applied.
    """

    USE_SKLEARN_16_DECIMAL_PRECISION: bool = False
    """Whether to round the probabilities to float 16 to match the precision of
     scikit-learn. This can help with reproducibility and compatibility with
     scikit-learn but is not recommended for general use. This is not exposed to the
     user or as a hyperparameter.
     To improve reproducibility,set `._sklearn_16_decimal_precision = True` before
     calling `.predict()` or `.predict_proba()`."""

    # TODO: move this somewhere else to support that this might change.
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

    @staticmethod
    def from_user_input(
        *,
        inference_config: dict | ModelInterfaceConfig | None,
    ) -> ModelInterfaceConfig:
        """Converts the user input to a `ModelInterfaceConfig` object.

        The input inference_config can be a dictionary, a `ModelInterfaceConfig` object,
        or None. If a dictionary is passed, the keys must match the attributes of
        `ModelInterfaceConfig`. If a `ModelInterfaceConfig` object is passed, it is
        returned as is. If None is passed, a new `ModelInterfaceConfig` object is
        created with default values.
        """
        if inference_config is None:
            interface_config_ = ModelInterfaceConfig()
        elif isinstance(inference_config, ModelInterfaceConfig):
            interface_config_ = deepcopy(inference_config)
        elif isinstance(inference_config, dict):
            interface_config_ = ModelInterfaceConfig()
            for key, value in inference_config.items():
                if not hasattr(interface_config_, key):
                    raise ValueError(
                        f"Unknown kwarg passed to model construction: {key}",
                    )
                setattr(interface_config_, key, value)
        else:
            raise ValueError(f"Unknown {inference_config=} passed to model.")

        return interface_config_


SKLEARN_16_DECIMAL_PRECISION = 16
PROBABILITY_EPSILON_ROUND_ZERO = 1e-3
REGRESSION_NAN_BORDER_LIMIT_UPPER = 1e3
REGRESSION_NAN_BORDER_LIMIT_LOWER = -1e3
AUTOCAST_DTYPE_BYTE_SIZE = 2  # bfloat16
DEFAULT_DTYPE_BYTE_SIZE = 4  # float32

# Otherwise, yoa-johnson double power can end up causing a lot of overflows...
DEFAULT_NUMPY_PREPROCESSING_DTYPE = np.float64

# TODO(eddiebergman): Maybe make these a parameter
MEMORY_SAFETY_FACTOR = 5.0  # Taken as default from function


# TODO(eddiebergman): Pulled from `def get_ensemble_configurations()`
ENSEMBLE_CONFIGURATION_MAX_STEP = 2
MAXIMUM_FEATURE_SHIFT = 1_000
CLASS_SHUFFLE_OVERESTIMATE_FACTOR = 3

# 1) Figure out whether this Joblib version supports "generator_unordered".
# For example, assume "generator_unordered" is officially supported in joblib >= 1.4.0
SUPPORTS_GENERATOR_UNORDERED = version.parse(joblib.__version__) >= version.parse(
    "1.4.0",
)
SUPPORTS_RETURN_AS = version.parse(joblib.__version__) >= version.parse(
    "1.3.0",
)
# 2) Define a mapping from your custom parallel mode to joblib's "return_as" parameter.
if SUPPORTS_GENERATOR_UNORDERED:
    # If the installed Joblib is new enough, allow "generator_unordered"
    PARALLEL_MODE_TO_RETURN_AS = {
        "block": "list",
        "in-order": "generator",
        "as-ready": "generator_unordered",
    }
else:
    # If the installed Joblib is older, fall back to "generator"
    PARALLEL_MODE_TO_RETURN_AS = {
        "block": "list",
        "in-order": "generator",
        # fallback to "generator" instead of "generator_unordered"
        "as-ready": "generator",
    }
