"""Additional configuration options for inference."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Literal

import pydantic

from tabpfn.constants import ModelVersion, TaskType
from tabpfn.preprocessing import (
    PreprocessorConfig,
    v2_5_classifier_preprocessor_configs,
    v2_5_regressor_preprocessor_configs,
    v2_classifier_preprocessor_configs,
    v2_regressor_preprocessor_configs,
)

DEFAULT_SOFTMAX_TEMPERATURE = 0.9
"""The softmax temperature of every checkpoint released before the temperature became
part of the inference config. Also the value used when the temperature cannot be
resolved from a checkpoint at all. Do not change this, see
`InferenceConfig.SOFTMAX_TEMPERATURE`."""

OVERRIDABLE_FIELDS: dict[str, tuple[str, str]] = {
    # field -> (how to name several of them, the estimator argument that sets one)
    "SOFTMAX_TEMPERATURE": ("softmax temperatures", "softmax_temperature"),
    "N_ESTIMATORS": ("numbers of estimators", "n_estimators"),
}
"""Fields a user can set from the estimator interface, and which the checkpoints of
one ensemble may therefore disagree on -- the user can break the tie. Every other
field has to match across an ensemble."""


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

    MIN_CARDINALITY_FOR_TEXT: int = 30
    """Number of distinct values above which a string column is read as text
    rather than as a category. A separate decision from
    `MAX_UNIQUE_FOR_CATEGORICAL_FEATURES`, which governs numerical-vs-categorical:
    that one describes when a *number* is few enough to be a category, this one
    describes when a *string* is varied enough to be text rather than a category,
    and there is no reason the two should move together.

    Text is expanded into numeric features with `TRANSFORM_TEXT`; off, it is
    ordinal-encoded as a high-cardinality category and `fit` warns about it."""

    SOFTMAX_TEMPERATURE: float = DEFAULT_SOFTMAX_TEMPERATURE
    """The temperature applied to the model's logits at predict time. Lower values
    make the predictions more confident, `1.0` is a no-op.

    The default is the value that shipped as the `softmax_temperature` argument of
    `TabPFNClassifier`/`TabPFNRegressor` before this became a config field, so
    checkpoints that predate the field (which is all of them up to and including the
    ones released with v8.5.0) keep their original behavior. Newer checkpoints are
    expected to store their own value and must do so explicitly.

    Setting this here overrides the checkpoint for every model in the ensemble, as
    does `TabPFNClassifier(softmax_temperature=...)`; naming a temperature both ways
    at once is rejected. With neither, the value comes from the checkpoint, and an
    ensemble whose checkpoints declare different temperatures is rejected too."""

    N_ESTIMATORS: int | Literal["auto"] = "auto"
    """How many estimators to run when the user leaves `n_estimators="auto"`.

    An estimator is one forward pass over a differently preprocessed view of the
    data; more of them costs proportionally more compute.

    This means exactly what the `n_estimators` argument of the estimators means, and
    is used in its place when the user names no count:
        - If an int, that many estimators run, and feature-coverage scaling never
          raises it -- a checkpoint that asks for a count gets that count, the same
          guarantee a user passing one gets.
        - If `"auto"` (the default), `DEFAULT_N_ESTIMATORS` estimators run, raised
          on wide tables so every feature is seen by some estimator (see
          `scale_n_estimators_for_feature_coverage`).

    The default leaves the decision where it was before this field existed, so
    checkpoints that predate it keep their original behavior. Newer checkpoints are
    expected to store a count of their own.

    Setting this here overrides the checkpoint, as does
    `TabPFNClassifier(n_estimators=...)`; naming a count both ways at once is
    rejected. With neither, the value comes from the checkpoint, and an ensemble
    whose checkpoints declare different counts is rejected too."""

    TRANSFORM_DATES: bool = False
    """Whether a column holding a genuine datetime dtype (`datetime64`, tz-aware,
    or `period`) is expanded into calendar features via `skrub.DatetimeEncoder`.
    Off, such a column is refused with an error naming it: cast or expand it
    yourself first. Only a real datetime dtype counts: a string column that merely
    looks like a date (e.g. "2020-01-01") is read as a plain category or text
    either way.

    On, the same columns have to hold datetimes at predict, in a DataFrame, and
    none of them may be listed in `categorical_features_indices`; each of these is
    refused with an error saying so rather than guessed at. The fine-tuning
    estimators do not run this conversion, whatever `inference_config` they are
    handed, so a datetime column has to be converted before fine-tuning."""

    TRANSFORM_TEXT: bool = False
    """Whether a text column, a pandas `string` or pyarrow string column with more
    than `MIN_CARDINALITY_FOR_TEXT` distinct values, is expanded into
    `TEXT_N_COMPONENTS` numeric features via `skrub.StringEncoder` (tf-idf over
    character n-grams, truncated SVD). Off, such a column is ordinal-encoded as a
    high-cardinality category and `fit` warns about it. An `object` column is
    never expanded. Not run by the fine-tuning estimators."""

    TEXT_N_COMPONENTS: int = 30
    """Features a text column is expanded into with `TRANSFORM_TEXT`: the leading
    components of a truncated SVD over its tf-idf matrix. Fewer when the column
    has fewer character n-grams than that."""

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

    ENABLE_GPU_PREPROCESSING: bool = False
    """Move quantile transform, SVD feature generation, and feature shuffling to
    GPU / torch.  When ``True``, these operations run on the same device as
    the model, which can be significantly faster for large datasets (>10 k rows).
    When ``False`` (default), all preprocessing runs on CPU / sklearn as before.

    Only ``quantile_uni*`` transforms are accelerated (the torch quantile
    transformer only supports uniform output).  Other transforms stay on CPU
    regardless of this flag.  SVD and shuffle always move to GPU / torch when this
    flag is set."""

    FEATURE_SUBSAMPLING_METHOD: Literal[
        "balanced",
        "random",
        "constant_and_balanced",
        "gini_feature_importance",
        "auto",
    ] = "balanced"
    """The method used to subsample features when the dataset has more features than
    max_features_per_estimator. The options are:
        - "random": Each estimator independently draws a random subset of features.
        - "balanced": Round-robin sampling from a shared shuffled pool so each feature
          appears approximately equally across estimators.
        - "constant_and_balanced": Always include the first N features (see
          FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT), then use balanced subsampling for
          the rest.
        - "gini_feature_importance": Use LightGBM gain importance to rank features.
          Always include the top-K most important features (see
          FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT), fill the rest via balanced
          round-robin sampling from the remaining features.
        - "auto": Automatically selects the method based on dataset size and whether
          feature subsampling is needed. Uses "gini_feature_importance" when
          n_samples > AUTO_FEATURE_SUBSAMPLING_IMPORTANCE_MIN_SAMPLES(=100_000) and
          subsampling is required (importance scoring is more accurate on larger
          datasets), otherwise falls back to "balanced".
    """
    FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT: int = 50
    """The number of leading features that are always included when using the
    'constant_and_balanced' feature subsampling method. Only used when
    FEATURE_SUBSAMPLING_METHOD is 'constant_and_balanced'."""

    FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT: int | float | Literal["auto"] = "auto"
    """Number of top important features always included per estimator when
    FEATURE_SUBSAMPLING_METHOD is an importance-based method. The remaining budget up to
    max_features_per_estimator is filled randomly from the remaining features.
        - If an int, that many features are always included.
        - If a float in (0, 1], resolved as ceil(value * n_total_features).
        - If "auto", uses top-k=AUTO_FEATURE_SUBSAMPLING_TOP_K(=150) when
          n_features > AUTO_FEATURE_SUBSAMPLING_TOP_K_MIN_FEATURES(=200);
          otherwise no importance filtering is done.
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

    MAX_CPU_SAMPLES: int = 1000
    """The number of samples above which CPU inference is disallowed by default due to
    slow performance. Raise via ignore_pretraining_limits or the
    TABPFN_ALLOW_CPU_LARGE_DATASET setting."""

    FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM: bool = True
    """Whether to repair any borders of the bar distribution in regression that are NaN
     after the transformation. This can happen due to multiple reasons and should in
     general always be done."""

    PASSTHROUGH_INF: bool = False
    """Whether to pass infinite values through to the model instead of rejecting them.
    When True, +/-inf are temporarily replaced with NaN for preprocessing and restored
    afterwards; when False, infinities are rejected at input validation."""

    _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD: float | None = None
    _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD: float = 12.0

    def override_with_user_input_and_resolve_auto(
        self, user_config: dict | InferenceConfig | None
    ) -> InferenceConfig:
        """Return a new config with fields specified in `user_config` overwritten.

        Args:
            user_config: Config provided by the user at inference time.
                If a dictionary, then the keys must match attributes of
                    `InferenceConfig` and will be used to override these attributes.
                If an `InferenceConfig` object, then the whole config is overridden with
                    the values from the user config. Deprecated.
                If None, then a copy of this config is returned with no fields changed.
        """
        if user_config is None:
            return deepcopy(self)
        if isinstance(user_config, InferenceConfig):
            warnings.warn(
                "Passing an `InferenceConfig` object as `inference_config` is "
                "deprecated and will be removed in a future version. It replaces "
                "the checkpoint's config as a whole, so any field not set on it "
                "takes a class default rather than the value the checkpoint "
                "declares, which can silently degrade predictions. Pass a dict "
                "naming only the settings to change instead, e.g. "
                '`inference_config={"POLYNOMIAL_FEATURES": "all"}`, which leaves '
                "every other field at the checkpoint's value. To keep replacing the "
                "whole config, pass `dataclasses.asdict(config)`.",
                FutureWarning,
                stacklevel=2,
            )
            return deepcopy(user_config)
        if isinstance(user_config, dict):
            return dataclasses.replace(self, **user_config)
        raise ValueError(
            f"{user_config=}\nUnknown user config provided, see config above."
        )

    def equals_ignoring_overridable_fields(self, other: InferenceConfig) -> bool:
        """Whether this config and `other` agree on every non-overridable field.

        A mismatch in one of `OVERRIDABLE_FIELDS` between the checkpoints of one
        ensemble gets its own error (see
        `raise_if_checkpoints_disagree_on_overridable_fields`), since the user can
        resolve it by naming a value; any other mismatch is unfixable.
        """
        mine = {field: getattr(self, field) for field in OVERRIDABLE_FIELDS}
        return dataclasses.replace(other, **mine) == self

    def get_resolved_outlier_removal_std(
        self,
        estimator_type: Literal["regressor", "classifier"],
    ) -> float | None:
        """Get the resolved outlier removal std."""
        if self.OUTLIER_REMOVAL_STD == "auto":
            return (
                self._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD
                if estimator_type == "regressor"
                else self._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD
            )

        if self.OUTLIER_REMOVAL_STD is not None and self.OUTLIER_REMOVAL_STD <= 0:
            raise ValueError("OUTLIER_REMOVAL_STD must be greater than 0")

        return self.OUTLIER_REMOVAL_STD

    @classmethod
    def get_default(
        cls, task_type: TaskType, model_version: ModelVersion
    ) -> InferenceConfig:
        """Return the default config for the given model version and task type.

        Note that for for model versions after v2, the inference config is stored in
        the checkpoints itself and this function is not called.
        """
        if model_version == ModelVersion.V2:
            if task_type == "multiclass":
                return _get_v2_config(v2_classifier_preprocessor_configs())
            if task_type == "regression":
                return _get_v2_config(v2_regressor_preprocessor_configs())
        elif model_version == ModelVersion.V2_5:
            if task_type == "multiclass":
                return _get_v2_5_config(v2_5_classifier_preprocessor_configs())
            if task_type == "regression":
                return _get_v2_5_config(v2_5_regressor_preprocessor_configs())
        raise ValueError(
            f"No inference config is configured for {model_version=}. "
            "Please make sure you are using a correct model checkpoint that contains "
            "the inference config."
        )


def raise_if_checkpoints_disagree_on_overridable_fields(
    inference_configs: Sequence[InferenceConfig],
    *,
    overrides: Mapping[str, float | int | None],
) -> None:
    """Reject an ensemble whose checkpoints disagree on an overridable field.

    One value of each is applied to the whole ensemble, so there is nothing sensible
    to do with two of them. The user can say which one to use, and then the
    checkpoints no longer have to agree.

    Args:
        inference_configs: The config of each model in the ensemble.
        overrides: What the user asked for, keyed by field name, applied to every
            model. A field is checked only where the user asked for nothing.
    """
    # A field left at "auto" by every checkpoint is not a disagreement, and neither
    # is one they all declare the same way.
    for field, (plural, argument) in OVERRIDABLE_FIELDS.items():
        if overrides.get(field) is not None:
            continue

        # `key=str` because a field can mix ints with the string "auto".
        values = sorted(
            {getattr(config, field) for config in inference_configs}, key=str
        )
        if len(values) > 1:
            raise ValueError(
                f"The given model checkpoints declare different {plural} "
                f"({values}), and one value is applied to the whole ensemble. Pick "
                f"the one to use for all of them and pass it explicitly, e.g. "
                f"`TabPFNClassifier({argument}={values[0]}, ...)`."
            )


def cpu_sample_limit(model_version: ModelVersion) -> int:
    """Max sample count allowed for CPU inference by default, per model version."""
    return 5000 if model_version == ModelVersion.V3 else 1000


def _get_v2_config(preprocessor_configs: list[PreprocessorConfig]) -> InferenceConfig:
    # SOFTMAX_TEMPERATURE is deliberately not listed here: these models predate the
    # field, so they take the class default, which is the temperature they have
    # always been run with.
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
        FEATURE_SUBSAMPLING_METHOD="random",
        FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT=50,
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


def _get_v2_5_config(preprocessor_configs: list[PreprocessorConfig]) -> InferenceConfig:
    # See the note in `_get_v2_config` about SOFTMAX_TEMPERATURE.
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
        FEATURE_SUBSAMPLING_METHOD="random",
        FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT=50,
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
