"""TabPFNRegressor class.

!!! example
    ```python
    import sklearn.datasets
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor()
    X, y = sklearn.datasets.make_regression(n_samples=50, n_features=10)

    model.fit(X, y)
    predictions = model.predict(X)
    ```
"""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import copy
import logging
import typing
import warnings
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, overload
from typing_extensions import Self, TypedDict, deprecated

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    check_is_fitted,
    clone,
)
from tqdm.auto import tqdm

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.base import (
    RegressorModelSpecs,
    create_inference_engine,
    determine_precision,
    estimator_to_device,
    expand_dates_and_text,
    get_embeddings,
    initialize_model_variables_helper,
    reject_categoricals_for_differentiable_input,
    resolved_n_estimators,
    resolved_softmax_temperature,
)
from tabpfn.constants import (
    REGRESSION_CONSTANT_TARGET_BORDER_EPSILON,
    ModelVersion,
)
from tabpfn.errors import TabPFNValidationError, handle_oom_errors
from tabpfn.inference import (
    InferenceEngineBatchedNoPreprocessing,
    _maybe_run_gpu_preprocessing,
)
from tabpfn.inference_tuning import (
    RegressorEvalMetrics,
    RegressorTuningConfig,
    find_regression_optimal_temperature,
    get_tuning_splits,
    resolve_tuning_config,
)
from tabpfn.model_loading import (
    ModelSource,
    load_fitted_tabpfn_model,
    prepend_cache_path,
    save_fitted_tabpfn_model,
)
from tabpfn.preprocessing import (
    EnsembleConfig,
    FeatureSubsamplingMethod,
    PreprocessorConfig,
    RegressorEnsembleConfig,
    clean_data,
    generate_regression_ensemble_configs,
)
from tabpfn.preprocessing.clean import clean_data_transform
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.datetimes import DateTransformer
from tabpfn.preprocessing.ensemble import (
    TabPFNEnsemblePreprocessor,
    scale_n_estimators_for_feature_coverage,
)
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.preprocessing.steps import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.text import TextTransformer
from tabpfn.utils import (
    DevicesSpecification,
    convert_batch_of_cat_ix_to_schema,
    infer_random_state,
    transform_borders_one,
    translate_probs_across_borders,
)
from tabpfn.validation import (
    check_input_shape_matches,
    ensure_compatible_fit_inputs,
    ensure_compatible_predict_input_sklearn,
    extract_input_shape,
    validate_categorical_features_indices,
    validate_dataset_size,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.pipeline import Pipeline
    from torch.types import _dtype

    from tabpfn.architectures.interface import (
        Architecture,
        ArchitectureConfig,
        PerformanceOptions,
    )
    from tabpfn.constants import MemorySavingMode, XType, YType
    from tabpfn.inference import InferenceEngine
    from tabpfn.inference_config import InferenceConfig
    from tabpfn.preprocessing.steps.preprocessing_helpers import (
        OrderPreservingColumnTransformer,
    )

    try:
        from sklearn.base import Tags
    except ImportError:
        Tags = Any


# --- Prediction Output Types and Constants ---

# 1. Tuples for runtime validation and internal logic.
# These are defined directly as tuples of strings for immediate clarity.
_OUTPUT_TYPES_BASIC = ("mean", "median", "mode")
_OUTPUT_TYPES_QUANTILES = ("quantiles",)
_OUTPUT_TYPES = _OUTPUT_TYPES_BASIC + _OUTPUT_TYPES_QUANTILES
_OUTPUT_TYPES_COMPOSITE = ("full", "main")
_USABLE_OUTPUT_TYPES = _OUTPUT_TYPES + _OUTPUT_TYPES_COMPOSITE


# 2. Type aliases for static type checking and IDE support.
OutputType = Literal["mean", "median", "mode", "quantiles", "full", "main"]
"""The type hint for the `output_type` parameter in `predict`."""


class MainOutputDict(TypedDict):
    """Specifies the return structure for `output_type="main"`."""

    mean: np.ndarray
    median: np.ndarray
    mode: np.ndarray
    quantiles: list[np.ndarray]


class FullOutputDict(MainOutputDict):
    """Specifies the return structure for `output_type="full"`."""

    criterion: FullSupportBarDistribution
    logits: torch.Tensor


RegressionResultType = np.ndarray | list[np.ndarray] | MainOutputDict | FullOutputDict
"""The type hint for the return value of the `predict` method."""

DEFAULT_REGRESSION_EVAL_METRIC = RegressorEvalMetrics.NLL


class TabPFNRegressor(RegressorMixin, BaseEstimator):
    """TabPFNRegressor class."""

    configs_: list[ArchitectureConfig]
    """The configurations of the loaded models to be used for inference.

    The concrete type of these configs is defined by the architectures in use and should
    be inspected at runtime, but they will be subclasses of ArchitectureConfig.
    """

    models_: list[Architecture]
    """The loaded models to be used for inference.

    The models can be different PyTorch modules, but will be subclasses of Architecture.
    """

    inference_config_: InferenceConfig
    """Additional configuration of inference for expert users."""

    devices_: tuple[torch.device, ...]
    """The devices determined to be used.

    The devices are determined based on the `device` argument to the constructor, and
    the devices available on the system. See the constructor documentation for details.
    """

    feature_names_in_: npt.NDArray[Any]
    """The feature names of the input data.

    May not be set if the input data does not have feature names,
    such as with a numpy array.
    """

    n_features_in_: int
    """The number of features in the input data used during `fit()`."""

    n_train_samples_: int
    """The number of training samples used during `fit()`."""

    inferred_feature_schema_: FeatureSchema
    """The inferred feature schema. This contains the feature modalities per column,
    using heuristics and user-provided indices for categorical features."""

    n_outputs_: Literal[1]  # We only support single output
    """The number of outputs the model supports. Only 1 for now"""

    znorm_space_bardist_: FullSupportBarDistribution
    """The bar distribution of the target variable, used by the model.
    This is the bar distribution in the normalized target space.
    """

    raw_space_bardist_: FullSupportBarDistribution
    """The bar distribution in the raw target space, used for computing the
    predictions."""

    use_autocast_: bool
    """Whether torch's autocast should be used."""

    forced_inference_dtype_: _dtype | None
    """The forced inference dtype for the model based on `inference_precision`."""

    executor_: InferenceEngine
    """The inference engine used to make predictions."""

    ordinal_encoder_: OrderPreservingColumnTransformer
    """The column transformer used to preprocess categorical data to be numeric."""

    date_transformer_: DateTransformer
    """The transformer that converted every temporal column before validation."""

    text_transformer_: TextTransformer
    """The transformer that expanded every text column before validation."""

    categorical_features_indices_: list[int] | None
    """`categorical_features_indices` as positions in the validated input, where
    an expanded date or text column has moved everything after it down."""

    eval_metric_: RegressorEvalMetrics
    """The validated evaluation metric to optimize for during prediction."""

    ensemble_softmax_temperature_: float
    """The temperature applied to the aggregated ensemble distribution at predict
    time, after the per-estimator `softmax_temperature`. This is `1.0`, a no-op, when
    no temperature calibration is done."""

    softmax_temperature_: float
    """The resolved per-estimator `softmax_temperature`, i.e. the one the checkpoint
    declares unless it was overridden."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int | Literal["auto"] = "auto",
        auto_scale_n_estimators: bool = True,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature: float | Literal["auto"] = "auto",
        average_before_softmax: bool = False,
        model_path: str
        | Path
        | list[str]
        | list[Path]
        | Literal["auto"]
        | RegressorModelSpecs
        | list[RegressorModelSpecs] = "auto",
        device: DevicesSpecification = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
            "batched",
        ] = "fit_preprocessors",
        memory_saving_mode: MemorySavingMode = "auto",
        keep_cache_on_device: bool = True,
        kv_cache_precision: Literal["auto", "int8", "fp8"] | None = None,
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        n_jobs: Annotated[int | None, deprecated("Use n_preprocessing_jobs")] = None,
        n_preprocessing_jobs: int = 1,
        inference_config: dict | InferenceConfig | None = None,
        differentiable_input: bool = False,
        eval_metric: str | RegressorEvalMetrics | None = None,
        tuning_config: dict | RegressorTuningConfig | None = None,
        show_progress_bar: bool = False,
    ) -> None:
        """Construct a TabPFN regressor.

        This constructs a regressor using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model.

        Args:
            n_estimators:
                The number of estimators in the TabPFN ensemble. We aggregate the
                predictions of `n_estimators`-many forward passes of TabPFN.
                Each forward pass has (slightly) different input data. Think of this
                as an ensemble of `n_estimators`-many "prompts" of the input data.
                With the default `"auto"`, the count comes from the checkpoint
                (`InferenceConfig.N_ESTIMATORS`), which is itself `"auto"` unless
                the checkpoint names a count. `"auto"` means
                `DEFAULT_N_ESTIMATORS`,
                raised on wide datasets so every feature is seen by some estimator
                (i.e. when the data has more than `max_features_per_estimator`
                features per estimator), to the smallest value that lets every
                feature appear in at least one ensemble member, emitting a warning
                when it does so. That auto-scaled value is capped at
                `MAX_AUTO_SCALED_N_ESTIMATORS`; beyond that some features may never
                be sampled unless you raise `n_estimators` yourself. An explicit
                integer — yours or the checkpoint's — is never overridden: if it
                is too small to cover every feature, a warning is emitted at fit
                time and the value is used as given. Your integer cannot be
                combined with an `N_ESTIMATORS` in `inference_config`, which is the
                other way of naming a count.

            auto_scale_n_estimators:
                Deprecated, removed in v9 — pass an explicit `n_estimators`
                instead. Only applies when `n_estimators="auto"`, where `False`
                keeps the auto value at `DEFAULT_N_ESTIMATORS` rather than raising
                it for feature coverage, exactly what passing that count as
                `n_estimators` does. Passing `False` emits a `FutureWarning` at
                fit time.

            categorical_features_indices:
                The indices of the columns that are suggested to be treated as
                categorical. If `None`, the model will infer the categorical columns.
                If provided, we might ignore some of the suggestion to better fit the
                data seen during pre-training.

                !!! note
                    The indices are 0-based and should represent the data passed to
                    `.fit()`. If the data changes between the initializations of the
                    model and the `.fit()`, consider setting the
                    `.categorical_features_indices` attribute after the model was
                    initialized and before `.fit()`.

            softmax_temperature:
                The temperature for the softmax function. This is used to control the
                confidence of the model's predictions. Lower values make the model's
                predictions more confident. This is only applied when predicting during
                a post-processing step. Set `softmax_temperature=1.0` for no effect.

                If `"auto"` (the default), the temperature is taken from the
                checkpoint (`InferenceConfig.SOFTMAX_TEMPERATURE`), which is `0.9` for
                every checkpoint released up to and including v8.5.0. Passing a float
                overrides the checkpoint for every model in the ensemble; it cannot be
                combined with a `SOFTMAX_TEMPERATURE` in `inference_config`, which is
                the other way of naming one.

            average_before_softmax:
                Only used if `n_estimators > 1`. Whether to average the predictions of
                the estimators before applying the softmax function. This can help to
                improve predictive performance when there are many classes or when
                calibrating the model's confidence. This is only applied when
                predicting during a post-processing.

                - If `True`, the predictions are averaged before applying the softmax
                  function. Thus, we average the logits of TabPFN and then apply the
                  softmax.
                - If `False`, the softmax function is applied to each set of logits.
                  Then, we average the resulting probabilities of each forward pass.

            model_path:
                The path to the TabPFN model file, i.e., the pre-trained weights.

                - If `"auto"`, the model will be downloaded upon first use. This
                  defaults to your system cache directory, but can be overwritten
                  with the use of an environment variable `TABPFN_MODEL_CACHE_DIR`.
                - If a path or a string of a path, the model will be loaded from
                  the user-specified location if available, otherwise it will be
                  downloaded to this location. Details on available checkpoints are
                  available in the repository README.

            device:
                The device(s) to use for inference.
                See the documentation of `.to()`.

            ignore_pretraining_limits:
                Whether to ignore the pre-training limits of the model. The TabPFN
                models have been pre-trained on a specific range of input data. If the
                input data is outside of this range, the model may not perform well.
                You may ignore our limits to use the model on data outside the
                pre-training range.

                - If `True`, the model will not raise an error if the input data is
                  outside the pre-training range. Also suppresses error when using
                  the model with a large dataset on CPU.
                - If `False`, you can use the model outside the pre-training range, but
                  the model could perform worse.

                !!! note

                    For version 2.5, the pre-training limits are:

                    - 50_000 samples/rows
                    - 2_000 features/columns (Note that for more than 500 features we
                        subsample 500 features per estimator. It is therefore important
                        to use a sufficiently large number of `n_estimators`.)

            device:
                The device to use for inference with TabPFN. If `"auto"`, the device is
                `"cuda"` if available, otherwise `"cpu"`.

                See PyTorch's documentation on devices for more information about
                supported devices.

            inference_precision:
                The precision to use for inference. This can dramatically affect the
                speed and reproducibility of the inference. Higher precision can lead to
                better reproducibility but at the cost of speed. By default, we optimize
                for speed and use torch's mixed-precision autocast. The options are:

                - If `torch.dtype`, we force precision of the model and data to be
                  the specified torch.dtype during inference. This can is particularly
                  useful for reproducibility. Here, we do not use mixed-precision.
                - If `"autocast"`, enable PyTorch's mixed-precision autocast. Ensure
                  that your device is compatible with mixed-precision.
                - If `"auto"`, we determine whether to use autocast or not depending on
                  the device type.

            fit_mode:
                Determine how the TabPFN model is "fitted". The mode determines how the
                data is preprocessed and cached for inference. This is unique to an
                in-context learning foundation model like TabPFN, as the "fitting" is
                technically the forward pass of the model. The options are:

                - If `"low_memory"`, the data is preprocessed on-demand during inference
                  when calling `.predict()` or `.predict_proba()`. This is the most
                  memory-efficient mode but can be slower for large datasets because
                  the data is (repeatedly) preprocessed on-the-fly.
                  Ideal with low GPU memory and/or a single call to `.fit()` and
                  `.predict()`.
                - If `"fit_preprocessors"`, the data is preprocessed and cached once
                  during the `.fit()` call. During inference, the cached preprocessing
                  (of the training data) is used instead of re-computing it.
                  Ideal with low GPU memory and multiple calls to `.predict()` with
                  the same training data.
                - If `"fit_with_cache"`, the data is preprocessed and cached once during
                  the `.fit()` call like in `fit_preprocessors`. Moreover, the
                  transformer key-value cache is also initialized, allowing for much
                  faster inference on the same data at a large cost of memory.
                  Ideal with very high GPU memory and multiple calls to `.predict()`
                  with the same training data.
                - If `"batched"`, the already pre-processed data is iterated over in
                  batches. This can only be done after the data has been preprocessed
                  with the get_preprocessed_datasets function. This is primarily used
                  only for inference with the InferenceEngineBatchedNoPreprocessing
                  class in Fine-Tuning. The fit_from_preprocessed() function sets this
                  attribute internally.

            memory_saving_mode:
                Enable GPU/CPU memory saving mode. This can both avoid out-of-memory
                errors and improve fit+predict speed by reducing memory pressure.

                It saves memory by automatically batching certain model computations
                within TabPFN.

                - If "auto": memory saving mode is enabled/disabled automatically based
                    on a heuristic
                - If True/False: memory saving mode is forced enabled/disabled.

                If speed is important to your application, you may wish to manually tune
                this option by comparing the time taken for fit+predict with it set to
                False and True.

                !!! warning
                    This does not batch the original input data. We still recommend to
                    batch the test set as necessary if you run out of memory.

            keep_cache_on_device:
                Only relevant when `fit_mode="fit_with_cache"`. If True
                (default), the key-value cache is kept on the inference
                device (e.g. GPU). Uses more device
                memory but gives lower latency. If False, the cache is stored on CPU.

            kv_cache_precision:
                Only relevant when `fit_mode="fit_with_cache"`. Resolved against
                what the model architecture supports. `None` (default) picks the
                architecture default (`"int8"` when it can quantize, e.g. TabPFN-3,
                else `"auto"`); `"int8"` quantizes the key-value cache to save
                memory; `"fp8"` stores it as 8-bit floats instead (same size,
                float rounding semantics; not supported on MPS);
                `"auto"` keeps the computed dtype. Requesting a
                quantized precision on an architecture that cannot quantize
                warns and falls back to `"auto"`.

            random_state:
                Controls the randomness of the model. Pass an int for reproducible
                results and see the scikit-learn glossary for more information.
                If `None`, the randomness is determined by the system when calling
                `.fit()`.

                !!! warning
                    We depart from the usual scikit-learn behavior in that by default
                    we provide a fixed seed of `0`.

                !!! note
                    Even if a seed is passed, we cannot always guarantee reproducibility
                    due to PyTorch's non-deterministic operations and general numerical
                    instability. To get the most reproducible results across hardware,
                    we recommend using a higher precision as well (at the cost of a
                    much higher inference time). Likewise, for scikit-learn, consider
                    passing `USE_SKLEARN_16_DECIMAL_PRECISION=True` as kwarg.

            n_jobs:
                Deprecated, use `n_preprocessing_jobs` instead.
                This parameter never had any effect.

            n_preprocessing_jobs:
                The number of worker processes to use for the preprocessing.

                If `1`, the preprocessing will be performed in the current process,
                parallelised across multiple CPU cores. If `>1` and `n_estimators > 1`,
                then different estimators will be dispatched to different processes.

                We strongly recommend setting this to 1, which has the lowest overhead
                and can often fully utilise the CPU. Values >1 can help if you have lots
                of CPU cores available, but can also be slower.

            inference_config:
                For advanced users, additional advanced arguments that adjust the
                behavior of the model interface.
                See [tabpfn.inference_config.InferenceConfig][] for details and options.

                - If `None`, the default InferenceConfig is used.
                - If `dict`, the key-value pairs are used to update the default
                  `InferenceConfig`. Raises an error if an unknown key is passed.
                - If `InferenceConfig`, the object replaces the checkpoint's config
                  as a whole, so any field not set on it takes a class default
                  rather than the value the checkpoint declares. Deprecated.

            differentiable_input:
                If true, preprocessing attempts to be end-to-end differentiable.
                Less relevant for standard regression fine-tuning compared to
                prompt-tuning.

            eval_metric:
                Metric by which predictions will be evaluated on test data for
                temperature calibration.
                For currently supported metrics, see
                [tabpfn.inference_tuning.RegressorEvalMetrics][].

            tuning_config:
                The settings to use to tune the model's predictions for the specified
                `eval_metric`. See
                [tabpfn.inference_tuning.RegressorTuningConfig][] for details
                and options.

            show_progress_bar:
                Whether to show a progress bar during inference. Defaults to False.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.auto_scale_n_estimators = auto_scale_n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
            inference_precision
        )
        self.fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
            "batched",
        ] = fit_mode
        self.show_progress_bar = show_progress_bar
        self.memory_saving_mode: MemorySavingMode = memory_saving_mode
        self.keep_cache_on_device = keep_cache_on_device
        self.kv_cache_precision = kv_cache_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.differentiable_input = differentiable_input
        self.eval_metric = eval_metric
        self.tuning_config = tuning_config

        if n_jobs is not None:
            warnings.warn(
                "TabPFNRegressor(n_jobs=...) is deprecated and has no effect. "
                "Use `n_preprocessing_jobs` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.n_jobs = n_jobs
        self.n_preprocessing_jobs = n_preprocessing_jobs

    @classmethod
    def create_default_for_version(cls, version: ModelVersion, **overrides) -> Self:
        """Construct a regressor that uses the given version of the model.

        In addition to selecting the model, this also configures certain settings to the
        default values associated with this model version.

        Any kwargs will override the default settings.
        """
        if version == ModelVersion.V2:
            options = {
                "model_path": prepend_cache_path(
                    ModelSource.get_regressor_v2().default_filename
                ),
                "n_estimators": "auto",
            }
        elif version == ModelVersion.V2_5:
            options = {
                "model_path": prepend_cache_path(
                    ModelSource.get_regressor_v2_5().default_filename
                ),
                "n_estimators": "auto",
            }
        elif version == ModelVersion.V2_6:
            options = {
                "model_path": prepend_cache_path(
                    ModelSource.get_regressor_v2_6().default_filename
                ),
                "n_estimators": "auto",
            }
        elif version == ModelVersion.V3:
            options = {
                "model_path": prepend_cache_path(
                    ModelSource.get_regressor_v3().default_filename
                ),
                "n_estimators": "auto",
            }
        else:
            raise ValueError(f"Unknown version: {version}")

        options.update(overrides)

        return cls(**options)

    @property
    def estimator_type(self) -> Literal["regressor"]:
        """The type of the model."""
        return "regressor"

    @property
    def model_(self) -> Architecture:
        """The model used for inference.

        This is set after the model is loaded and initialized.
        """
        if not hasattr(self, "models_"):
            raise ValueError(
                "The model has not been initialized yet. Please initialize the model "
                "before using the `model_` property."
            )
        if len(self.models_) > 1:
            raise ValueError(
                "The `model_` property is not supported when multiple models are used. "
                "Use `models_` instead."
            )
        return self.models_[0]

    @property
    def norm_bardist_(self) -> FullSupportBarDistribution:
        """WARNING: DEPRECATED. Please use `raw_space_bardist_` instead.
        This attribute will be removed in a future version.
        """
        warnings.warn(
            "`norm_bardist_` is deprecated and will be removed in a future version. "
            "Please use `raw_space_bardist_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.raw_space_bardist_

    @norm_bardist_.setter
    def norm_bardist_(self, value: FullSupportBarDistribution) -> None:
        warnings.warn(
            "`norm_bardist_` is deprecated and will be removed in a future version. "
            "Please use `raw_space_bardist_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.raw_space_bardist_ = value

    @property
    def bardist_(self) -> FullSupportBarDistribution:
        """WARNING: DEPRECATED. Please use `znorm_space_bardist_` instead.
        This attribute will be removed in a future version.
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
        warnings.warn(
            "`bardist_` is deprecated and will be removed in a future version. "
            "Please use `znorm_space_bardist_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.znorm_space_bardist_ = value

    def get_inference_config(self) -> InferenceConfig:
        """Load the model if needed and return the active inference config.

        Loads the model checkpoint without requiring fit data so the config can be
        inspected before calling `fit()`. Any ``inference_config`` override
        passed to the constructor is considered.

        Returns:
            A deep copy of the active inference config.
        """
        if not hasattr(self, "inference_config_"):
            self._initialize_model_variables()
        return copy.deepcopy(self.inference_config_)

    # TODO: We can remove this from scikit-learn lower bound of 1.6
    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = self.estimator_type
        return tags

    def _initialize_model_variables(self) -> int:
        """Initializes the model and configurations.

        Returns:
            The determined byte_size.
        """
        return initialize_model_variables_helper(self, self.estimator_type)

    def _rebuild_raw_space_bardist(self) -> None:
        """Rebuild ``raw_space_bardist_`` from current ``y_train_mean_``/std_.

        Detaches the znorm-space borders so the rebuilt buffer never holds a
        y autograd graph — required for the differentiable-input path and a
        no-op for the standard path. Both ``y_train_mean_`` and
        ``y_train_std_`` must already be set as Python floats.
        """
        borders = self.znorm_space_bardist_.borders.detach()
        self.raw_space_bardist_ = FullSupportBarDistribution(
            borders * self.y_train_std_ + self.y_train_mean_,
        ).float()

    def _build_ensemble_preprocessor_and_executor(
        self,
        *,
        X: Any,
        y: Any,
        ensemble_configs: list[RegressorEnsembleConfig],
        static_seed: int,
        byte_size: int,
        n_preprocessing_jobs: int,
        inference_mode: bool,
    ) -> None:
        """Build ``self.ensemble_preprocessor_`` and ``self.executor_``.

        Shared between the standard fit path and the differentiable-input
        path. The two paths differ only in ``n_preprocessing_jobs``
        (forced to 1 in the differentiable path so the autograd graph on
        ``X`` survives joblib's process-boundary pickling) and
        ``inference_mode`` (False under differentiable input so backprop
        works through the executor).
        """
        self.ensemble_preprocessor_ = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            n_samples=X.shape[0],
            feature_schema=self.inferred_feature_schema_,
            # Use static_seed so we're independent of any random generation
            # inside the initialize functions above.
            random_state=static_seed,
            n_preprocessing_jobs=n_preprocessing_jobs,
            keep_fitted_cache=(self.fit_mode == "fit_with_cache"),
            enable_gpu_preprocessing=self.inference_config_.ENABLE_GPU_PREPROCESSING,
            feature_subsampling_method=FeatureSubsamplingMethod(
                self.inference_config_.FEATURE_SUBSAMPLING_METHOD
            ),
            constant_feature_count=self.inference_config_.FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT,
            subsample_samples=self.inference_config_.SUBSAMPLE_SAMPLES,
            importance_top_k_count=self.inference_config_.FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT,
            X_train=X,
            y_train=y,
            task_type=self.estimator_type,
        )
        self.executor_ = create_inference_engine(
            fit_mode=self.fit_mode,
            X_train=X,
            y_train=y,
            ensemble_preprocessor=self.ensemble_preprocessor_,
            models=self.models_,
            devices_=self.devices_,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            task_type="regression",
            keep_cache_on_device=self.keep_cache_on_device,
            kv_cache_precision=self.kv_cache_precision,
            inference_mode=inference_mode,
        )

    def _initialize_for_differentiable_input(
        self,
        X: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[list[RegressorEnsembleConfig], torch.Tensor]:
        """First-call setup for the differentiable path.

        Mirrors the classifier-side helper so that gradients can flow from a
        loss back to upstream torch modules feeding ``X`` (and optionally
        ``y``). Skips the standard numpy preprocessing path and uses a
        differentiable identity preprocessor. y-target normalization happens
        every call inside ``fit_with_differentiable_input``; this helper is
        only for the cached feature-schema and ensemble-config setup.
        """
        # Minimal preprocessing for prompt tuning: no categorical features,
        # all-numerical schema, identity preprocessor that preserves grads.
        reject_categoricals_for_differentiable_input(self.categorical_features_indices)
        n_features = X.shape[1]
        # One Feature instance per column — list multiplication would share
        # the same dataclass and any later in-place update would leak across
        # columns.
        features = [
            Feature(name=None, modality=FeatureModality.NUMERICAL)
            for _ in range(n_features)
        ]
        self.inferred_feature_schema_ = FeatureSchema(features=features)
        # A tensor holds no dates or strings, so these fit nothing; set anyway, so
        # every predict path converts through them without first checking.
        self.date_transformer_ = DateTransformer().fit(X)
        self.text_transformer_ = TextTransformer().fit(X)
        self.n_features_in_ = n_features

        preprocessor_configs = [PreprocessorConfig("none", differentiable=True)]
        self.n_estimators_ = scale_n_estimators_for_feature_coverage(
            n_estimators=resolved_n_estimators(self),
            n_total_features=n_features,
            preprocessor_configs=preprocessor_configs,
            auto_scale_n_estimators=self.auto_scale_n_estimators,
        )
        # Polynomial features go through sklearn StandardScaler on numpy and
        # are not differentiable; force "no" regardless of the runtime default
        # (the regressor config defaults to a non-zero value).
        ensemble_configs = generate_regression_ensemble_configs(
            num_estimators=self.n_estimators_,
            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
            polynomial_features="no",
            preprocessor_configs=preprocessor_configs,
            target_transforms=[None],
            random_state=rng,
            num_models=len(self.models_),
            outlier_removal_std=self.inference_config_.get_resolved_outlier_removal_std(
                estimator_type=self.estimator_type
            ),
        )
        assert len(ensemble_configs) == self.n_estimators_

        return ensemble_configs, X

    def _initialize_dataset_preprocessing(
        self,
        X: XType,
        y: YType,
        random_state: int | np.random.Generator,
    ) -> tuple[
        list[RegressorEnsembleConfig],
        np.ndarray,
        np.ndarray,
        FullSupportBarDistribution,
    ]:
        """Prepare ensemble configs and validate X, y for one dataset/chunk.

        Handle the preprocessing of the input (X and y). We also return the
        BarDistribution here, since it is vital for computing the standardized
        target variable in the DatasetCollectionWithPreprocessing class.
        """
        # feature_names_in_/n_features_in_ have to describe what the caller passed,
        # not the wider frame expansion can make of it, so they come off the raw
        # input here, before any conversion.
        self.feature_names_in_, self.n_features_in_ = extract_input_shape(X)

        validate_categorical_features_indices(self.categorical_features_indices)
        X, date_transformer, text_transformer, feature_names, categorical_indices = (
            expand_dates_and_text(
                X,
                categorical_features_indices=self.categorical_features_indices,
                inference_config=self.inference_config_,
            )
        )

        X, y, _ = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=self,
            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
            max_cpu_samples=self.inference_config_.MAX_CPU_SAMPLES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            ensure_y_numeric=True,
            devices=self.devices_,
        )
        # Set class variables for sklearn compatibility
        self.n_train_samples_ = len(X)

        feature_schema = detect_feature_modalities(
            X=X,
            feature_names=feature_names,
            provided_categorical_indices=categorical_indices,
            min_samples_for_inference=self.inference_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.inference_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
            min_cardinality_for_text=self.inference_config_.MIN_CARDINALITY_FOR_TEXT,
        )
        X, ordinal_encoder, feature_schema = clean_data(
            X=X,
            feature_schema=feature_schema,
            passthrough_inf=self.get_inference_config().PASSTHROUGH_INF,
        )
        self.inferred_feature_schema_ = feature_schema
        self.ordinal_encoder_ = ordinal_encoder
        self.date_transformer_ = date_transformer
        self.text_transformer_ = text_transformer
        self.categorical_features_indices_ = categorical_indices

        # TODO: Introduce regressor target transformer that also keeps track of
        # target name
        possible_target_transforms = get_all_reshape_feature_distribution_preprocessors(
            num_examples=y.shape[0],  # Use length of validated y
            random_state=random_state,  # Use the provided rng
        )
        target_preprocessors: list[TransformerMixin | Pipeline | None] = []
        for (
            y_target_preprocessor
        ) in self.inference_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS:
            if y_target_preprocessor is not None:
                preprocessor = possible_target_transforms[y_target_preprocessor]
            else:
                preprocessor = None
            target_preprocessors.append(preprocessor)

        preprocessor_configs = self.inference_config_.PREPROCESS_TRANSFORMS
        self.n_estimators_ = scale_n_estimators_for_feature_coverage(
            n_estimators=resolved_n_estimators(self),
            n_total_features=feature_schema.num_columns,
            preprocessor_configs=preprocessor_configs,
            auto_scale_n_estimators=self.auto_scale_n_estimators,
        )
        ensemble_configs = generate_regression_ensemble_configs(
            num_estimators=self.n_estimators_,
            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.inference_config_.POLYNOMIAL_FEATURES,
            preprocessor_configs=preprocessor_configs,
            target_transforms=target_preprocessors,
            random_state=random_state,
            num_models=len(self.models_),
            outlier_removal_std=self.inference_config_.get_resolved_outlier_removal_std(
                estimator_type=self.estimator_type
            ),
            passthrough_inf=self.get_inference_config().PASSTHROUGH_INF,
        )

        self.znorm_space_bardist_ = self.znorm_space_bardist_.to(self.devices_[0])

        assert len(ensemble_configs) == self.n_estimators_

        return ensemble_configs, X, y, self.znorm_space_bardist_

    def _get_tuning_regressor(self, **overwrite_kwargs: Any) -> TabPFNRegressor:
        """Return a fresh regressor configured for holdout tuning."""
        params = self.get_params(deep=False)

        # Avoids sharing mutable config across instances
        for key in params:
            try:
                if isinstance(params.get(key), dict):
                    params[key] = copy.deepcopy(params[key])
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "Error during initialization of tuning regressor when trying "
                    f"to deepcopy configuration with name `{key}`: {e}. "
                    "Falling back to original configuration"
                )

        forced = {
            "fit_mode": "fit_preprocessors",
            "differentiable_input": False,
            "tuning_config": None,  # never tune inside tuning
            # Fit on the already-expanded array, where a declared column may
            # have moved down past an expanded date or text column.
            "categorical_features_indices": self.categorical_features_indices_,
        }

        params.update(forced)
        params.update(overwrite_kwargs)

        return TabPFNRegressor(**params)

    def fit_from_preprocessed(
        self,
        X_preprocessed: list[torch.Tensor],
        y_preprocessed: list[torch.Tensor],  # These y are standardized
        cat_ix: list[list[list[int]]],
        configs: list[list[EnsembleConfig]],  # Should be RegressorEnsembleConfig
        *,
        performance_options: PerformanceOptions,
        no_refit: bool = True,
    ) -> TabPFNRegressor:
        """Used in Fine-Tuning. Fit the model to preprocessed inputs from torch
        dataloader inside a training loop a Dataset provided by
        get_preprocessed_datasets. This function always uses the "batched" fit_mode.

        Args:
            X_preprocessed: The input features obtained from the preprocessed Dataset
                The list contains one item for each ensemble predictor.
                use tabpfn.utils.collate_for_tabpfn_dataset to use this function with
                batch sizes of more than one dataset (see examples/tabpfn_finetune.py)
            y_preprocessed: The target variable obtained from the preprocessed Dataset
            cat_ix: categorical indices obtained from the preprocessed Dataset
            configs: Ensemble configurations obtained from the preprocessed Dataset
            performance_options: Performance and memory options forwarded to the
                model on each forward call inside the resulting executor.
            no_refit: if True, the classifier will not be reinitialized when calling
                fit multiple times.
        """
        if self.fit_mode != "batched":
            logging.warning(
                "The model was not in 'batched' mode. "
                "Automatically switching to 'batched' mode for finetuning."
            )
            self.fit_mode = "batched"

        # If there is a model, and we are lazy, we skip reinitialization
        if not hasattr(self, "models_") or not no_refit:
            byte_size = self._initialize_model_variables()
        else:
            _, _, byte_size = determine_precision(
                self.inference_precision, self.devices_
            )

        feature_schema = convert_batch_of_cat_ix_to_schema(
            batch_of_cat_indices=cat_ix,
            num_features=X_preprocessed[0].shape[1],
        )

        # Preprocessed tensors hold no dates or strings either, so these fit
        # nothing: the date transformer still refuses a date and converts a
        # duration, like any fitted one.
        self.date_transformer_ = DateTransformer().fit(X_preprocessed[0])
        self.text_transformer_ = TextTransformer().fit(X_preprocessed[0])

        self.n_estimators_ = len(configs[0])
        self.executor_ = InferenceEngineBatchedNoPreprocessing(
            X_trains=X_preprocessed,
            y_trains=y_preprocessed,
            feature_schema=feature_schema,
            ensemble_configs=configs,
            models=self.models_,
            devices=self.devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=self.forced_inference_dtype_,
            save_peak_mem=self.memory_saving_mode,
            inference_mode=True,
            performance_options=performance_options,
        )

        return self

    def fit_with_differentiable_input(self, X: torch.Tensor, y: torch.Tensor) -> Self:
        """Fit the model with differentiable input.

        Mirror of ``TabPFNClassifier.fit_with_differentiable_input``. Lets
        gradients flow from a downstream loss back through ``X`` (and ``y``,
        if it carries grads) into upstream torch modules. Use this instead
        of ``fit`` when ``differentiable_input=True``.

        Args:
            X: The input data as a torch tensor.
            y: The target variable as a torch tensor.

        Returns:
            self
        """
        if self.fit_mode != "fit_preprocessors":
            logging.warning(
                "The model was not in 'fit_preprocessors' mode. "
                "Automatically switching to 'fit_preprocessors' mode for differentiable"
                " input."
            )
            self.fit_mode = "fit_preprocessors"

        static_seed, rng = infer_random_state(self.random_state)

        is_first_fit_call = not hasattr(self, "models_")
        if is_first_fit_call:
            byte_size = self._initialize_model_variables()
            ensemble_configs, X = self._initialize_for_differentiable_input(
                X=X, rng=rng
            )
            self.ensemble_configs_ = ensemble_configs  # Store for prompt tuning reuse
        else:
            _, _, byte_size = determine_precision(
                self.inference_precision, self.devices_
            )
            ensemble_configs = self.ensemble_configs_  # Reuse from first fit
            # Mirror classifier.py: re-assert n_estimators_ from cached
            # configs so a subsequent call after pickling restores it.
            self.n_estimators_ = len(ensemble_configs)

        # Refresh target stats and rebuild the raw-space bardist on every
        # call so they track the current fit data; cached state is only the
        # model load, feature schema, and ensemble configs above.
        validate_dataset_size(
            X=X,
            y=y,
            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
            max_cpu_samples=self.inference_config_.MAX_CPU_SAMPLES,
            devices=self.devices_,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        self.n_train_samples_ = int(X.shape[0])

        y_float = (
            y.float()
            if isinstance(y, torch.Tensor)
            else torch.as_tensor(y, dtype=torch.float32)
        )
        y_mean = y_float.mean()
        # Match the standard fit's np.std (population std, ddof=0). torch.std
        # defaults to correction=1 and returns NaN for N=1; clamp keeps the
        # divisor non-zero. The constant-target guard below catches the
        # remaining bardist-collapse case.
        y_std = torch.clamp(y_float.std(correction=0), min=1e-20)
        if y_std.detach().item() <= 1e-12:
            raise ValueError(
                "Constant or near-constant target (std≈0) is not supported "
                "by fit_with_differentiable_input; there is no signal to "
                "predict differentiably. Use fit() for constant-target data."
            )
        # Detach when storing as Python floats — raw_space_bardist_ is a
        # frozen lookup and must not hold a y autograd graph. Users who need
        # fully differentiable target scaling should z-normalise y themselves
        # before calling so the mean/std are constants here.
        self.y_train_mean_ = y_mean.detach().item()
        self.y_train_std_ = y_std.detach().item()
        y = (y_float - y_mean) / y_std
        self._rebuild_raw_space_bardist()

        # Force sequential preprocessing: with differentiable input X carries
        # an autograd graph that does not survive joblib's process-boundary
        # pickling. Sequential execution preserves the graph in-process.
        self._build_ensemble_preprocessor_and_executor(
            X=X,
            y=y,
            ensemble_configs=ensemble_configs,
            static_seed=static_seed,
            byte_size=byte_size,
            n_preprocessing_jobs=1,
            inference_mode=False,
        )

        return self

    @config_context(transform_output="default")  # type: ignore
    def fit(self, X: XType, y: YType) -> Self:
        """Fit the model.

        Args:
            X: The input data.
            y: The target variable.

        Returns:
            self
        """
        # Validate eval_metric here instead of in __init__ as per sklearn convention
        self.eval_metric_ = _validate_eval_metric(self.eval_metric)
        # Set here as well as in `_maybe_calibrate_ensemble_temperature` below, so
        # that the constant-target fit, which returns before calibration, still
        # exposes the attribute.
        self.ensemble_softmax_temperature_ = 1.0

        if self.differentiable_input:
            raise ValueError(
                "differentiable_input=True requires fit_with_differentiable_input "
                "with torch tensor X and y, not fit()."
            )

        if self.fit_mode == "batched":
            logging.warning(
                "The model was in 'batched' mode, likely after finetuning. "
                "Automatically switching to 'fit_preprocessors' mode for standard "
                "prediction. The model will be re-initialized."
            )
            self.fit_mode = "fit_preprocessors"

        static_seed, _ = infer_random_state(self.random_state)
        byte_size = self._initialize_model_variables()
        ensemble_configs, X, y, znorm_space_bardist = (
            self._initialize_dataset_preprocessing(X=X, y=y, random_state=static_seed)
        )
        self.znorm_space_bardist_ = znorm_space_bardist
        self.ensemble_configs_ = ensemble_configs

        assert len(ensemble_configs) == self.n_estimators_

        self.is_constant_target_ = np.unique(y).size == 1
        self.constant_value_ = y[0] if self.is_constant_target_ else None

        if self.is_constant_target_:
            # Use relative epsilon, s.t. it works for small and large constant values
            border_adjustment = max(
                abs(self.constant_value_ * REGRESSION_CONSTANT_TARGET_BORDER_EPSILON),
                REGRESSION_CONSTANT_TARGET_BORDER_EPSILON,
            )

            self.znorm_space_bardist_ = FullSupportBarDistribution(
                borders=torch.tensor(
                    [
                        self.constant_value_ - border_adjustment,
                        self.constant_value_ + border_adjustment,
                    ]
                )
            )
            # No need to create an inference engine for a constant prediction
            return self

        # Must run on the raw `y`, before the z-normalisation below: each fold's
        # tuning regressor derives its own mean/std from its own training split.
        self._maybe_calibrate_ensemble_temperature(X=X, y=y)

        mean, std = np.mean(y), np.std(y)
        # TODO: y_train_std_ and y_train_mean_ don't seem to be used anywhere else.
        self.y_train_std_ = std.item() + 1e-20
        self.y_train_mean_ = mean.item()
        y = (y - self.y_train_mean_) / self.y_train_std_
        self._rebuild_raw_space_bardist()

        self._build_ensemble_preprocessor_and_executor(
            X=X,
            y=y,
            ensemble_configs=ensemble_configs,
            static_seed=static_seed,
            byte_size=byte_size,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            # TODO: Standard fit usually uses inference_mode=True, before it was enabled
            inference_mode=True,
        )

        return self

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["mean", "median", "mode"] = "mean",
        quantiles: list[float] | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["quantiles"],
        quantiles: list[float] | None = None,
    ) -> list[np.ndarray]: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["main"],
        quantiles: list[float] | None = None,
    ) -> MainOutputDict: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["full"],
        quantiles: list[float] | None = None,
    ) -> FullOutputDict: ...

    @config_context(transform_output="default")  # type: ignore
    def predict(
        self,
        X: XType,
        *,
        # TODO: support "ei", "pi"
        output_type: OutputType = "mean",
        quantiles: list[float] | None = None,
    ) -> RegressionResultType:
        """Runs the forward() method and then transform the logits
        from the binning space in order to predict target variable.

        Args:
            X: The input data.
            output_type:
                Determines the type of output to return.

                - If `"mean"`, we return the mean over the predicted distribution.
                - If `"median"`, we return the median over the predicted distribution.
                - If `"mode"`, we return the mode over the predicted distribution.
                - If `"quantiles"`, we return the quantiles of the predicted
                    distribution. The parameter `quantiles` determines which
                    quantiles are returned.
                - If `"main"`, we return the all output types above in a dict.
                - If `"full"`, we return the full output of the model, including the
                  logits and the criterion, and all the output types from "main".

            quantiles:
                The quantiles to return if `output="quantiles"`.

                By default, the `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
                quantiles are returned. The predictions per quantile match
                the input order.

        Returns:
            The prediction, which can be a numpy array, a list of arrays (for
            quantiles), or a dictionary with detailed outputs.
        """
        check_is_fitted(self)

        # TODO: Move these at some point to InferenceEngine
        check_input_shape_matches(X, estimator=self)
        X = self.date_transformer_.transform(X)
        X = self.text_transformer_.transform(X)
        X = ensure_compatible_predict_input_sklearn(X, self)

        check_is_fitted(self)

        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif not all((0 <= q <= 1) and isinstance(q, float) for q in quantiles):
            raise TabPFNValidationError(
                "All quantiles must be between 0 and 1 and floats."
            )
        if output_type not in _USABLE_OUTPUT_TYPES:
            raise TabPFNValidationError(f"Invalid output type: {output_type}")

        if hasattr(self, "is_constant_target_") and self.is_constant_target_:
            return self._handle_constant_target(X.shape[0], output_type, quantiles)

        logits = self._compute_aggregated_logits(X)

        # Determine and return intended output type
        logit_to_output = partial(
            _logits_to_output,
            logits=logits,
            criterion=self.raw_space_bardist_,
            quantiles=quantiles,
        )
        if output_type in ["full", "main"]:
            # Create a dictionary of outputs with proper typing via TypedDict
            # Get individual outputs with proper typing
            mean_out = typing.cast("np.ndarray", logit_to_output(output_type="mean"))
            median_out = typing.cast(
                "np.ndarray", logit_to_output(output_type="median")
            )
            mode_out = typing.cast("np.ndarray", logit_to_output(output_type="mode"))
            quantiles_out = typing.cast(
                "list[np.ndarray]",
                logit_to_output(output_type="quantiles"),
            )

            main_outputs = MainOutputDict(
                mean=mean_out,
                median=median_out,
                mode=mode_out,
                quantiles=quantiles_out,
            )

            if output_type == "full":
                # Return full output with criterion and logits
                return FullOutputDict(
                    **main_outputs,
                    criterion=self.raw_space_bardist_,
                    logits=logits,
                )

            return main_outputs

        return logit_to_output(output_type=output_type)

    def _maybe_calibrate_ensemble_temperature(self, X: XType, y: YType) -> None:
        """If a `tuning_config` was given, calibrate the ensemble temperature.

        The regressor analogue of the classifier's
        `_maybe_calibrate_temperature_and_tune_decision_thresholds`. Fits
        throwaway regressors on cross-validation splits of `X`/`y`, sweeps the
        temperature against `eval_metric_` on their pooled holdout rows, and
        stores the winner in `ensemble_softmax_temperature_`, which
        `_compute_aggregated_logits` applies to every prediction.

        `y` must be the *raw* target: each fold's regressor performs its own
        z-normalisation, and passing an already-normalised `y` would normalise
        it twice.

        Unlike the classifier's counterpart there are no metric-specific
        warnings, as NLL is the only supported metric and calibrating the
        temperature always targets it directly.
        """
        assert self.eval_metric_ is not None

        # Always set this, so the attribute exists on every fitted estimator
        # regardless of whether tuning runs (sklearn compatibility). 1.0 is a
        # no-op, leaving predictions bit-identical to an uncalibrated fit.
        self.ensemble_softmax_temperature_ = 1.0

        tuning_config_resolved = resolve_tuning_config(
            tuning_config=self.tuning_config,
            num_samples=X.shape[0],
            config_cls=RegressorTuningConfig,
        )
        if tuning_config_resolved is None:
            return

        if not tuning_config_resolved.calibrate_temperature:
            return

        holdout_folds = self._compute_holdout_validation_data(
            X=X,
            y=y,
            holdout_frac=float(tuning_config_resolved.tuning_holdout_frac),
            n_folds=int(tuning_config_resolved.tuning_n_folds),
        )

        # Falls back to the current no-op temperature if every fold was
        # dropped, e.g. because each training split had a constant target.
        self.ensemble_softmax_temperature_ = find_regression_optimal_temperature(
            holdout_folds=holdout_folds,
            metric_name=self.eval_metric_,
            current_default_temperature=self.ensemble_softmax_temperature_,
        )

    def _compute_holdout_validation_data(
        self,
        X: XType,
        y: YType,
        holdout_frac: float,
        n_folds: int,
    ) -> list[tuple[torch.Tensor, FullSupportBarDistribution, torch.Tensor]]:
        """Compute holdout validation data, one entry per cross-validation fold.

        Folds are kept separate rather than concatenated: every
        fold normalises the target with its own mean and standard deviation, so
        every fold's `raw_space_bardist_` has its own borders and only scores that
        fold's own logits correctly.

        Returns:
            One `(logits, raw_space_bardist, y_holdout)` triple per usable fold.
            `logits` has shape `[n_holdout_samples, n_buckets]`, and `y_holdout`
            shape `[n_holdout_samples]` in the *original* units of the target, to
            match the borders of the fold's `raw_space_bardist`. Folds whose
            training target turned out to be constant are omitted, so the list may
            be shorter than `n_folds` and may be empty.
        """
        splits = get_tuning_splits(
            X=copy.deepcopy(X),
            y=copy.deepcopy(y),
            holdout_frac=holdout_frac,
            random_state=self.random_state,
            n_splits=n_folds,
            task_type="regressor",
        )

        holdout_folds: list[
            tuple[torch.Tensor, FullSupportBarDistribution, torch.Tensor]
        ] = []
        # suffixes: Nt=num_train_samples, F=num_features, Nh=num_holdout_samples,
        # B=num buckets
        for X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh in splits:
            tuning_regressor = self._get_tuning_regressor()
            with warnings.catch_warnings():
                # Filter expected warnings during tuning
                warnings.filterwarnings(
                    "ignore",
                    message=".*haven't specified any tuning configuration*",
                    category=UserWarning,
                )
                tuning_regressor.fit(X_train_NtF, y_train_Nt)

            if tuning_regressor.is_constant_target_:
                # This fold's training split is single-valued, so its regressor
                # returned early with a degenerate two-border distribution and no
                # executor. There is nothing to calibrate against; skip the fold.
                continue

            # The tuning regressor has no `ensemble_softmax_temperature_`, so these
            # are the untempered aggregated logits, with the per-estimator
            # `softmax_temperature` correctly still applied.
            check_input_shape_matches(X_holdout_NhF, estimator=tuning_regressor)
            X_holdout_NhF = tuning_regressor.date_transformer_.transform(  # noqa: PLW2901
                X_holdout_NhF
            )
            X_holdout_NhF = tuning_regressor.text_transformer_.transform(  # noqa: PLW2901
                X_holdout_NhF
            )
            X_holdout_NhF = ensure_compatible_predict_input_sklearn(  # noqa: PLW2901
                X_holdout_NhF, tuning_regressor
            )
            logits_NhB = tuning_regressor._compute_aggregated_logits(X_holdout_NhF)

            # `raw_space_bardist_` undoes this fold's normalisation, so the targets
            # are scored in their original units and need no transformation --
            # only a tensor of the dtype and device the bar distribution expects.
            raw_space_bardist = tuning_regressor.raw_space_bardist_
            y_holdout_Nh_tensor = torch.as_tensor(
                y_holdout_Nh,
                dtype=raw_space_bardist.borders.dtype,
                device=logits_NhB.device,
            )
            holdout_folds.append((logits_NhB, raw_space_bardist, y_holdout_Nh_tensor))

        return holdout_folds

    def _compute_aggregated_logits(self, X: XType) -> torch.Tensor:
        """Run the ensemble and aggregate it into one log-probability tensor.

        Each estimator's bucket probabilities are translated onto the
        `znorm_space_bardist_` borders, averaged across the ensemble
        (before or after the softmax, per `average_before_softmax` flat),
        and returned in log space.

        Shared by `predict` and `_maybe_calibrate_ensemble_temperature`.

        `X` must already have passed `ensure_compatible_predict_input_sklearn`;
        the remaining dtype/NA preprocessing happens here. Constant-target
        models have no executor and must be routed to
        `_handle_constant_target` by the caller instead.

        Args:
            X: The validated input data.

        Returns:
            A `[n_samples, n_buckets]` tensor of log-probabilities over the
            `znorm_space_bardist_` buckets, with the calibrated
            `ensemble_softmax_temperature_` applied.
        """
        cat_indices = self.inferred_feature_schema_.indices_for(
            FeatureModality.CATEGORICAL
        )
        X = clean_data_transform(
            X,
            cat_indices=cat_indices,
            ord_encoder=getattr(self, "ordinal_encoder_", None),
            passthrough_inf=self.get_inference_config().PASSTHROUGH_INF,
        )

        n_estimators = 0
        accumulated_logits: torch.Tensor | None = None
        with handle_oom_errors(
            self.devices_,
            X,
            model_type="regressor",
            n_train_samples=getattr(self, "n_train_samples_", None),
        ):
            for borders_t, output in tqdm(
                self._iter_forward_executor(X, use_inference_mode=True),
                total=self.n_estimators_,
                desc="TabPFN inference",
                unit="estimator",
                disable=not self.show_progress_bar,
            ):
                transformed = translate_probs_across_borders(
                    output,
                    frm=torch.as_tensor(borders_t, device=output.device),
                    to=self.znorm_space_bardist_.borders.to(output.device),
                )

                if self.average_before_softmax:
                    transformed = transformed.log()

                if accumulated_logits is None:
                    accumulated_logits = transformed
                else:
                    accumulated_logits = accumulated_logits + transformed
                n_estimators += 1

        assert n_estimators > 0

        if accumulated_logits is None:
            raise ValueError(
                "Cannot make predictions, possibly due to `n_estimators=0`."
            )

        return self._reduce_accumulated_logits(accumulated_logits, n_estimators)

    def _reduce_accumulated_logits(
        self,
        accumulated_logits: torch.Tensor,
        n_estimators: int,
    ) -> torch.Tensor:
        """Average the ensemble's accumulated output and apply the temperature.

        Args:
            accumulated_logits:
                Summed per-estimator output for one dataset. Already in log space
                if `average_before_softmax` is True.
            n_estimators: How many estimators contributed to the sum.

        Returns:
            A `[n_samples, n_buckets]` tensor of log-probabilities with
            `ensemble_softmax_temperature_` applied.
        """
        if self.average_before_softmax:
            logits = (accumulated_logits / n_estimators).softmax(dim=-1)
        else:
            logits = accumulated_logits / n_estimators

        # Post-process the logits
        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()

        # The calibrated temperature acts on the aggregated logits of n_estimators:
        # softmax(log(sum(probs_i)/T)). The softmax is implicit, as
        # `FullSupportBarDistribution.compute_scaled_log_probs` applies
        # `log_softmax` to whatever it receives. `getattr` keeps models pickled
        # before this attribute existed loadable; the guard keeps the
        # uncalibrated path allocation-free and bit-identical.
        temperature = getattr(self, "ensemble_softmax_temperature_", 1.0)
        if temperature != 1.0:
            logits = logits / temperature

        return logits

    def predict_batched(  # noqa: C901, PLR0912
        self,
        X_train_list: list[XType],
        y_train_list: list[YType],
        X_test_list: list[XType],
        *,
        output_type: OutputType = "mean",
        quantiles: list[float] | None = None,
    ) -> list[RegressionResultType]:
        """Predict for several independent datasets in one pass.

        Each triple is preprocessed exactly as ``fit()`` + ``predict()`` does,
        then compatible model-input shapes are fused. Heterogeneous
        post-preprocessing shapes run in separate groups. This is equivalent to
        independent prediction and leaves ``self`` unchanged.

        Datasets need not share a target scale (each is decoded with its own
        bar distribution) but must share raw array shapes. Fitted transforms may
        produce different shapes; these are grouped internally without padding.

        Args:
            X_train_list: Training features, one array per dataset (all same shape).
            y_train_list: Training targets, one array per dataset.
            X_test_list: Test features, one array per dataset (all same shape).
            output_type: As in :meth:`predict`, applied to every dataset.
            quantiles: As in :meth:`predict`.

        Returns:
            One entry per dataset, in input order, each being what :meth:`predict`
            would return for that dataset.

        Raises:
            ValueError: If the input lists have unequal or zero length, or the
                training (or test) arrays do not all share one shape.
            TabPFNValidationError: If ``output_type`` or ``quantiles`` are invalid.
            NotImplementedError: If ``tuning_config`` is configured on the
                estimator -- the calibrated ensemble temperature is per-dataset
                state and cannot be applied correctly across a shared batch, so
                score those datasets individually with :meth:`predict`. Also
                raised for ``inference_precision=torch.float64``, which the
                fused forward does not support.

        Note:
            Constant-target datasets are answered analytically and take no part
            in the fused forward.
        """
        # Imported here rather than at module scope: importing
        # architectures.interface at runtime is circular, as is `finetuning`,
        # which imports TabPFNRegressor.
        from tabpfn.architectures.interface import (  # noqa: PLC0415
            PerformanceOptions,
        )
        from tabpfn.finetuning.data_util import (  # noqa: PLC0415
            RegressorBatch,
            _collate_same_shape_for_batched_inference,
            _group_batches_by_shape,
        )

        if not len(X_train_list) == len(y_train_list) == len(X_test_list):
            raise ValueError(
                "X_train_list, y_train_list and X_test_list must have equal length."
            )
        if len(X_train_list) == 0:
            raise ValueError("Nothing to predict: empty dataset list.")

        # `ensemble_softmax_temperature_` is calibrated per dataset on that
        # dataset's own holdout, so there is no single temperature to apply to a
        # shared batch. Mirrors the same guard in
        # `TabPFNClassifier.predict_proba_batched`.
        if self.tuning_config is not None:
            raise NotImplementedError(
                "predict_batched does not support tuning_config (ensemble "
                "temperature calibration); the calibrated temperature is fitted "
                "per dataset and cannot be applied across a shared batch. Score "
                "datasets individually with predict."
            )

        if self.inference_precision == torch.float64:
            raise NotImplementedError(
                "predict_batched does not support inference_precision=torch.float64; "
                "the fused forward runs at float32. Use predict per dataset instead."
            )

        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif not all((0 <= q <= 1) and isinstance(q, float) for q in quantiles):
            raise TabPFNValidationError(
                "All quantiles must be between 0 and 1 and floats."
            )
        if output_type not in _USABLE_OUTPUT_TYPES:
            raise TabPFNValidationError(f"Invalid output type: {output_type}")

        # Padding ragged datasets would feed the model fake context rows and leave
        # padded query rows untrimmed, silently corrupting results.
        train_shapes = {np.asarray(X).shape for X in X_train_list}
        test_shapes = {np.asarray(X).shape for X in X_test_list}
        if len(train_shapes) > 1 or len(test_shapes) > 1:
            raise ValueError(
                "predict_batched requires all training arrays to share one shape "
                "and all test arrays to share one shape (ragged batches are not "
                f"supported); got train shapes {sorted(train_shapes)} and test "
                f"shapes {sorted(test_shapes)}. Group datasets by shape and call "
                "once per group."
            )

        # Fit on a clone so a prior fit on ``self`` survives and the batched
        # executor is dropped with the clone. The clone shares the model via the
        # ``model_path`` param, so there is no reload.
        worker = clone(self)
        # "fit_preprocessors" caches the fitted members on the executor, so the
        # loop below reuses them instead of preprocessing twice.
        worker.fit_mode = "fit_preprocessors"

        results: list[RegressionResultType | None] = [None] * len(X_train_list)
        items: list[RegressorBatch] = []
        item_indices: list[int] = []
        znorm_borders: torch.Tensor | None = None

        for idx, (X, y, X_test) in enumerate(
            zip(X_train_list, y_train_list, X_test_list, strict=True)
        ):
            worker.fit(X, y)

            if worker.is_constant_target_:
                # fit() returns before building an executor here, so there is
                # nothing to score.
                results[idx] = worker._handle_constant_target(
                    len(np.asarray(X_test)), output_type, quantiles
                )
                continue

            # Rebuilt fresh on every fit, so each dataset keeps its own object.
            # Fixed by the checkpoint, so identical across datasets.
            if znorm_borders is None:
                znorm_borders = worker.znorm_space_bardist_.borders.clone()

            # Clean X_test as the standard predict path does, so DataFrames,
            # categoricals and NaNs behave identically.
            check_input_shape_matches(X_test, estimator=worker)
            X_test = worker.date_transformer_.transform(X_test)  # noqa: PLW2901
            X_test = worker.text_transformer_.transform(X_test)  # noqa: PLW2901
            X_test = ensure_compatible_predict_input_sklearn(X_test, worker)  # noqa: PLW2901
            X_test = clean_data_transform(  # noqa: PLW2901
                X_test,
                cat_indices=worker.inferred_feature_schema_.indices_for(
                    FeatureModality.CATEGORICAL
                ),
                ord_encoder=getattr(worker, "ordinal_encoder_", None),
                passthrough_inf=worker.inference_config_.PASSTHROUGH_INF,
            )
            members = worker.executor_.ensemble_members
            x_context, x_query, cat_indices = [], [], []
            y_context = [
                torch.as_tensor(np.asarray(m.y_train), dtype=torch.float32)
                for m in members
            ]
            device = worker.devices_[0]
            for member in members:
                x_tr = torch.as_tensor(np.asarray(member.X_train), dtype=torch.float32)
                x_te = torch.as_tensor(
                    np.asarray(member.transform_X_test(X_test)), dtype=torch.float32
                )
                full, schema = _maybe_run_gpu_preprocessing(
                    torch.cat([x_tr, x_te], dim=0).to(device),
                    member.gpu_preprocessor,
                    member.feature_schema,
                    num_train_rows=x_tr.shape[0],
                )
                n = x_tr.shape[0]
                x_context.append(full[:n])
                x_query.append(full[n:])
                cat_indices.append(
                    schema.indices_for(FeatureModality.CATEGORICAL) or []
                )
            n_test = x_query[0].shape[0]
            items.append(
                RegressorBatch(
                    X_context=x_context,
                    X_query=x_query,
                    y_context=y_context,
                    y_query=torch.zeros(n_test),
                    cat_indices=cat_indices,
                    # Must come from the members, not ``worker.ensemble_configs_``:
                    # with n_preprocessing_jobs > 1 ``target_transform`` is fitted in
                    # a worker process, so only the member's copy carries the fitted
                    # transform the border mapping needs.
                    configs=[m.config for m in members],
                    raw_space_bardist=worker.raw_space_bardist_,
                    znorm_space_bardist=worker.znorm_space_bardist_,
                    X_query_raw=torch.zeros(n_test, 1),
                    y_query_raw=torch.zeros(n_test),
                )
            )
            item_indices.append(idx)

        if items:
            worker.fit_mode = "batched"
            assert znorm_borders is not None
            std_borders = znorm_borders.cpu().numpy()
            groups = _group_batches_by_shape(items)
            if len(groups) > 1:
                logger.debug(
                    "predict_batched split %d datasets into %d post-preprocessing "
                    "shape groups with sizes %s; running one inference pass per "
                    "group may be slower than homogeneous batched inference.",
                    len(items),
                    len(groups),
                    [len(group) for group in groups],
                )
            for group in groups:
                positions, group_items = zip(*group, strict=True)
                batch = _collate_same_shape_for_batched_inference(list(group_items))
                worker.fit_from_preprocessed(
                    batch.X_context,
                    batch.y_context,
                    batch.cat_indices,
                    batch.configs,
                    performance_options=PerformanceOptions(),
                )
                accumulated: list[torch.Tensor | None] = [None] * len(group)
                n_estimators = 0
                for output, configs in worker.executor_.iter_outputs(
                    batch.X_query,
                    autocast=worker.use_autocast_,
                    task_type="regression",
                ):
                    for lane in range(len(group)):
                        contribution = worker._translate_batched_logits(
                            output=output[:, lane, :],
                            config=configs[lane],
                            znorm_borders=znorm_borders,
                            std_borders=std_borders,
                        )
                        previous = accumulated[lane]
                        accumulated[lane] = (
                            contribution
                            if previous is None
                            else previous + contribution
                        )
                    n_estimators += 1
                    del output

                for lane, position in enumerate(positions):
                    logits = accumulated[lane]
                    assert logits is not None
                    # Release each lane's bucket logits as it is decoded instead of
                    # retaining every dataset's tensor through the whole decode loop.
                    accumulated[lane] = None
                    item = items[position]
                    results[item_indices[position]] = worker._decode_batched_dataset(
                        accumulated_logits=logits,
                        n_estimators=n_estimators,
                        raw_space_bardist=item.raw_space_bardist,
                        output_type=output_type,
                        quantiles=quantiles,
                    )

        assert all(r is not None for r in results)
        return typing.cast("list[RegressionResultType]", results)

    def _translate_batched_logits(
        self,
        *,
        output: torch.Tensor,
        config: RegressorEnsembleConfig,
        znorm_borders: torch.Tensor,
        std_borders: np.ndarray,
    ) -> torch.Tensor:
        """Map one estimator's output for one dataset onto the shared borders.

        Same border translation as :meth:`predict`, for a single
        (estimator, dataset) pair of the fused forward.
        """
        out_d = output.float()
        temperature = resolved_softmax_temperature(self)
        if temperature != 1:
            out_d = out_d / temperature
        if config.target_transform is None:
            borders_t = std_borders.copy()
            logit_cancel_mask = None
        else:
            logit_cancel_mask, descending_borders, borders_t = transform_borders_one(
                std_borders,
                target_transform=config.target_transform,
                repair_nan_borders_after_transform=self.inference_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
            )
            if descending_borders:
                borders_t = borders_t.flip(-1)  # type: ignore
        if logit_cancel_mask is not None:
            out_d = out_d.clone()
            out_d[..., logit_cancel_mask] = float("-inf")

        transformed = translate_probs_across_borders(
            out_d,
            frm=torch.as_tensor(borders_t, device=out_d.device),
            to=znorm_borders.to(out_d.device),
        )
        return transformed.log() if self.average_before_softmax else transformed

    def _decode_batched_dataset(
        self,
        *,
        accumulated_logits: torch.Tensor,
        n_estimators: int,
        raw_space_bardist: FullSupportBarDistribution,
        output_type: OutputType,
        quantiles: list[float],
    ) -> RegressionResultType:
        """Turn one dataset's accumulated logits into its prediction output.

        Shares :meth:`_reduce_accumulated_logits` with :meth:`predict`, so the
        two paths average identically, and decodes with this dataset's own
        raw-space bar distribution as the criterion.
        """
        assert n_estimators > 0
        # `predict_batched` rejects `tuning_config`, so the temperature applied
        # here is always the 1.0 no-op; the call is shared with `predict` so the
        # two reductions cannot drift apart again.
        logits = self._reduce_accumulated_logits(accumulated_logits, n_estimators)

        logit_to_output = partial(
            _logits_to_output,
            logits=logits,
            criterion=raw_space_bardist,
            quantiles=quantiles,
        )
        if output_type in _OUTPUT_TYPES_COMPOSITE:
            main_outputs = MainOutputDict(
                mean=typing.cast("np.ndarray", logit_to_output(output_type="mean")),
                median=typing.cast("np.ndarray", logit_to_output(output_type="median")),
                mode=typing.cast("np.ndarray", logit_to_output(output_type="mode")),
                quantiles=typing.cast(
                    "list[np.ndarray]", logit_to_output(output_type="quantiles")
                ),
            )
            if output_type == "full":
                return FullOutputDict(
                    **main_outputs, criterion=raw_space_bardist, logits=logits
                )
            return main_outputs

        return logit_to_output(output_type=output_type)

    def _iter_forward_executor(
        self,
        X: list[torch.Tensor] | XType,
        *,
        use_inference_mode: bool = False,
    ) -> Iterator[tuple[np.ndarray, torch.Tensor]]:
        # Scenario 1: Standard inference path
        is_standard_inference = use_inference_mode and not isinstance(
            self.executor_, InferenceEngineBatchedNoPreprocessing
        )

        # Scenario 2: Batched path, typically for fine-tuning with gradients
        is_batched_for_grads = (
            not use_inference_mode
            and isinstance(self.executor_, InferenceEngineBatchedNoPreprocessing)
            and isinstance(X, list)
            and (not X or isinstance(X[0], torch.Tensor))
        )

        assert is_standard_inference or is_batched_for_grads, (
            "Invalid forward pass: Bad combination of inference mode, input X, "
            "or executor type. Ensure call is from standard predict or a "
            "batched fine-tuning context."
        )

        # Specific check for float64 incompatibility if the batched engine is being
        # used, now framed as an assertion that the problematic condition is NOT met.
        assert not (
            isinstance(self.executor_, InferenceEngineBatchedNoPreprocessing)
            and self.forced_inference_dtype_ == torch.float64
        ), (
            "Batched engine error: float64 precision is not supported for the "
            "fine-tuning workflow (requires float32 for backpropagation)."
        )

        check_is_fitted(self)
        # Ensure torch.inference_mode is OFF to allow gradients
        if self.fit_mode in ["fit_preprocessors", "batched"]:
            # only these two modes support this option.
            # Don't enable inference mode when differentiable_input=True (prompt
            # tuning) to allow gradients to flow through.
            actual_inference_mode = use_inference_mode and not self.differentiable_input
            self.executor_.use_torch_inference_mode(use_inference=actual_inference_mode)
        std_borders = self.znorm_space_bardist_.borders.cpu().numpy()
        temperature = resolved_softmax_temperature(self)
        for output, config in self.executor_.iter_outputs(
            X, autocast=self.use_autocast_, task_type="regression"
        ):
            output = output.float()  # noqa: PLW2901
            if temperature != 1:
                output = output / temperature  # noqa: PLW2901

            # BSz.= 1 Scenario, the same as normal predict() function
            # Handled by first if-statement
            config_for_ensemble = config
            if isinstance(config, list) and len(config) == 1:
                single_config = config[0]
                config_for_ensemble = single_config

            if isinstance(config_for_ensemble, RegressorEnsembleConfig):
                borders_t: np.ndarray
                logit_cancel_mask: np.ndarray | None
                descending_borders: bool

                # TODO(eddiebergman): Maybe this could be parallelized or done in fit
                # but I somehow doubt it takes much time to be worth it.
                # One reason to make it worth it is if you want fast predictions, i.e.
                # don't re-do this each time.
                # However it gets a bit more difficult as you need to line up the
                # outputs from `iter_outputs` above (which may be in arbitrary order),
                # along with the specific config the output belongs to. This is because
                # the transformation done to the borders for a given output is dependant
                # upon the target_transform of the config.
                if config_for_ensemble.target_transform is None:
                    borders_t = std_borders.copy()
                    logit_cancel_mask = None
                    descending_borders = False
                else:
                    logit_cancel_mask, descending_borders, borders_t = (
                        transform_borders_one(
                            std_borders,
                            target_transform=config_for_ensemble.target_transform,
                            repair_nan_borders_after_transform=self.inference_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                        )
                    )
                    if descending_borders:
                        borders_t = borders_t.flip(-1)  # type: ignore

                if logit_cancel_mask is not None:
                    output = output.clone()  # noqa: PLW2901
                    output[..., logit_cancel_mask] = float("-inf")
                yield borders_t, output
            else:
                raise ValueError(
                    "Unexpected config format "
                    "and Batch prediction is not supported yet!"
                )

    def forward(
        self,
        X: list[torch.Tensor] | XType,
        *,
        use_inference_mode: bool = False,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor], list[np.ndarray]]:
        """Forward pass for TabPFNRegressor Inference Engine.
        Used in fine-tuning and prediction. Called directly
        in FineTuning training loop or by predict() function
        with the use_inference_mode flag explicitly set to True.

        Iterates over outputs of InferenceEngine.

        Args:
            X: list[torch.Tensor] in fine-tuning, XType in normal predictions.
            use_inference_mode: Flag for inference mode., default at False since
            it is called within predict. During FineTuning forward() is called
            directly by user, so default should be False here.

        Returns:
            A tuple containing:
                - Averaged logits over the ensemble (for fine-tuning).
                - Raw outputs from each estimator in the ensemble.
                - Borders used for each estimator.
        """
        outputs: list[torch.Tensor] = []
        borders: list[np.ndarray] = []

        for border, output in tqdm(
            self._iter_forward_executor(X, use_inference_mode=use_inference_mode),
            total=self.n_estimators_,
            desc="TabPFN inference",
            unit="estimator",
            disable=not self.show_progress_bar,
        ):
            borders.append(border)
            outputs.append(output)

        averaged_logits = None
        all_logits = None
        if outputs:
            all_logits = torch.stack(outputs, dim=0)  # [N_est, N_sampls, N_bord]
            averaged_logits_over_ensemble = torch.mean(all_logits, dim=0)
            averaged_logits = averaged_logits_over_ensemble.transpose(0, 1)

        return averaged_logits, outputs, borders

    def _handle_constant_target(
        self, n_samples: int, output_type: OutputType, quantiles: list[float]
    ) -> RegressionResultType:
        """Handles prediction when the training target `y` was a constant value."""
        constant_prediction = np.full(n_samples, self.constant_value_)
        if output_type in _OUTPUT_TYPES_BASIC:
            return constant_prediction
        if output_type == "quantiles":
            return [np.copy(constant_prediction) for _ in quantiles]

        # Handle "main" and "full"
        main_outputs = MainOutputDict(
            mean=constant_prediction,
            median=np.copy(constant_prediction),
            mode=np.copy(constant_prediction),
            quantiles=[np.copy(constant_prediction) for _ in quantiles],
        )
        if output_type == "full":
            return FullOutputDict(
                **main_outputs,
                criterion=self.znorm_space_bardist_,
                logits=torch.zeros((n_samples, 1)),
            )
        return main_outputs

    def get_embeddings(
        self,
        X: XType,
        data_source: Literal["train", "test"] = "test",
    ) -> np.ndarray:
        """Gets the embeddings for the input data `X`.

        Args:
            X : XType
                The input data.
            data_source : {"train", "test"}, default="test"
                Select the transformer output to return. Use ``"train"`` to obtain
                embeddings from the training tokens and ``"test"`` for the test
                tokens. When ``n_estimators > 1`` the returned array has shape
                ``(n_estimators, n_samples, embedding_dim)``. ``"train"`` is not
                available with ``fit_mode="fit_with_cache"``; see
                :func:`tabpfn.base.get_embeddings`.

        Returns:
            np.ndarray
                The computed embeddings for each fitted estimator.
        """
        return get_embeddings(self, X, data_source)

    def save_fit_state(self, path: Path | str) -> None:
        """Save a fitted regressor, light wrapper around save_fitted_tabpfn_model."""
        save_fitted_tabpfn_model(self, path)

    @classmethod
    def load_from_fit_state(
        cls, path: Path | str, *, device: str | torch.device = "cpu"
    ) -> TabPFNRegressor:
        """Restore a fitted regressor, light wrapper around load_fitted_tabpfn_model."""
        est = load_fitted_tabpfn_model(path, device=device)
        if not isinstance(est, cls):
            raise TypeError(
                f"Attempting to load a '{est.__class__.__name__}' as '{cls.__name__}'"
            )
        return est

    def to(self, device: DevicesSpecification) -> None:
        """Move the estimator to the given device(s).

        If "auto": devices are selected based on availability in the
        following order of priority: all available CUDA GPUs, "mps", "cpu".

        To manually select a single device: specify a PyTorch device string e.g.
        "cuda:1". See PyTorch's documentation for information about supported
        devices.

        To use several GPUs: specify a list of PyTorch GPU device strings, e.g.
        ["cuda:0", "cuda:1"]. This can dramatically speed up inference for
        larger datasets, by executing the estimators in parallel on the GPUs.
        Multiple GPUs are only used when `fit_mode="fit_preprocessors"` or
        `fit_mode="low_memory"`. In other cases, only the first GPU is used.

        Note:
            The specified device is only used once the model is initialized. This occurs
            during the first .fit() call.
        """
        estimator_to_device(self, device)
        if hasattr(self, "znorm_space_bardist_"):
            self.znorm_space_bardist_.to(self.devices_[0])
        if hasattr(self, "raw_space_bardist_"):
            self.raw_space_bardist_.to(self.devices_[0])


def _logits_to_output(
    *,
    output_type: str,
    logits: torch.Tensor,
    criterion: FullSupportBarDistribution,
    quantiles: list[float],
) -> np.ndarray | list[np.ndarray]:
    """Converts raw model logits to the desired prediction format."""
    if output_type == "quantiles":
        return [criterion.icdf(logits, q).cpu().detach().numpy() for q in quantiles]

    # TODO: support
    #   "pi": criterion.pi(logits, np.max(self.y)),
    #   "ei": criterion.ei(logits),
    if output_type == "mean":
        output = criterion.mean(logits)
    elif output_type == "median":
        output = criterion.median(logits)
    elif output_type == "mode":
        output = criterion.mode(logits)
    else:
        raise ValueError(f"Invalid output type: {output_type}")

    return output.cpu().detach().numpy()


def _validate_eval_metric(
    eval_metric: str | RegressorEvalMetrics | None,
) -> RegressorEvalMetrics:
    if eval_metric is None:
        return DEFAULT_REGRESSION_EVAL_METRIC
    if isinstance(eval_metric, RegressorEvalMetrics):
        return eval_metric
    try:
        return RegressorEvalMetrics(eval_metric)  # Convert string to Enum
    except ValueError as err:
        valid_values = [e.value for e in RegressorEvalMetrics]
        raise ValueError(
            f"Invalid eval_metric: `{eval_metric}`. Must be one of {valid_values}"
        ) from err
