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

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import logging
import typing
import warnings
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union
from typing_extensions import Self, TypedDict, deprecated, overload

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    check_is_fitted,
)
from tabpfn_common_utils.telemetry import track_model_call

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.base import (
    RegressorModelSpecs,
    check_cpu_warning,
    create_inference_engine,
    determine_precision,
    estimator_to_device,
    initialize_model_variables_helper,
    initialize_telemetry,
)
from tabpfn.constants import REGRESSION_CONSTANT_TARGET_BORDER_EPSILON, ModelVersion
from tabpfn.errors import TabPFNValidationError
from tabpfn.inference import InferenceEngine, InferenceEngineBatchedNoPreprocessing
from tabpfn.model_loading import (
    ModelSource,
    load_fitted_tabpfn_model,
    log_model_init_params,
    prepend_cache_path,
    save_fitted_tabpfn_model,
)
from tabpfn.preprocessing import (
    EnsembleConfig,
    RegressorEnsembleConfig,
    generate_regression_ensemble_configs,
)
from tabpfn.preprocessing.steps import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder
from tabpfn.utils import (
    DevicesSpecification,
    fix_dtypes,
    get_embeddings,
    infer_categorical_features,
    infer_random_state,
    process_text_na_dataframe,
    transform_borders_one,
    translate_probs_across_borders,
    validate_X_predict,
    validate_Xy_fit,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from torch.types import _dtype

    from tabpfn.architectures.base.memory import MemorySavingMode
    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
    from tabpfn.constants import XType, YType
    from tabpfn.inference import InferenceEngine
    from tabpfn.inference_config import InferenceConfig

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


RegressionResultType = Union[
    np.ndarray, list[np.ndarray], MainOutputDict, FullOutputDict
]
"""The type hint for the return value of the `predict` method."""


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

    inferred_categorical_indices_: list[int]
    """The indices of the columns that were inferred to be categorical,
    as a product of any features deemed categorical by the user and what would
    work best for the model.
    """

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

    preprocessor_: ColumnTransformer
    """The column transformer used to preprocess the input data to be numeric."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int = 8,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature: float = 0.9,
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
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        n_jobs: Annotated[int | None, deprecated("Use n_preprocessing_jobs")] = None,
        n_preprocessing_jobs: int = 1,
        inference_config: dict | InferenceConfig | None = None,
        differentiable_input: bool = False,
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
                  the model with more than 1000 samples on CPU.
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
                - If `InferenceConfig`, the object is used as the configuration.

            differentiable_input:
                If true, preprocessing attempts to be end-to-end differentiable.
                Less relevant for standard regression fine-tuning compared to
                prompt-tuning.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
            inference_precision
        )
        self.fit_mode: Literal["low_memory", "fit_preprocessors", "batched"] = fit_mode
        self.memory_saving_mode = memory_saving_mode
        self.random_state = random_state
        self.inference_config = inference_config
        self.differentiable_input = differentiable_input

        if n_jobs is not None:
            warnings.warn(
                "TabPFNRegressor(n_jobs=...) is deprecated and has no effect. "
                "Use `n_preprocessing_jobs` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.n_jobs = n_jobs
        self.n_preprocessing_jobs = n_preprocessing_jobs
        initialize_telemetry()

        # Only anonymously record `fit_mode` usage
        log_model_init_params(self, {"fit_mode": self.fit_mode})

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
                "n_estimators": 8,
                "softmax_temperature": 0.9,
            }
        elif version == ModelVersion.V2_5:
            options = {
                "model_path": prepend_cache_path(
                    ModelSource.get_regressor_v2_5().default_filename
                ),
                "n_estimators": 8,
                "softmax_temperature": 0.9,
            }
        else:
            raise ValueError(f"Unknown version: {version}")

        options.update(overrides)

        return cls(**options)

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

    # TODO: We can remove this from scikit-learn lower bound of 1.6
    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        return tags

    def _initialize_model_variables(self) -> tuple[int, np.random.Generator]:
        """Initializes the model, returning byte_size and RNG object."""
        return initialize_model_variables_helper(self, "regressor")

    def _initialize_dataset_preprocessing(
        self, X: XType, y: YType, rng: np.random.Generator
    ) -> tuple[list[RegressorEnsembleConfig], XType, YType, FullSupportBarDistribution]:
        """Prepare ensemble configs and validate X, y for one dataset/chunk.
        Handle the preprocessing of the input (X and y). We also return the
        BarDistribution here, since it is vital for computing the standardized
        target variable in the DatasetCollectionWithPreprocessing class.
        Sets self.inferred_categorical_indices_.
        """
        if self.differentiable_input:
            raise ValueError(
                "Differentiable input is not supported for regressors yet."
            )

        X, y, feature_names_in, n_features_in = validate_Xy_fit(
            X,
            y,
            estimator=self,
            ensure_y_numeric=True,
            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )

        assert isinstance(X, np.ndarray)
        check_cpu_warning(
            self.devices_, X, allow_cpu_override=self.ignore_pretraining_limits
        )

        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in

        self.inferred_categorical_indices_ = infer_categorical_features(
            X=X,
            provided=self.categorical_features_indices,
            min_samples_for_inference=self.inference_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.inference_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        )

        # Will convert inferred categorical indices to category dtype,
        # to be picked up by the ord_encoder, as well
        # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
        X = fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
        # Ensure categories are ordinally encoded
        ord_encoder = get_ordinal_encoder()
        X = process_text_na_dataframe(
            X,
            ord_encoder=ord_encoder,
            fit_encoder=True,  # type: ignore
        )
        self.preprocessor_ = ord_encoder

        possible_target_transforms = get_all_reshape_feature_distribution_preprocessors(
            num_examples=y.shape[0],  # Use length of validated y
            random_state=rng,  # Use the provided rng
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

        ensemble_configs = generate_regression_ensemble_configs(
            num_estimators=self.n_estimators,
            subsample_samples=self.inference_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.inference_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=self.inference_config_.PREPROCESS_TRANSFORMS,
            target_transforms=target_preprocessors,
            random_state=rng,
            num_models=len(self.models_),
        )

        self.znorm_space_bardist_ = self.znorm_space_bardist_.to(self.devices_[0])

        assert len(ensemble_configs) == self.n_estimators

        return ensemble_configs, X, y, self.znorm_space_bardist_

    @track_model_call("fit", param_names=["X_preprocessed", "y_preprocessed"])
    def fit_from_preprocessed(
        self,
        X_preprocessed: list[torch.Tensor],
        y_preprocessed: list[torch.Tensor],  # These y are standardized
        cat_ix: list[list[int]],
        configs: list[list[EnsembleConfig]],  # Should be RegressorEnsembleConfig
        *,
        no_refit: bool = True,
    ) -> TabPFNRegressor:
        """Used in Fine-Tuning. Fit the model to preprocessed inputs from torch
        dataloader inside a training loop a Dataset provided by
        get_preprocessed_datasets. This function sets the fit_mode attribute
        to "batched" internally.

        Args:
            X_preprocessed: The input features obtained from the preprocessed Dataset
                The list contains one item for each ensemble predictor.
                use tabpfn.utils.collate_for_tabpfn_dataset to use this function with
                batch sizes of more than one dataset (see examples/tabpfn_finetune.py)
            y_preprocessed: The target variable obtained from the preprocessed Dataset
            cat_ix: categorical indices obtained from the preprocessed Dataset
            configs: Ensemble configurations obtained from the preprocessed Dataset
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
            byte_size, rng = self._initialize_model_variables()
        else:
            _, _, byte_size = determine_precision(
                self.inference_precision, self.devices_
            )
            rng = None

        # Create the inference engine
        self.executor_ = create_inference_engine(
            X_train=X_preprocessed,
            y_train=y_preprocessed,
            models=self.models_,
            ensemble_configs=configs,
            cat_ix=cat_ix,
            fit_mode="batched",
            devices_=self.devices_,
            rng=rng,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            inference_mode=not self.differentiable_input,  # False if differentiable
            # needed (prompt tune)
        )

        return self

    @config_context(transform_output="default")  # type: ignore
    @track_model_call(model_method="fit", param_names=["X", "y"])
    def fit(self, X: XType, y: YType) -> Self:
        """Fit the model.

        Args:
            X: The input data.
            y: The target variable.

        Returns:
            self
        """
        ensemble_configs: list[RegressorEnsembleConfig]

        if self.fit_mode == "batched":
            logging.warning(
                "The model was in 'batched' mode, likely after finetuning. "
                "Automatically switching to 'fit_preprocessors' mode for standard "
                "prediction. The model will be re-initialized."
            )
            self.fit_mode = "fit_preprocessors"

        if not hasattr(self, "models_") or not self.differentiable_input:
            byte_size, rng = self._initialize_model_variables()
            ensemble_configs, X, y, self.znorm_space_bardist_ = (
                self._initialize_dataset_preprocessing(X, y, rng)
            )
        else:  # already fitted and prompt_tuning mode: no cat. features
            _, rng = infer_random_state(self.random_state)
            _, _, byte_size = determine_precision(
                self.inference_precision, self.devices_
            )

        assert len(ensemble_configs) == self.n_estimators

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

        mean, std = np.mean(y), np.std(y)
        self.y_train_std_ = std.item() + 1e-20
        self.y_train_mean_ = mean.item()
        y = (y - self.y_train_mean_) / self.y_train_std_
        self.raw_space_bardist_ = FullSupportBarDistribution(
            self.znorm_space_bardist_.borders * self.y_train_std_ + self.y_train_mean_,
        ).float()

        # Create the inference engine
        self.executor_ = create_inference_engine(
            X_train=X,
            y_train=y,
            models=self.models_,
            ensemble_configs=ensemble_configs,
            cat_ix=self.inferred_categorical_indices_,
            fit_mode=self.fit_mode,
            devices_=self.devices_,
            rng=rng,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            # TODO: Standard fit usually uses inference_mode=True, before it was enabled
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
    @track_model_call(model_method="predict", param_names=["X"])
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
        X = validate_X_predict(X, self)

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

        X = fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
        X = process_text_na_dataframe(X, ord_encoder=self.preprocessor_)  # type: ignore

        # Runs over iteration engine
        (
            _,
            outputs,  # list of tensors [N_est, N_samples, N_borders] (after forward)
            borders,  # list of numpy arrays containing borders for each estimator
        ) = self.forward(X, use_inference_mode=True)

        # --- Translate probs, average, get final logits ---
        transformed_logits = [
            translate_probs_across_borders(
                logits,
                frm=torch.as_tensor(borders_t, device=logits.device),
                to=self.znorm_space_bardist_.borders.to(logits.device),
            )
            for logits, borders_t in zip(outputs, borders)
        ]
        stacked_logits = torch.stack(transformed_logits, dim=0)
        if self.average_before_softmax:
            logits = stacked_logits.log().mean(dim=0).softmax(dim=-1)
        else:
            logits = stacked_logits.mean(dim=0)

        # Post-process the logits
        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()

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

            # Create our typed dictionary
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

        # Ensure torch.inference_mode is OFF to allow gradients
        if self.fit_mode in ["fit_preprocessors", "batched"]:
            # only these two modes support this option
            self.executor_.use_torch_inference_mode(use_inference=use_inference_mode)

        check_is_fitted(self)

        std_borders = self.znorm_space_bardist_.borders.cpu().numpy()
        outputs: list[torch.Tensor] = []
        borders: list[np.ndarray] = []

        # Iterate over estimators
        for output, config in self.executor_.iter_outputs(
            X, autocast=self.use_autocast_
        ):
            output = output.float()  # noqa: PLW2901
            if self.softmax_temperature != 1:
                output = output / self.softmax_temperature  # noqa: PLW2901

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

                borders.append(borders_t)

                if logit_cancel_mask is not None:
                    output = output.clone()  # noqa: PLW2901
                    output[..., logit_cancel_mask] = float("-inf")

            else:
                raise ValueError(
                    "Unexpected config format "
                    "and Batch prediction is not supported yet!"
                )

            outputs.append(output)  # type: ignore

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
                ``(n_estimators, n_samples, embedding_dim)``.

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

        If "auto": a single device is selected based on availability in the
        following order of priority: "cuda:0", "mps", "cpu".

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
