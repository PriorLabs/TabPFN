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
from collections.abc import Callable, Generator, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union
from typing_extensions import Self, TypedDict, overload

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    check_is_fitted,
)

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.base import (
    RegressorModelSpecs,
    check_cpu_warning,
    create_inference_engine,
    determine_precision,
    get_preprocessed_datasets_helper,
    initialize_model_variables_helper,
)
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from tabpfn.preprocessing import (
    DatasetCollectionWithPreprocessing,
    EnsembleConfig,
    PreprocessorConfig,
    RegressorEnsembleConfig,
    ReshapeFeatureDistributionsStep,
    default_regressor_preprocessor_configs,
)
from tabpfn.utils import (
    fix_dtypes,
    get_embeddings,
    get_ordinal_encoder,
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

    from tabpfn.architectures.interface import ArchitectureConfig
    from tabpfn.config import ModelInterfaceConfig
    from tabpfn.constants import XType, YType
    from tabpfn.inference import InferenceEngine

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

    # Default quantiles returned by the predict method
    _DEFAULT_REGRESSION_QUANTILES: typing.ClassVar[list[float]] = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    config_: ArchitectureConfig
    """The configuration of the loaded model to be used for inference.

    The concrete type of this config is defined by the arhitecture in use and should be
    inspected at runtime, but it will be a subclass of ArchitectureConfig.
    """

    interface_config_: ModelInterfaceConfig
    """Additional configuration of the interface for expert users."""

    device_: torch.device
    """The device determined to be used."""

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
        model_path: str | Path | Literal["auto"] | RegressorModelSpecs = "auto",
        device: str | torch.device | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
            "batched",
        ] = "fit_preprocessors",
        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        n_jobs: int = -1,
        inference_config: dict | ModelInterfaceConfig | None = None,
        differentiable_input: bool = False,
    ) -> None:
        """A TabPFN interface for regression.

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
                  downloaded to this location.

            device:
                The device to use for inference with TabPFN. If set to "auto", the
                device is selected based on availability in the following order of
                priority: "cuda", "mps", and then "cpu". You can also set the device
                manually to one of these options.

                See PyTorch's documentation on devices for more information about
                supported devices.

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

                    The current pre-training limits are:

                    - 10_000 samples/rows
                    - 500 features/columns

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
                Enable GPU/CPU memory saving mode. This can help to prevent
                out-of-memory errors that result from computations that would consume
                more memory than available on the current device. We save memory by
                automatically batching certain model computations within TabPFN to
                reduce the total required memory. The options are:

                - If `bool`, enable/disable memory saving mode.
                - If `"auto"`, we will estimate the amount of memory required for the
                  forward pass and apply memory saving if it is more than the
                  available GPU/CPU memory. This is the recommended setting as it
                  allows for speed-ups and prevents memory errors depending on
                  the input data.
                - If `float` or `int`, we treat this value as the maximum amount of
                  available GPU/CPU memory (in GB). We will estimate the amount
                  of memory required for the forward pass and apply memory saving
                  if it is more than this value. Passing a float or int value for
                  this parameter is the same as setting it to True and explicitly
                  specifying the maximum free available memory

                !!! warning
                    This does not batch the original input data. We still recommend to
                    batch this as necessary if you run into memory errors! For example,
                    if the entire input data does not fit into memory, even the memory
                    save mode will not prevent memory errors.

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
                The number of workers for tasks that can be parallelized across CPU
                cores. Currently, this is used for preprocessing the data in parallel
                (if `n_estimators > 1`).

                - If `-1`, all available CPU cores are used.
                - If `int`, the number of CPU cores to use is determined by `n_jobs`.

            inference_config:
                For advanced users, additional advanced arguments that adjust the
                behavior of the model interface.
                See [tabpfn.constants.ModelInterfaceConfig][] for details and options.

                - If `None`, the default ModelInterfaceConfig is used.
                - If `dict`, the key-value pairs are used to update the default
                  `ModelInterfaceConfig`. Raises an error if an unknown key is passed.
                - If `ModelInterfaceConfig`, the object is used as the configuration.

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
        self.memory_saving_mode: bool | Literal["auto"] | float | int = (
            memory_saving_mode
        )
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inference_config = inference_config
        self.differentiable_input = differentiable_input

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

    def get_preprocessed_datasets(
        self,
        X_raw: XType | list[XType],
        y_raw: YType | list[YType],
        split_fn: Callable,
        max_data_size: None | int = 10000,
    ) -> DatasetCollectionWithPreprocessing:
        """Transforms raw input data into a collection of datasets,
        with varying preprocessings.

        The helper function initializes an RNG. This RNG is passed to the
        `DatasetCollectionWithPreprocessing` class. When an item (dataset)
        is retrieved, the collection's preprocessing routine uses this stored
        RNG to generate seeds for its individual workers/pipelines, ensuring
        reproducible stochastic transformations from a fixed initial state.

        Args:
            X_raw: single or list of input dataset features, in case of single it
            is converted to list inside get_preprocessed_datasets_helper()
            y_raw: single or list of input dataset labels, in case of single it
            is converted to list inside get_preprocessed_datasets_helper()
            split_fn: A function to dissect a dataset into train and test partition.
            max_data_size: Maximum allowed number of samples within one dataset.
            If None, datasets are not splitted.
        """
        return get_preprocessed_datasets_helper(
            self,
            X_raw,
            y_raw,
            split_fn,
            max_data_size,
            model_type="regressor",
        )

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
            ensure_y_numeric=False,
            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )

        assert isinstance(X, np.ndarray)
        check_cpu_warning(
            self.device, X, allow_cpu_override=self.ignore_pretraining_limits
        )

        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in

        self.inferred_categorical_indices_ = infer_categorical_features(
            X=X,
            provided=self.categorical_features_indices,
            min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
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

        possible_target_transforms = (
            ReshapeFeatureDistributionsStep.get_all_preprocessors(
                num_examples=y.shape[0],  # Use length of validated y
                random_state=rng,  # Use the provided rng
            )
        )
        target_preprocessors: list[TransformerMixin | Pipeline | None] = []
        for (
            y_target_preprocessor
        ) in self.interface_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS:
            if y_target_preprocessor is not None:
                preprocessor = possible_target_transforms[y_target_preprocessor]
            else:
                preprocessor = None
            target_preprocessors.append(preprocessor)
        preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS

        ensemble_configs = EnsembleConfig.generate_for_regression(
            n=self.n_estimators,
            subsample_size=self.interface_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=typing.cast(
                "Sequence[PreprocessorConfig]",
                preprocess_transforms
                if preprocess_transforms is not None
                else default_regressor_preprocessor_configs(),
            ),
            target_transforms=target_preprocessors,
            random_state=rng,
        )

        self.znorm_space_bardist_ = self.znorm_space_bardist_.to(self.device_)

        assert len(ensemble_configs) == self.n_estimators

        return ensemble_configs, X, y, self.znorm_space_bardist_

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
        if not hasattr(self, "model_") or not no_refit:
            byte_size, rng = self._initialize_model_variables()
        else:
            _, _, byte_size = determine_precision(
                self.inference_precision, self.device_
            )
            rng = None

        # Create the inference engine
        self.executor_ = create_inference_engine(
            X_train=X_preprocessed,
            y_train=y_preprocessed,
            model=self.model_,
            ensemble_configs=configs,
            cat_ix=cat_ix,
            fit_mode="batched",
            device_=self.device_,
            rng=rng,
            n_jobs=self.n_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            inference_mode=not self.differentiable_input,  # False if differentiable
            # needed (prompt tune)
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
        ensemble_configs: list[RegressorEnsembleConfig]

        if self.fit_mode == "batched":
            logging.warning(
                "The model was in 'batched' mode, likely after finetuning. "
                "Automatically switching to 'fit_preprocessors' mode for standard "
                "prediction. The model will be re-initialized."
            )
            self.fit_mode = "fit_preprocessors"

        if not hasattr(self, "model_") or not self.differentiable_input:
            byte_size, rng = self._initialize_model_variables()
            ensemble_configs, X, y, self.znorm_space_bardist_ = (
                self._initialize_dataset_preprocessing(X, y, rng)
            )
        else:  # already fitted and prompt_tuning mode: no cat. features
            _, rng = infer_random_state(self.random_state)
            _, _, byte_size = determine_precision(
                self.inference_precision, self.device_
            )

        assert len(ensemble_configs) == self.n_estimators

        self.is_constant_target_ = np.unique(y).size == 1
        self.constant_value_ = y[0] if self.is_constant_target_ else None

        if self.is_constant_target_:
            self.znorm_space_bardist_ = FullSupportBarDistribution(
                borders=torch.tensor(
                    [self.constant_value_ - 1e-5, self.constant_value_ + 1e-5]
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
            model=self.model_,
            ensemble_configs=ensemble_configs,
            cat_ix=self.inferred_categorical_indices_,
            fit_mode=self.fit_mode,
            device_=self.device_,
            rng=rng,
            n_jobs=self.n_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            # TODO: Standard fit usually uses inference_mode=True, before it was enabled
        )

        return self

    def _raw_predict(self, X: XType) -> torch.Tensor:
        """Handles preprocessing and calls the forward pass to get final predictions."""
        # 1. Preprocess the input data
        X_processed = validate_X_predict(X, self)
        X_processed = fix_dtypes(
            X_processed, cat_indices=self.inferred_categorical_indices_
        )
        X_processed = process_text_na_dataframe(
            X_processed, ord_encoder=self.preprocessor_
        )

        # 2. Get final logits directly from the efficient forward pass
        final_logits = self.forward(X_processed, use_inference_mode=True)

        if final_logits is None:
            raise ValueError("Prediction failed: the model produced no output.")

        return final_logits

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

        if quantiles is None:
            quantiles = self._DEFAULT_REGRESSION_QUANTILES.copy()

        assert all(
            (0 <= q <= 1) and (isinstance(q, float)) for q in quantiles
        ), "All quantiles must be between 0 and 1 and floats."

        if output_type not in _USABLE_OUTPUT_TYPES:
            raise ValueError(f"Invalid output type: {output_type}")

        if hasattr(self, "is_constant_target_") and self.is_constant_target_:
            return self._handle_constant_target(
                validate_X_predict(X, self).shape[0], output_type, quantiles
            )

        # Get the final logits from our single, powerful helper method
        logits = self._raw_predict(X)

        # Convert final logits to the requested output format
        logit_to_output = partial(
            _logits_to_output,
            logits=logits,
            criterion=self.raw_space_bardist_,
            quantiles=quantiles,
        )
        if output_type in ["full", "main"]:
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
                return FullOutputDict(
                    **main_outputs,
                    criterion=self.raw_space_bardist_,
                    logits=logits,
                )
            return main_outputs

        return logit_to_output(output_type=output_type)

    def _get_raw_output_generator(
        self, X: XType, *, use_inference_mode: bool
    ) -> Generator[tuple[torch.Tensor, RegressorEnsembleConfig], None, None]:
        """A generator that executes the model and yields the raw output tensor
        and its corresponding config for each estimator in the ensemble.
        """
        # This method is only supported for fit_modes that might switch between
        # training and inference.
        if self.fit_mode in ["fit_preprocessors", "batched"]:
            self.executor_.use_torch_inference_mode(use_inference=use_inference_mode)

        for output, config in self.executor_.iter_outputs(
            X, device=self.device_, autocast=self.use_autocast_
        ):
            config_for_ensemble = config[0] if isinstance(config, list) else config
            if not isinstance(config_for_ensemble, RegressorEnsembleConfig):
                raise TypeError(
                    f"Expected RegressorEnsembleConfig, "
                    f"but got {type(config_for_ensemble).__name__}."
                )

            yield output, config_for_ensemble

    def _process_one_output_for_prediction(
        self, output: torch.Tensor, config: RegressorEnsembleConfig
    ) -> torch.Tensor:
        """Processes a single raw model output for the prediction path.

        This includes border transformation, temperature scaling, and probability
        translation to the standardized target distribution.
        """
        std_borders = self.bardist_.borders.cpu().numpy()

        # Check if a target transform exists before applying it
        if config.target_transform is None:
            borders_t = std_borders
            logit_cancel_mask = None
            descending_borders = False
        else:
            # A transform exists, so call the helper
            (
                logit_cancel_mask,
                descending_borders,
                borders_t,
            ) = transform_borders_one(
                std_borders,
                target_transform=config.target_transform,
                repair_nan_borders_after_transform=(
                    self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM
                ),
            )

        # Apply transformations based on the results
        if descending_borders:
            # Some target transforms can reverse the bin order
            borders_t = borders_t.flip(-1)

        if logit_cancel_mask is not None:
            # Clone to avoid modifying the original tensor which might be used elsewhere
            output = output.clone()
            output[..., logit_cancel_mask] = float("-inf")

        # Apply temperature
        if self.softmax_temperature != 1.0:
            output = output.float() / self.softmax_temperature

        # Translate to standardized borders and return probabilities
        return translate_probs_across_borders(
            output,
            frm=torch.as_tensor(borders_t, device=self.device_),
            to=self.bardist_.borders.to(self.device_),
        )

    def forward(
        self,
        X: list[torch.Tensor] | XType,
        *,
        use_inference_mode: bool = False,
    ) -> torch.Tensor | None:
        """Performs a memory-efficient forward pass for fine-tuning or prediction.
        Includes an optimized fast path for fine-tuning when no target transforms
        are used.
        """
        check_is_fitted(self)

        # --- Pre-flight Checks and Assertions ---
        # This import is only needed for type checking
        from tabpfn.inference import InferenceEngineBatchedNoPreprocessing

        # Scenario 1: Standard inference path for predict()
        is_standard_inference = use_inference_mode and not isinstance(
            self.executor_, InferenceEngineBatchedNoPreprocessing
        )
        # Scenario 2: Batched path for fine-tuning with gradients
        is_batched_for_grads = (
            not use_inference_mode
            and isinstance(self.executor_, InferenceEngineBatchedNoPreprocessing)
            and isinstance(X, list)
            and (not X or isinstance(X[0], torch.Tensor))
        )
        assert is_standard_inference or is_batched_for_grads, (
            "Invalid forward pass: Bad combination of inference mode, input X, "
            "or executor type. Ensure the call is from the standard predict() method "
            "or a batched fine-tuning context."
        )
        # Specific check for float64 incompatibility with the fine-tuning workflow.
        assert not (
            isinstance(self.executor_, InferenceEngineBatchedNoPreprocessing)
            and self.forced_inference_dtype_ == torch.float64
        ), (
            "Batched engine error: float64 precision is not supported for the "
            "fine-tuning workflow, which requires float32 for backpropagation."
        )

        # Ensure torch.inference_mode is OFF to allow gradients
        if self.fit_mode in ["fit_preprocessors", "batched"]:
            # only these two modes support this option
            self.executor_.use_torch_inference_mode(use_inference=use_inference_mode)

        # --- Check for Fast Path Optimization ---
        configs = self.executor_.ensemble_configs

        # Create a single iterator that handles both flat and nested lists
        # TODO: We can use the fast path as long as the target transforms are the same
        #   to make it simpler we disable the fast path for now
        #   We could even merge transformed logits but this would likely be numerically
        #   unstable.
        can_use_fast_path = len(configs) == 1

        output_generator = self._get_raw_output_generator(
            X, use_inference_mode=use_inference_mode
        )

        final_logits: torch.Tensor | None

        if can_use_fast_path and not use_inference_mode:
            # --- Fast Path for Fine-Tuning - gradients are cleaner, no sotftmax ---
            sum_of_logits = None
            estimator_count = 0
            for raw_output, _ in output_generator:
                estimator_count += 1
                sum_of_logits = (
                    raw_output if sum_of_logits is None else sum_of_logits + raw_output
                )
            final_logits = (
                (sum_of_logits / estimator_count) if sum_of_logits is not None else None
            )
        elif not can_use_fast_path and not use_inference_mode:
            raise NotImplementedError(
                "The fast path for fine-tuning is only supported when using a single "
                "ensemble config for now."
            )
        else:
            # --- General Path (for Prediction OR if transforms used with finetune) ---
            accumulator = None
            estimator_count = 0
            for raw_output, config in output_generator:
                estimator_count += 1
                translated_proba = self._process_one_output_for_prediction(
                    raw_output, config
                )
                current_val = (
                    translated_proba.log()
                    if self.average_before_softmax
                    else translated_proba
                )
                accumulator = (
                    current_val if accumulator is None else accumulator + current_val
                )

            avg_val = accumulator / estimator_count
            final_logits = (
                torch.nn.functional.log_softmax(avg_val, dim=-1)
                if self.average_before_softmax
                else avg_val.log()
            )

        # --- Final Processing and Return ---
        if final_logits.dtype == torch.float16:
            return final_logits.float()

        if not use_inference_mode:
            return final_logits.transpose(0, 1)

        return final_logits

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
