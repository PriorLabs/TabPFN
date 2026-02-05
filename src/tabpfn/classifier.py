"""TabPFNClassifier class.

!!! example
    ```python
    import sklearn.datasets
    from tabpfn import TabPFNClassifier

    model = TabPFNClassifier()

    X, y = sklearn.datasets.load_iris(return_X_y=True)

    model.fit(X, y)
    predictions = model.predict(X)
    ```
"""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from typing_extensions import Self, deprecated

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from tabpfn_common_utils.telemetry import track_model_call

from tabpfn.base import (
    ClassifierModelSpecs,
    create_inference_engine,
    determine_precision,
    estimator_to_device,
    get_embeddings,
    initialize_model_variables_helper,
    initialize_telemetry,
)
from tabpfn.constants import (
    PROBABILITY_EPSILON_ROUND_ZERO,
    SKLEARN_16_DECIMAL_PRECISION,
    ModelVersion,
    XType,
    YType,
)
from tabpfn.errors import handle_oom_errors
from tabpfn.inference import (
    InferenceEngine,
    InferenceEngineBatchedNoPreprocessing,
    InferenceEngineCachePreprocessing,
)
from tabpfn.inference_tuning import (
    ClassifierEvalMetrics,
    ClassifierTuningConfig,
    find_optimal_classification_thresholds,
    find_optimal_temperature,
    get_tuning_splits,
    resolve_tuning_config,
)
from tabpfn.model_loading import (
    ModelSource,
    load_fitted_tabpfn_model,
    log_model_init_params,
    prepend_cache_path,
    save_fitted_tabpfn_model,
)
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    PreprocessorConfig,
    clean_data,
    generate_classification_ensemble_configs,
)
from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.label_encoder import TabPFNLabelEncoder
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.utils import (
    DevicesSpecification,
    balance_probas_by_class_counts,
    convert_batch_of_cat_ix_to_schema,
    infer_random_state,
    remove_non_differentiable_preprocessing_from_models,
)
from tabpfn.validation import (
    ensure_compatible_fit_inputs,
    ensure_compatible_predict_input_sklearn,
    validate_dataset_size,
    validate_num_classes,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.compose import ColumnTransformer
    from torch.types import _dtype

    from tabpfn.architectures.base.memory import MemorySavingMode
    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
    from tabpfn.inference_config import InferenceConfig

    try:
        from sklearn.base import Tags
    except ImportError:
        Tags = Any

DEFAULT_CLASSIFICATION_EVAL_METRIC = ClassifierEvalMetrics.ACCURACY


class TabPFNClassifier(ClassifierMixin, BaseEstimator):
    """TabPFNClassifier class."""

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
    """Additional configuration of the interface for expert users."""

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

    inferred_feature_schema_: FeatureSchema
    """The inferred feature schema. This contains the feature modalities per column,
    using heuristics and user-provided indices for categorical features."""

    classes_: npt.NDArray[Any]
    """The unique classes found in the target data during `fit()`."""

    n_classes_: int
    """The number of classes found in the target data during `fit()`."""

    class_counts_: npt.NDArray[Any]
    """The number of classes per class found in the target data during `fit()`."""

    n_outputs_: Literal[1]
    """The number of outputs the model has. Only 1 for now"""

    use_autocast_: bool
    """Whether torch's autocast should be used."""

    forced_inference_dtype_: _dtype | None
    """The forced inference dtype for the model based on `inference_precision`."""

    executor_: InferenceEngine
    """The inference engine used to make predictions."""

    label_encoder_: TabPFNLabelEncoder
    """The label encoder used to encode the target variable."""

    ordinal_encoder_: ColumnTransformer
    """The column transformer used to preprocess categorical data to be numeric."""

    tuned_classification_thresholds_: npt.NDArray[Any] | None
    """The tuned classification thresholds for each class or None if no tuning is
    specified."""

    eval_metric_: ClassifierEvalMetrics
    """The validated evaluation metric to optimize for during prediction."""

    softmax_temperature_: float
    """The softmax temperature used for prediction. This is set to the default softmax
    temperature if no temperature tuning is done"""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int = 8,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        model_path: str
        | Path
        | list[str]
        | list[Path]
        | Literal["auto"]
        | ClassifierModelSpecs
        | list[ClassifierModelSpecs] = "auto",
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
        eval_metric: str | ClassifierEvalMetrics | None = None,
        tuning_config: dict | ClassifierTuningConfig | None = None,
    ) -> None:
        """Construct a TabPFN classifier.

        This constructs a classifier using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model.

        Args:
            n_estimators:
                The number of estimators in the TabPFN ensemble. We aggregate the
                 predictions of `n_estimators`-many forward passes of TabPFN. Each
                 forward pass has (slightly) different input data. Think of this as an
                 ensemble of `n_estimators`-many "prompts" of the input data.

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
                a post-processing step. Set `softmax_temperature=1.0` for no effect. Be
                advised that `.predict()` does not currently sample, so this setting is
                only relevant for `.predict_proba()` and `.predict_logits()`.

            balance_probabilities:
                Whether to balance the probabilities based on the class distribution
                in the training data. This can help to improve predictive performance
                when the classes are highly imbalanced and the metric of interest is
                insensitive to class imbalance (e.g., balanced accuracy, balanced log
                loss, roc-auc macro ovo, etc.). This is only applied when predicting
                during a post-processing step.

            average_before_softmax:
                Only used if `n_estimators > 1`. Whether to average the predictions of
                the estimators before applying the softmax function. This can help to
                improve predictive performance when there are many classes or when
                calibrating the model's confidence. This is only applied when predicting
                during a post-processing.

                - If `True`, the predictions are averaged before applying the softmax
                  function. Thus, we average the logits of TabPFN and then apply the
                  softmax.
                - If `False`, the softmax function is applied to each set of logits.
                  Then, we average the resulting probabilities of each forward pass.

            model_path:
                The path to the TabPFN model file, i.e., the pre-trained weights.
                Can be a list of paths to load multiple models. If a list is provided,
                the models are applied across different estimators.

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
                    - 10 classes, this is not ignorable and will raise an error
                      if the model is used with more classes.

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
                results and see the scikit-learn glossary for more information. If
                `None`, the randomness is determined by the system when calling
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
                If true, the preprocessing will be adapted to be end-to-end
                differentiable with PyTorch.
                This is useful for explainability and prompt-tuning, essential
                in the prompttuning code.

            eval_metric:
                Metric by which predictions will be ultimately evaluated on test data.
                This can be used to improve this metric on validation data by
                calibrating the model's probabilities and tuning the decision
                thresholds during the `fit()/predict()` calls. The tuning can be
                enabled by configuring the `tuning_config` argument, see below.
                For currently supported metrics, see
                [tabpfn.classifier.ClassifierEvalMetrics][].

            tuning_config:
                The settings to use to tune the model's predictions for the specified
                `eval_metric`. See
                [tabpfn.inference_tuning.ClassifierTuningConfig][] for details
                and options.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
            inference_precision
        )
        self.fit_mode = fit_mode
        self.memory_saving_mode: MemorySavingMode = memory_saving_mode
        self.random_state = random_state
        self.inference_config = inference_config
        self.differentiable_input = differentiable_input

        if n_jobs is not None:
            warnings.warn(
                "TabPFNClassifier(n_jobs=...) is deprecated and has no effect. "
                "Use `n_preprocessing_jobs` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.n_jobs = n_jobs
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.eval_metric = eval_metric
        self.tuning_config = tuning_config
        initialize_telemetry()

        # Only anonymously record `fit_mode` usage
        log_model_init_params(self, {"fit_mode": self.fit_mode})

    @classmethod
    def create_default_for_version(cls, version: ModelVersion, **overrides) -> Self:
        """Construct a classifier that uses the given version of the model.

        In addition to selecting the model, this also configures certain settings to the
        default values associated with this model version.

        Any kwargs will override the default settings.
        """
        if version == ModelVersion.V2:
            options = {
                "model_path": prepend_cache_path(
                    ModelSource.get_classifier_v2().default_filename
                ),
                "n_estimators": 8,
                "softmax_temperature": 0.9,
            }
        elif version == ModelVersion.V2_5:
            options = {
                "model_path": prepend_cache_path(
                    ModelSource.get_classifier_v2_5().default_filename
                ),
                "n_estimators": 8,
                "softmax_temperature": 0.9,
            }
        else:
            raise ValueError(f"Unknown version: {version}")

        options.update(overrides)

        return cls(**options)

    @property
    def estimator_type(self) -> Literal["classifier"]:
        """The type of the model."""
        return "classifier"

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

    # TODO: We can remove this from scikit-learn lower bound of 1.6
    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
            "multilabel": False,
        }

    def __sklearn_tags__(self) -> Tags:  # type: ignore
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = self.estimator_type
        return tags

    def _initialize_model_variables(self) -> tuple[int, np.random.Generator]:
        """Perform initialization of the model, return determined byte_size
        and RNG object.
        """
        return initialize_model_variables_helper(self, self.estimator_type)

    def _initialize_for_differentiable_input(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[list[ClassifierEnsembleConfig], torch.Tensor, torch.Tensor]:
        """Initialize the model for differentiable input."""
        validate_dataset_size(
            X=X,
            y=y,
            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
            devices=self.devices_,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )

        # We use the convention that the class labels are [0, ..., n-1]
        # for differentiable input.
        if not hasattr(self, "n_classes_"):
            self.n_classes_ = int(torch.max(y).item()) + 1
        self.classes_ = torch.arange(self.n_classes_)

        validate_num_classes(
            num_classes=self.n_classes_,
            max_num_classes=self.inference_config_.MAX_NUMBER_OF_CLASSES,
        )

        # Minimal preprocessing for prompt tuning
        if (
            self.categorical_features_indices is not None
            and len(self.categorical_features_indices) > 0
        ):
            raise ValueError(
                "Categorical features are not supported for differentiable input."
            )
        n_features = X.shape[1]
        features = [Feature(name=None, modality=FeatureModality.NUMERICAL)] * n_features
        self.inferred_feature_schema_ = FeatureSchema(features=features)
        preprocessor_configs = [PreprocessorConfig("none", differentiable=True)]

        ensemble_configs = generate_classification_ensemble_configs(
            num_estimators=self.n_estimators,
            subsample_samples=self.inference_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.inference_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=preprocessor_configs,
            class_shift_method=None,
            n_classes=self.n_classes_,
            random_state=rng,
            num_models=len(self.models_),
            outlier_removal_std=self.inference_config_.get_resolved_outlier_removal_std(
                estimator_type=self.estimator_type
            ),
        )
        assert len(ensemble_configs) == self.n_estimators

        return ensemble_configs, X, y

    def _initialize_dataset_preprocessing(
        self,
        X: XType,
        y: YType,
        rng: np.random.Generator,
    ) -> tuple[list[ClassifierEnsembleConfig], np.ndarray, np.ndarray]:
        """Initialize the model for standard input."""
        # Data validation and cleaning
        X, y, feature_names, n_features, original_y_name = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=self,
            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            ensure_y_numeric=False,
            devices=self.devices_,
        )

        feature_schema = detect_feature_modalities(
            X=X,
            feature_names=feature_names,
            provided_categorical_indices=self.categorical_features_indices,
            min_samples_for_inference=self.inference_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.inference_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        )
        X, ordinal_encoder, feature_schema = clean_data(
            X=X, feature_schema=feature_schema
        )
        self.inferred_feature_schema_ = feature_schema
        self.ordinal_encoder_ = ordinal_encoder
        self.feature_names_in_ = feature_names
        self.n_features_in_ = n_features

        # Label encoding
        self.label_encoder_ = TabPFNLabelEncoder(original_target_name=original_y_name)
        y, label_metadata = self.label_encoder_.fit_transform(
            y=y, max_num_classes=self.inference_config_.MAX_NUMBER_OF_CLASSES
        )
        self.classes_ = label_metadata.classes
        self.n_classes_ = label_metadata.n_classes
        self.class_counts_ = label_metadata.class_counts

        # Ensemble definition
        preprocessor_configs = self.inference_config_.PREPROCESS_TRANSFORMS
        ensemble_configs = generate_classification_ensemble_configs(
            num_estimators=self.n_estimators,
            subsample_samples=self.inference_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.inference_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=preprocessor_configs,
            class_shift_method=self.inference_config_.CLASS_SHIFT_METHOD,
            n_classes=self.n_classes_,
            random_state=rng,
            num_models=len(self.models_),
            outlier_removal_std=self.inference_config_.get_resolved_outlier_removal_std(
                estimator_type=self.estimator_type
            ),
        )
        assert len(ensemble_configs) == self.n_estimators

        return ensemble_configs, X, y

    def _get_tuning_classifier(self, **overwrite_kwargs: Any) -> TabPFNClassifier:
        """Return a fresh classifier configured for holdout tuning."""
        params = self.get_params(deep=False)

        # Avoids sharing mutable config across instances
        for key in params:
            try:
                if isinstance(params.get(key), dict):
                    params[key] = copy.deepcopy(params[key])
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "Error during initialization of tuning classifier when trying "
                    f"to deepcopy configuration with name `{key}`: {e}. "
                    "Falling back to original configuration"
                )

        forced = {
            "fit_mode": "fit_preprocessors",
            "differentiable_input": False,
            "tuning_config": None,  # never tune inside tuning
        }

        params.update(forced)
        params.update(overwrite_kwargs)

        return TabPFNClassifier(**params)

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
        # Validate eval_metric here instead of in __init__ as per sklearn convention
        self.eval_metric_ = _validate_eval_metric(self.eval_metric)

        if self.fit_mode == "batched":
            logging.warning(
                "The model was in 'batched' mode, likely after finetuning. "
                "Automatically switching to 'fit_preprocessors' mode for standard "
                "prediction. The model will be re-initialized."
            )
            self.fit_mode: Literal[
                "low_memory",
                "fit_preprocessors",
                "fit_with_cache",
                "batched",
            ] = "fit_preprocessors"

        byte_size, rng = self._initialize_model_variables()
        ensemble_configs, X, y = self._initialize_dataset_preprocessing(X, y, rng)
        self.ensemble_configs_ = ensemble_configs

        self._maybe_calibrate_temperature_and_tune_decision_thresholds(X=X, y=y)

        self.ensemble_preprocessor_ = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            rng=rng,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            keep_fitted_cache=(self.fit_mode == "fit_with_cache"),
        )

        self.executor_ = create_inference_engine(
            fit_mode=self.fit_mode,
            X_train=X,
            y_train=y,
            feature_schema=self.inferred_feature_schema_,
            models=self.models_,
            ensemble_preprocessor=self.ensemble_preprocessor_,
            devices_=self.devices_,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            inference_mode=True,
        )

        return self

    @track_model_call("fit", param_names=["X_preprocessed", "y_preprocessed"])
    def fit_from_preprocessed(
        self,
        X_preprocessed: list[torch.Tensor],
        y_preprocessed: list[torch.Tensor],
        cat_ix: list[list[list[int]]],
        configs: list[list[EnsembleConfig]],
        *,
        no_refit: bool = True,
    ) -> TabPFNClassifier:
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
            byte_size, _ = self._initialize_model_variables()
        else:
            _, _, byte_size = determine_precision(
                self.inference_precision, self.devices_
            )

        feature_schema = convert_batch_of_cat_ix_to_schema(
            batch_of_cat_indices=cat_ix,
            num_features=X_preprocessed[0].shape[1],
        )

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
            inference_mode=not self.differentiable_input,
        )

        return self

    @track_model_call(model_method="fit", param_names=["X", "y"])
    def fit_with_differentiable_input(self, X: torch.Tensor, y: torch.Tensor) -> Self:
        """Fit the model with differentiable input.

        Args:
            X: The input data.
            y: The target variable.

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

        is_first_fit_call = not hasattr(self, "models_")
        if is_first_fit_call:
            byte_size, rng = self._initialize_model_variables()
            ensemble_configs, X, y = self._initialize_for_differentiable_input(
                X=X, y=y, rng=rng
            )
            self.ensemble_configs_ = ensemble_configs  # Store for prompt tuning reuse
            remove_non_differentiable_preprocessing_from_models(models=self.models_)
        else:
            _, rng = infer_random_state(self.random_state)
            _, _, byte_size = determine_precision(
                self.inference_precision, self.devices_
            )
            ensemble_configs = self.ensemble_configs_  # Reuse from first fit

        self.ensemble_preprocessor_ = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            rng=rng,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
        )

        self.executor_ = InferenceEngineCachePreprocessing(
            X_train=X,
            y_train=y,
            feature_schema=self.inferred_feature_schema_,
            models=self.models_,
            ensemble_preprocessor=self.ensemble_preprocessor_,
            devices=self.devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=self.forced_inference_dtype_,
            save_peak_mem=self.memory_saving_mode,
            inference_mode=False,
        )

        return self

    def _maybe_calibrate_temperature_and_tune_decision_thresholds(
        self,
        X: XType,
        y: YType,
    ) -> None:
        """If this class was initialized with a 'tuning_config', calibrate and tune.

        This first computes scores on validation holdout data and then calibrates the
        softmax temperature and tunes the decision thresholds as per the tuning
        configuration. Results are stored in the 'tuned_classification_thresholds_' and
        'softmax_temperature_' attributes.
        """
        assert self.eval_metric_ is not None

        # Always set this to stay compatible with sklearn interface.
        self.tuned_classification_thresholds_ = None
        self.softmax_temperature_ = self.softmax_temperature

        tuning_config_resolved = resolve_tuning_config(
            tuning_config=self.tuning_config,
            num_samples=X.shape[0],
        )
        if tuning_config_resolved is None:
            if self.eval_metric_ is ClassifierEvalMetrics.F1:
                warnings.warn(
                    f"You specified '{self.eval_metric_}' as the eval metric but "
                    "haven't specified any tuning configuration. Consider configuring "
                    "tuning via the `tuning_config` argument of the TabPFNClassifier "
                    "to improve predictive performance.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.eval_metric_ is ClassifierEvalMetrics.BALANCED_ACCURACY:
                warnings.warn(
                    f"You specified '{self.eval_metric_}' as the eval metric but "
                    "haven't specified any tuning configuration. "
                    f"For metric '{self.eval_metric_}' we recommend "
                    "balancing the probabilities by class counts which can be achieved "
                    "by setting `balance_probabilities` to True.",
                    UserWarning,
                    stacklevel=2,
                )
            return

        if self.eval_metric_ is ClassifierEvalMetrics.ROC_AUC:
            warnings.warn(
                f"You specified '{self.eval_metric_}' as the eval metric with "
                "threshold tuning or temperature calibration enabled. "
                "ROC AUC is independent of these tunings and they will not "
                "improve this metric. Consider disabling them.",
                UserWarning,
                stacklevel=2,
            )

        holdout_raw_logits, holdout_y_true = self._compute_holdout_validation_data(
            X=X,
            y=y,
            holdout_frac=float(tuning_config_resolved.tuning_holdout_frac),
            n_folds=int(tuning_config_resolved.tuning_n_folds),
        )

        # WARNING: ensure the calibration happens before threshold tuning!
        if tuning_config_resolved.calibrate_temperature:
            calibrated_softmax_temperature = self._get_calibrated_softmax_temperature(
                holdout_raw_logits=holdout_raw_logits,
                holdout_y_true=holdout_y_true,
            )
            self.softmax_temperature_ = calibrated_softmax_temperature

        if tuning_config_resolved.tune_decision_thresholds:
            holdout_probas = (
                self.logits_to_probabilities(holdout_raw_logits)
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            tuned_classification_thresholds = find_optimal_classification_thresholds(
                metric_name=self.eval_metric_,
                y_true=holdout_y_true,
                y_pred_probas=holdout_probas,
                n_classes=self.n_classes_,
            )
            self.tuned_classification_thresholds_ = tuned_classification_thresholds

    def _compute_holdout_validation_data(
        self,
        X: XType,
        y: YType,
        holdout_frac: float,
        n_folds: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute holdout validation data.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - holdout_raw_logits: Array of holdout raw logits
                    (shape `[n_estimators, n_holdout_samples, n_classes]`).
                - holdout_y_true: Array of holdout y true labels
                    (shape `[n_holdout_samples]`).
        """
        splits = get_tuning_splits(
            X=copy.deepcopy(X),
            y=copy.deepcopy(y),
            holdout_frac=holdout_frac,
            random_state=self.random_state,
            n_splits=n_folds,
        )

        holdout_raw_logits = []
        holdout_y_true = []
        # suffixes: Nt=num_train_samples, F=num_features, Nh=num_holdout_samples
        for X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh in splits:
            holdout_y_true.append(y_holdout_Nh)
            calibration_classifier = self._get_tuning_classifier()
            with warnings.catch_warnings():
                # Filter expected warnings during tuning
                warnings.filterwarnings(
                    "ignore",
                    message=".*haven't specified any tuning configuration*",
                    category=UserWarning,
                )
                calibration_classifier.fit(X_train_NtF, y_train_Nt)

            # E=num estimators, Nh=num holdout samples, C=num classes
            raw_logits_ENhC = calibration_classifier.predict_raw_logits(X=X_holdout_NhF)
            holdout_raw_logits.append(raw_logits_ENhC)

        holdout_raw_logits_all = np.concatenate(holdout_raw_logits, axis=1)
        holdout_y_true__all = np.concatenate(holdout_y_true, axis=0)
        return holdout_raw_logits_all, holdout_y_true__all

    def _raw_predict(
        self,
        X: XType,
        *,
        return_logits: bool,
        return_raw_logits: bool = False,
    ) -> torch.Tensor:
        """Internal method to run prediction.

        Handles input validation, preprocessing, and the forward pass.
        Returns the raw torch.Tensor output (either logits or probabilities)
        before final detachment and conversion to NumPy.

        Args:
            X: The input data for prediction.
            return_logits: If True, the logits are returned. Otherwise,
                           probabilities are returned after softmax and other
                           post-processing steps.
            return_raw_logits: If True, returns the raw logits without
                averaging estimators or temperature scaling.

        Returns:
            The raw torch.Tensor output, either logits or probabilities,
            depending on `return_logits` and `return_raw_logits`.
        """
        check_is_fitted(self)

        if not self.differentiable_input:
            X = ensure_compatible_predict_input_sklearn(X, self)
            X = fix_dtypes(
                X,
                cat_indices=self.inferred_feature_schema_.indices_for(
                    FeatureModality.CATEGORICAL
                ),
            )
            X = process_text_na_dataframe(
                X=X,
                ord_encoder=getattr(self, "ordinal_encoder_", None),
            )

        with handle_oom_errors(self.devices_, X, model_type="classifier"):
            return self.forward(
                X,
                use_inference_mode=True,
                return_logits=return_logits,
                return_raw_logits=return_raw_logits,
            )

    @track_model_call(model_method="predict", param_names=["X"])
    def predict(self, X: XType) -> np.ndarray:
        """Predict the class labels for the provided input samples.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted class labels as a NumPy array.
        """
        probas = self._predict_proba(X=X)
        y_pred = np.argmax(probas, axis=1)
        if hasattr(self, "label_encoder_") and self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(y_pred)

        return y_pred

    @config_context(transform_output="default")
    @track_model_call(model_method="predict", param_names=["X"])
    def predict_logits(self, X: XType) -> np.ndarray:
        """Predict the raw logits for the provided input samples.

        Logits represent the unnormalized log-probabilities of the classes
        before the softmax activation function is applied.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted logits as a NumPy array. Shape (n_samples, n_classes).
        """
        logits_tensor = self._raw_predict(X, return_logits=True)
        return logits_tensor.float().detach().cpu().numpy()

    @config_context(transform_output="default")
    @track_model_call(model_method="predict", param_names=["X"])
    def predict_raw_logits(self, X: XType) -> np.ndarray:
        """Predict the raw logits for the provided input samples.

        Logits represent the unnormalized log-probabilities of the classes
        before the softmax activation function is applied. In contrast to the
        `predict_logits` method, this method returns the raw logits for each
        estimator, without averaging estimators or temperature scaling.

        Args:
            X: The input data for prediction.

        Returns:
            An array of predicted logits for each estimator,
            Shape (n_estimators, n_samples, n_classes).
        """
        logits_tensor = self._raw_predict(
            X,
            return_logits=False,
            return_raw_logits=True,
        )
        return logits_tensor.float().detach().cpu().numpy()

    @track_model_call(model_method="predict", param_names=["X"])
    def predict_proba(self, X: XType) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input samples.

        This is a wrapper around the `_predict_proba` method.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted probabilities of the classes as a NumPy array.
            Shape (n_samples, n_classes).
        """
        return self._predict_proba(X)

    @config_context(transform_output="default")  # type: ignore
    def _predict_proba(self, X: XType) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input samples.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted probabilities of the classes as a NumPy array.
            Shape (n_samples, n_classes).
        """
        probas = (
            self._raw_predict(X, return_logits=False).float().detach().cpu().numpy()
        )
        probas = self._maybe_reweight_probas(probas=probas)
        if self.inference_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
            probas = np.around(probas, decimals=SKLEARN_16_DECIMAL_PRECISION)
            probas = np.where(probas < PROBABILITY_EPSILON_ROUND_ZERO, 0.0, probas)

        # Ensure probabilities sum to 1 in case of minor floating point inaccuracies
        # going from torch to numpy
        return probas / probas.sum(axis=1, keepdims=True)  # type: ignore

    def _get_calibrated_softmax_temperature(
        self,
        holdout_raw_logits: np.ndarray,
        holdout_y_true: np.ndarray,
    ) -> float:
        """Calibrate temperature based on the holdout logits and true labels."""

        def logits_to_probabilities_fn(
            raw_logits: np.ndarray | torch.Tensor,
            softmax_temperature: float,
        ) -> np.ndarray:
            return (
                self.logits_to_probabilities(
                    raw_logits=raw_logits,
                    softmax_temperature=softmax_temperature,
                    average_before_softmax=self.average_before_softmax,
                    balance_probabilities=self.balance_probabilities,
                )
                .float()
                .detach()
                .cpu()
                .numpy()
            )

        return find_optimal_temperature(
            raw_logits=holdout_raw_logits,
            y_true=holdout_y_true,
            logits_to_probabilities_fn=logits_to_probabilities_fn,
            current_default_temperature=self.softmax_temperature_,
        )

    def _maybe_reweight_probas(self, probas: np.ndarray) -> np.ndarray:
        """Reweights the probabilities if a target_metric is specified.

        If a target metric is specified, the probabilities are reweighted based on
        the true holdout sets labels and predicted logits. This is done to tune the
        threshold for classification to the specified target metric.

        Args:
            probas: The predicted probabilities of the classes as a NumPy array.
                Shape (n_samples, n_classes).

        Returns:
            The input probas if no tuning is done, otherwise the reweighted
            probabilities.
        """
        if getattr(self, "tuned_classification_thresholds_", None) is None:
            return probas

        probas = probas / np.maximum(self.tuned_classification_thresholds_, 1e-8)
        return probas / probas.sum(axis=1, keepdims=True)

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by the softmax temperature."""
        temp = getattr(self, "softmax_temperature_", self.softmax_temperature)
        if temp != 1.0:
            return logits / temp
        return logits

    def _average_across_estimators(self, tensors: torch.Tensor) -> torch.Tensor:
        """Averages a tensor across the estimator dimension (dim=0)."""
        return tensors.mean(dim=0)

    def _apply_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies the softmax function to the last dimension."""
        return torch.nn.functional.softmax(logits, dim=-1)

    def _apply_balancing(self, probas: torch.Tensor) -> torch.Tensor:
        """Applies class balancing to a probability tensor."""
        counts = getattr(self, "class_counts_", None)
        if counts is None:
            return probas
        return balance_probas_by_class_counts(probas, counts)

    def logits_to_probabilities(
        self,
        raw_logits: np.ndarray | torch.Tensor,
        *,
        softmax_temperature: float | None = None,
        average_before_softmax: bool | None = None,
        balance_probabilities: bool | None = None,
    ) -> torch.Tensor:
        """Convert logits to probabilities using the classifier's post-processing.

        Args:
            raw_logits: Logits with shape (n_estimators, n_samples, n_classes) or
                (n_samples, n_classes). If the logits have three dimensions, they are
                averaged across the estimator dimension (dim=0).
            softmax_temperature: Optional override for temperature scaling.
            average_before_softmax: Optional override for averaging order.
            balance_probabilities: Optional override for probability balancing.

        Returns:
            Probabilities with shape (n_samples, n_classes).
        """
        raw_logits = (
            raw_logits
            if isinstance(raw_logits, torch.Tensor)
            else torch.from_numpy(np.asarray(raw_logits))
        )
        used_temperature = (
            softmax_temperature
            if softmax_temperature is not None
            else getattr(self, "softmax_temperature_", self.softmax_temperature)
        )
        use_average_before_softmax = (
            self.average_before_softmax
            if average_before_softmax is None
            else average_before_softmax
        )
        use_balance = (
            self.balance_probabilities
            if balance_probabilities is None
            else balance_probabilities
        )

        steps: list[Callable[[torch.Tensor], torch.Tensor]] = []

        if used_temperature != 1.0:

            def apply_temp(t: torch.Tensor) -> torch.Tensor:
                return t / used_temperature

            steps.append(apply_temp)

        if raw_logits.ndim >= 3:
            if use_average_before_softmax:
                steps.append(self._average_across_estimators)
                steps.append(self._apply_softmax)
            else:
                steps.append(self._apply_softmax)
                steps.append(self._average_across_estimators)
        elif raw_logits.ndim == 2:
            steps.append(self._apply_softmax)
        else:
            raise ValueError(
                f"Expected logits with 2 or more dims, got {raw_logits.ndim}"
            )

        if use_balance:
            steps.append(self._apply_balancing)

        output = raw_logits
        for fn in steps:
            output = fn(output)

        return output

    def forward(  # noqa: C901, PLR0912
        self,
        X: list[torch.Tensor] | torch.Tensor,
        *,
        use_inference_mode: bool = False,
        return_logits: bool = False,
        return_raw_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass returning predicted probabilities or logits
        for TabPFNClassifier Inference Engine. Used in
        Fine-Tuning and prediction. Called directly
        in FineTuning training loop or by predict() function
        with the use_inference_mode flag explicitly set to True.

        Iterates over outputs of InferenceEngine.

        Args:
            X: list[torch.Tensor] in fine-tuning, XType in normal predictions.
            use_inference_mode: Flag for inference mode., default at False since
            it is called within predict. During FineTuning forward() is called
            directly by user, so default should be False here.
            return_logits: If True, returns logits averaged across estimators.
                Otherwise, probabilities are returned.
            return_raw_logits: If True, returns the raw logits, without
                averaging estimators or temperature scaling.

        Returns:
            The predicted probabilities or logits of the classes as a torch.Tensor.
            - If `use_inference_mode` is True: Shape (N_samples, N_classes)
            - If `use_inference_mode` is False (e.g., for training/fine-tuning):
              Shape (Batch_size, N_classes, N_samples), suitable for NLLLoss.
            - If `return_raw_logits` is True: Shape (n_estimators, n_samples, n_classes)
        """
        if return_logits and return_raw_logits:
            raise ValueError(
                "Cannot return both logits and raw logits. Please specify only one."
            )

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
            f"Invalid forward pass: Bad combination of inference mode "
            f"({use_inference_mode=}), input X, "
            f"or executor type ({type(self.executor_)}). Ensure call is from standard "
            f"predict ({is_standard_inference=}) or a batched fine-tuning context."
            f"({is_batched_for_grads=})."
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

        if self.fit_mode in ["fit_preprocessors", "batched"]:
            # Don't enable inference mode when differentiable_input=True (prompt tuning)
            # to allow gradients to flow through
            actual_inference_mode = use_inference_mode and not self.differentiable_input
            self.executor_.use_torch_inference_mode(use_inference=actual_inference_mode)

        outputs = []
        for output, config in self.executor_.iter_outputs(
            X, autocast=self.use_autocast_
        ):
            original_ndim = output.ndim

            # This block correctly handles both single configs and lists of configs
            if original_ndim == 2:
                # Shape is [Nsamples, NClasses] -> [Nsamples, 1,  NClasses]
                processed_output = output.unsqueeze(1)
                config_list = [config]
            elif original_ndim == 3:
                # Shape is [Nsamples, batch_size, NClasses]
                processed_output = output
                config_list = config
            else:
                raise ValueError(
                    f"Output tensor must be 2d or 3d, got {original_ndim}d"
                )

            # Process the config_list (which is now guaranteed to be a list)
            output_batch = []
            for i, batch_config in enumerate(config_list):
                assert isinstance(batch_config, ClassifierEnsembleConfig)
                # If class_permutation is None - class shifting is disabled
                # So we slice to self.n_classes_ to ensure the output tensor matches
                # the expected number of classes
                if batch_config.class_permutation is None:
                    output_batch.append(processed_output[:, i, : self.n_classes_])
                else:
                    # make sure the processed_output num_classes are the same.
                    if len(batch_config.class_permutation) != self.n_classes_:
                        use_perm = np.arange(self.n_classes_)
                        use_perm[: len(batch_config.class_permutation)] = (
                            batch_config.class_permutation
                        )
                    else:
                        use_perm = batch_config.class_permutation

                    output_batch.append(processed_output[:, i, use_perm])

            outputs.append(torch.stack(output_batch, dim=1))

        # --- Post-processing ---
        stacked_outputs = torch.stack(outputs)

        if return_logits:
            temp_scaled = self._apply_temperature(stacked_outputs)
            output = self._average_across_estimators(temp_scaled)
        elif return_raw_logits:
            output = stacked_outputs
        else:
            output = self.logits_to_probabilities(stacked_outputs)

        # --- Final output shaping ---
        if output.ndim > 2 and use_inference_mode:
            output = output.squeeze(1) if not return_raw_logits else output.squeeze(2)

        if not use_inference_mode:
            # This case is primarily for fine-tuning where NLLLoss expects [B, C, N]
            if output.ndim == 2:  # was likely [N, C]
                output = output.unsqueeze(0)  # [1, N, C]
            output = output.transpose(0, 1).transpose(1, 2)

        return output

    def get_embeddings(
        self,
        X: XType,
        data_source: Literal["train", "test"] = "test",
    ) -> np.ndarray:
        """Get embeddings for the input data ``X``.

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
        """Save a fitted classifier, light wrapper around save_fitted_tabpfn_model."""
        save_fitted_tabpfn_model(self, path)

    @classmethod
    def load_from_fit_state(
        cls, path: Path | str, *, device: str | torch.device = "cpu"
    ) -> TabPFNClassifier:
        """Restore a fitted clf, light wrapper around load_fitted_tabpfn_model."""
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


def _validate_eval_metric(
    eval_metric: str | ClassifierEvalMetrics | None,
) -> ClassifierEvalMetrics:
    if eval_metric is None:
        return DEFAULT_CLASSIFICATION_EVAL_METRIC
    if isinstance(eval_metric, ClassifierEvalMetrics):
        return eval_metric
    try:
        return ClassifierEvalMetrics(eval_metric)  # Convert string to Enum
    except ValueError as err:
        valid_values = [e.value for e in ClassifierEvalMetrics]
        raise ValueError(
            f"Invalid eval_metric: `{eval_metric}`. Must be one of {valid_values}"
        ) from err
