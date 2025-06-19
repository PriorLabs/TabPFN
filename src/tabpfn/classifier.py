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

import typing
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import Self

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.preprocessing import LabelEncoder

from tabpfn.base import (
    _initialize_model_variables_helper,
    check_cpu_warning,
    create_inference_engine,
    determine_precision,
    get_preprocessed_datasets_helper,
)
from tabpfn.constants import (
    PROBABILITY_EPSILON_ROUND_ZERO,
    SKLEARN_16_DECIMAL_PRECISION,
    XType,
    YType,
)
from tabpfn.inference import InferenceEngineBatchedNoPreprocessing
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    DatasetCollectionWithPreprocessing,
    EnsembleConfig,
    PreprocessorConfig,
    default_classifier_preprocessor_configs,
)
from tabpfn.utils import (
    _fix_dtypes,
    _get_embeddings,
    _get_ordinal_encoder,
    _process_text_na_dataframe,
    infer_categorical_features,
    infer_random_state,
    validate_X_predict,
    validate_Xy_fit,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.compose import ColumnTransformer
    from torch.types import _dtype

    from tabpfn.config import ModelInterfaceConfig
    from tabpfn.inference import InferenceEngine
    from tabpfn.model.config import ModelConfig
    from tabpfn.preprocessing import (
        ClassifierEnsembleConfig,
        DatasetCollectionWithPreprocessing,
    )

    try:
        from sklearn.base import Tags
    except ImportError:
        Tags = Any


class TabPFNClassifier(ClassifierMixin, BaseEstimator):
    """TabPFNClassifier class."""

    config_: ModelConfig
    """The configuration of the loaded model to be used for inference."""

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

    label_encoder_: LabelEncoder
    """The label encoder used to encode the target variable."""

    preprocessor_: ColumnTransformer
    """The column transformer used to preprocess the input data to be numeric."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int = 4,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        model_path: str | Path | Literal["auto"] = "auto",
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
        """A TabPFN interface for classification.

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
                a post-processing step. Set `softmax_temperature=1.0` for no effect.

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

                - If `"auto"`, the model will be downloaded upon first use. This
                  defaults to your system cache directory, but can be overwritten
                  with the use of an environment variable `TABPFN_MODEL_CACHE_DIR`.
                - If a path or a string of a path, the model will be loaded from
                  the user-specified location if available, otherwise it will be
                  downloaded to this location.

            device:
                The device to use for inference with TabPFN. If `"auto"`, the device
                is `"cuda"` if available, otherwise `"cpu"`.

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
                technically the
                forward pass of the model. The options are:

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
                  specifying the maximum free available memory.

                !!! warning
                    This does not batch the original input data. We still recommend to
                    batch this as necessary if you run into memory errors! For example,
                    if the entire input data does not fit into memory, even the memory
                    save mode will not prevent memory errors.

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
                If true, the preprocessing will be adapted to be end-to-end
                differentiable with PyTorch.
                This is useful for explainability and prompt-tuning, essential
                in the prompttuning code.
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
        self.fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = (
            fit_mode
        )
        self.memory_saving_mode: bool | Literal["auto"] | float | int = (
            memory_saving_mode
        )
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inference_config = inference_config
        self.differentiable_input = differentiable_input

    # TODO: We can remove this from scikit-learn lower bound of 1.6
    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
            "multilabel": False,
        }

    def __sklearn_tags__(self) -> Tags:  # type: ignore
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "classifier"
        return tags

    def get_preprocessed_datasets(
        self,
        X_raw: XType | list[XType],
        y_raw: YType | list[YType],
        split_fn,
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
            max_data_size: Maximum allowed number of samples in one dataset.
            If None, datasets are not splitted.
        """
        return get_preprocessed_datasets_helper(
            self,
            X_raw,
            y_raw,
            split_fn,
            max_data_size,
            model_type="classifier",
        )

    def _initialize_model_variables(self) -> tuple[int, np.random.Generator]:
        """Perform initialization of the model, return determined byte_size
        and RNG object.
        """
        return _initialize_model_variables_helper(self, "classifier")

    def _initialize_dataset_preprocessing(
        self, X: XType, y: YType, rng
    ) -> tuple[list[ClassifierEnsembleConfig], XType, YType]:
        """Internal preprocessing method for input arguments.
        Returns ClassifierEnsembleConfigs, inferred categorical indices,
        and modelfied features X and labels y.
        Sets self.inferred_categorical_indices_.
        """
        X, y, feature_names_in, n_features_in = validate_Xy_fit(
            X,
            y,
            estimator=self,
            ensure_y_numeric=False,
            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )

        check_cpu_warning(
            self.device, X, allow_cpu_override=self.ignore_pretraining_limits
        )

        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in

        # Ensure that the y values are ordinally encoded
        # TODO(eddiebergman): Ensure the counts here line up with
        #   the actual classes after label encoder.
        if not self.differentiable_input:
            _, counts = np.unique(y, return_counts=True)
            self.class_counts_ = counts
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_  # type: ignore
            self.n_classes_ = len(self.classes_)
        else:
            # if pt_diffable, it is a convention that the class
            # labels are [0, ..., n-1].
            self.label_encoder_ = None
            if not hasattr(self, "n_classes_"):
                self.n_classes_ = int(torch.max(y).item()) + 1
            self.classes_ = torch.arange(self.n_classes_)

        # TODO: Support more classes with a fallback strategy.
        if self.n_classes_ > self.interface_config_.MAX_NUMBER_OF_CLASSES:
            raise ValueError(
                f"Number of classes {self.n_classes_} exceeds the maximal number of "
                "classes supported by TabPFN. Consider using a strategy to reduce "
                "the number of classes. For code see "
                "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/"
                "tabpfn_extensions/many_class/many_class_classifier.py",
            )

        # Will convert specified categorical indices to category dtype, as well
        # as handle `np.object` arrays or otherwise `object` dtype pandas columns.

        if not self.differentiable_input:
            self.inferred_categorical_indices_ = infer_categorical_features(
                X=X,
                provided=self.categorical_features_indices,
                min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
                max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
                min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
            )
            preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS

            # Will convert inferred categorical indices to category dtype,
            # to be picked up by the ord_encoder, as well
            # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
            X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
            # Ensure categories are ordinally encoded
            ord_encoder = _get_ordinal_encoder()
            X = _process_text_na_dataframe(X, ord_encoder=ord_encoder, fit_encoder=True)

            assert isinstance(X, np.ndarray)
            self.preprocessor_ = ord_encoder

        else:  # Minimal preprocessing for prompt tuning
            self.inferred_categorical_indices_ = []
            self.preprocessor_ = None
            preprocess_transforms = [PreprocessorConfig("none", differentiable=True)]

        ensemble_configs = EnsembleConfig.generate_for_classification(
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
                else default_classifier_preprocessor_configs(),
            ),
            class_shift_method=self.interface_config_.CLASS_SHIFT_METHOD
            if not self.differentiable_input
            else None,
            n_classes=self.n_classes_,
            random_state=rng,
        )
        assert len(ensemble_configs) == self.n_estimators
        return ensemble_configs, X, y

    def fit_from_preprocessed(
        self,
        X_preprocessed: list[torch.Tensor],
        y_preprocessed: list[torch.Tensor],
        cat_ix: list[list[int]],
        configs: list[list[EnsembleConfig]],
        *,
        no_refit=True,
    ) -> TabPFNClassifier:
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
        # If there isa model, and we are lazy, we skip reinitialization
        if not hasattr(self, "model_") or not no_refit:
            byte_size, rng = self._initialize_model_variables()
        else:
            _, _, byte_size = determine_precision(
                self.inference_precision, self.device_
            )
            rng = None

        if not self.fit_mode == "batched":
            raise ValueError(
                "The fit_from_preprocessed function"
                " is only supported in the batched fit_mode."
                " Since in other fit_modes the preprocessing"
                " is done as part of the inference engine"
            )

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
            inference_mode=not self.differentiable_input,
        )

        return self

    @config_context(transform_output="default")  # type: ignore
    def fit(self, X: XType, y: YType) -> Self:
        """Fit the model.

        Args:
            X: The input data.
            y: The target variable.
        """
        if not hasattr(self, "model_") or not self.differentiable_input:
            byte_size, rng = self._initialize_model_variables()
            ensemble_configs, X, y = self._initialize_dataset_preprocessing(X, y, rng)
        else:  # already fitted and prompt_tuning mode: no cat. features
            _, rng = infer_random_state(self.random_state)
            _, _, byte_size = determine_precision(
                self.inference_precision, self.device_
            )

        if self.fit_mode == "batched":
            raise ValueError(
                "The fit() function is currently not supported in the batched fit_mode."
            )

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
            inference_mode=not self.differentiable_input,
        )

        return self

    def predict(self, X: XType) -> np.ndarray:
        """Predict the class labels for the provided input samples.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)
        if self.label_encoder_:
            return self.label_encoder_.inverse_transform(y)  # type: ignore

        return y

    @config_context(transform_output="default")  # type: ignore
    def predict_proba(self, X: XType) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input samples.

        Args:
            X: The input data.

        Returns:
            The predicted probabilities of the classes.
        """
        check_is_fitted(self)

        if not self.differentiable_input:
            X = validate_X_predict(X, self)
            X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
            X = _process_text_na_dataframe(X, ord_encoder=self.preprocessor_)

        output = self.forward(X, use_inference_mode=True)
        output = output.float().detach().cpu().numpy()

        if self.interface_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
            output = np.around(output, decimals=SKLEARN_16_DECIMAL_PRECISION)
            output = np.where(output < PROBABILITY_EPSILON_ROUND_ZERO, 0.0, output)

        # Normalize to guarantee proba sum to 1, required due to precision issues and
        # going from torch to numpy
        return output / output.sum(axis=1, keepdims=True)  # type: ignore

    # TODO: reduce complexity to remove noqa C901, PLR0912
    def forward(  # noqa: C901, PLR0912
        self,
        X: list[torch.Tensor] | torch.Tensor,
        *,
        use_inference_mode: bool = False,
    ) -> torch.Tensor:
        """Forward pass returning predicted probabilities
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

        Returns:
            The predicted probabilities of the classes.
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

        if self.fit_mode in ["fit_preprocessors", "batched"]:
            # only these two modes support this option
            self.executor_.use_torch_inference_mode(use_inference=use_inference_mode)

        outputs = []
        for output, config in self.executor_.iter_outputs(
            X,
            device=self.device_,
            autocast=self.use_autocast_,
        ):
            original_ndim = output.ndim

            if original_ndim == 2:
                # Shape is [Nsamples, NClasses] -> [Nsamples, 1,  NClasses]
                processed_output = output.unsqueeze(1)
                config_list = [config]
            elif original_ndim == 3:
                # Shape is [Nsamples, batch_size, NClasses] noqa ERA001
                processed_output = output
                config_list = config
            else:
                raise ValueError(
                    f"Output tensor must be 2d or 3d, got {original_ndim}d"
                )

            num_target_classes = processed_output.shape[-1]

            if self.softmax_temperature != 1:
                processed_output = (
                    processed_output[:, :, :num_target_classes].float()
                    / self.softmax_temperature
                )

            if config_list is not None:
                output_batch = []
                for i, batch_config in enumerate(config_list):
                    # make sure the processed_output num_classes are the same.
                    if len(batch_config.class_permutation) != self.n_classes_:
                        use_perm = np.arange(self.n_classes_)
                        use_perm[: len(batch_config.class_permutation)] = (
                            batch_config.class_permutation
                        )
                    else:
                        use_perm = batch_config.class_permutation
                    output_batch.append(processed_output[:, i, use_perm])

                output_all = torch.stack(output_batch, dim=1)

            outputs.append(output_all)

        if self.average_before_softmax:
            output = torch.stack(outputs).mean(dim=0)
            output = torch.nn.functional.softmax(output, dim=-1)
        else:
            outputs = [torch.nn.functional.softmax(o, dim=-1) for o in outputs]
            output = torch.stack(outputs).mean(dim=0)

        if self.balance_probabilities:
            class_prob_in_train = self.class_counts_ / self.class_counts_.sum()
            output = output / torch.Tensor(class_prob_in_train).to(self.device_)
            output = output / output.sum(dim=-1, keepdim=True)

        if use_inference_mode:
            output = output.squeeze(1)  # [N, B, C] -> [N, C]
        else:
            output = output.transpose(0, 1).transpose(1, 2)  # for NLLLoss [B, C, N]

        return output

    def get_embeddings(
        self,
        X: XType,
        data_source: Literal["train", "test"] = "test",
    ) -> np.ndarray:
        """Get the embeddings for the input data `X`.

        Parameters:
            X (XType): The input data.
            data_source str: Extract either the train or test embeddings
        Returns:
            np.ndarray: The computed embeddings for each fitted estimator.
        """
        return _get_embeddings(self, X, data_source)
