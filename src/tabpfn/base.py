"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, Union

import torch

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution

# --- TabPFN imports ---
from tabpfn.constants import (
    AUTOCAST_DTYPE_BYTE_SIZE,
    DEFAULT_DTYPE_BYTE_SIZE,
    ModelPath,
    XType,
    YType,
)
from tabpfn.inference import (
    InferenceEngine,
    InferenceEngineBatchedNoPreprocessing,
    InferenceEngineCacheKV,
    InferenceEngineCachePreprocessing,
    InferenceEngineOnDemand,
)
from tabpfn.model_loading import load_model_criterion_config
from tabpfn.preprocessing import (
    BaseDatasetConfig,
    ClassifierDatasetConfig,
    DatasetCollectionWithPreprocessing,
    RegressorDatasetConfig,
)
from tabpfn.settings import settings
from tabpfn.utils import (
    infer_devices,
    infer_fp16_inference_mode,
    infer_random_state,
    split_large_data,
    update_encoder_params,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
    from tabpfn.classifier import TabPFNClassifier
    from tabpfn.inference_config import InferenceConfig
    from tabpfn.regressor import TabPFNRegressor


class BaseModelSpecs:
    """Base class for model specifications."""

    def __init__(
        self,
        model: Architecture,
        arch_config: ArchitectureConfig,
        inference_config: InferenceConfig,
    ):
        self.model = model
        self.architecture_config = arch_config
        self.inference_config = inference_config


class ClassifierModelSpecs(BaseModelSpecs):
    """Model specs for classifiers."""

    norm_criterion = None


class RegressorModelSpecs(BaseModelSpecs):
    """Model specs for regressors."""

    def __init__(
        self,
        model: Architecture,
        arch_config: ArchitectureConfig,
        inference_config: InferenceConfig,
        norm_criterion: FullSupportBarDistribution,
    ):
        super().__init__(model, arch_config, inference_config)
        self.norm_criterion = norm_criterion


ModelSpecs = Union[RegressorModelSpecs, ClassifierModelSpecs]


def initialize_tabpfn_model(
    model_path: ModelPath
    | list[ModelPath]
    | RegressorModelSpecs
    | ClassifierModelSpecs
    | list[RegressorModelSpecs]
    | list[ClassifierModelSpecs],
    which: Literal["classifier", "regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
) -> tuple[
    list[Architecture],
    list[ArchitectureConfig],
    FullSupportBarDistribution | None,
    InferenceConfig,
]:
    """Initializes a TabPFN model based on the provided configuration.

    Args:
        model_path: Path or directive ("auto") to load the pre-trained model from.
            If a list of paths is provided, the models are applied across different
            estimators. If a RegressorModelSpecs or ClassifierModelSpecs object is
            provided, the model is loaded from the object.

        which: Which TabPFN model to load.
        fit_mode: Determines caching behavior.

    Returns:
        a list of models,
        a list of architecture configs (associated with each model),
        if regression, the bar distribution, otherwise None,
        the inference config
    """
    if isinstance(model_path, RegressorModelSpecs) and which == "regressor":
        return (
            [model_path.model],
            [model_path.architecture_config],
            model_path.norm_criterion,
            model_path.inference_config,
        )

    if isinstance(model_path, ClassifierModelSpecs) and which == "classifier":
        return (
            [model_path.model],
            [model_path.architecture_config],
            None,
            model_path.inference_config,
        )

    if (
        isinstance(model_path, list)
        and len(model_path) > 0
        and all(isinstance(spec, RegressorModelSpecs) for spec in model_path)
    ):
        _assert_inference_configs_equal(model_path)
        return (  # pyright: ignore[reportReturnType]
            [spec.model for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
            [spec.architecture_config for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
            model_path[0].norm_criterion,  # pyright: ignore[reportAttributeAccessIssue]
            model_path[0].inference_config,
        )

    if (
        isinstance(model_path, list)
        and len(model_path) > 0
        and all(isinstance(spec, ClassifierModelSpecs) for spec in model_path)
    ):
        _assert_inference_configs_equal(model_path)
        return (
            [spec.model for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
            [spec.architecture_config for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
            None,
            model_path[0].inference_config,
        )

    if (
        model_path is None
        or model_path == "auto"
        or isinstance(model_path, (str, pathlib.Path, list))  # pyright: ignore[reportArgumentType]
    ):
        if isinstance(model_path, str) and model_path == "auto":
            model_path = None  # type: ignore

        if isinstance(model_path, list) and len(model_path) == 0:
            raise ValueError(
                "You provided a list of model paths with no entries. "
                "Please provide a valid `model_path` argument, or use 'auto' to use "
                "the default model."
            )

        download_if_not_exists = True

        if which == "classifier":
            models, _, arch_configs, inference_config = load_model_criterion_config(
                model_path=model_path,  # pyright: ignore[reportArgumentType]
                # The classifier's bar distribution is not used
                check_bar_distribution_criterion=False,
                cache_trainset_representation=(fit_mode == "fit_with_cache"),
                which="classifier",
                version="v2",
                download_if_not_exists=download_if_not_exists,
            )
            norm_criterion = None
        else:
            models, bardist, arch_configs, inference_config = (
                load_model_criterion_config(
                    model_path=model_path,  # pyright: ignore[reportArgumentType]
                    # The regressor's bar distribution is required
                    check_bar_distribution_criterion=True,
                    cache_trainset_representation=(fit_mode == "fit_with_cache"),
                    which="regressor",
                    version="v2",
                    download_if_not_exists=download_if_not_exists,
                )
            )
            norm_criterion = bardist

        return models, arch_configs, norm_criterion, inference_config

    raise TypeError(
        "Received ModelSpecs via 'model_path', but 'which' parameter is set to '"
        + which
        + "'. Expected 'classifier' or 'regressor'. and model_path"
        + "is of of type"
        + str(type(model_path))
    )


def _assert_inference_configs_equal(
    model_specs: list[ClassifierModelSpecs] | list[RegressorModelSpecs],
) -> None:
    if not all(
        spec.inference_config == model_specs[0].inference_config for spec in model_specs
    ):
        raise ValueError("All models must have the same inference config")


def determine_precision(
    inference_precision: torch.dtype | Literal["autocast", "auto"],
    devices_: Sequence[torch.device],
) -> tuple[bool, torch.dtype | None, int]:
    """Decide whether to use autocast or a forced precision dtype.

    Args:
        inference_precision:

            - If `"auto"`, decide automatically based on the device.
            - If `"autocast"`, explicitly use PyTorch autocast (mixed precision).
            - If a `torch.dtype`, force that precision.

        devices_: The devices which will be used for inference.

    Returns:
        use_autocast_:
            True if mixed-precision autocast will be used.
        forced_inference_dtype_:
            If not None, the forced precision dtype for the model.
        byte_size:
            The byte size per element for the chosen precision.
    """
    if inference_precision in ["autocast", "auto"]:
        use_autocast_ = infer_fp16_inference_mode(
            devices=devices_,
            enable=True if (inference_precision == "autocast") else None,
        )
        forced_inference_dtype_ = None
        byte_size = (
            AUTOCAST_DTYPE_BYTE_SIZE if use_autocast_ else DEFAULT_DTYPE_BYTE_SIZE
        )
    elif isinstance(inference_precision, torch.dtype):
        use_autocast_ = False
        forced_inference_dtype_ = inference_precision
        byte_size = inference_precision.itemsize
    else:
        raise ValueError(f"Unknown inference_precision={inference_precision}")

    return use_autocast_, forced_inference_dtype_, byte_size


def create_inference_engine(  # noqa: PLR0913
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: list[Architecture],
    ensemble_configs: Any,
    cat_ix: list[int],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    devices_: Sequence[torch.device],
    rng: np.random.Generator,
    n_preprocessing_jobs: int,
    byte_size: int,
    forced_inference_dtype_: torch.dtype | None,
    memory_saving_mode: bool | Literal["auto"] | float | int,
    use_autocast_: bool,
    inference_mode: bool = True,
) -> InferenceEngine:
    """Creates the appropriate TabPFN inference engine based on `fit_mode`.

    Each execution mode will perform slightly different operations based on the mode
    specified by the user. In the case where preprocessors will be fit after `prepare`,
    we will use them to further transform the associated borders with each ensemble
    config member.

    Args:
        X_train: Training features
        y_train: Training target
        models: The loaded TabPFN models.
        ensemble_configs: The ensemble configurations to create multiple "prompts".
        cat_ix: Indices of inferred categorical features.
        fit_mode: Determines how we prepare inference (pre-cache or not).
        devices_: The devices for inference.
        rng: Numpy random generator.
        n_preprocessing_jobs: Number of parallel CPU workers to use for the
            preprocessing.
        byte_size: Byte size for the chosen inference precision.
        forced_inference_dtype_: If not None, the forced dtype for inference.
        memory_saving_mode: GPU/CPU memory saving settings.
        use_autocast_: Whether we use torch.autocast for inference.
        inference_mode: Whether to use torch.inference_mode (set False if
            backprop is needed)
    """
    engine: (
        InferenceEngineOnDemand
        | InferenceEngineCachePreprocessing
        | InferenceEngineCacheKV
        | InferenceEngineBatchedNoPreprocessing
    )
    if fit_mode == "low_memory":
        engine = InferenceEngineOnDemand.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            ensemble_configs=ensemble_configs,
            rng=rng,
            models=models,
            n_preprocessing_jobs=n_preprocessing_jobs,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
        )
    elif fit_mode == "fit_preprocessors":
        engine = InferenceEngineCachePreprocessing.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            ensemble_configs=ensemble_configs,
            models=models,
            n_preprocessing_jobs=n_preprocessing_jobs,
            rng=rng,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            inference_mode=inference_mode,
        )
    elif fit_mode == "fit_with_cache":
        engine = InferenceEngineCacheKV.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            models=models,
            ensemble_configs=ensemble_configs,
            n_preprocessing_jobs=n_preprocessing_jobs,
            devices=devices_,
            dtype_byte_size=byte_size,
            rng=rng,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            autocast=use_autocast_,
        )
    elif fit_mode == "batched":
        engine = InferenceEngineBatchedNoPreprocessing.prepare(
            X_trains=X_train,
            y_trains=y_train,
            cat_ix=cat_ix,
            models=models,
            ensemble_configs=ensemble_configs,
            force_inference_dtype=forced_inference_dtype_,
            inference_mode=inference_mode,
            save_peak_mem=memory_saving_mode,
            dtype_byte_size=byte_size,
        )
    else:
        raise ValueError(f"Invalid fit_mode: {fit_mode}")

    return engine


def check_cpu_warning(
    devices: Sequence[torch.device],
    X: np.ndarray | torch.Tensor | pd.DataFrame,
    *,
    allow_cpu_override: bool = False,
) -> None:
    """Check if using CPU with large datasets and warn or error appropriately.

    Args:
        devices: The torch devices being used
        X: The input data (NumPy array, Pandas DataFrame, or Torch Tensor)
        allow_cpu_override: If True, allow CPU usage with large datasets.
    """
    allow_cpu_override = allow_cpu_override or settings.tabpfn.allow_cpu_large_dataset

    if allow_cpu_override:
        return

    # Determine number of samples
    try:
        num_samples = X.shape[0]
    except AttributeError:
        return

    if any(device.type == "cpu" for device in devices):
        if num_samples > 1000:
            raise RuntimeError(
                "Running on CPU with more than 1000 samples is not allowed "
                "by default due to slow performance.\n"
                "To override this behavior, set the environment variable "
                "TABPFN_ALLOW_CPU_LARGE_DATASET=1 or "
                "set ignore_pretraining_limits=True.\n"
                "Alternatively, consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client"
            )
        if num_samples > 200:
            warnings.warn(
                "Running on CPU with more than 200 samples may be slow.\n"
                "Consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client",
                stacklevel=2,
            )


def get_preprocessed_datasets_helper(
    calling_instance: Any,  # Union[TabPFNClassifier, TabPFNRegressor],
    X_raw: XType | list[XType],
    y_raw: YType | list[YType],
    split_fn: Callable,
    max_data_size: int | None,
    model_type: Literal["regressor", "classifier"],
    *,
    equal_split_size: bool,
) -> DatasetCollectionWithPreprocessing:
    """Helper function to create a DatasetCollectionWithPreprocessing.
    Relies on methods from the calling_instance for specific initializations.
    Modularises Code for both Regressor and Classifier.

    Args:
        calling_instance: The instance of the TabPFNRegressor or TabPFNClassifier.
        X_raw: individual or list of input dataset features
        y_raw: individual or list of input dataset labels
        split_fn: A function to dissect a dataset into train and test partition.
        max_data_size: Maximum allowed number of samples within one dataset.
        If None, datasets are not splitted.
        model_type: The type of the model.
        equal_split_size: If True, splits data into equally sized chunks under
            max_data_size.
            If False, splits into chunks of size `max_data_size`, with
            the last chunk having the remainder samples but is dropped if its
            size is less than 2.
    """
    if not isinstance(X_raw, list):
        X_raw = [X_raw]
    if not isinstance(y_raw, list):
        y_raw = [y_raw]
    assert len(X_raw) == len(y_raw), "X and y lists must have the same length."

    if not hasattr(calling_instance, "models_") or calling_instance.models_ is None:
        _, rng = calling_instance._initialize_model_variables()
    else:
        _static_seed, rng = infer_random_state(calling_instance.random_state)

    X_split, y_split = [], []
    for X_item, y_item in zip(X_raw, y_raw):
        if max_data_size is not None:
            Xparts, yparts = split_large_data(
                X_item, y_item, max_data_size, equal_split_size=equal_split_size
            )
        else:
            Xparts, yparts = [X_item], [y_item]
        X_split.extend(Xparts)
        y_split.extend(yparts)

    dataset_config_collection: list[BaseDatasetConfig] = []
    for X_item, y_item in zip(X_split, y_split):
        if model_type == "classifier":
            ensemble_configs, X_mod, y_mod = (
                calling_instance._initialize_dataset_preprocessing(X_item, y_item, rng)
            )
            current_cat_ix = calling_instance.inferred_categorical_indices_

            dataset_config = ClassifierDatasetConfig(
                config=ensemble_configs,
                X_raw=X_mod,
                y_raw=y_mod,
                cat_ix=current_cat_ix,
            )
        elif model_type == "regressor":
            ensemble_configs, X_mod, y_mod, bardist_ = (
                calling_instance._initialize_dataset_preprocessing(X_item, y_item, rng)
            )
            current_cat_ix = calling_instance.inferred_categorical_indices_
            dataset_config = RegressorDatasetConfig(
                config=ensemble_configs,
                X_raw=X_mod,
                y_raw=y_mod,
                cat_ix=current_cat_ix,
                znorm_space_bardist_=bardist_,
            )
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        dataset_config_collection.append(dataset_config)

    return DatasetCollectionWithPreprocessing(split_fn, rng, dataset_config_collection)


def initialize_model_variables_helper(
    calling_instance: TabPFNRegressor | TabPFNClassifier,
    model_type: Literal["regressor", "classifier"],
) -> tuple[int, np.random.Generator]:
    """Set attributes on the given model to prepare it for inference.

    This includes selecting the device and the inference precision.

    Returns:
        a tuple (byte_size, rng), where byte_size is the number of bytes in the selected
        dtype, and rng is a NumPy random Generator for use during inference.
    """
    static_seed, rng = infer_random_state(calling_instance.random_state)
    models, arch_configs, maybe_bardist, inference_config_ = initialize_tabpfn_model(
        model_path=calling_instance.model_path,  # pyright: ignore[reportArgumentType]
        which=model_type,
        fit_mode=calling_instance.fit_mode,  # pyright: ignore[reportArgumentType]
    )
    calling_instance.models_ = models
    calling_instance.configs_ = arch_configs
    if model_type == "regressor" and maybe_bardist is not None:
        calling_instance.znorm_space_bardist_ = maybe_bardist

    calling_instance.devices_ = infer_devices(calling_instance.device)
    (
        calling_instance.use_autocast_,
        calling_instance.forced_inference_dtype_,
        byte_size,
    ) = determine_precision(
        calling_instance.inference_precision, calling_instance.devices_
    )

    inference_config_ = inference_config_.override_with_user_input(
        user_config=calling_instance.inference_config
    )

    calling_instance.inference_config_ = inference_config_

    outlier_removal_std = inference_config_.OUTLIER_REMOVAL_STD
    if outlier_removal_std == "auto":
        default_stds = {
            "regressor": inference_config_._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD,
            "classifier": inference_config_._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD,
        }
        try:
            outlier_removal_std = default_stds[model_type]
        except KeyError as e:
            raise ValueError(f"Invalid model_type: {model_type}") from e

    update_encoder_params(  # Use the renamed function if available, or original one
        models=calling_instance.models_,
        remove_outliers_std=outlier_removal_std,
        seed=static_seed,
        differentiable_input=calling_instance.differentiable_input,
    )
    return byte_size, rng
