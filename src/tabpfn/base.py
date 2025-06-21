"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, overload

import torch

from tabpfn.config import ModelInterfaceConfig

# --- TabPFN imports ---
from tabpfn.constants import (
    AUTOCAST_DTYPE_BYTE_SIZE,
    DEFAULT_DTYPE_BYTE_SIZE,
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
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.model.loading import load_model_criterion_config
from tabpfn.preprocessing import (
    BaseDatasetConfig,
    ClassifierDatasetConfig,
    DatasetCollectionWithPreprocessing,
    RegressorDatasetConfig,
)
from tabpfn.utils import (
    infer_device_and_type,
    infer_fp16_inference_mode,
    infer_random_state,
    split_large_data,
    update_encoder_params,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    from tabpfn.model.config import ModelConfig
    from tabpfn.model.transformer import PerFeatureTransformer


class BaseModelSpecs:
    """Base class for model specifications."""

    def __init__(self, model: PerFeatureTransformer, config: ModelConfig):
        self.model = model
        self.config = config


class ClassifierModelSpecs(BaseModelSpecs):
    """Model specs for classifiers."""

    norm_criterion = None


class RegressorModelSpecs(BaseModelSpecs):
    """Model specs for regressors."""

    def __init__(
        self,
        model: PerFeatureTransformer,
        config: ModelConfig,
        norm_criterion: FullSupportBarDistribution,
    ):
        super().__init__(model, config)
        self.norm_criterion = norm_criterion


ModelSpecs = Union[RegressorModelSpecs, ClassifierModelSpecs]


@overload
def initialize_tabpfn_model(
    model_path: (str | Path | Literal["auto"] | RegressorModelSpecs),
    which: Literal["regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
) -> RegressorModelSpecs: ...


@overload
def initialize_tabpfn_model(
    model_path: (str | Path | Literal["auto"] | ClassifierModelSpecs),
    which: Literal["classifier"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
) -> ClassifierModelSpecs: ...


def initialize_tabpfn_model(
    model_path: str
    | Path
    | Literal["auto"]
    | RegressorModelSpecs
    | ClassifierModelSpecs,
    which: Literal["classifier", "regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
) -> ModelSpecs:
    """Initializes a TabPFN model based on the provided configuration.

    Args:
        model_path: Path or directive ("auto") to load the pre-trained model from.
        which: Which TabPFN model to load.
        fit_mode: Determines caching behavior.

    Returns:
        model: The loaded TabPFN model.
        config: The configuration object associated with the loaded model.
        bar_distribution: The BarDistribution for regression (`None` if classifier).
    """
    model, config, norm_criterion = None, None, None
    if isinstance(model_path, RegressorModelSpecs) and which == "regressor":
        model = model_path.model
        config = model_path.config
        norm_criterion = model_path.norm_criterion
    elif isinstance(model_path, ClassifierModelSpecs) and which == "classifier":
        model = model_path.model
        config = model_path.config
    elif model_path is None or isinstance(model_path, (str, Path)):
        # (after processing 'auto')
        download = True
        if isinstance(model_path, str) and model_path == "auto":
            model_path = None  # type: ignore

        # Load model with potential caching
        if which == "classifier":
            # The classifier's bar distribution is not used;
            # pass check_bar_distribution_criterion=False
            model, _, config_ = load_model_criterion_config(
                model_path=model_path,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=(fit_mode == "fit_with_cache"),
                which="classifier",
                version="v2",
                download=download,
            )
            norm_criterion = None
        else:
            # The regressor's bar distribution is required
            model, bardist, config_ = load_model_criterion_config(
                model_path=model_path,
                check_bar_distribution_criterion=True,
                cache_trainset_representation=(fit_mode == "fit_with_cache"),
                which="regressor",
                version="v2",
                download=download,
            )
            norm_criterion = bardist
    else:
        raise TypeError(
            "Received ModelSpecs via 'model_path', but 'which' parameter is set to '"
            + which
            + "'. Expected 'classifier' or 'regressor'. and model_path"
            + "is of of type"
            + str(type(model_path))
        )

    return model, config, norm_criterion


def determine_precision(
    inference_precision: torch.dtype | Literal["autocast", "auto"],
    device_: torch.device,
) -> tuple[bool, torch.dtype | None, int]:
    """Decide whether to use autocast or a forced precision dtype.

    Args:
        inference_precision:

            - If `"auto"`, decide automatically based on the device.
            - If `"autocast"`, explicitly use PyTorch autocast (mixed precision).
            - If a `torch.dtype`, force that precision.

        device_: The device on which inference is run.

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
            device=device_,
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
    model: PerFeatureTransformer,
    ensemble_configs: Any,
    cat_ix: list[int],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    device_: torch.device,
    rng: np.random.Generator,
    n_jobs: int,
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
        model: The loaded TabPFN model.
        ensemble_configs: The ensemble configurations to create multiple "prompts".
        cat_ix: Indices of inferred categorical features.
        fit_mode: Determines how we prepare inference (pre-cache or not).
        device_: The device for inference.
        rng: Numpy random generator.
        n_jobs: Number of parallel CPU workers.
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
            model=model,
            n_workers=n_jobs,
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
            n_workers=n_jobs,
            model=model,
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
            model=model,
            ensemble_configs=ensemble_configs,
            n_workers=n_jobs,
            device=device_,
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
            model=model,
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
    device: str | torch.device,
    X: np.ndarray | torch.Tensor | pd.DataFrame,
    *,
    allow_cpu_override: bool = False,
) -> None:
    """Check if using CPU with large datasets and warn or error appropriately.

    Args:
        device: The torch device being used
        X: The input data (NumPy array, Pandas DataFrame, or Torch Tensor)
        allow_cpu_override: If True, allow CPU usage with large datasets.
    """
    allow_cpu_override = allow_cpu_override or (
        os.getenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "0") == "1"
    )

    if allow_cpu_override:
        return

    device_mapped = infer_device_and_type(device)

    # Determine number of samples
    try:
        num_samples = X.shape[0]
    except AttributeError:
        return

    if torch.device(device_mapped).type == "cpu":
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
    """
    if not isinstance(X_raw, list):
        X_raw = [X_raw]
    if not isinstance(y_raw, list):
        y_raw = [y_raw]
    assert len(X_raw) == len(y_raw), "X and y lists must have the same length."

    if not hasattr(calling_instance, "model_") or calling_instance.model_ is None:
        _, rng = calling_instance._initialize_model_variables()
    else:
        static_seed, rng = infer_random_state(calling_instance.random_state)

    X_split, y_split = [], []
    for X_item, y_item in zip(X_raw, y_raw):
        if max_data_size is not None:
            Xparts, yparts = split_large_data(X_item, y_item, max_data_size)
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
                bardist_=bardist_,
            )
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        dataset_config_collection.append(dataset_config)

    return DatasetCollectionWithPreprocessing(split_fn, rng, dataset_config_collection)


def _initialize_model_variables_helper(
    calling_instance: Any,
    model_type: Literal["regressor", "classifier"],
) -> tuple[int, np.random.Generator]:
    """Helper function to perform initialization
    of the model, return determined byte_size
    and RNG object.
    """
    static_seed, rng = infer_random_state(calling_instance.random_state)

    if model_type == "regressor":
        (
            calling_instance.model_,
            calling_instance.config_,
            calling_instance.bardist_,
        ) = initialize_tabpfn_model(
            model_path=calling_instance.model_path,
            which="regressor",
            fit_mode=calling_instance.fit_mode,  # Use the instance's fit_mode
        )
    elif model_type == "classifier":
        (calling_instance.model_, calling_instance.config_, _) = (
            initialize_tabpfn_model(
                model_path=calling_instance.model_path,
                which="classifier",
                fit_mode=calling_instance.fit_mode,  # Use the instance's fit_mode
            )
        )
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    calling_instance.device_ = infer_device_and_type(calling_instance.device)
    (
        calling_instance.use_autocast_,
        calling_instance.forced_inference_dtype_,
        byte_size,
    ) = determine_precision(
        calling_instance.inference_precision, calling_instance.device_
    )
    calling_instance.model_.to(calling_instance.device_)

    # Build the interface_config
    _config = ModelInterfaceConfig.from_user_input(
        inference_config=calling_instance.inference_config,
    )  # shorter alias
    calling_instance.interface_config_ = _config

    outlier_removal_std = _config.OUTLIER_REMOVAL_STD
    if outlier_removal_std == "auto":
        default_stds = {
            "regressor": _config._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD,
            "classifier": _config._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD,
        }
        try:
            outlier_removal_std = default_stds[model_type]
        except KeyError as e:
            raise ValueError(f"Invalid model_type: {model_type}") from e

    update_encoder_params(  # Use the renamed function if available, or original one
        model=calling_instance.model_,
        remove_outliers_std=outlier_removal_std,
        seed=static_seed,
        inplace=True,
        differentiable_input=calling_instance.differentiable_input,
    )
    return byte_size, rng
