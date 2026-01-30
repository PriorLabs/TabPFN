"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import pathlib
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import torch
from sklearn.base import (
    check_is_fitted,
)
from tabpfn_common_utils.telemetry.interactive import capture_session, ping

# --- TabPFN imports ---
from tabpfn.constants import (
    AUTOCAST_DTYPE_BYTE_SIZE,
    DEFAULT_DTYPE_BYTE_SIZE,
    ModelPath,
    XType,
)
from tabpfn.errors import TabPFNValidationError
from tabpfn.inference import (
    InferenceEngine,
    InferenceEngineBatchedNoPreprocessing,
    InferenceEngineCacheKV,
    InferenceEngineCachePreprocessing,
    InferenceEngineOnDemand,
)
from tabpfn.model_loading import load_model_criterion_config, resolve_model_version
from tabpfn.preprocessing.clean import fix_dtypes
from tabpfn.utils import (
    DevicesSpecification,
    infer_devices,
    infer_fp16_inference_mode,
    infer_random_state,
)
from tabpfn.validation import ensure_compatible_predict_input_sklearn

if TYPE_CHECKING:
    from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
    from tabpfn.architectures.base.memory import MemorySavingMode
    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
    from tabpfn.classifier import TabPFNClassifier
    from tabpfn.inference_config import InferenceConfig
    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
    from tabpfn.regressor import TabPFNRegressor


class BaseModelSpecs:
    """Base class for model specifications."""

    def __init__(
        self,
        model: Architecture,
        architecture_config: ArchitectureConfig,
        inference_config: InferenceConfig,
    ):
        self.model = model
        self.architecture_config = architecture_config
        self.inference_config = inference_config


class ClassifierModelSpecs(BaseModelSpecs):
    """Model specs for classifiers."""

    norm_criterion = None


class RegressorModelSpecs(BaseModelSpecs):
    """Model specs for regressors."""

    def __init__(
        self,
        model: Architecture,
        architecture_config: ArchitectureConfig,
        inference_config: InferenceConfig,
        norm_criterion: FullSupportBarDistribution,
    ):
        super().__init__(model, architecture_config, inference_config)
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
        if isinstance(model_path, list) and len(model_path) == 0:
            raise ValueError(
                "You provided a list of model paths with no entries. "
                "Please provide a valid `model_path` argument, or use 'auto' to use "
                "the default model."
            )

        if isinstance(model_path, str) and model_path == "auto":
            model_path = None  # type: ignore

        version = resolve_model_version(model_path)  # type: ignore
        download_if_not_exists = True

        if which == "classifier":
            models, _, architecture_configs, inference_config = (
                load_model_criterion_config(
                    model_path=model_path,  # pyright: ignore[reportArgumentType]
                    # The classifier's bar distribution is not used
                    check_bar_distribution_criterion=False,
                    cache_trainset_representation=(fit_mode == "fit_with_cache"),
                    which="classifier",
                    version=version.value,
                    download_if_not_exists=download_if_not_exists,
                )
            )
            norm_criterion = None
        else:
            models, bardist, architecture_configs, inference_config = (
                load_model_criterion_config(
                    model_path=model_path,  # pyright: ignore[reportArgumentType]
                    # The regressor's bar distribution is required
                    check_bar_distribution_criterion=True,
                    cache_trainset_representation=(fit_mode == "fit_with_cache"),
                    which="regressor",
                    version=version.value,
                    download_if_not_exists=download_if_not_exists,
                )
            )
            norm_criterion = bardist

        return models, architecture_configs, norm_criterion, inference_config

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
        raise TabPFNValidationError(
            f"Unknown inference_precision={inference_precision}"
        )

    return use_autocast_, forced_inference_dtype_, byte_size


def create_inference_engine(  # noqa: PLR0913
    *,
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_schema: FeatureSchema,
    ensemble_preprocessor: TabPFNEnsemblePreprocessor,
    models: list[Architecture],
    devices_: Sequence[torch.device],
    byte_size: int,
    forced_inference_dtype_: torch.dtype | None,
    memory_saving_mode: MemorySavingMode,
    use_autocast_: bool,
    inference_mode: bool = True,
) -> InferenceEngine:
    """Create the appropriate TabPFN inference engine based on `fit_mode`.

    Each execution mode will perform slightly different operations based on the mode
    specified by the user. In the case where preprocessors will be fit during
    initialization, we will use them to further transform the associated borders with
    each ensemble config member.

    Args:
        fit_mode: Determines how we prepare inference (pre-cache or not).
        X_train: Training features
        y_train: Training target
        feature_schema: The feature schema.
        ensemble_preprocessor: The ensemble preprocessor to use.
        models: The loaded TabPFN models.
        devices_: The devices for inference.
        byte_size: Byte size for the chosen inference precision.
        forced_inference_dtype_: If not None, the forced dtype for inference.
        memory_saving_mode: GPU/CPU memory saving settings.
        use_autocast_: Whether we use torch.autocast for inference.
        inference_mode: Whether to use torch.inference_mode (set False if
            backprop is needed)
    """
    if fit_mode == "low_memory":
        return InferenceEngineOnDemand(
            X_train=X_train,
            y_train=y_train,
            feature_schema=feature_schema,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
        )
    if fit_mode == "fit_preprocessors":
        return InferenceEngineCachePreprocessing(
            X_train=X_train,
            y_train=y_train,
            feature_schema=feature_schema,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            inference_mode=inference_mode,
        )
    if fit_mode == "fit_with_cache":
        return InferenceEngineCacheKV(
            X_train=X_train,
            y_train=y_train,
            feature_schema=feature_schema,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            autocast=use_autocast_,
        )
    if fit_mode == "batched":
        return InferenceEngineBatchedNoPreprocessing(
            X_trains=X_train,  # pyright: ignore[reportArgumentType]
            y_trains=y_train,  # pyright: ignore[reportArgumentType]
            feature_schema=feature_schema,  # pyright: ignore[reportArgumentType]
            ensemble_configs=ensemble_preprocessor.configs,  # pyright: ignore[reportArgumentType]
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            inference_mode=inference_mode,
        )

    raise ValueError(f"Invalid fit_mode: {fit_mode}")


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
    _, rng = infer_random_state(calling_instance.random_state)
    models, architecture_configs, maybe_bardist, inference_config = (
        initialize_tabpfn_model(
            model_path=calling_instance.model_path,  # pyright: ignore[reportArgumentType]
            which=model_type,
            fit_mode=calling_instance.fit_mode,  # pyright: ignore[reportArgumentType]
        )
    )
    calling_instance.models_ = models
    calling_instance.configs_ = architecture_configs
    if model_type == "regressor" and maybe_bardist is not None:
        calling_instance.znorm_space_bardist_ = maybe_bardist

    byte_size = estimator_to_device(calling_instance, calling_instance.device)

    inference_config = inference_config.override_with_user_input_and_resolve_auto(
        user_config=calling_instance.inference_config,
    )

    calling_instance.inference_config_ = inference_config

    return byte_size, rng


def estimator_to_device(
    estimator: TabPFNClassifier | TabPFNRegressor, device: DevicesSpecification
) -> int:
    """Move the given estimator to the given device(s)."""
    parsed_devices = infer_devices(device)

    estimator.device = device
    estimator.devices_ = parsed_devices
    estimator.use_autocast_, estimator.forced_inference_dtype_, byte_size = (
        determine_precision(estimator.inference_precision, estimator.devices_)
    )

    if hasattr(estimator, "executor_"):
        estimator.executor_.to(
            parsed_devices, estimator.forced_inference_dtype_, byte_size
        )

    return byte_size


def initialize_telemetry() -> None:
    """Initialize telemetry and acknowledge anonymous session.

    If user opted out of telemetry using `TABPFN_DISABLE_TELEMETRY`,
    no action is taken.
    """
    ping()
    capture_session()


def get_embeddings(
    model: TabPFNClassifier | TabPFNRegressor,
    X: XType,
    data_source: Literal["train", "test"] = "test",
) -> np.ndarray:
    """Extract embeddings from a fitted TabPFN model.

    Args:
        model : TabPFNClassifier | TabPFNRegressor
            The fitted classifier or regressor.
        X : XType
            The input data.
        data_source : {"train", "test"}, default="test"
            Select the transformer output to return. Use ``"train"`` to obtain
            embeddings from the training tokens and ``"test"`` for the test tokens.

    Returns:
        np.ndarray
            The computed embeddings for each fitted estimator.
            When ``n_estimators > 1`` the returned array has shape
            ``(n_estimators, n_samples, embedding_dim)``. You can average over the
            first axis or reshape to concatenate the estimators, e.g.:

                emb = get_embeddings(model, X)
                emb_avg = emb.mean(axis=0)
                emb_concat = emb.reshape(emb.shape[1], -1)
    """
    check_is_fitted(model)

    data_map = {"train": "train_embeddings", "test": "test_embeddings"}

    selected_data = data_map[data_source]

    # Avoid circular imports
    from tabpfn.preprocessing import (  # noqa: PLC0415
        ClassifierEnsembleConfig,
        RegressorEnsembleConfig,
    )

    X = ensure_compatible_predict_input_sklearn(X, model)
    X = fix_dtypes(X, cat_indices=model.categorical_features_indices)
    X = model.ordinal_encoder_.transform(X)

    embeddings: list[np.ndarray] = []

    # Cast executor to Any to bypass the iter_outputs signature check
    for output, config in model.executor_.iter_outputs(
        X,
        autocast=model.use_autocast_,
        only_return_standard_out=False,
    ):
        # Cast output to Any to allow dict-like access
        output_dict = typing.cast("dict[str, torch.Tensor]", output)
        embed = output_dict[selected_data].squeeze(1)
        assert isinstance(config, (ClassifierEnsembleConfig, RegressorEnsembleConfig))
        assert embed.ndim == 2
        embeddings.append(embed.squeeze().cpu().numpy())

    return np.array(embeddings)
