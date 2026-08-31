"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import dataclasses
import pathlib
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from sklearn.base import (
    check_is_fitted,
)

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
    InferenceEngineCachePreprocessing,
    InferenceEngineExplicitKVCache,
    InferenceEngineOnDemand,
)
from tabpfn.inference_config import (
    DEFAULT_SOFTMAX_TEMPERATURE,
    InferenceConfig,
    cpu_sample_limit,
    raise_if_softmax_temperatures_differ,
)
from tabpfn.model_loading import (
    load_model_criterion_config,
    resolve_model_version,
)
from tabpfn.preprocessing.clean import fix_dtypes
from tabpfn.utils import (
    DevicesSpecification,
    infer_autocast_inference_mode,
    infer_devices,
)
from tabpfn.validation import ensure_compatible_predict_input_sklearn

if TYPE_CHECKING:
    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    from tabpfn.classifier import TabPFNClassifier
    from tabpfn.constants import MemorySavingMode
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


ModelSpecs = RegressorModelSpecs | ClassifierModelSpecs


def initialize_tabpfn_model(
    model_path: ModelPath
    | list[ModelPath]
    | RegressorModelSpecs
    | ClassifierModelSpecs
    | list[RegressorModelSpecs]
    | list[ClassifierModelSpecs],
    which: Literal["classifier", "regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    softmax_temperature_override: float | None = None,
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
        softmax_temperature_override: The temperature the caller will apply to every
            model, or None if they did not ask for one. Only used to decide whether
            checkpoints are allowed to disagree on their temperature; the override
            itself is applied by the caller.

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
        _assert_inference_configs_equal(model_path, softmax_temperature_override)
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
        _assert_inference_configs_equal(model_path, softmax_temperature_override)
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
                    estimator_type="classifier",
                    version=version.value,
                    download_if_not_exists=download_if_not_exists,
                    softmax_temperature_override=softmax_temperature_override,
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
                    estimator_type="regressor",
                    version=version.value,
                    download_if_not_exists=download_if_not_exists,
                    softmax_temperature_override=softmax_temperature_override,
                )
            )
            norm_criterion = bardist

        inference_config = dataclasses.replace(
            inference_config, MAX_CPU_SAMPLES=cpu_sample_limit(version)
        )
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
    softmax_temperature_override: float | None,
) -> None:
    # A temperature mismatch is reported separately, as the user can fix that one by
    # naming a temperature.
    if not all(
        spec.inference_config.equals_ignoring_softmax_temperature(
            model_specs[0].inference_config
        )
        for spec in model_specs
    ):
        raise ValueError("All models must have the same inference config")
    raise_if_softmax_temperatures_differ(
        [spec.inference_config for spec in model_specs],
        softmax_temperature_override=softmax_temperature_override,
    )


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
        use_autocast_ = infer_autocast_inference_mode(
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
    ensemble_preprocessor: TabPFNEnsemblePreprocessor,
    models: list[Architecture],
    devices_: Sequence[torch.device],
    byte_size: int,
    forced_inference_dtype_: torch.dtype | None,
    memory_saving_mode: MemorySavingMode,
    use_autocast_: bool,
    task_type: str,
    inference_mode: bool = True,
    keep_cache_on_device: bool = True,
    kv_cache_precision: Literal["auto", "int8", "fp8"] | None = None,
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
        ensemble_preprocessor: The ensemble preprocessor to use.
        models: The loaded TabPFN models.
        devices_: The devices for inference.
        byte_size: Byte size for the chosen inference precision.
        forced_inference_dtype_: If not None, the forced dtype for inference.
        memory_saving_mode: GPU/CPU memory saving settings.
        use_autocast_: Whether we use torch.autocast for inference.
        task_type: The task type, e.g. "multiclass" or "regression". Only used
            for ``fit_mode="fit_with_cache"``, where the cache is built during
            initialization and is task-specific.
        inference_mode: Whether to use torch.inference_mode (set False if
            backprop is needed)
        keep_cache_on_device: Only relevant for ``fit_mode="fit_with_cache"``.
            If True (default), each per-estimator KV cache stays on the
            inference device. If False, caches are offloaded to CPU as they
            are built and moved back on demand during inference, lowering
            resident device memory at the cost of per-call transfers.
        kv_cache_precision: Only for ``fit_mode="fit_with_cache"``. Resolved
            against what the architecture supports. ``None`` (default) picks the
            architecture default (``"int8"`` when it can quantize, else
            ``"auto"``); ``"int8"`` quantizes the KV cache to save memory;
            ``"fp8"`` stores it as 8-bit floats (same size, float rounding
            semantics); ``"auto"`` keeps the computed dtype.
    """
    if fit_mode == "low_memory":
        return InferenceEngineOnDemand(
            X_train=X_train,
            y_train=y_train,
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
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            inference_mode=inference_mode,
        )
    if fit_mode == "fit_with_cache":
        return InferenceEngineExplicitKVCache(
            X_train=X_train,
            y_train=y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=devices_,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            autocast=use_autocast_,
            task_type=task_type,
            keep_cache_on_device=keep_cache_on_device,
            kv_cache_precision=kv_cache_precision,
        )
    if fit_mode == "batched":
        raise ValueError(
            "InferenceEngineBatchedNoPreprocessing should be initialized directly "
            "rather than through create_inference_engine."
        )

    raise ValueError(f"Invalid fit_mode: {fit_mode}")


def reject_categoricals_for_differentiable_input(
    categorical_features_indices: Sequence[int] | None,
) -> None:
    """Reject categorical features in the differentiable-input fit path.

    The differentiable path uses an identity preprocessor (no
    ordinal-encoding step), so categorical columns have no valid handling
    and would corrupt the prompt-tuning signal.
    """
    if (
        categorical_features_indices is not None
        and len(categorical_features_indices) > 0
    ):
        raise ValueError(
            "Categorical features are not supported for differentiable input."
        )


def initialize_model_variables_helper(
    calling_instance: TabPFNRegressor | TabPFNClassifier,
    model_type: Literal["regressor", "classifier"],
) -> int:
    """Set attributes on the given model to prepare it for inference.

    This includes selecting the device and the inference precision.

    Returns:
        a tuple (byte_size, rng), where byte_size is the number of bytes in the selected
        dtype, and rng is a NumPy random Generator for use during inference.
    """
    user_config = calling_instance.inference_config
    # Resolved before loading: checkpoints only have to agree on their temperature
    # when the user has not named one.
    softmax_temperature_override = _resolve_softmax_temperature_override(
        softmax_temperature=calling_instance.softmax_temperature,
        user_config=user_config,
    )

    models, architecture_configs, maybe_bardist, inference_config = (
        initialize_tabpfn_model(
            model_path=calling_instance.model_path,  # pyright: ignore[reportArgumentType]
            which=model_type,
            fit_mode=calling_instance.fit_mode,  # pyright: ignore[reportArgumentType]
            softmax_temperature_override=softmax_temperature_override,
        )
    )
    calling_instance.models_ = models
    calling_instance.configs_ = architecture_configs
    if model_type == "regressor" and maybe_bardist is not None:
        calling_instance.znorm_space_bardist_ = maybe_bardist

    byte_size = estimator_to_device(calling_instance, calling_instance.device)

    inference_config = inference_config.override_with_user_input_and_resolve_auto(
        user_config=user_config,
    )
    # Only the `softmax_temperature` argument still has to be applied here; an
    # override that came from `user_config` was applied by the call above, and the two
    # cannot both be given.
    if softmax_temperature_override is not None:
        inference_config = dataclasses.replace(
            inference_config, SOFTMAX_TEMPERATURE=softmax_temperature_override
        )

    calling_instance.inference_config_ = inference_config
    calling_instance.softmax_temperature_ = inference_config.SOFTMAX_TEMPERATURE

    return byte_size


def _resolve_softmax_temperature_override(
    *,
    softmax_temperature: float | Literal["auto"],
    user_config: dict | InferenceConfig | None,
) -> float | None:
    """Return the temperature the user asked for, or None if they asked for none.

    A temperature named either way wins over what the checkpoint declares. Naming it
    both ways is rejected rather than resolved, since the winner would not be
    apparent from the call.

    Raises:
        ValueError: If the temperature is given as the `softmax_temperature` argument
            and through `inference_config` at the same time.
    """
    config_temperature: float | None = None
    if isinstance(user_config, InferenceConfig):
        # A hand-built config replaces the checkpoint's wholesale, temperature
        # included, so it always names one -- even when the caller never mentioned
        # the temperature and only got the field default. Passing a config object is
        # deprecated for exactly that reason; drop this branch along with it, after
        # which only a dict that spells out SOFTMAX_TEMPERATURE names one.
        config_temperature = user_config.SOFTMAX_TEMPERATURE
    elif isinstance(user_config, dict) and "SOFTMAX_TEMPERATURE" in user_config:
        config_temperature = float(user_config["SOFTMAX_TEMPERATURE"])

    if softmax_temperature == "auto":
        return config_temperature

    if config_temperature is not None:
        raise ValueError(
            f"The softmax temperature was given twice: "
            f"`softmax_temperature={softmax_temperature}` and "
            f"`inference_config` with SOFTMAX_TEMPERATURE={config_temperature}. "
            f"Pass it one way or the other, so which one applies is unambiguous."
        )
    return float(softmax_temperature)


def resolved_softmax_temperature(
    estimator: TabPFNClassifier | TabPFNRegressor,
) -> float:
    """The softmax temperature `estimator` applies at predict time.

    Reads the value resolved by `initialize_model_variables_helper`, falling back to
    the unresolved argument for estimators pickled before that resolution existed,
    and to the temperature of the checkpoints predating `SOFTMAX_TEMPERATURE` when
    even that is unset (`"auto"` on an estimator that was never initialized).
    """
    temperature = getattr(
        estimator, "softmax_temperature_", estimator.softmax_temperature
    )
    return DEFAULT_SOFTMAX_TEMPERATURE if temperature == "auto" else float(temperature)


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
            ``"train"`` requires a fit mode that keeps the training rows around;
            it is not available with ``fit_mode="fit_with_cache"``, whose predict
            pass never runs the training rows through the transformer.

    Raises:
        TabPFNValidationError: If ``data_source="train"`` and the model was
            fitted with ``fit_mode="fit_with_cache"``.

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

    if data_source == "train" and isinstance(
        model.executor_, InferenceEngineExplicitKVCache
    ):
        # The cached predict pass only ever sees the test rows: the cache holds
        # the ICL key/value pairs and the projected decoder keys, not the train
        # embeddings themselves, so there is nothing to return here.
        raise TabPFNValidationError(
            'get_embeddings(..., data_source="train") is not supported with '
            'fit_mode="fit_with_cache", because the cached predict pass does not '
            "run the training rows through the transformer. Refit the model with "
            'fit_mode="fit_preprocessors" to obtain training embeddings.'
        )

    data_map = {"train": "train_embeddings", "test": "test_embeddings"}

    selected_data = data_map[data_source]

    # Avoid circular imports
    from tabpfn.preprocessing import (  # noqa: PLC0415
        ClassifierEnsembleConfig,
        RegressorEnsembleConfig,
    )
    from tabpfn.regressor import TabPFNRegressor  # noqa: PLC0415

    task_type = "regression" if isinstance(model, TabPFNRegressor) else "multiclass"

    X = ensure_compatible_predict_input_sklearn(X, model)
    X = fix_dtypes(X, cat_indices=model.categorical_features_indices)
    X = model.ordinal_encoder_.transform(X)

    embeddings: list[np.ndarray] = []

    for output, config in model.executor_.iter_outputs(
        X,
        autocast=model.use_autocast_,
        task_type=task_type,
        only_return_standard_out=False,
    ):
        # Cast output to Any to allow dict-like access
        output_dict = typing.cast("dict[str, torch.Tensor]", output)
        embed = output_dict[selected_data].squeeze(1)
        assert isinstance(config, (ClassifierEnsembleConfig, RegressorEnsembleConfig))
        assert embed.ndim == 2
        embeddings.append(embed.squeeze().cpu().numpy())

    return np.array(embeddings)
