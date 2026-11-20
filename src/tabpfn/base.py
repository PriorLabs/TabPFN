"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import dataclasses
import functools
import pathlib
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.base import (
    check_is_fitted,
)

# --- TabPFN imports ---
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
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
    OVERRIDABLE_FIELDS,
    InferenceConfig,
    cpu_sample_limit,
    raise_if_checkpoints_disagree_on_overridable_fields,
)
from tabpfn.model_loading import (
    load_model_criterion_config,
    resolve_model_version,
)
from tabpfn.preprocessing.clean import (
    ColumnRangeEncoders,
    clean_data,
    clean_data_transform,
    empty_fit_frame,
    may_overwrite_fit_input,
    narrow_fit_input_in_place,
    overwriting_fit_input,
)
from tabpfn.preprocessing.datamodel import (
    FeatureModality,
    FeatureSchema,
    build_input_feature_names,
)
from tabpfn.preprocessing.datetimes import DateTransformer
from tabpfn.preprocessing.group_ids import as_group_feature
from tabpfn.preprocessing.modality_detection import (
    detect_feature_modalities,
    named_feature_modalities,
    warn_on_text,
)
from tabpfn.preprocessing.pipeline_factory import pipelines_keep_float32
from tabpfn.preprocessing.text import TextTransformer
from tabpfn.utils import (
    DevicesSpecification,
    infer_autocast_inference_mode,
    infer_devices,
)
from tabpfn.validation import (
    check_input_shape_matches,
    ensure_compatible_fit_columns,
    ensure_compatible_fit_inputs,
    ensure_compatible_fit_inputs_sklearn,
    ensure_compatible_predict_input_sklearn,
    validate_categorical_features_indices,
    validate_dataset_size,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
    from tabpfn.classifier import TabPFNClassifier
    from tabpfn.constants import MemorySavingMode, YType
    from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
    from tabpfn.preprocessing.steps.preprocessing_helpers import (
        OrderPreservingColumnTransformer,
    )
    from tabpfn.regressor import TabPFNRegressor


@dataclasses.dataclass
class ModelSpecs:
    """An in-memory model and the metadata needed by either TabPFN estimator.

    Pass this object (or a list of objects) as ``model_path``. The estimator
    selects the task, so the same specs can back both tasks of a multitask model.
    The model is used by reference, not copied; create an isolated evaluation
    model before wrapping live training weights.

    Args:
        model: The neural network used for inference.
        architecture_config: The configuration associated with the model.
        inference_config: Preprocessing and ensembling settings.
        norm_criterion: Optional regression distribution in normalized target
            space. When omitted, regression initialization constructs it from
            ``model.regression_borders`` (available from v3 onward). Required for
            older models without those borders, including v2/v2.5 finetuning;
            preserve the estimator's ``znorm_space_bardist_`` in that case.
            An explicit distribution takes precedence over embedded borders.
            Classification ignores this field. This is not the training objective
            or the fitted distribution in raw target units.
    """

    model: Architecture
    architecture_config: ArchitectureConfig
    inference_config: InferenceConfig
    norm_criterion: FullSupportBarDistribution | None = None


def _regression_distribution_from_specs(spec: ModelSpecs) -> FullSupportBarDistribution:
    if getattr(spec.model, "task_type", None) == "multiclass":
        raise ValueError("A classification-only model cannot be used for regression")
    if spec.norm_criterion is not None:
        return spec.norm_criterion
    borders = getattr(spec.model, "regression_borders", None)
    if borders is None:
        raise ValueError(
            "Regression ModelSpecs requires norm_criterion when the model has no "
            "regression_borders. For legacy models, pass the distribution returned "
            "by the loader or the fitted estimator's znorm_space_bardist_."
        )
    # Inference may have cast the shared model to half precision. Keep target-space
    # arithmetic in float32 so large target means do not overflow.
    return FullSupportBarDistribution(
        borders.detach().to(torch.float32).clone(), ignore_nan_targets=True
    )


def initialize_tabpfn_model(
    model_path: ModelPath | list[ModelPath] | ModelSpecs | list[ModelSpecs],
    which: Literal["classifier", "regressor"],
    *,
    softmax_temperature_override: float | None = None,
    n_estimators_override: int | None = None,
    average_before_softmax_override: bool | None = None,
    devices: Sequence[torch.device] | None = None,
    force_inference_dtype: torch.dtype | None = None,
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
            estimators. A ModelSpecs object or list of objects uses in-memory models
            without loading a checkpoint. Regression distributions are derived
            from embedded borders unless explicitly supplied in the specs.

        which: Which TabPFN model to load.
        softmax_temperature_override: The temperature the caller will apply to every
            model, or None if they did not ask for one. Only used to decide whether
            checkpoints are allowed to disagree on their temperature; the override
            itself is applied by the caller.
        n_estimators_override: Likewise for the number of estimators.
        average_before_softmax_override: Likewise for the ensemble averaging order.
        devices: Where the caller will place the models; part of the cache key.
        force_inference_dtype: The dtype the caller will cast them to; likewise.

    Returns:
        a list of models,
        a list of architecture configs (associated with each model),
        if regression, the bar distribution, otherwise None,
        the inference config
    """
    if isinstance(model_path, ModelSpecs):
        model_path = [model_path]

    if (
        isinstance(model_path, list)
        and model_path
        and all(isinstance(spec, ModelSpecs) for spec in model_path)
    ):
        specs = typing.cast("list[ModelSpecs]", model_path)
        _assert_inference_configs_equal(
            specs,
            softmax_temperature_override,
            n_estimators_override,
            average_before_softmax_override=average_before_softmax_override,
        )
        norm_criterion = None
        if which == "regressor":
            distributions = [
                _regression_distribution_from_specs(spec) for spec in specs
            ]
            norm_criterion = distributions[0]
            if any(
                not norm_criterion.has_equal_borders(distribution)
                for distribution in distributions[1:]
            ):
                raise ValueError("All regression ModelSpecs must have the same borders")
        return (
            [spec.model for spec in specs],
            [spec.architecture_config for spec in specs],
            norm_criterion,
            specs[0].inference_config,
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
                    estimator_type="classifier",
                    version=version.value,
                    download_if_not_exists=download_if_not_exists,
                    softmax_temperature_override=softmax_temperature_override,
                    n_estimators_override=n_estimators_override,
                    average_before_softmax_override=average_before_softmax_override,
                    devices=devices,
                    force_inference_dtype=force_inference_dtype,
                )
            )
            norm_criterion = None
        else:
            models, bardist, architecture_configs, inference_config = (
                load_model_criterion_config(
                    model_path=model_path,  # pyright: ignore[reportArgumentType]
                    # The regressor's bar distribution is required
                    check_bar_distribution_criterion=True,
                    estimator_type="regressor",
                    version=version.value,
                    download_if_not_exists=download_if_not_exists,
                    softmax_temperature_override=softmax_temperature_override,
                    n_estimators_override=n_estimators_override,
                    average_before_softmax_override=average_before_softmax_override,
                    devices=devices,
                    force_inference_dtype=force_inference_dtype,
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
    model_specs: list[ModelSpecs],
    softmax_temperature_override: float | None,
    n_estimators_override: int | None,
    *,
    average_before_softmax_override: bool | None,
) -> None:
    # A mismatch in an overridable field is reported separately, as the user can fix
    # those by naming a value.
    if not all(
        spec.inference_config.equals_ignoring_overridable_fields(
            model_specs[0].inference_config
        )
        for spec in model_specs
    ):
        raise ValueError("All models must have the same inference config")
    raise_if_checkpoints_disagree_on_overridable_fields(
        [spec.inference_config for spec in model_specs],
        overrides={
            "SOFTMAX_TEMPERATURE": softmax_temperature_override,
            "N_ESTIMATORS": n_estimators_override,
            "AVERAGE_BEFORE_SOFTMAX": average_before_softmax_override,
        },
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
    kv_cache_precision: Literal["auto", "int8", "fp8", "adaptive"] | None = None,
    enable_torch_compile: bool = False,
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
        enable_torch_compile: Enable the architecture's compiled inference regions.
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
            semantics); ``"auto"`` keeps the computed dtype; ``"adaptive"``
            stores the grid an attention backend already rounded the keys and
            values to (``kv_grid_dtype``), else ``"int8"``.
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
            enable_torch_compile=enable_torch_compile,
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
            enable_torch_compile=enable_torch_compile,
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
            enable_torch_compile=enable_torch_compile,
        )
    if fit_mode == "batched":
        raise ValueError(
            "InferenceEngineBatchedNoPreprocessing should be initialized directly "
            "rather than through create_inference_engine."
        )

    raise ValueError(f"Invalid fit_mode: {fit_mode}")


def resolve_categorical_features_indices(
    X: XType,
    categorical_features_indices: Sequence[int] | None,
) -> list[int] | None:
    """Validate declared indices and merge pandas `category` column positions.

    A `category` dtype states the same intent as an entry in
    `categorical_features_indices`, so the two are merged here, before any column
    moves. The merged declarations drive date/text expansion and modality
    detection; numeric columns remain subject to the categorical cardinality cap.

    Returns:
        The sorted union of both, or `None` when neither names a column.
    """
    validate_categorical_features_indices(categorical_features_indices)
    declared = set(categorical_features_indices or ())
    if isinstance(X, pd.DataFrame):
        typed = {
            i
            for i, dtype in enumerate(X.dtypes)
            if isinstance(dtype, pd.CategoricalDtype)
        }
        declared.update(typed)
    return sorted(declared) if declared else None


def expand_dates_and_text(
    X: XType,
    *,
    categorical_features_indices: Sequence[int] | None,
    inference_config: InferenceConfig,
) -> tuple[XType, DateTransformer, TextTransformer, list[str] | None, list[int] | None]:
    """Expand the datetime and text columns of a fit input, before validation.

    An expanded column is dropped and its features appended, so every column
    after it moves down. The returned labels and categorical positions describe
    the returned input, so no caller needs to know which transformer ran last.

    Returns:
        The expanded input, the two fitted transformers, the expanded input's
        column labels (`None` when `X` is not a `DataFrame`), and the declared
        categorical positions in it (`None` when none were declared).
    """
    date_transformer = DateTransformer(
        categorical_indices=categorical_features_indices,
        transform_dates=inference_config.TRANSFORM_DATES,
    )
    X = date_transformer.fit_transform(X)
    categorical_indices = date_transformer.output_indices(categorical_features_indices)
    text_transformer = TextTransformer(
        categorical_indices=categorical_indices,
        transform_text=inference_config.TRANSFORM_TEXT,
        min_cardinality_for_text=inference_config.MIN_CARDINALITY_FOR_TEXT,
        n_components=inference_config.TEXT_N_COMPONENTS,
    )
    X = text_transformer.fit_transform(X)
    return (
        X,
        date_transformer,
        text_transformer,
        text_transformer.feature_names_out_,
        text_transformer.output_indices(categorical_indices),
    )


def transform_columns(
    model: TabPFNClassifier | TabPFNRegressor, X: XType
) -> tuple[XType, np.ndarray | None]:
    """Apply the group, date and text transformers fitted on the fit input.

    Returns:
        The transformed input, and each row's group within `X` when the
        predictions are to be pooled per group, else `None`.
    """
    groups = model.group_encoder_.row_groups(X)
    X = model.group_encoder_.transform(X)
    X = model.date_transformer_.transform(X)
    X = model.text_transformer_.transform(X)
    if groups is None:
        return X, None
    X = model.group_aggregator_.transform(X, groups)
    return X, groups if model.group_aggregator_.pool_ else None


def cleaned_table_dtype(estimator: TabPFNClassifier | TabPFNRegressor) -> np.dtype:
    """The dtype the cleaned fit input is kept in: float32 when every member reads
    it as float32, float64 otherwise.

    A member reads float32 when its CPU pipeline only selects, reorders, compares and
    encodes values, and the model takes float32. Narrowing a wider input first then
    changes what reaches the model only where values that differ round to the same
    float32. In `"batched"` mode the table goes to the fine-tuning datasets, whose
    members are built without GPU preprocessing.
    """
    config = estimator.inference_config_
    reads_float32 = estimator.forced_inference_dtype_ in (
        None,
        torch.float32,
    ) and pipelines_keep_float32(
        config.PREPROCESS_TRANSFORMS,
        polynomial_features=config.POLYNOMIAL_FEATURES,
        add_fingerprint_feature=config.FINGERPRINT_FEATURE,
        enable_gpu_preprocessing=config.ENABLE_GPU_PREPROCESSING
        and estimator.fit_mode != "batched",
    )
    return np.dtype(np.float32 if reads_float32 else np.float64)


def validate_and_clean_fit_input(
    estimator: TabPFNClassifier | TabPFNRegressor,
    X: XType,
    y: YType,
    *,
    X_given: XType,
    feature_names: list[str] | None,
    categorical_indices: list[int] | None,
    group_feature_index: int | None,
    ensure_y_numeric: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str | None,
    FeatureSchema,
    OrderPreservingColumnTransformer | ColumnRangeEncoders,
]:
    """Validate the expanded fit input, decide its modalities and clean it.

    Args:
        estimator: The estimator being fit.
        X: The fit input after date, text and group expansion.
        y: The targets.
        X_given: The input the caller passed to `fit`.
        feature_names: `X`'s column labels, `None` for an array.
        categorical_indices: Positions in `X` declared categorical.
        group_feature_index: Position of the group feature, which is cleaned as a
            numerical feature and marked categorical after.
        ensure_y_numeric: Whether the targets must be numeric.

    Returns:
        The cleaned table, the validated targets, the targets' name when they came as
        a `Series`, the feature schema, and the fitted encoders `clean_predict_input`
        reapplies.
    """
    config = estimator.inference_config_
    dtype = cleaned_table_dtype(estimator)
    min_samples = config.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE
    thresholds = {
        "min_samples_for_inference": min_samples,
        "max_unique_for_category": config.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
        "min_unique_for_numerical": config.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        "min_cardinality_for_text": config.MIN_CARDINALITY_FOR_TEXT,
    }
    passthrough_inf = estimator.get_inference_config().PASSTHROUGH_INF
    if not _cleans_by_column_ranges(X, dtype):
        X, y, y_name = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=estimator,
            max_num_samples=config.MAX_NUMBER_OF_SAMPLES,
            max_num_features=config.MAX_NUMBER_OF_FEATURES,
            max_cpu_samples=config.MAX_CPU_SAMPLES,
            ignore_pretraining_limits=estimator.ignore_pretraining_limits,
            ensure_y_numeric=ensure_y_numeric,
            devices=estimator.devices_,
        )
        feature_schema = detect_feature_modalities(
            X=X,
            feature_names=feature_names,
            provided_categorical_indices=categorical_indices,
            **thresholds,
        )
        # Numerical through cleaning, so neither the ordinal encoder nor the
        # constant-column restore rewrites an unseen group's NaN; categorical
        # after it, whatever the number of groups.
        feature_schema = as_group_feature(
            feature_schema, group_feature_index, FeatureModality.NUMERICAL
        )
        X, encoder, feature_schema = clean_data(
            X=X,
            feature_schema=feature_schema,
            passthrough_inf=passthrough_inf,
            overwrite_input=may_overwrite_fit_input(X, X_given),
        )
        if X.dtype != dtype and dtype == np.float32:
            X = X.astype(np.float32)
        return (
            X,
            y,
            y_name,
            as_group_feature(
                feature_schema, group_feature_index, FeatureModality.CATEGORICAL
            ),
            encoder,
        )

    y_name = str(y.name) if isinstance(y, pd.Series) else None
    n_rows, n_columns = X.shape
    clean_ranges = functools.partial(
        _clean_ranges,
        estimator,
        X,
        y,
        names=build_input_feature_names(feature_names, n_columns),
        declared=sorted(categorical_indices or ()),
        group_feature_index=group_feature_index,
        ensure_y_numeric=ensure_y_numeric,
        thresholds=thresholds,
        passthrough_inf=passthrough_inf,
    )
    table = None
    if isinstance(X, np.ndarray):
        # Cleaning that encodes no column leaves every value as it is, so the table
        # is `X` narrowed, in its own memory where the caller lets go of it, else
        # cast, which reads it row by row rather than in strided ranges; the ranges
        # are read without a table to write into.
        y_clean, ranges, feature_schema = clean_ranges(None)
        if not any(encoder.selected_columns() for _, _, encoder in ranges):
            table = (
                narrow_fit_input_in_place(X)
                if _narrows_in_place(X, X_given, dtype)
                else X.astype(dtype)
            )
    if table is None:
        table = np.empty((n_rows, n_columns), dtype=dtype)
        y_clean, ranges, feature_schema = clean_ranges(table)
        if overwriting_fit_input() and isinstance(X_given, pd.DataFrame):
            empty_fit_frame(X_given)
    warn_on_text(feature_schema)
    return (
        table,
        y_clean,
        y_name,
        as_group_feature(
            feature_schema, group_feature_index, FeatureModality.CATEGORICAL
        ),
        ColumnRangeEncoders(ranges=ranges, dtype=dtype),
    )


def _clean_ranges(
    estimator: TabPFNClassifier | TabPFNRegressor,
    X: pd.DataFrame | np.ndarray,
    y: YType,
    table: np.ndarray | None,
    *,
    names: list[str],
    declared: list[int],
    group_feature_index: int | None,
    ensure_y_numeric: bool,
    thresholds: dict[str, int],
    passthrough_inf: bool,
) -> tuple[
    np.ndarray,
    list[tuple[int, int, OrderPreservingColumnTransformer]],
    FeatureSchema,
]:
    """Validate, type and clean `X` a range of columns at a time, into `table`.

    Returns the validated targets, each range's fitted encoder and the schema.
    """
    config = estimator.inference_config_
    ranges: list[tuple[int, int, OrderPreservingColumnTransformer]] = []
    features = []
    # A numeric array is validated whole, in place: a range of it is a strided view,
    # which reads slowly. A frame is validated a range at a time, which converts it.
    whole = isinstance(X, np.ndarray)
    for start, stop in _column_ranges(X):
        part = _columns(X, start, stop)
        if start == 0:
            # The targets are validated with the first range, and the size limits
            # checked before the rest is read.
            validated, y = ensure_compatible_fit_inputs_sklearn(
                X if whole else part,
                y,
                estimator=estimator,
                ensure_y_numeric=ensure_y_numeric,
            )
            if not whole:
                part = validated
            validate_dataset_size(
                X=X,
                y=y,
                max_num_samples=config.MAX_NUMBER_OF_SAMPLES,
                max_num_features=config.MAX_NUMBER_OF_FEATURES,
                max_cpu_samples=config.MAX_CPU_SAMPLES,
                devices=estimator.devices_,
                ignore_pretraining_limits=estimator.ignore_pretraining_limits,
            )
        elif not whole:
            part = ensure_compatible_fit_columns(part, estimator=estimator)
        schema = named_feature_modalities(
            part,
            names[start:stop],
            provided_categorical_indices=[
                i - start for i in declared if start <= i < stop
            ],
            **thresholds,
        )
        if group_feature_index is not None and start <= group_feature_index < stop:
            schema = as_group_feature(
                schema, group_feature_index - start, FeatureModality.NUMERICAL
            )
        # Each range is a copy or a view the caller still owns, never written to.
        cleaned, encoder, schema = clean_data(
            X=part, feature_schema=schema, passthrough_inf=passthrough_inf
        )
        if table is not None:
            table[:, start:stop] = cleaned
        ranges.append((start, stop, encoder))
        features += schema.features
    return np.asarray(y), ranges, FeatureSchema(features=features)


def clean_predict_input(
    estimator: TabPFNClassifier | TabPFNRegressor, X: XType
) -> np.ndarray:
    """Validate and clean predict input the way `validate_and_clean_fit_input` did
    the fit input.
    """
    encoder = getattr(estimator, "ordinal_encoder_", None)
    cat_indices = estimator.inferred_feature_schema_.indices_for(
        FeatureModality.CATEGORICAL
    )
    passthrough_inf = estimator.get_inference_config().PASSTHROUGH_INF
    if (
        not isinstance(encoder, ColumnRangeEncoders)
        or not isinstance(X, (pd.DataFrame, np.ndarray))
        or X.ndim != 2
    ):
        X = ensure_compatible_predict_input_sklearn(X, estimator)
    if not isinstance(encoder, ColumnRangeEncoders):
        return clean_data_transform(
            X,
            cat_indices=cat_indices,
            ord_encoder=encoder,
            passthrough_inf=passthrough_inf,
        )
    table = np.empty((X.shape[0], encoder.n_columns), dtype=encoder.dtype)
    for start, stop, range_encoder in encoder.ranges:
        part = ensure_compatible_predict_input_sklearn(
            _columns(X, start, stop), estimator
        )
        table[:, start:stop] = clean_data_transform(
            part,
            cat_indices=[i - start for i in cat_indices if start <= i < stop],
            ord_encoder=range_encoder,
            passthrough_inf=passthrough_inf,
        )
    return table


_RANGE_BYTES = 1 << 30
"""Transient bytes a range of columns takes, at most, while it is validated and
cleaned. Wider ranges cost memory; narrower ones pay the validation and the encoder
fit of a range more often."""

_NUMERIC_CELL_BYTES = 32
"""Bytes a numeric cell can take while its range is cleaned: a few copies, the widest
in float64."""

_OBJECT_CELL_BYTES = 256
"""Bytes any other cell can take while its range is cleaned: it passes through
Python objects and a few frames."""


def _cleans_by_column_ranges(X: XType, dtype: np.dtype) -> bool:
    """Whether the fit input is validated and cleaned a range of columns at a time.

    So it is when reading it whole would build a converted copy of all of it: a frame
    of more than one block or of a non-numeric one, or a numeric frame or array of
    another dtype than a float32 table. A numeric frame of one block or a numeric
    array otherwise is read as it is, without a copy. An input without a column is
    left to the whole-table validation, which refuses it.
    """
    if len(getattr(X, "shape", ())) == 2 and X.shape[1] == 0:
        return False
    if isinstance(X, pd.DataFrame):
        blocks = getattr(getattr(X, "_mgr", None), "blocks", None)
        if blocks is None or len(blocks) != 1:
            return True
        # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
        values = blocks[0].values  # noqa: PD011
        if not isinstance(values, np.ndarray) or values.dtype.kind not in "biuf":
            return True
        return dtype == np.float32 and values.dtype != np.float32
    return (
        isinstance(X, np.ndarray)
        and X.ndim == 2
        and dtype == np.float32
        and X.dtype.kind in "biuf"
        and X.dtype != np.float32
    )


def _narrows_in_place(X: XType, X_given: XType, dtype: np.dtype) -> bool:
    """Whether the float32 table may be `X` narrowed in its own memory.

    So it may when `X` is the caller's own writeable C-ordered numeric array, of at
    least four bytes per value, given to a fit inside `overwrite_fit_input()`.
    """
    return (
        dtype == np.float32
        and isinstance(X, np.ndarray)
        and X.ndim == 2
        and X.dtype.kind in "iuf"
        and X.dtype.itemsize >= 4
        and X.dtype != np.float32
        and X.flags.c_contiguous
        and may_overwrite_fit_input(X, X_given)
    )


def _column_ranges(X: pd.DataFrame | np.ndarray) -> Iterator[tuple[int, int]]:
    """`(start, stop)` ranges covering `X`'s columns, each of one kind, numeric or
    not, and as wide as `_RANGE_BYTES` allows.
    """
    n_rows, n_columns = X.shape
    if isinstance(X, pd.DataFrame):
        numeric = [getattr(dtype, "kind", "O") in "biuf" for dtype in X.dtypes]
    else:
        numeric = [True] * n_columns
    start = 0
    while start < n_columns:
        cell_bytes = _NUMERIC_CELL_BYTES if numeric[start] else _OBJECT_CELL_BYTES
        width = max(1, _RANGE_BYTES // max(1, n_rows * cell_bytes))
        stop = start + 1
        while (
            stop < n_columns
            and stop - start < width
            and numeric[stop] == numeric[start]
        ):
            stop += 1
        yield start, stop
        start = stop


def _columns(X: pd.DataFrame | np.ndarray, start: int, stop: int) -> XType:
    """`X`'s columns from `start` to `stop`, as the same kind of container."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, start:stop]
    return X[:, start:stop]


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


GROUPED_PRESET: dict[str, Any] = {
    "APPEND_GROUP_AGGREGATES": True,
    "POOL_GROUP_PREDICTIONS": "mean",
}
"""The `"grouped"` preset of both estimators."""


def resolve_presets(
    presets: str | Sequence[str],
    user_config: dict | InferenceConfig | None,
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The `inference_config` values of the presets, without the keys `user_config`
    names.

    Raises:
        TabPFNValidationError: If a preset is unknown, or two presets set one key to
            different values.
    """
    if not isinstance(user_config, dict | None):
        raise TabPFNValidationError(
            "`presets` needs `inference_config` as a dict or None."
        )
    values: dict[str, Any] = {}
    setter: dict[str, str] = {}
    for name in [presets] if isinstance(presets, str) else presets:
        if name not in registry:
            raise TabPFNValidationError(
                f"Unknown preset {name!r}. Choose from {sorted(registry)}."
            )
        for key, value in registry[name].items():
            if key in values and values[key] != value:
                raise TabPFNValidationError(
                    f"The presets {setter[key]!r} and {name!r} cannot be combined: "
                    f"they set {key} to {values[key]!r} and {value!r}."
                )
            values[key] = value
            setter[key] = name
    return {k: v for k, v in values.items() if k not in (user_config or {})}


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
    # Resolved before loading: checkpoints only have to agree on a field when the
    # user has not named a value for it.
    overrides = _resolve_overrides(calling_instance, user_config)

    devices = infer_devices(calling_instance.device)
    _, forced_inference_dtype, _ = determine_precision(
        calling_instance.inference_precision, devices
    )
    models, architecture_configs, maybe_bardist, inference_config = (
        initialize_tabpfn_model(
            model_path=calling_instance.model_path,  # pyright: ignore[reportArgumentType]
            which=model_type,
            softmax_temperature_override=overrides["SOFTMAX_TEMPERATURE"],
            n_estimators_override=overrides["N_ESTIMATORS"],
            average_before_softmax_override=overrides["AVERAGE_BEFORE_SOFTMAX"],
            devices=devices,
            force_inference_dtype=forced_inference_dtype,
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
    # Only the `softmax_temperature` and `average_before_softmax` arguments still have
    # to be applied here; an override that came from `user_config` was applied by the
    # call above, and the two cannot both be given.
    if overrides["SOFTMAX_TEMPERATURE"] is not None:
        inference_config = dataclasses.replace(
            inference_config, SOFTMAX_TEMPERATURE=overrides["SOFTMAX_TEMPERATURE"]
        )
    if overrides["AVERAGE_BEFORE_SOFTMAX"] is not None:
        inference_config = dataclasses.replace(
            inference_config,
            AVERAGE_BEFORE_SOFTMAX=overrides["AVERAGE_BEFORE_SOFTMAX"],
        )

    # An `n_estimators` argument is deliberately *not* written back into the config.
    # `inference_config_` is what `save_tabpfn_model` persists into a checkpoint, so
    # it has to keep describing the model rather than this run's compute budget --
    # otherwise fine-tuning, which runs a handful of estimators on purpose, would
    # bake that handful into every checkpoint it writes. The argument still wins,
    # applied where the count is used (see `_initialize_dataset_preprocessing`).

    calling_instance.inference_config_ = inference_config
    calling_instance.softmax_temperature_ = inference_config.SOFTMAX_TEMPERATURE
    calling_instance.average_before_softmax_ = inference_config.AVERAGE_BEFORE_SOFTMAX

    return byte_size


def _resolve_overrides(
    estimator: TabPFNClassifier | TabPFNRegressor,
    user_config: dict | InferenceConfig | None,
) -> dict[str, float | int | None]:
    """What the user asked for per overridable field, None where they asked nothing.

    Each of `OVERRIDABLE_FIELDS` can be named by its estimator argument or through
    `inference_config`, and either wins over what the checkpoint declares. Naming one
    both ways is rejected rather than resolved, since the winner would not be
    apparent from the call.

    Raises:
        ValueError: If a field is named by its argument and through
            `inference_config` at the same time.
    """
    overrides: dict[str, float | int | None] = {}
    for field, (_, argument) in OVERRIDABLE_FIELDS.items():
        from_config: float | int | None = None
        if isinstance(user_config, InferenceConfig):
            # A hand-built config replaces the checkpoint's wholesale, so it carries
            # a value for every field, this one included.
            from_config = getattr(user_config, field)
        elif isinstance(user_config, dict) and field in user_config:
            from_config = user_config[field]
        if from_config == "auto":
            # "auto" is the absence of a choice, wherever it comes from.
            from_config = None

        from_argument = getattr(estimator, argument)
        if from_argument == "auto":
            overrides[field] = from_config
            continue

        if from_config is not None:
            raise ValueError(
                f"`{argument}` was given twice: `{argument}={from_argument}` and "
                f"`inference_config` with {field}={from_config}. Pass it one way or "
                f"the other, so which one applies is unambiguous."
            )
        overrides[field] = from_argument

    return overrides


def resolved_n_estimators(
    estimator: TabPFNClassifier | TabPFNRegressor,
) -> int | Literal["auto"]:
    """How many estimators `estimator` should run, before feature-coverage scaling.

    The `n_estimators` argument wins; left at `"auto"` the count comes from the
    checkpoint's `InferenceConfig.N_ESTIMATORS`, which may itself be `"auto"`. Both
    mean the same thing, so the result is passed straight to
    `scale_n_estimators_for_feature_coverage`: an int is used as given, `"auto"`
    resolves to `DEFAULT_N_ESTIMATORS` and may be raised for feature coverage.
    """
    if estimator.n_estimators != "auto":
        return estimator.n_estimators
    return estimator.inference_config_.N_ESTIMATORS


def resolved_average_before_softmax(
    estimator: TabPFNClassifier | TabPFNRegressor,
) -> bool:
    """Whether `estimator` averages the estimators' logits before the softmax.

    Reads the value resolved by `initialize_model_variables_helper`, falling back to
    the unresolved argument for estimators that were never initialized or were pickled
    before that resolution existed, where `"auto"` means averaging probabilities.
    """
    value = getattr(
        estimator, "average_before_softmax_", estimator.average_before_softmax
    )
    return False if value == "auto" else bool(value)


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

    check_input_shape_matches(X, estimator=model)
    X, _ = transform_columns(model, X)
    X = clean_predict_input(model, X)

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
