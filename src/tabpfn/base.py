"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import dataclasses
import functools
import pathlib
import typing
import warnings
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
    DELETING_COLUMNS_COPIES,
    ColumnRangeEncoders,
    check_fit_input,
    clean_data,
    clean_data_transform,
    coerce_nullable_dtypes_to_numpy,
    column_range,
    delete_columns,
    empty_fit_frame,
    is_private_copy,
    mark_fit_memory,
    may_overwrite_fit_input,
    may_write_fit_memory,
    narrow_fit_input_in_place,
    overwriting_fit_input,
    release_rows,
)
from tabpfn.preprocessing.configs import (
    FeatureSubsamplingMethod,
    SampleSubsamplingMethod,
)
from tabpfn.preprocessing.datamodel import (
    FeatureModality,
    FeatureSchema,
    build_input_feature_names,
)
from tabpfn.preprocessing.datetimes import DateTransformer
from tabpfn.preprocessing.ensemble import (
    draw_fit_rows,
    scale_n_estimators_for_feature_coverage,
)
from tabpfn.preprocessing.group_ids import as_group_feature
from tabpfn.preprocessing.modality_detection import (
    named_feature_modalities,
    warn_on_text,
)
from tabpfn.preprocessing.pipeline_factory import pipelines_keep_float32
from tabpfn.preprocessing.source_decisions import SourceDecisions
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
    from collections.abc import Callable, Iterator

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
    kv_cache_on_host: bool | None = None,
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
        kv_cache_on_host: Only for ``fit_mode="fit_with_cache"``. Whether the KV
            cache keeps its keys and values in host memory, from where a predict
            copies to the device only what it reads. ``None`` (default) lets the
            architecture decide; an architecture whose cache cannot be read from
            host memory ignores it.
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
            kv_cache_on_host=kv_cache_on_host,
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
    """The dtype of the cleaned fit table: float32 if every member reads float32.

    A member reads float32 when the model takes float32 and the member's CPU pipeline
    only selects, reorders, compares and encodes values. Its values then reach the
    model as they would from a float64 table. Its decisions can still differ, because
    different values can round to the same float32. `SourceDecisions` takes those
    decisions before the cast to float32.
    """
    if estimator.tuning_config is not None:
        # Tuning's holdout models fit their members on this table without
        # `SourceDecisions`, so they take their decisions on its values.
        return np.dtype(np.float64)
    config = estimator.inference_config_
    reads_float32 = estimator.forced_inference_dtype_ in (
        None,
        torch.float32,
    ) and pipelines_keep_float32(
        config.PREPROCESS_TRANSFORMS,
        polynomial_features=config.POLYNOMIAL_FEATURES,
        add_fingerprint_feature=config.FINGERPRINT_FEATURE,
        enable_gpu_preprocessing=config.ENABLE_GPU_PREPROCESSING,
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
    random_state: int | np.random.Generator,
    may_cast_to_float32: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str | None,
    FeatureSchema,
    OrderPreservingColumnTransformer | ColumnRangeEncoders,
    SourceDecisions | None,
]:
    """Validate the expanded fit input, decide its modalities and clean it.

    Args:
        estimator: The estimator being fit.
        X: The fit input after date, text and group expansion.
        y: The targets.
        X_given: The input the caller passed to `fit`.
        feature_names: `X`'s column labels, `None` for an array.
        categorical_indices: Positions in `X` declared categorical.
        group_feature_index: Position of the group feature. It is cleaned as
            numerical and then marked categorical.
        ensure_y_numeric: Whether the targets must be numeric.
        random_state: The seed the ensemble preprocessor draws its rows with.
        may_cast_to_float32: Whether the caller passes the returned decisions on to
            the ensemble. If not, the table is float64.

    Returns:
        The cleaned table, the validated targets, the targets' name if `y` is a
        `Series`, the feature schema, the fitted encoders that `clean_predict_input`
        applies again, and the ensemble's decisions taken before the cast to float32
        (`None` if the input was not cast).
    """
    config = estimator.inference_config_
    dtype = (
        cleaned_table_dtype(estimator) if may_cast_to_float32 else np.dtype(np.float64)
    )
    if dtype == np.float32 and _rows_of(X) > _FLOAT32_EXACT_INTEGERS:
        # Category and group codes count up from 0 and are fewer than the rows. So
        # only an input with more rows can have a code that float32 cannot hold.
        dtype = np.dtype(np.float64)
    min_samples = config.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE
    thresholds = {
        "min_samples_for_inference": min_samples,
        "max_unique_for_category": config.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
        "min_unique_for_numerical": config.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        "min_cardinality_for_text": config.MIN_CARDINALITY_FOR_TEXT,
    }
    passthrough_inf = estimator.get_inference_config().PASSTHROUGH_INF
    check_fit_input(X_given)
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
        feature_schema = named_feature_modalities(
            X,
            build_input_feature_names(feature_names, X.shape[1]),
            provided_categorical_indices=categorical_indices,
            **thresholds,
        )
        # Warn from here, as the range path does, so that the warning's `stacklevel`
        # points at the caller's `fit` in both paths.
        warn_on_text(feature_schema)
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
        decisions = None
        if X.dtype != dtype and dtype == np.float32:
            decisions = _source_decisions(estimator, y, X.shape, random_state)
            decisions.take(0, X)
            X = X.astype(np.float32)
        if not is_private_copy(X, X_given):
            # The fitted model can read the caller's memory without a copy, so no
            # later fit may write into it.
            mark_fit_memory(X, overwritten=False)
        return (
            X,
            y,
            y_name,
            as_group_feature(
                feature_schema, group_feature_index, FeatureModality.CATEGORICAL
            ),
            encoder,
            decisions,
        )

    y_name = str(y.name) if isinstance(y, pd.Series) else None
    _n_rows, n_columns = X.shape
    cleaner = _RangeCleaner(
        estimator=estimator,
        X=X,
        names=build_input_feature_names(feature_names, n_columns),
        declared=sorted(categorical_indices or ()),
        group_feature_index=group_feature_index,
        thresholds=thresholds,
        passthrough_inf=passthrough_inf,
        check_finite=_whole_table_checks_finite(X),
    )
    # Validate the targets before the fit empties or overwrites the input, so that a
    # fit that fails here leaves the input unchanged.
    y = cleaner.validate_targets(y, ensure_y_numeric=ensure_y_numeric)
    decisions = (
        _source_decisions(estimator, y, X.shape, random_state)
        if dtype == np.float32 and _has_wider_columns(X)
        else None
    )
    table, ranges, feature_schema = _clean_into_table(
        dataclasses.replace(cleaner, decisions=decisions), X_given, dtype
    )
    warn_on_text(feature_schema)
    return (
        table,
        y,
        y_name,
        as_group_feature(
            feature_schema, group_feature_index, FeatureModality.CATEGORICAL
        ),
        ColumnRangeEncoders(ranges=ranges, dtype=dtype),
        decisions,
    )


@dataclasses.dataclass
class _RangeCleaner:
    """Validates the fit input, detects its modalities and cleans it, range by range."""

    estimator: TabPFNClassifier | TabPFNRegressor
    X: pd.DataFrame | np.ndarray
    names: list[str]
    declared: list[int]
    group_feature_index: int | None
    thresholds: dict[str, int]
    passthrough_inf: bool
    check_finite: bool
    decisions: SourceDecisions | None = None
    """The ensemble's decisions, taken on each range before the cast to float32."""

    def validate_targets(self, y: YType, *, ensure_y_numeric: bool) -> np.ndarray:
        """Validate `y` against the input and check the size limits."""
        config = self.estimator.inference_config_
        # Validate a numeric array whole and in place, because a range of it is a
        # strided view that is slow to read. Validation converts a frame, so validate
        # only its first column here, with the targets.
        _, y = ensure_compatible_fit_inputs_sklearn(
            self.X if isinstance(self.X, np.ndarray) else self.X.iloc[:, :1],
            y,
            estimator=self.estimator,
            ensure_y_numeric=ensure_y_numeric,
            check_finite=self.check_finite,
        )
        validate_dataset_size(
            X=self.X,
            y=y,
            max_num_samples=config.MAX_NUMBER_OF_SAMPLES,
            max_num_features=config.MAX_NUMBER_OF_FEATURES,
            max_cpu_samples=config.MAX_CPU_SAMPLES,
            devices=self.estimator.devices_,
            ignore_pretraining_limits=self.estimator.ignore_pretraining_limits,
        )
        return np.asarray(y)

    def clean(
        self, part: XType, start: int, stop: int
    ) -> tuple[np.ndarray, OrderPreservingColumnTransformer, list]:
        """Clean `part`, the columns `start:stop`.

        Returns:
            The cleaned values, the fitted encoder and the features.
        """
        if not isinstance(self.X, np.ndarray):
            part = ensure_compatible_fit_columns(
                part, estimator=self.estimator, check_finite=self.check_finite
            )
        schema = named_feature_modalities(
            part,
            self.names[start:stop],
            provided_categorical_indices=[
                i - start for i in self.declared if start <= i < stop
            ],
            **self.thresholds,
        )
        group = self.group_feature_index
        if group is not None and start <= group < stop:
            schema = as_group_feature(schema, group - start, FeatureModality.NUMERICAL)
        # `part` is a copy or a view of the caller's data. Cleaning does not write
        # into it.
        cleaned, encoder, schema = clean_data(
            X=part, feature_schema=schema, passthrough_inf=self.passthrough_inf
        )
        if self.decisions is not None:
            self.decisions.take(start, cleaned)
        return cleaned, encoder, schema.features


def _clean_into_table(
    cleaner: _RangeCleaner, X_given: XType, dtype: np.dtype
) -> tuple[
    np.ndarray,
    list[tuple[int, int, OrderPreservingColumnTransformer]],
    FeatureSchema,
]:
    """Clean the fit input into a table of `dtype`, one range of columns at a time.

    Inside `overwrite_fit_input()`, a wider array is cast to float32 in its own memory
    to become the table, and a frame is freed as its columns are copied into the table.

    Returns:
        The table, the fitted encoder of each range and the schema.
    """
    X = cleaner.X
    n_rows, n_columns = X.shape
    if isinstance(X, np.ndarray):
        ranges, feature_schema, encoded = _clean_array_ranges(cleaner)
        # Cleaning keeps every value of a range that encodes no column. So the table
        # is `X` cast to `dtype`, with the encoded ranges written over it. If no range
        # is encoded and the caller gave `X` up, the cast is in the memory of `X`.
        # Casting reads `X` row by row instead of in strided ranges.
        table = (
            narrow_fit_input_in_place(X)
            if not encoded and _narrows_in_place(X, X_given, dtype)
            else X.astype(dtype)
        )
        for start, stop, cleaned in encoded:
            table[:, start:stop] = cleaned
        return table, ranges, feature_schema
    given_up = overwriting_fit_input() and isinstance(X_given, pd.DataFrame)
    if given_up:
        # Empty the caller's frame now, so that a fit that fails later leaves it
        # empty, not partly cleaned. `cleaner.X` keeps the blocks until each is
        # copied into the table.
        if X is X_given:
            cleaner = dataclasses.replace(cleaner, X=X.copy(deep=False))
        empty_fit_frame(X_given)
    table = np.empty((n_rows, n_columns), dtype=dtype)
    ranges, feature_schema = _clean_ranges(
        cleaner, table, release=given_up and not DELETING_COLUMNS_COPIES
    )
    return table, ranges, feature_schema


def _rows_of(X: XType) -> int:
    """The number of rows of `X`, without converting it.

    An object without rows (e.g. sklearn's `_NotAnArray`, a 0-d array) counts as 0:
    it is small enough to convert, and its conversion and validation come later.
    """
    if hasattr(X, "shape"):
        return int(X.shape[0]) if len(X.shape) > 0 else 0
    return len(X) if hasattr(X, "__len__") else 0


def _has_wider_columns(X: pd.DataFrame | np.ndarray) -> bool:
    """Whether `X` has a column whose dtype is not float32."""
    dtypes = X.dtypes if isinstance(X, pd.DataFrame) else [X.dtype]
    return any(dtype != np.float32 for dtype in dtypes)


_FLOAT32_EXACT_INTEGERS = 2**24
"""Float32 holds every integer up to this one exactly."""


def _source_decisions(
    estimator: TabPFNClassifier | TabPFNRegressor,
    y: np.ndarray,
    shape: tuple[int, int],
    random_state: int | np.random.Generator,
) -> SourceDecisions:
    """The decisions, not yet taken, for the rows that the ensemble of `estimator`
    draws with `random_state` from a cleaned fit input of `shape`.
    """
    config = estimator.inference_config_
    n_rows, n_columns = shape
    with warnings.catch_warnings():
        # `fit` warns about the ensemble size later, when it sets the size.
        warnings.simplefilter("ignore")
        n_members = scale_n_estimators_for_feature_coverage(
            n_estimators=resolved_n_estimators(estimator),
            n_total_features=n_columns,
            preprocessor_configs=config.PREPROCESS_TRANSFORMS,
            auto_scale_n_estimators=estimator.auto_scale_n_estimators,
        )
    member_rows, importance_rows = draw_fit_rows(
        random_state=random_state,
        n_members=n_members,
        n_samples=n_rows,
        n_features=n_columns,
        y=np.asarray(y),
        task_type=estimator.estimator_type,
        max_features_per_estimator=[
            c.max_features_per_estimator for c in config.PREPROCESS_TRANSFORMS
        ],
        feature_subsampling_method=FeatureSubsamplingMethod(
            config.FEATURE_SUBSAMPLING_METHOD
        ),
        importance_top_k_count=config.FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT,
        subsample_samples=config.SUBSAMPLE_SAMPLES,
        sample_subsampling_method=SampleSubsamplingMethod(
            config.SAMPLE_SUBSAMPLING_METHOD
        ),
    )
    return SourceDecisions.for_rows(member_rows, importance_rows, n_columns)


def _clean_array_ranges(
    cleaner: _RangeCleaner,
) -> tuple[
    list[tuple[int, int, OrderPreservingColumnTransformer]],
    FeatureSchema,
    list[tuple[int, int, np.ndarray]],
]:
    """Clean a numeric array one range of columns at a time.

    Returns:
        The fitted encoder of each range, the schema, and the cleaned values of the
        ranges that encode a column. Cleaning does not change the other ranges.
    """
    ranges, features, encoded = [], [], []
    for start, stop in _column_ranges(cleaner.X):
        cleaned, encoder, range_features = cleaner.clean(
            column_range(cleaner.X, start, stop), start, stop
        )
        ranges.append((start, stop, encoder))
        features += range_features
        if encoder.selected_columns():
            encoded.append((start, stop, cleaned))
    return ranges, FeatureSchema(features=features), encoded


def _clean_ranges(
    cleaner: _RangeCleaner, table: np.ndarray, *, release: bool
) -> tuple[list[tuple[int, int, OrderPreservingColumnTransformer]], FeatureSchema]:
    """Clean the fit input, a frame, into `table` one range of columns at a time.

    `release` is for a frame that the caller gave up. Then the order is:

    1. If `_StreamedBlock.of` accepts the largest block, clean its ranges. Copy the
       block into `table` a slice of rows at a time, and release each slice after
       its copy.
    2. Clean the ranges of the other blocks, smallest block first.

    A block is deleted from the frame once its last range is cleaned, which frees
    its memory.

    Returns:
        The fitted encoder of each range and the schema.
    """
    X = cleaner.X
    streamed = _StreamedBlock.of(X) if release else None
    # The streamed block's columns get ranges of their own.
    order = list(
        _column_ranges(X, apart=None if streamed is None else streamed.positions)
    )
    if release:
        order.sort(key=_largest_block_of(X))
        # Label the columns by position. These labels do not change when columns are
        # deleted.
        X.columns = pd.RangeIndex(X.shape[1])
        blocks = [block.mgr_locs.as_array for block in X._mgr.blocks]
        block_of = np.empty(X.shape[1], dtype=np.intp)
        for number, positions in enumerate(blocks):
            block_of[positions] = number
        left = np.array([len(positions) for positions in blocks])
    by_range: dict[int, tuple[int, OrderPreservingColumnTransformer, list]] = {}

    def clean(start: int, stop: int) -> tuple[np.ndarray, bool]:
        part = X[list(range(start, stop))] if release else column_range(X, start, stop)
        cleaned, encoder, features = cleaner.clean(part, start, stop)
        by_range[start] = (stop, encoder, features)
        return cleaned, bool(encoder.selected_columns())

    if streamed is not None:
        order = _stream(streamed, order, clean, table, X)
    for start, stop in order:
        table[:, start:stop], _ = clean(start, stop)
        if release:
            np.subtract.at(left, block_of[start:stop], 1)
            for number in np.unique(block_of[start:stop]):
                if left[number] == 0:
                    _delete_columns(X, blocks[number])
    ranges = sorted(by_range.items())
    return (
        [(start, stop, encoder) for start, (stop, encoder, _) in ranges],
        FeatureSchema(features=[f for _, (_, _, fs) in ranges for f in fs]),
    )


def _stream(
    streamed: _StreamedBlock,
    order: list[tuple[int, int]],
    clean: Callable[[int, int], tuple[np.ndarray, bool]],
    table: np.ndarray,
    X: pd.DataFrame,
) -> list[tuple[int, int]]:
    """Clean the ranges of `streamed`, copy it into `table` and delete it from `X`.

    The ranges are cleaned before any write to `table`. A write to a range of
    columns touches every page of `table`, so the system then holds the whole table
    in memory. Only the ranges that encode a column are written again, over the copy.

    Returns:
        The ranges of `order` that are not in `streamed`.
    """
    encoded = []
    for start, stop in [r for r in order if streamed.holds(*r)]:
        cleaned, encodes = clean(start, stop)
        if encodes:
            encoded.append((start, stop, cleaned))
    streamed.copy_into(table)
    for start, stop, cleaned in encoded:
        table[:, start:stop] = cleaned
    _delete_columns(X, streamed.positions)
    return [r for r in order if not streamed.holds(*r)]


def _delete_columns(X: pd.DataFrame, labels: np.ndarray) -> None:
    """Delete the columns `labels` from `X` in one call.

    Deleting part of a block splits it into one block per column, and each later
    delete then scans every block. Deleting whole blocks leaves the others as they
    are.
    """
    positions = X.columns.get_indexer(labels)
    # -1 marks a missing label, which `idelete` would take as the last column.
    if (positions < 0).any():
        raise RuntimeError(f"Columns {labels} are not all in the frame.")
    delete_columns(X, positions)


def _largest_block_of(X: pd.DataFrame) -> Callable[[tuple[int, int]], int]:
    """A sort key: for a range of columns of `X`, the width of the largest block that
    holds one of its columns.
    """
    sizes = np.zeros(X.shape[1], dtype=np.intp)
    for block in X._mgr.blocks:
        positions = block.mgr_locs.as_array
        sizes[positions] = len(positions)
    return lambda columns: int(sizes[columns[0] : columns[1]].max())


def clean_predict_input(
    estimator: TabPFNClassifier | TabPFNRegressor, X: XType
) -> np.ndarray:
    """Validate and clean predict input the way `validate_and_clean_fit_input` did
    the fit input.
    """
    cat_indices = estimator.inferred_feature_schema_.indices_for(
        FeatureModality.CATEGORICAL
    )
    encoder = getattr(estimator, "ordinal_encoder_", None)
    passthrough_inf = estimator.get_inference_config().PASSTHROUGH_INF
    validate = functools.partial(
        ensure_compatible_predict_input_sklearn,
        estimator=estimator,
        check_finite=_whole_table_checks_finite(X),
    )
    if isinstance(encoder, ColumnRangeEncoders):
        return encoder.transform(
            X,
            cat_indices=cat_indices,
            passthrough_inf=passthrough_inf,
            validate=validate,
            # Each range adds a fixed cost to validation, so an input whose converted
            # copy fits in one range validates its ranges together.
            validate_runs=_converted_bytes(X) <= _RANGE_BYTES,
        )
    return clean_data_transform(
        X,
        cat_indices=cat_indices,
        ord_encoder=encoder,
        passthrough_inf=passthrough_inf,
        validate=validate,
    )


_RANGE_BYTES = 1 << 30
"""Maximum temporary bytes that one range of columns uses while it is validated and
cleaned. Wider ranges use more memory. Narrower ranges repeat the per-range validation
and encoder fit more often."""

_NUMERIC_CELL_BYTES = 32
"""Bytes that one numeric cell can use while its range is cleaned. This allows for a
few copies, the widest of them float64."""

_OBJECT_CELL_BYTES = 256
"""Bytes that one non-numeric cell can use while its range is cleaned. Such a cell
goes through Python objects and a few frames."""


def _cleans_by_column_ranges(X: XType, dtype: np.dtype) -> bool:
    """Whether to validate and clean the fit input one range of columns at a time.

    This is the case when reading the input whole would make a converted copy of all
    of it:

    - a frame with more than one block, unless all are float32, or with one
      non-numeric block;
    - a numeric frame or array whose dtype is not float32, for a float32 table.

    Other numeric frames and arrays are read without a copy. A frame of float32
    blocks is read into one float32 copy, which cleaning keeps as the table. An input
    with no columns goes to the whole-table validation, which refuses it. An input
    whose converted copy fits in one range is read whole.
    """
    if _converted_bytes(X) <= _RANGE_BYTES:
        # Each range adds a fixed cost to validation and cleaning. This also sends
        # an input with no columns, and anything not a 2-D frame or array, to the
        # whole-table validation.
        return False
    if isinstance(X, pd.DataFrame):
        # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
        arrays = [block.values for block in X._mgr.blocks]  # noqa: PD011
        if len(arrays) != 1:
            return not all(
                isinstance(values, np.ndarray) and values.dtype == np.float32
                for values in arrays
            )
        values = arrays[0]
        if not isinstance(values, np.ndarray) or values.dtype.kind not in "biuf":
            return True
        return dtype == np.float32 and values.dtype != np.float32
    return dtype == np.float32 and X.dtype.kind in "biuf" and X.dtype != np.float32


def _whole_table_checks_finite(X: XType) -> bool:
    """Whether the whole-table validation refuses infinities in `X`, so validation
    by ranges must refuse them too.

    It reads a frame as one array, and `check_array` refuses infinities only in a
    numeric array, not in an array of objects.
    """
    return not isinstance(X, pd.DataFrame) or _interleaved_dtype(X).kind != "O"


def _interleaved_dtype(X: pd.DataFrame) -> np.dtype:
    """The dtype of the one array that `check_array` makes from all columns of `X`."""
    return coerce_nullable_dtypes_to_numpy(X.iloc[:0]).to_numpy().dtype


def _numeric_columns(X: pd.DataFrame | np.ndarray) -> list[bool]:
    """Whether each of `X`'s columns is numeric."""
    if isinstance(X, pd.DataFrame):
        return [getattr(dtype, "kind", "O") in "biuf" for dtype in X.dtypes]
    return [True] * X.shape[1]


def _converted_bytes(X: XType) -> int:
    """The bytes that reading all of `X` at once can use, as `_column_ranges` counts."""
    if not isinstance(X, (pd.DataFrame, np.ndarray)) or X.ndim != 2:
        return 0
    numeric = sum(_numeric_columns(X))
    cell_bytes = numeric * _NUMERIC_CELL_BYTES
    cell_bytes += (X.shape[1] - numeric) * _OBJECT_CELL_BYTES
    return X.shape[0] * cell_bytes


def _narrows_in_place(X: np.ndarray, X_given: XType, dtype: np.dtype) -> bool:
    """Whether the float32 table may be `X`, a numeric array cleaned by ranges, cast
    in its own memory.

    This needs a C-ordered `X` with at least four bytes per value, which
    `may_overwrite_fit_input` lets the fit write into.
    """
    return (
        dtype == np.float32
        and X.dtype.itemsize >= 4
        and X.flags.c_contiguous
        and may_overwrite_fit_input(X, X_given)
    )


def _column_ranges(
    X: pd.DataFrame | np.ndarray, *, apart: np.ndarray | None = None
) -> Iterator[tuple[int, int]]:
    """`(start, stop)` ranges that cover the columns of `X`.

    Each range is all numeric or all non-numeric, and as wide as `_RANGE_BYTES` allows.
    No range mixes columns in `apart` with columns not in `apart`.
    """
    n_rows, n_columns = X.shape
    numerics = _numeric_columns(X)
    in_apart = np.zeros(n_columns, dtype=bool)
    if apart is not None:
        in_apart[apart] = True
    kind = list(zip(numerics, in_apart.tolist(), strict=True))
    start = 0
    while start < n_columns:
        numeric = kind[start][0]
        cell_bytes = _NUMERIC_CELL_BYTES if numeric else _OBJECT_CELL_BYTES
        width = max(1, _RANGE_BYTES // max(1, n_rows * cell_bytes))
        stop = start + 1
        while stop < n_columns and stop - start < width and kind[stop] == kind[start]:
            stop += 1
        yield start, stop
        start = stop


@dataclasses.dataclass
class _StreamedBlock:
    """The largest block of a given-up frame, copied into the table a slice of rows at
    a time.

    Each slice is released after its copy, so the block and the whole table are never
    in memory together.
    """

    rows: np.ndarray
    """The block's values, `(rows, columns)`, C-ordered."""
    positions: np.ndarray
    """The position of each of its columns in the frame and in the table."""

    @classmethod
    def of(cls, X: pd.DataFrame) -> _StreamedBlock | None:
        """The largest block of `X`, if it is numeric and row-major, and its memory
        may be released.
        """
        block = max(X._mgr.blocks, key=lambda b: len(b.mgr_locs))
        # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
        values = block.values  # noqa: PD011
        if (
            not isinstance(values, np.ndarray)
            or values.dtype.kind not in "biuf"
            or not values.T.flags.c_contiguous
            or not may_write_fit_memory(values)
        ):
            return None
        # Mark before any page is released, so that a later fit on this memory is
        # refused.
        mark_fit_memory(values, overwritten=True)
        return cls(values.T, block.mgr_locs.as_array)

    def holds(self, start: int, stop: int) -> bool:
        """Whether the columns `start:stop` are this block's."""
        return bool(np.isin(np.arange(start, stop), self.positions).all())

    def copy_into(self, table: np.ndarray) -> None:
        """Copy the block into `table`, and release each slice of rows once copied."""
        # Copying into a slice of columns is much faster than into an index array.
        columns: slice | np.ndarray = (
            slice(int(self.positions[0]), int(self.positions[-1]) + 1)
            if bool((np.diff(self.positions) == 1).all())
            else self.positions
        )
        step = max(1, _STREAM_BLOCK_BYTES // max(1, self.rows.strides[0]))
        for start in range(0, len(self.rows), step):
            stop = min(start + step, len(self.rows))
            table[start:stop, columns] = self.rows[start:stop]
            release_rows(self.rows, start, stop)


_STREAM_BLOCK_BYTES = 1 << 28
"""Bytes of the slice of rows of a streamed block that is copied into the table at a
time. Each slice is released after its copy. Larger slices keep more memory in use
next to the table. Smaller slices pay the per-copy overhead more often."""


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
