"""Module that defines different ways to run inference with TabPFN."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import deepcopy
from functools import partial
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import override

import joblib
import torch

from tabpfn.architectures.base.memory import (
    DEFAULT_SAVE_PEAK_MEMORY_FACTOR,
    MemorySavingMode,
    should_save_peak_mem,
)
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.parallel_execute import parallel_execute
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.utils import get_autocast_context

if TYPE_CHECKING:
    import numpy as np

    from tabpfn.architectures.interface import Architecture
    from tabpfn.preprocessing import EnsembleConfig
    from tabpfn.preprocessing.ensemble import (
        TabPFNEnsembleMember,
        TabPFNEnsemblePreprocessor,
    )
    from tabpfn.preprocessing.torch import (
        FeatureSchema,
        TorchPreprocessingPipeline,
    )


_T = TypeVar("_T")


class _TimedIterator(Iterator[_T]):
    """Wraps an iterator, accumulating wall-clock time spent in ``__next__``."""

    def __init__(self, inner: Iterator[_T]) -> None:
        super().__init__()
        self._inner = inner
        self.elapsed_seconds: float = 0.0

    @override
    def __next__(self) -> _T:
        start = time.perf_counter()
        value = next(self._inner)
        self.elapsed_seconds += time.perf_counter() - start
        return value

    @override
    def __iter__(self) -> _TimedIterator[_T]:
        return self


def _model_expectes_task_type_arg(model: Architecture) -> bool:
    """Check if the model's forward function expects a task_type argument.

    This is a check for backwards compatibility.
    """
    return "task_type" in signature(model.forward).parameters


class InferenceEngine(ABC):
    """Base class defining how TabPFN inference can be run.

    As there are many things that can be cached, with multiple ways to parallelize,
    `InferenceEngine` defines three primary things:

    1. What to cache:

        As we can prepare a lot of the transformers context, there is a tradeoff in
        terms of how much memory to be spent in caching. This memory is used during
        initialization (in `__init__`), usually called from `fit()`.

    2. Using the cached data for inference:

        Based on what has been prepared for the transformer context,
        `iter_outputs()` will use this cached information to make predictions.

    3. Controlling parallelism:

        As we have trivially parallel parts for inference, we can parallelize them.
        However as the GPU is typically a bottle-neck in most systems, we can define,
        where and how we would like to parallelize the inference.

    The InferenceEngineBatchedNoPreprocessing and InferenceEngineCachePreprocessing
    engines also support toggling `torch.use_torch_inference_mode` via
    `use_torch_inference_mode` to enable/disable gradient tracking during prediction.
    """

    def __init__(
        self,
        *,
        save_peak_mem: MemorySavingMode,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
    ) -> None:
        """Initialize the inference engine.

        Args:
            save_peak_mem: Whether to save peak memory usage.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: If not None, inference will be performed using this
                dtype. Otherwise, the default dtype will be used.
        """
        super().__init__()
        self.save_peak_mem = save_peak_mem
        self.dtype_byte_size = dtype_byte_size
        self.force_inference_dtype = force_inference_dtype
        self._speed_metrics: dict[str, float] = {}

    @abstractmethod
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
        task_type: str,
    ) -> Iterator[tuple[torch.Tensor, EnsembleConfig]]:
        """Iterate over the outputs of the model for each ensemble configuration.

        Depending on the InferenceEngine used, this will run the forward pass of the
        model for each estimator.

        Args:
            X: The input data to make predictions on.
            autocast: Whether to use torch.autocast during inference.
            task_type: The task type, e.g. "multiclass" or "regression".
        """
        ...

    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
        """Enable/Disable `torch.inference_mode`.

        Disabling allows backpropagation (gradients) but is slower and uses more
        memory during prediction. Enabling is faster for pure inference.

        Only `InferenceEngineBatchedNoPreprocessing` and
        `InferenceEngineCachePreprocessing` currently support this method. Other
        engines will raise `NotImplementedError`.

        Called internally by methods like
        `TabPFNClassifier.predict_proba_from_preprocessed` (for batched engine) and
        `TabPFNRegressor.forward` (for batched & fit_preprocessors engines)
        when gradients might be needed (e.g., for fine-tuning) or when pure
        inference speed is desired.

        """
        raise NotImplementedError(
            "This inference engine does not support torch.inference_mode changes."
        )

    def save_state_except_model_weights(self, path: str | Path) -> None:
        """Persist the executor state to ``path`` without the model weights.

        This does not support the KV cache, and will raise an error if this is an
        InferenceEngineCacheKV.
        """
        _raise_if_kv_cache_enabled_on_save_or_load(self)
        joblib.dump(self._create_copy_for_pickling(), path)

    @abstractmethod
    def _create_copy_for_pickling(self) -> InferenceEngine:
        """Return a copy of the inference engine ready for pickling.

        This should remove the models, which we don't want to include. in the pickled
        file.
        """
        ...

    @staticmethod
    def load_state(path: str | Path, models: list[Architecture]) -> InferenceEngine:
        """Load an executor saved to disk with save_state_except_model_weights().

        The state on disk does not include the models, so these must be provided as the
        `models` parameter.
        """
        engine: InferenceEngine = joblib.load(Path(path))
        _raise_if_kv_cache_enabled_on_save_or_load(engine)
        engine._set_models(models)
        return engine

    @abstractmethod
    def _set_models(self, models: list[Architecture]) -> None:
        """Set the models in the inference engine.

        This is called, when the inference engine is unpickled from disk, to restore the
        models. These are not included in the pickled file.
        """
        ...

    def to(
        self,
        devices: Sequence[torch.device],
        force_inference_dtype: torch.dtype | None,
        dtype_byte_size: int,
    ) -> None:
        """Move the inference engine to the given set of devices.

        Args:
            devices: The devices to use.
            force_inference_dtype: The dtype to use for inference, as supported by the
                specified devices.
            dtype_byte_size: The size of the dtype in bytes.
        """
        self.force_inference_dtype = force_inference_dtype
        self.dtype_byte_size = dtype_byte_size
        self._move_models_to_devices(devices)

    @abstractmethod
    def _move_models_to_devices(self, devices: Sequence[torch.device]) -> None:
        """Move the models to the given devices. Used when .to() is called."""
        ...


def _raise_if_kv_cache_enabled_on_save_or_load(engine: InferenceEngine) -> None:
    if isinstance(engine, (InferenceEngineCacheKV, InferenceEngineExplicitKVCache)):
        raise NotImplementedError(
            "Saving and loading fitted models that use "
            '`fit_mode="fit_with_cache"` is not currently supported.'
        )


class SingleDeviceInferenceEngine(InferenceEngine):
    """Inference engine that uses a single device to execute the model."""

    def __init__(
        self,
        *,
        models: list[Architecture],
        save_peak_mem: MemorySavingMode,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
    ) -> None:
        """Initialize the single device inference engine.

        Args:
            models: The models to use for inference.
            save_peak_mem: Whether to save peak memory usage.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: If not None, inference will be performed using this
                dtype. Otherwise, the default dtype will be used.
        """
        super().__init__(
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )
        self.models = models

    @override
    def _create_copy_for_pickling(self) -> InferenceEngine:
        state_copy = deepcopy(self)
        state_copy.models = None  # type: ignore
        return state_copy

    @override
    def _set_models(self, models: list[Architecture]) -> None:
        self.models = models


class MultiDeviceInferenceEngine(InferenceEngine):
    """Inference engine that parallelizes the members of the ensemble across devices."""

    def __init__(
        self,
        *,
        model_caches: list[_PerDeviceModelCache],
        save_peak_mem: MemorySavingMode,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
    ) -> None:
        """Initialize the multi-device inference engine.

        Args:
            model_caches: Per-device model caches for each model.
            save_peak_mem: Whether to save peak memory usage.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: If not None, inference will be performed using this
                dtype. Otherwise, the default dtype will be used.
        """
        super().__init__(
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )
        self.model_caches = model_caches

    @override
    def _create_copy_for_pickling(self) -> InferenceEngine:
        state_copy = deepcopy(self)
        state_copy.model_caches = None  # type: ignore
        return state_copy

    @override
    def _set_models(self, models: list[Architecture]) -> None:
        self.model_caches = [_PerDeviceModelCache(model) for model in models]

    @override
    def _move_models_to_devices(self, devices: Sequence[torch.device]) -> None:
        for model_cache in self.model_caches:
            model_cache.to(devices)

    def get_devices(self) -> list[torch.device]:
        """Return the devices that the models are on."""
        # We always keep all the models on the same set of devices, so this is safe.
        return self.model_caches[0].get_devices()


class InferenceEngineOnDemand(MultiDeviceInferenceEngine):
    """Inference engine that does not cache anything, computes everything as needed.

    This is one of the slowest ways to run inference, as computation that could be
    cached is recomputed on every call. However the memory demand is lowest and
    can be more trivially parallelized across GPUs with some work.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        ensemble_preprocessor: TabPFNEnsemblePreprocessor,
        models: list[Architecture],
        devices: Sequence[torch.device],
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: MemorySavingMode,
    ) -> None:
        """Initialize the on-demand inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            ensemble_preprocessor: The ensemble preprocessor to use.
            models: The models to use.
            devices: A list of the devices to use for inference. If multiple devices are
                specified, then the inference engine will parallelize the members of the
                ensemble across the devices.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
        """
        super().__init__(
            model_caches=[_PerDeviceModelCache(model) for model in models],
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.X_train = X_train
        self.y_train = y_train
        self.ensemble_preprocessor = ensemble_preprocessor

        self.to(devices, self.force_inference_dtype, self.dtype_byte_size)

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
        task_type: str,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        devices = self.get_devices()

        save_peak_mem = should_save_peak_mem(
            memory_saving_mode=self.save_peak_mem,
            X_train_shape=self.X_train.shape,
            X_test_shape=X.shape,
            devices=devices,
            dtype_byte_size=self.dtype_byte_size,
        )

        if self.force_inference_dtype is not None:
            for model_cache in self.model_caches:
                model_cache.set_dtype(self.force_inference_dtype)

        ensemble_members_iterator = (
            self.ensemble_preprocessor.fit_transform_ensemble_members_iterator(
                X_train=self.X_train,
                y_train=self.y_train,
                parallel_mode="in-order",
            )
        )

        model_forward_functions = (
            partial(
                self._call_model,
                X_train=em.X_train,
                X_test=em.transform_X_test(X),
                y_train=em.y_train,
                feature_schema=em.feature_schema,
                only_return_standard_out=only_return_standard_out,
                autocast=autocast,
                model_index=em.config._model_index,
                save_peak_mem=save_peak_mem,
                gpu_preprocessor=em.gpu_preprocessor,
                task_type=task_type,
            )
            for em in ensemble_members_iterator
        )

        timed_outputs = _TimedIterator(
            parallel_execute(devices, model_forward_functions)
        )

        for config, output in zip(self.ensemble_preprocessor.configs, timed_outputs):
            yield _move_and_squeeze_output(output, devices[0]), config

        self._speed_metrics["predict_model_forward_seconds"] = (
            timed_outputs.elapsed_seconds
        )

    def _call_model(  # noqa: PLR0913
        self,
        *,
        device: torch.device,
        X_train: torch.Tensor | np.ndarray,
        X_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        feature_schema: FeatureSchema,
        autocast: bool,
        only_return_standard_out: bool,
        model_index: int,
        save_peak_mem: bool,
        gpu_preprocessor: TorchPreprocessingPipeline | None,
        task_type: str,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        model = self.model_caches[model_index].get(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )

        performance_options = model.get_default_performance_options()
        performance_options = dataclasses.replace(
            performance_options,
            save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
            if save_peak_mem
            else None,
        )

        X_full, feature_schema = _maybe_run_gpu_preprocessing(
            X_full,
            gpu_preprocessor=gpu_preprocessor,
            num_train_rows=X_train.shape[0],
            feature_schema=feature_schema,
        )
        batched_cat_ix = [feature_schema.indices_for(FeatureModality.CATEGORICAL)]

        kwargs = {}
        if _model_expectes_task_type_arg(model):
            kwargs["task_type"] = task_type

        with get_autocast_context(device, enabled=autocast), torch.inference_mode():
            return model(
                X_full,
                y_train,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=batched_cat_ix,
                performance_options=performance_options,
                **kwargs,
            )


class InferenceEngineBatchedNoPreprocessing(SingleDeviceInferenceEngine):
    """Inference engine that uses preprocessed inputs, and allows batched predictions
    on several datasets at once.
    """

    def __init__(
        self,
        X_trains: list[torch.Tensor],
        y_trains: list[torch.Tensor],
        *,
        feature_schema: list[list[FeatureSchema]],
        ensemble_configs: list[list[EnsembleConfig]],
        models: list[Architecture],
        devices: Sequence[torch.device],
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: MemorySavingMode,
        inference_mode: bool,
    ) -> None:
        """Initialize the batched inference engine without preprocessing.

        Args:
            X_trains: The training data.
            y_trains: The training target.
            feature_schema: The feature schema.
            models: The models to use.
            devices: A list of devices, the first of which will be used to run the
                model. The other devices will be ignored.
            ensemble_configs: The ensemble configurations to use.
            inference_mode: Whether to use torch inference mode.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
        """
        for ensemble_config in ensemble_configs:
            if len(ensemble_config) > 1:
                raise ValueError(
                    "Batched inference does not support multiple ensemble"
                    " configurations because no preprocessing is applied."
                )

        super().__init__(
            models=models,
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.X_trains = X_trains
        self.y_trains = y_trains
        self.feature_schema_list = feature_schema
        self.ensemble_configs = ensemble_configs
        self.inference_mode = inference_mode

        self.to(devices, self.force_inference_dtype, self.dtype_byte_size)

    @override
    def iter_outputs(
        self,
        X: list[torch.Tensor],
        *,
        autocast: bool,
        task_type: str,
    ) -> Iterator[tuple[torch.Tensor | dict, list[EnsembleConfig]]]:
        device = _get_current_device(self.models[0])
        batch_size = len(self.X_trains)
        forward_time = 0.0
        for i in range(batch_size):
            train_x_full = torch.cat([self.X_trains[i], X[i]], dim=-2)
            train_y_batch = self.y_trains[i]
            train_x_full = train_x_full.to(device)
            train_y_batch = train_y_batch.to(device)
            if self.force_inference_dtype is not None:
                train_x_full = train_x_full.type(self.force_inference_dtype)
                train_y_batch = train_y_batch.type(self.force_inference_dtype)  # type: ignore

            model = self.models[self.ensemble_configs[i][0]._model_index]
            kwargs = {}
            if _model_expectes_task_type_arg(model):
                kwargs["task_type"] = task_type
            forward_start = time.perf_counter()
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                output = model(
                    train_x_full.transpose(0, 1),
                    train_y_batch.transpose(0, 1),
                    only_return_standard_out=True,
                    categorical_inds=list(  # noqa: C411
                        [
                            cat_item[i].indices_for(FeatureModality.CATEGORICAL)
                            for cat_item in self.feature_schema_list
                        ]
                    ),
                    **kwargs,
                )
            forward_time += time.perf_counter() - forward_start

            yield output, self.ensemble_configs[i]

        self._speed_metrics["predict_model_forward_seconds"] = forward_time

    @override
    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
        self.inference_mode = use_inference

    @override
    def _move_models_to_devices(self, devices: Sequence[torch.device]) -> None:
        # As this inference engine only supports one device, just take the first.
        device = devices[0]
        for model in self.models:
            model.to(device)


class InferenceEngineCachePreprocessing(MultiDeviceInferenceEngine):
    """Inference engine that caches the preprocessing for feeding as model context on
    predict.

    This will fit the preprocessors on the training data, as well as cache the
    transformed training data on RAM (not GPU RAM).

    This saves some time on each predict call, at the cost of increasing the amount
    of memory in RAM. The main functionality performed at `predict()` time is to
    forward pass through the model which is currently done sequentially.
    """

    def __init__(
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        ensemble_preprocessor: TabPFNEnsemblePreprocessor,
        models: list[Architecture],
        devices: Sequence[torch.device],
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: MemorySavingMode,
        inference_mode: bool,
        no_preprocessing: bool = False,
    ) -> None:
        """Initialize the cache preprocessing inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            ensemble_preprocessor: The ensemble preprocessor to use.
            models: The models to use.
            devices: A list of the devices to use for inference. If multiple devices are
                specified, then the inference engine will parallelize the members of the
                ensemble across the devices.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to use torch.inference mode
                (this is quicker but disables backpropagation)
            no_preprocessing: If True, skip preprocessing on test data.
                Used for differentiability.
        """
        super().__init__(
            model_caches=[_PerDeviceModelCache(model) for model in models],
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.inference_mode = inference_mode
        self.no_preprocessing = no_preprocessing
        self.X_train_shape_before_preprocessing = X_train.shape

        fit_preprocess_start = time.perf_counter()
        self.ensemble_members: list[TabPFNEnsembleMember] = (
            ensemble_preprocessor.fit_transform_ensemble_members(
                X_train=X_train,
                y_train=y_train,
            )
        )
        self._speed_metrics["fit_preprocessing_seconds"] = (
            time.perf_counter() - fit_preprocess_start
        )

        self.to(devices, self.force_inference_dtype, self.dtype_byte_size)

    @override
    def iter_outputs(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        autocast: bool,
        task_type: str,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        devices = self.get_devices()

        if self.force_inference_dtype is not None:
            for model_cache in self.model_caches:
                model_cache.set_dtype(self.force_inference_dtype)

        if self.inference_mode:
            save_peak_mem = should_save_peak_mem(
                memory_saving_mode=self.save_peak_mem,
                X_train_shape=tuple[int, int](self.X_train_shape_before_preprocessing),
                X_test_shape=tuple[int, int](X.shape),
                devices=devices,
                dtype_byte_size=self.dtype_byte_size,
            )
        else:
            save_peak_mem = False

        def _transform_X_test(
            ensemble_member: TabPFNEnsembleMember,
        ) -> np.ndarray | torch.Tensor:
            return X if self.no_preprocessing else ensemble_member.transform_X_test(X)

        model_forward_functions = (
            partial(
                self._call_model,
                X_train=ensemble_member.X_train,
                X_test=_transform_X_test(ensemble_member),
                y_train=ensemble_member.y_train,
                feature_schema=ensemble_member.feature_schema,
                autocast=autocast,
                only_return_standard_out=only_return_standard_out,
                model_index=ensemble_member.config._model_index,
                save_peak_mem=save_peak_mem,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
                task_type=task_type,
            )
            for ensemble_member in self.ensemble_members
        )

        timed_outputs = _TimedIterator(
            parallel_execute(devices, model_forward_functions)
        )

        for output, ensemble_member in zip(timed_outputs, self.ensemble_members):
            yield _move_and_squeeze_output(output, devices[0]), ensemble_member.config

        self._speed_metrics["predict_model_forward_seconds"] = (
            timed_outputs.elapsed_seconds
        )

    def _call_model(  # noqa: PLR0913
        self,
        *,
        device: torch.device,
        X_train: torch.Tensor | np.ndarray,
        X_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        feature_schema: FeatureSchema,
        autocast: bool,
        only_return_standard_out: bool,
        model_index: int,
        save_peak_mem: bool,
        gpu_preprocessor: TorchPreprocessingPipeline | None,
        task_type: str,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        model = self.model_caches[model_index].get(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )

        performance_options = model.get_default_performance_options()
        performance_options = dataclasses.replace(
            performance_options,
            save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
            if save_peak_mem
            else None,
        )

        X_full, feature_schema = _maybe_run_gpu_preprocessing(
            X_full,
            gpu_preprocessor=gpu_preprocessor,
            num_train_rows=X_train.shape[0],
            feature_schema=feature_schema,
        )
        batched_cat_ix = [feature_schema.indices_for(FeatureModality.CATEGORICAL)]

        kwargs = {}
        if _model_expectes_task_type_arg(model):
            kwargs["task_type"] = task_type

        with (
            get_autocast_context(device, enabled=autocast),
            torch.inference_mode(self.inference_mode),
        ):
            return model(
                X_full,
                y_train,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=batched_cat_ix,
                performance_options=performance_options,
                **kwargs,
            )

    @override
    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
        self.inference_mode = use_inference


class InferenceEngineCacheKV(SingleDeviceInferenceEngine):
    """Inference engine that caches the actual KV cache calculated from the context
    of the processed training data.

    This is by far the most memory intensive inference engine, as for each ensemble
    member we store the full KV cache of that model. For now this is held in CPU RAM.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        ensemble_preprocessor: TabPFNEnsemblePreprocessor,
        models: list[Architecture],
        devices: Sequence[torch.device],
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: MemorySavingMode,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> None:
        """Initialize the KV cache inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            ensemble_preprocessor: The ensemble configurations to use.
            models: The models to use.
            devices: A list of devices, the first of which will be used to run the
                model. The other devices will be ignored.
            dtype_byte_size: Size of the dtype in bytes.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            autocast: Whether to use torch.autocast during inference.
            only_return_standard_out: Whether to only return the standard output
        """
        # This engine currently only supports one device, so just take the first.
        device = devices[0]

        ensemble_members_iterator = (
            ensemble_preprocessor.fit_transform_ensemble_members_iterator(
                X_train=X_train,
                y_train=y_train,
                parallel_mode="as-ready",
            )
        )

        ens_models: list[Architecture] = []
        ensemble_members: list[TabPFNEnsembleMember] = []
        # Wrap the iterator to capture CPU preprocessing time (which runs
        # inside __next__ of the ensemble_members_iterator).
        timed_cpu_preprocess = _TimedIterator(ensemble_members_iterator)
        fit_gpu_preprocess_time = 0.0
        fit_forward_time = 0.0

        for ensemble_member in timed_cpu_preprocess:
            ensemble_members.append(ensemble_member)

            ens_model = deepcopy(models[ensemble_member.config._model_index])
            ens_model = ens_model.to(device)

            gpu_preprocess_start = time.perf_counter()
            X = ensemble_member.X_train
            y = ensemble_member.y_train

            # Use force_inference_dtype when set (e.g. float64) so GPU
            # preprocessing sees the same precision as fit_preprocessors mode.
            tensor_dtype = force_inference_dtype or torch.float32
            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X, dtype=tensor_dtype, device=device)
            X = X.unsqueeze(1)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=tensor_dtype, device=device)

            X, updated_schema = _maybe_run_gpu_preprocessing(
                X,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
                feature_schema=ensemble_member.feature_schema,
            )
            batched_preprocessor_cat_ix = [
                updated_schema.indices_for(FeatureModality.CATEGORICAL)
            ]

            if force_inference_dtype is not None:
                ens_model.type(force_inference_dtype)
                X = X.type(force_inference_dtype)
                y = y.type(force_inference_dtype)
            fit_gpu_preprocess_time += time.perf_counter() - gpu_preprocess_start

            # We do not reset the peak memory for cache_kv mode
            # because the entire data has to be passed through the model
            # at once to generate the KV cache
            forward_start = time.perf_counter()
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(),
            ):
                ens_model.forward(
                    X,
                    y,
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_preprocessor_cat_ix,
                )
            fit_forward_time += time.perf_counter() - forward_start

            ens_model.cpu()

            ens_models.append(ens_model)

        super().__init__(
            models=ens_models,
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )
        self._speed_metrics["fit_preprocessing_seconds"] = (
            timed_cpu_preprocess.elapsed_seconds + fit_gpu_preprocess_time
        )
        self._speed_metrics["fit_model_forward_seconds"] = fit_forward_time

        self.device = device
        self.ensemble_members = ensemble_members

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
        task_type: str,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        preprocess_time = 0.0
        forward_time = 0.0
        for ensemble_member, model in zip(self.ensemble_members, self.models):
            preprocess_start = time.perf_counter()
            model.to(self.device)
            X_test = ensemble_member.transform_X_test(X)
            tensor_dtype = self.force_inference_dtype or torch.float32
            X_test = torch.as_tensor(X_test, dtype=tensor_dtype, device=self.device)
            X_test = X_test.unsqueeze(1)
            X_test, updated_schema = _maybe_run_gpu_preprocessing(
                X_test,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
                num_train_rows=0,
                use_fitted_cache=True,
                feature_schema=ensemble_member.feature_schema,
            )
            batched_cat_ix = [updated_schema.indices_for(FeatureModality.CATEGORICAL)]

            if self.force_inference_dtype is not None:
                model.type(self.force_inference_dtype)
                X_test = X_test.type(self.force_inference_dtype)
            preprocess_time += time.perf_counter() - preprocess_start

            kwargs = {}
            if _model_expectes_task_type_arg(model):
                kwargs["task_type"] = task_type
            # When the KV cache is enabled, we assume we are under memory
            # pressure and enable the saving mode.
            # TODO: Use the heuristic in this case also.
            PerformanceOptions(save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR)
            forward_start = time.perf_counter()
            with (
                get_autocast_context(self.device, enabled=autocast),
                torch.inference_mode(),
            ):
                output = model(
                    X_test,
                    y=None,
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_cat_ix,
                    # When the KV cache is enabled, we assume we are under memory
                    # pressure and enable the saving mode.
                    # TODO: Use the heuristic in this case also.
                    performance_options=PerformanceOptions(
                        save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
                    ),
                    **kwargs,
                )
            forward_time += time.perf_counter() - forward_start

            model.cpu()

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, ensemble_member.config

        self._speed_metrics["predict_preprocessing_seconds"] = preprocess_time
        self._speed_metrics["predict_model_forward_seconds"] = forward_time

    @override
    def _move_models_to_devices(self, devices: Sequence[torch.device]) -> None:
        # Various things in the model do not currently respect the `.to()` function, and
        # just stay on the device where they were created.
        raise NotImplementedError(
            "fit_mode 'fit_with_cache' does not currently support .to() after .fit()"
        )


class InferenceEngineExplicitKVCache(MultiDeviceInferenceEngine):
    """Inference engine with explicit KV cache passed through forward().

    Unlike :class:`InferenceEngineCacheKV`, the KV cache is stored externally
    (not inside the model) and passed explicitly to the model's forward pass.
    This avoids deepcopying the model per ensemble member and keeps the model
    stateless.

    Each ensemble member (estimator) has its own KV cache.
    Ensemble members are dispatched across available GPUs
    via :func:`parallel_execute`.

    When ``keep_cache_on_device=True``, each per-estimator cache is kept
    on the GPU for subsequent prediction calls, avoiding CPU↔GPU transfers.

    At predict, only test rows are preprocessed (CPU and GPU) and
    uploaded. The model is called with ``x_is_test_only=True``. ``y``
    still carries the full train labels for the many-class decoder.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        ensemble_preprocessor: TabPFNEnsemblePreprocessor,
        models: list[Architecture],
        devices: Sequence[torch.device],
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: MemorySavingMode,
        autocast: bool,
        keep_cache_on_device: bool = True,
    ) -> None:
        """Initialize the explicit KV cache inference engine.

        Builds a KV cache per ensemble member in parallel across devices.
        Each estimator uses a different data permutation, so each cache is
        unique.

        Args:
            X_train: The training data.
            y_train: The training target.
            feature_schema: The feature schema.
            ensemble_preprocessor: The ensemble preprocessor to use.
            models: The models to use.
            devices: Devices to use for inference. If multiple devices are
                specified, ensemble members are parallelised across them
                during both cache build and prediction.
            dtype_byte_size: Size of the dtype in bytes.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            autocast: Whether to use torch.autocast during cache build.
            keep_cache_on_device: If True, keep each per-estimator KV cache
                on the device where it was built.  Uses more device memory
                but avoids CPU↔GPU transfers, giving lower latency.  When
                False (default), caches are moved to CPU after building and
                transferred to the target device on every predict call.
        """
        super().__init__(
            model_caches=[_PerDeviceModelCache(model) for model in models],
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.keep_cache_on_device = keep_cache_on_device
        self.X_train_shape_before_preprocessing = X_train.shape

        # Place model copies on all devices before building caches
        self.to(devices, self.force_inference_dtype, self.dtype_byte_size)

        # Preprocess ensemble members (CPU work)
        fit_preprocess_start = time.perf_counter()
        self.ensemble_members: list[TabPFNEnsembleMember] = (
            ensemble_preprocessor.fit_transform_ensemble_members(
                X_train=X_train,
                y_train=y_train,
            )
        )
        self._speed_metrics["fit_preprocessing_seconds"] = (
            time.perf_counter() - fit_preprocess_start
        )

        # Build per-estimator caches in parallel across devices
        build_functions = (
            partial(
                self._build_cache,
                X_train=ensemble_member.X_train,
                y_train=ensemble_member.y_train,
                feature_schema=ensemble_member.feature_schema,
                model_index=ensemble_member.config._model_index,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
                autocast=autocast,
                save_peak_mem=save_peak_mem,
            )
            for ensemble_member in self.ensemble_members
        )
        timed_caches = _TimedIterator(parallel_execute(devices, build_functions))
        self.caches: list = list(timed_caches)
        self._speed_metrics["fit_model_forward_seconds"] = timed_caches.elapsed_seconds

    def _build_cache(
        self,
        *,
        device: torch.device,
        X_train: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        feature_schema: FeatureSchema,
        model_index: int,
        gpu_preprocessor: TorchPreprocessingPipeline | None,
        autocast: bool,
        save_peak_mem: bool,
    ) -> object:
        """Build KV cache for one ensemble member on the given device.

        Called via :func:`parallel_execute` — may run on different devices
        in parallel threads.
        """
        model = self.model_caches[model_index].get(device)

        # Cast model weights to match force_inference_dtype (else linear
        # layers throw a Half/Float mismatch — matches CacheKV).
        if self.force_inference_dtype is not None:
            model.type(self.force_inference_dtype)

        tensor_dtype = self.force_inference_dtype or torch.float32
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.as_tensor(X_train, dtype=tensor_dtype, device=device)
        else:
            X_train = X_train.to(device)
        X = X_train.unsqueeze(1)
        if not isinstance(y_train, torch.Tensor):
            y = torch.as_tensor(y_train, dtype=tensor_dtype, device=device)
        else:
            y = y_train.to(device)

        # Fit the gpu preprocessor on train data (populates fitted_cache
        # when keep_fitted_cache=True, so predict can use_fitted_cache=True)
        X, feature_schema = _maybe_run_gpu_preprocessing(
            X,
            gpu_preprocessor=gpu_preprocessor,
            feature_schema=feature_schema,
        )
        batched_cat_ix = [feature_schema.indices_for(FeatureModality.CATEGORICAL)]

        if self.force_inference_dtype is not None:
            X = X.type(self.force_inference_dtype)
            y = y.type(self.force_inference_dtype)

        performance_options = model.get_default_performance_options()
        performance_options = dataclasses.replace(
            performance_options,
            save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
            if save_peak_mem
            else None,
        )

        with (
            get_autocast_context(device, enabled=autocast),
            torch.inference_mode(),
        ):
            _, cache = model(
                X,
                y,
                only_return_standard_out=True,
                categorical_inds=batched_cat_ix,
                performance_options=performance_options,
                return_kv_cache=True,
            )

        assert cache is not None
        if self.keep_cache_on_device:
            return cache
        return cache.to("cpu")

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
        task_type: str,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        devices = self.get_devices()

        if self.force_inference_dtype is not None:
            for model_cache in self.model_caches:
                model_cache.set_dtype(self.force_inference_dtype)

        save_peak_mem = should_save_peak_mem(
            memory_saving_mode=self.save_peak_mem,
            X_train_shape=tuple[int, int](self.X_train_shape_before_preprocessing),
            X_test_shape=tuple[int, int](X.shape),
            devices=devices,
            dtype_byte_size=self.dtype_byte_size,
        )

        model_forward_functions = (
            partial(
                self._call_model,
                cache_index=i,
                X_test=ensemble_member.transform_X_test(X),
                y_train=ensemble_member.y_train,
                feature_schema=ensemble_member.feature_schema,
                autocast=autocast,
                only_return_standard_out=only_return_standard_out,
                model_index=ensemble_member.config._model_index,
                save_peak_mem=save_peak_mem,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
                task_type=task_type,
            )
            for i, ensemble_member in enumerate(self.ensemble_members)
        )
        timed_outputs = _TimedIterator(
            parallel_execute(devices, model_forward_functions)
        )

        for output, ensemble_member in zip(timed_outputs, self.ensemble_members):
            yield _move_and_squeeze_output(output, devices[0]), ensemble_member.config

        self._speed_metrics["predict_model_forward_seconds"] = (
            timed_outputs.elapsed_seconds
        )

    def _call_model(  # noqa: PLR0913
        self,
        *,
        device: torch.device,
        cache_index: int,
        X_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        feature_schema: FeatureSchema,
        autocast: bool,
        only_return_standard_out: bool,
        model_index: int,
        save_peak_mem: bool,
        gpu_preprocessor: TorchPreprocessingPipeline | None,
        task_type: str,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Each ensemble member (estimator) has its own KV cache at
        ``self.caches[cache_index]``.  When ``keep_cache_on_device`` is True,
        the cache is moved to ``device`` on the first call and kept there.

        Only X_test is uploaded and preprocessed — the v3 model's cache
        fast path never reads train rows when ``x_is_test_only=True``.

        May be executed in parallel across threads, one per device.
        """
        model = self.model_caches[model_index].get(device)

        dtype = self.force_inference_dtype or torch.float32
        X_test_tensor = torch.as_tensor(X_test, dtype=dtype, device=device).unsqueeze(1)
        y_train = torch.as_tensor(y_train, dtype=dtype, device=device)

        X_test_tensor, feature_schema = _maybe_run_gpu_preprocessing(
            X_test_tensor,
            gpu_preprocessor=gpu_preprocessor,
            num_train_rows=0,
            use_fitted_cache=True,
            feature_schema=feature_schema,
        )

        # Cast post-preproc tensors back to force_inference_dtype if set —
        # GPU preprocessing may emit fp32 internally (SVD/quantile for
        # numerical stability) even under a fp16 run. Matches CacheKV.
        if self.force_inference_dtype is not None:
            X_test_tensor = X_test_tensor.type(self.force_inference_dtype)
            y_train = y_train.type(self.force_inference_dtype)

        batched_cat_ix = [feature_schema.indices_for(FeatureModality.CATEGORICAL)]

        performance_options = model.get_default_performance_options()
        performance_options = dataclasses.replace(
            performance_options,
            save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
            if save_peak_mem
            else None,
        )

        kwargs = {}
        if _model_expectes_task_type_arg(model):
            kwargs["task_type"] = task_type

        # Get cache for this estimator and move to target device
        cache = self.caches[cache_index]
        cache_on_device = cache.to(device)
        if self.keep_cache_on_device:
            # Persist on-device copy so subsequent calls skip the transfer
            self.caches[cache_index] = cache_on_device

        with (
            get_autocast_context(device, enabled=autocast),
            torch.inference_mode(),
        ):
            return model(
                X_test_tensor,
                y_train,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=batched_cat_ix,
                performance_options=performance_options,
                kv_cache=cache_on_device,
                x_is_test_only=True,
                **kwargs,
            )


def _prepare_model_inputs(
    device: torch.device,
    force_inference_dtype: torch.dtype | None,
    X_train: torch.Tensor | np.ndarray,
    X_test: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = force_inference_dtype if force_inference_dtype else torch.float32
    X_train = torch.as_tensor(X_train, dtype=dtype, device=device)
    X_test = torch.as_tensor(X_test, dtype=dtype, device=device)
    X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
    y_train = torch.as_tensor(y_train, dtype=dtype, device=device)
    return X_full, y_train


def _move_and_squeeze_output(
    output: dict | torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor] | torch.Tensor:
    if isinstance(output, dict):
        return {k: v.to(device) for k, v in output.items()}
    return output.squeeze(1).to(device)


def _maybe_run_gpu_preprocessing(
    X: torch.Tensor,
    gpu_preprocessor: TorchPreprocessingPipeline | None,
    feature_schema: FeatureSchema,
    *,
    num_train_rows: int | None = None,
    use_fitted_cache: bool = False,
) -> tuple[torch.Tensor, FeatureSchema]:
    """Run GPU preprocessing if a pipeline is provided.

    Args:
        X: Input tensor.
        gpu_preprocessor: The GPU preprocessing pipeline (or None).
        feature_schema: Feature schema from CPU preprocessing.
        num_train_rows: Number of training rows for fit.
        use_fitted_cache: Reuse previously fitted state.

    Returns:
        Tuple of (transformed tensor, updated feature schema).
    """
    if gpu_preprocessor is None:
        return X, feature_schema

    result = gpu_preprocessor(
        X,
        feature_schema=feature_schema,
        num_train_rows=num_train_rows,
        use_fitted_cache=use_fitted_cache,
    )
    return result.x, result.feature_schema


class _PerDeviceModelCache:
    """Maintains a copy of a PyTorch model on a set of devices."""

    def __init__(self, model: Architecture) -> None:
        """Create a new instance."""
        super().__init__()
        self._models: dict[torch.device, Architecture] = {
            _get_current_device(model): model
        }

    def to(self, devices: Sequence[torch.device]) -> None:
        """Load copies of the model on the given devices.

        This function will re-use any existing copies of the model, moving them to new
        devices as needed, before creating new copies. Thus, the called should discard
        any references to models previously obtained with .get_model() after calling
        this function.
        """
        spare_models = [
            model for device, model in self._models.items() if device not in devices
        ]

        def get_on_device(device: torch.device) -> Architecture:
            """Get the model on the given device. Try to reuse existing models."""
            if device in self._models:
                return self._models[device]
            if len(spare_models) > 0:
                return spare_models.pop().to(device)
            existing_model = next(iter(self._models.values()))
            return deepcopy(existing_model).to(device)

        self._models = {device: get_on_device(device) for device in devices}

    def get(self, device: torch.device) -> Architecture:
        """Return the model on the given device.

        Raises:
            KeyError: If a device is specified that was not included in the last call to
                .to()
        """
        return self._models[device]

    def set_dtype(self, dtype: torch.dtype) -> None:
        """Set the dtype of the model's parameters."""
        for model in self._models.values():
            model.type(dtype)

    def get_devices(self) -> list[torch.device]:
        """Return the devices that are in use."""
        return list(self._models.keys())


def _get_current_device(model: Architecture) -> torch.device:
    """Return the device that the model parameters are on."""
    # Assume the model is in a good state: all parameters are on the same device.
    return next(iter(model.parameters())).device
