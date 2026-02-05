"""Module that defines different ways to run inference with TabPFN."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import override

import joblib
import numpy as np
import torch

from tabpfn.architectures.base.memory import (
    DEFAULT_SAVE_PEAK_MEMORY_FACTOR,
    MemorySavingMode,
    should_save_peak_mem,
)
from tabpfn.parallel_execute import parallel_execute
from tabpfn.preprocessing.datamodel import Feature, FeatureModality
from tabpfn.preprocessing.torch import (
    FeatureSchema,
    TorchPreprocessingPipeline,
)
from tabpfn.utils import get_autocast_context

if TYPE_CHECKING:
    from tabpfn.architectures.interface import Architecture
    from tabpfn.preprocessing import EnsembleConfig
    from tabpfn.preprocessing.ensemble import (
        TabPFNEnsembleMember,
        TabPFNEnsemblePreprocessor,
    )


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

    @abstractmethod
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor, EnsembleConfig]]:
        """Iterate over the outputs of the model for each ensemble configuration.

        Depending on the InferenceEngine used, this will run the forward pass of the
        model for each estimator.

        Args:
            X: The input data to make predictions on.
            autocast: Whether to use torch.autocast during inference.
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
    if isinstance(engine, InferenceEngineCacheKV):
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
        feature_schema: FeatureSchema,
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
            feature_schema: The feature schema.
            ensemble_preprocessor: The ensemble preprocessor to use.
            models: The models to use.
            devices: A list of the devices to use for inference. If multiple devices are
                specified, then the inference engine will parallelize the members of the
                ensemble across the devices.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
        """
        # We save it as a static seed to be reproducible across predicts
        static_seed = ensemble_preprocessor.next_static_seed()

        super().__init__(
            model_caches=[_PerDeviceModelCache(model) for model in models],
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.X_train = X_train
        self.y_train = y_train
        self.feature_schema = feature_schema
        self.static_seed = static_seed
        self.ensemble_preprocessor = ensemble_preprocessor

        self.to(devices, self.force_inference_dtype, self.dtype_byte_size)

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
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
                feature_schema=self.feature_schema,
                parallel_mode="in-order",
                override_random_state=np.random.default_rng(self.static_seed),
            )
        )

        model_forward_functions = (
            partial(
                self._call_model,
                X_train=ensemble_member.X_train,
                X_test=ensemble_member.transform_X_test(X),
                y_train=ensemble_member.y_train,
                feature_schema=ensemble_member.feature_schema,
                only_return_standard_out=only_return_standard_out,
                autocast=autocast,
                model_index=ensemble_member.config._model_index,
                save_peak_mem=save_peak_mem,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
            )
            for ensemble_member in ensemble_members_iterator
        )
        outputs = parallel_execute(devices, model_forward_functions)

        for config, output in zip(self.ensemble_preprocessor.configs, outputs):
            yield _move_and_squeeze_output(output, devices[0]), config

    def _call_model(
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
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        model = self.model_caches[model_index].get(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )
        batched_cat_ix = [feature_schema.indices_for(FeatureModality.CATEGORICAL)]

        save_peak_memory_factor = (
            DEFAULT_SAVE_PEAK_MEMORY_FACTOR if save_peak_mem else None
        )

        X_full = _maybe_run_gpu_preprocessing(
            X_full,
            gpu_preprocessor=gpu_preprocessor,
            num_train_rows=X_train.shape[0],
        )

        with get_autocast_context(device, enabled=autocast), torch.inference_mode():
            return model(
                X_full,
                y_train,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=batched_cat_ix,
                save_peak_memory_factor=save_peak_memory_factor,
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
    ) -> Iterator[tuple[torch.Tensor | dict, list[EnsembleConfig]]]:
        device = _get_current_device(self.models[0])
        batch_size = len(self.X_trains)
        for i in range(batch_size):
            train_x_full = torch.cat([self.X_trains[i], X[i]], dim=-2)
            train_y_batch = self.y_trains[i]
            train_x_full = train_x_full.to(device)
            train_y_batch = train_y_batch.to(device)
            if self.force_inference_dtype is not None:
                train_x_full = train_x_full.type(self.force_inference_dtype)
                train_y_batch = train_y_batch.type(self.force_inference_dtype)  # type: ignore

            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                output = self.models[self.ensemble_configs[i][0]._model_index](
                    train_x_full.transpose(0, 1),
                    train_y_batch.transpose(0, 1),
                    only_return_standard_out=True,
                    categorical_inds=list(  # noqa: C411
                        [
                            cat_item[i].indices_for(FeatureModality.CATEGORICAL)
                            for cat_item in self.feature_schema_list
                        ]
                    ),
                )

            yield output, self.ensemble_configs[i]

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

    def __init__(  # noqa: PLR0913
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        feature_schema: FeatureSchema,
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
            feature_schema: The feature schema.
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
        self.feature_schema = feature_schema

        self.ensemble_members: list[TabPFNEnsembleMember] = (
            ensemble_preprocessor.fit_transform_ensemble_members(
                X_train=X_train,
                y_train=y_train,
                feature_schema=feature_schema,
            )
        )

        self.to(devices, self.force_inference_dtype, self.dtype_byte_size)

    @override
    def iter_outputs(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        autocast: bool,
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
            )
            for ensemble_member in self.ensemble_members
        )
        outputs = parallel_execute(devices, model_forward_functions)

        for output, ensemble_member in zip(outputs, self.ensemble_members):
            yield _move_and_squeeze_output(output, devices[0]), ensemble_member.config

    def _call_model(
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
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        model = self.model_caches[model_index].get(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )
        batched_cat_ix = [feature_schema.indices_for(FeatureModality.CATEGORICAL)]

        save_peak_memory_factor = (
            DEFAULT_SAVE_PEAK_MEMORY_FACTOR if save_peak_mem else None
        )

        X_full = _maybe_run_gpu_preprocessing(
            X_full,
            gpu_preprocessor=gpu_preprocessor,
            num_train_rows=X_train.shape[0],
        )

        with (
            get_autocast_context(device, enabled=autocast),
            torch.inference_mode(self.inference_mode),
        ):
            return model(
                X_full,
                y_train,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=batched_cat_ix,
                save_peak_memory_factor=save_peak_memory_factor,
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

    def __init__(  # noqa: PLR0913
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        feature_schema: FeatureSchema,
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
            feature_schema: The feature schema.
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
                feature_schema=feature_schema,
                parallel_mode="as-ready",
            )
        )

        ens_models: list[Architecture] = []
        ensemble_members: list[TabPFNEnsembleMember] = []

        for ensemble_member in ensemble_members_iterator:
            ensemble_members.append(ensemble_member)

            ens_model = deepcopy(models[ensemble_member.config._model_index])
            ens_model = ens_model.to(device)
            X = ensemble_member.X_train
            y = ensemble_member.y_train

            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X, dtype=torch.float32, device=device)
            X = X.unsqueeze(1)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.float32, device=device)

            batched_preprocessor_cat_ix = [
                ensemble_member.feature_schema.indices_for(FeatureModality.CATEGORICAL)
            ]

            X = _maybe_run_gpu_preprocessing(
                X,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
            )

            # We do not reset the peak memory for cache_kv mode
            # because the entire data has to be passed through the model
            # at once to generate the KV cache
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

            ens_model.cpu()

            ens_models.append(ens_model)

        super().__init__(
            models=ens_models,
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.device = device
        self.ensemble_members = ensemble_members

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        for ensemble_member, model in zip(self.ensemble_members, self.models):
            model.to(self.device)
            X_test = ensemble_member.transform_X_test(X)
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
            X_test = X_test.unsqueeze(1)
            batched_cat_ix = [
                ensemble_member.feature_schema.indices_for(FeatureModality.CATEGORICAL)
            ]

            X_test = _maybe_run_gpu_preprocessing(
                X_test,
                gpu_preprocessor=ensemble_member.gpu_preprocessor,
                use_fitted_cache=True,
            )

            if self.force_inference_dtype is not None:
                model.type(self.force_inference_dtype)
                X_test = X_test.type(self.force_inference_dtype)

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
                    save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR,
                )

            model.cpu()

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, ensemble_member.config

    @override
    def _move_models_to_devices(self, devices: Sequence[torch.device]) -> None:
        # Various things in the model do not currently respect the `.to()` function, and
        # just stay on the device where they were created.
        raise NotImplementedError(
            "fit_mode 'fit_with_cache' does not currently support .to() after .fit()"
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
    *,
    num_train_rows: int | None = None,
    use_fitted_cache: bool = False,
) -> torch.Tensor:
    if gpu_preprocessor is None:
        return X

    # TODO: Currently, we construct the metadata on-the-fly.
    # In a follow-up, this will become part of a DatasetView object
    # parsed to the inference engine class.
    n_cols = X.shape[-1]
    features = [Feature(name=None, modality=FeatureModality.NUMERICAL)] * n_cols
    feature_schema = FeatureSchema(features=features)

    return gpu_preprocessor(
        X,
        feature_schema=feature_schema,
        num_train_rows=num_train_rows,
        use_fitted_cache=use_fitted_cache,
    ).x


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
