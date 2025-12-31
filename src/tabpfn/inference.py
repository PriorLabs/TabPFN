"""Module that defines different ways to run inference with TabPFN."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

import joblib
import numpy as np
import torch

from tabpfn.architectures.base.memory import (
    DEFAULT_SAVE_PEAK_MEMORY_FACTOR,
    should_save_peak_mem,
)
from tabpfn.parallel_execute import parallel_execute
from tabpfn.preprocessing import fit_preprocessing
from tabpfn.utils import get_autocast_context

if TYPE_CHECKING:
    from tabpfn.architectures.interface import Architecture
    from tabpfn.preprocessing import EnsembleConfig, SequentialFeatureTransformer


@dataclass
class InferenceEngine(ABC):
    """These define how tabpfn inference can be run.

    As there are many things that can be cached, with multiple ways to parallelize,
    `Executor` defines three primary things:

    Most will define a method `prepare()` which is specific to that inference engine.
    These do not share a common interface.

    1. What to cache:

        As we can prepare a lot of the transformers context, there is a tradeoff in
        terms of how much memory to be spent in caching. This memory is used when
        `prepare()` is called, usually in `fit()`.

    2. Using the cached data for inference:

        Based on what has been prepared for the transformer context,
        `iter_outputs()` will use this cached information to make predictions.

    3. Controlling parallelism:

        As we have trivially parallel parts for inference, we can parallelize them.
        However as the GPU is typically a bottle-neck in most systems, we can define,
        where and how we would like to parallelize the inference.

    The InferenceEngineBatchedNoPreprocessing
    InferenceEngineCachePreprocessing engines also support toggling
    `torch.use_torch_inference_mode` via `use_torch_inference_mode`
    to enable/disable gradient tracking during prediction.

    Attributes:
        save_peak_mem: Whether to save peak memory usage.
        dtype_byte_size: The byte size of the dtype.
        force_inference_dtype: If not None, inference will be performed using this
            dtype. Otherwise, the default dtype will be used.
        models: The models to use for inference.
    """

    save_peak_mem: bool | Literal["auto"] | float | int
    dtype_byte_size: int
    force_inference_dtype: torch.dtype | None

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


@dataclass
class SingleDeviceInferenceEngine(InferenceEngine):
    """Inference engine that uses a single device to execute the model."""

    models: list[Architecture]

    @override
    def _create_copy_for_pickling(self) -> InferenceEngine:
        state_copy = deepcopy(self)
        state_copy.models = None  # type: ignore
        return state_copy

    @override
    def _set_models(self, models: list[Architecture]) -> None:
        self.models = models


@dataclass
class MultiDeviceInferenceEngine(InferenceEngine):
    """Inference engine that parallelizes the members of the ensemble across devices."""

    model_caches: list[_PerDeviceModelCache]

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


@dataclass
class InferenceEngineOnDemand(MultiDeviceInferenceEngine):
    """Inference engine that does not cache anything, computes everything as needed.

    This is one of the slowest ways to run inference, as computation that could be
    cached is recomputed on every call. However the memory demand is lowest and
    can be more trivially parallelized across GPUs with some work.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    cat_ix: list[int]
    static_seed: int
    n_preprocessing_jobs: int
    ensemble_configs: list[EnsembleConfig]

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
        models: list[Architecture],
        devices: Sequence[torch.device],
        ensemble_configs: Sequence[EnsembleConfig],
        rng: np.random.Generator,
        n_preprocessing_jobs: int,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
    ) -> InferenceEngineOnDemand:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            models: The models to use.
            devices: A list of the devices to use for inference. If multiple devices are
                specified, then the inference engine will parallelize the members of the
                ensemble across the devices.
            ensemble_configs: The ensemble configurations to use.
            rng: The random number generator.
            n_preprocessing_jobs: The number of workers to use.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
        """
        # We save it as a static seed to be reproducible across predicts
        static_seed = rng.integers(0, int(np.iinfo(np.int32).max))
        engine = cls(
            X_train=X_train,
            y_train=y_train,
            ensemble_configs=list(ensemble_configs),
            cat_ix=cat_ix,
            model_caches=[_PerDeviceModelCache(model) for model in models],
            static_seed=static_seed,
            n_preprocessing_jobs=n_preprocessing_jobs,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
        )
        engine.to(devices, force_inference_dtype, dtype_byte_size)
        return engine

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        rng = np.random.default_rng(self.static_seed)
        devices = self.get_devices()

        preprocessed_data_iterator = fit_preprocessing(
            configs=self.ensemble_configs,
            X_train=self.X_train,
            y_train=self.y_train,
            random_state=rng,
            cat_ix=self.cat_ix,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            parallel_mode="in-order",
        )

        save_peak_mem = should_save_peak_mem(
            self.save_peak_mem,
            X_train_shape=self.X_train.shape,
            X_test_shape=X.shape,
            devices=devices,
            dtype_byte_size=self.dtype_byte_size,
        )

        ensemble_configs, preprocessings = itertools.tee(preprocessed_data_iterator)

        if self.force_inference_dtype is not None:
            for model_cache in self.model_caches:
                model_cache.set_dtype(self.force_inference_dtype)

        model_forward_functions = (
            partial(
                self._call_model,
                X_train=X_train,
                X_test=preprocessor.transform(X).X,
                y_train=y_train,
                cat_ix=cat_ix,
                only_return_standard_out=only_return_standard_out,
                autocast=autocast,
                model_index=config._model_index,
                save_peak_mem=save_peak_mem,
            )
            for config, preprocessor, X_train, y_train, cat_ix in preprocessings
        )
        outputs = parallel_execute(devices, model_forward_functions)

        for (config, _, _, _, _), output in zip(ensemble_configs, outputs):
            yield _move_and_squeeze_output(output, devices[0]), config

    def _call_model(
        self,
        *,
        device: torch.device,
        X_train: torch.Tensor | np.ndarray,
        X_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        cat_ix: list[int],
        autocast: bool,
        only_return_standard_out: bool,
        model_index: int,
        save_peak_mem: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        model = self.model_caches[model_index].get(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )
        batched_cat_ix = [cat_ix]

        save_peak_memory_factor = (
            DEFAULT_SAVE_PEAK_MEMORY_FACTOR if save_peak_mem else None
        )

        with get_autocast_context(device, enabled=autocast), torch.inference_mode():
            return model(
                X_full,
                y_train,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=batched_cat_ix,
                save_peak_memory_factor=save_peak_memory_factor,
            )


@dataclass
class InferenceEngineBatchedNoPreprocessing(SingleDeviceInferenceEngine):
    """Inference engine that uses preprocessed inputs, and allows batched predictions
    on several datasets at once.

    Args:
            X_trains: The training data.
            y_trains    : The training target.
            cat_ix: The categorical indices.
            ensemble_configs: The ensemble configurations to use.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to enable torch inference mode.
    """

    X_trains: list[torch.Tensor]
    y_trains: list[torch.Tensor]
    cat_ix: list[list[list[int]]]
    ensemble_configs: list[list[EnsembleConfig]]
    inference_mode: bool

    @classmethod
    def prepare(
        cls,
        X_trains: list[torch.Tensor],
        y_trains: list[torch.Tensor],
        *,
        cat_ix: list[list[list[int]]],
        models: list[Architecture],
        devices: Sequence[torch.device],
        ensemble_configs: list[list[EnsembleConfig]],
        force_inference_dtype: torch.dtype | None,
        inference_mode: bool,
        dtype_byte_size: int,
        save_peak_mem: bool | Literal["auto"] | float | int,
    ) -> InferenceEngineBatchedNoPreprocessing:
        """Prepare the inference engine.

        Args:
            X_trains: The training data.
            y_trains: The training target.
            cat_ix: The categorical indices.
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

        # We save it as a static seed to be reproducible across predicts
        engine = cls(
            X_trains=X_trains,
            y_trains=y_trains,
            cat_ix=cat_ix,
            models=models,
            ensemble_configs=ensemble_configs,
            force_inference_dtype=force_inference_dtype,
            inference_mode=inference_mode,
            dtype_byte_size=dtype_byte_size,
            save_peak_mem=save_peak_mem,
        )
        engine.to(devices, force_inference_dtype, dtype_byte_size)
        return engine

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
                    categorical_inds=list([cat_item[i] for cat_item in self.cat_ix]),  # noqa: C411
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


@dataclass
class InferenceEngineCachePreprocessing(MultiDeviceInferenceEngine):
    """Inference engine that caches the preprocessing for feeding as model context on
    predict.

    This will fit the preprocessors on the training data, as well as cache the
    transformed training data on RAM (not GPU RAM).

    This saves some time on each predict call, at the cost of increasing the amount
    of memory in RAM. The main functionality performed at `predict()` time is to
    forward pass through the model which is currently done sequentially.
    """

    X_trains: Sequence[np.ndarray | torch.Tensor]
    y_trains: Sequence[np.ndarray | torch.Tensor]
    X_train_shape_before_preprocessing: tuple[int, int]
    cat_ixs: Sequence[list[int]]
    preprocessors: Sequence[SequentialFeatureTransformer]
    inference_mode: bool
    ensemble_configs: list[EnsembleConfig]
    no_preprocessing: bool = False

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        cat_ix: list[int],
        models: list[Architecture],
        devices: Sequence[torch.device],
        ensemble_configs: Sequence[EnsembleConfig],
        n_preprocessing_jobs: int,
        rng: np.random.Generator,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        inference_mode: bool,
        no_preprocessing: bool = False,
    ) -> InferenceEngineCachePreprocessing:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            models: The models to use.
            devices: A list of the devices to use for inference. If multiple devices are
                specified, then the inference engine will parallelize the members of the
                ensemble across the devices.
            ensemble_configs: The ensemble configurations to use.
            n_preprocessing_jobs: The number of workers to use.
            rng: The random number generator.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to use torch.inference mode
                (this is quicker but disables backpropagation)
            no_preprocessing: If turned of, the preprocessing on the test
                tensors is tuned off. Used for differentiablity.

        Returns:
            The prepared inference engine.
        """
        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_preprocessing_jobs=n_preprocessing_jobs,
            parallel_mode="block",
        )
        configs, preprocessors, X_trains, y_trains, cat_ixs = list(zip(*itr))
        engine = InferenceEngineCachePreprocessing(
            X_trains=X_trains,
            y_trains=y_trains,
            X_train_shape_before_preprocessing=tuple[int, int](X_train.shape),
            model_caches=[_PerDeviceModelCache(model) for model in models],
            cat_ixs=cat_ixs,
            ensemble_configs=configs,
            preprocessors=preprocessors,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
            inference_mode=inference_mode,
            no_preprocessing=no_preprocessing,
        )
        engine.to(devices, force_inference_dtype, dtype_byte_size)
        return engine

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
                X_train_shape=self.X_train_shape_before_preprocessing,
                X_test_shape=tuple[int, int](X.shape),
                devices=devices,
                dtype_byte_size=self.dtype_byte_size,
            )
        else:
            save_peak_mem = False

        def _transform_X_test(i: int) -> np.ndarray | torch.Tensor:
            return X if self.no_preprocessing else self.preprocessors[i].transform(X).X

        model_forward_functions = (
            partial(
                self._call_model,
                X_train=self.X_trains[i],
                X_test=_transform_X_test(i),
                y_train=self.y_trains[i],
                cat_ix=self.cat_ixs[i],
                autocast=autocast,
                only_return_standard_out=only_return_standard_out,
                model_index=self.ensemble_configs[i]._model_index,
                save_peak_mem=save_peak_mem,
            )
            for i in range(len(self.ensemble_configs))
        )
        outputs = parallel_execute(devices, model_forward_functions)

        for output, i in zip(outputs, range(len(self.ensemble_configs))):
            yield _move_and_squeeze_output(output, devices[0]), self.ensemble_configs[i]

    def _call_model(
        self,
        *,
        device: torch.device,
        X_train: torch.Tensor | np.ndarray,
        X_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        cat_ix: list[int],
        autocast: bool,
        only_return_standard_out: bool,
        model_index: int,
        save_peak_mem: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        model = self.model_caches[model_index].get(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )
        batched_cat_ix = [cat_ix]

        save_peak_memory_factor = (
            DEFAULT_SAVE_PEAK_MEMORY_FACTOR if save_peak_mem else None
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


@dataclass
class InferenceEngineCacheKV(SingleDeviceInferenceEngine):
    """Inference engine that caches the actual KV cache calculated from the context
    of the processed training data.

    This is by far the most memory intensive inference engine, as for each ensemble
    member we store the full KV cache of that model. For now this is held in CPU RAM
    (TODO(eddiebergman): verify)
    """

    preprocessors: list[SequentialFeatureTransformer]
    cat_ixs: Sequence[list[int]]
    n_train_samples: list[int]
    ensemble_configs: list[EnsembleConfig]
    device: torch.device

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
        ensemble_configs: Sequence[EnsembleConfig],
        n_preprocessing_jobs: int,
        models: list[Architecture],
        devices: Sequence[torch.device],
        rng: np.random.Generator,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> InferenceEngineCacheKV:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            ensemble_configs: The ensemble configurations to use.
            n_preprocessing_jobs: The number of workers to use.
            models: The models to use.
            devices: A list of devices, the first of which will be used to run the
                model. The other devices will be ignored.
            rng: The random number generator.
            dtype_byte_size: Size of the dtype in bytes.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            autocast: Whether to use torch.autocast during inference.
            only_return_standard_out: Whether to only return the standard output
        """
        # This engine currently only supports one device, so just take the first.
        device = devices[0]

        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_preprocessing_jobs=n_preprocessing_jobs,
            parallel_mode="as-ready",
        )
        ens_models: list[Architecture] = []
        preprocessors: list[SequentialFeatureTransformer] = []
        correct_order_configs: list[EnsembleConfig] = []
        cat_ixs: Sequence[list[int]] = []
        n_train_samples: list[int] = []

        for config, preprocessor, X, y, preprocessor_cat_ix in itr:
            cat_ixs.append(preprocessor_cat_ix)
            preprocessors.append(preprocessor)
            correct_order_configs.append(config)
            n_train_samples.append(len(y))

            ens_model = deepcopy(models[config._model_index])
            ens_model = ens_model.to(device)
            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X, dtype=torch.float32, device=device)  # noqa: PLW2901
            X = X.unsqueeze(1)  # noqa: PLW2901
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.float32, device=device)  # noqa: PLW2901

            batched_preprocessor_cat_ix = [preprocessor_cat_ix]

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

        return InferenceEngineCacheKV(
            preprocessors=preprocessors,
            ensemble_configs=correct_order_configs,
            cat_ixs=cat_ixs,
            n_train_samples=n_train_samples,
            models=ens_models,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
            device=device,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        for preprocessor, model, config, cat_ix, _X_train_len in zip(
            self.preprocessors,
            self.models,
            self.ensemble_configs,
            self.cat_ixs,
            self.n_train_samples,
        ):
            model.to(self.device)
            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
            X_test = X_test.unsqueeze(1)
            batched_cat_ix = [cat_ix]

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

            yield output, config

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
