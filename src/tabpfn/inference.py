"""Module that defines different ways to run inference with TabPFN."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

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

from tabpfn.architectures.base.memory import MemoryUsageEstimator
from tabpfn.parallel_execute import parallel_execute
from tabpfn.preprocessing import fit_preprocessing
from tabpfn.utils import get_autocast_context

if TYPE_CHECKING:
    from tabpfn.architectures.base.preprocessing import SequentialFeatureTransformer
    from tabpfn.architectures.interface import Architecture
    from tabpfn.preprocessing import EnsembleConfig


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
    """

    save_peak_mem: bool | Literal["auto"] | float | int
    dtype_byte_size: int
    ensemble_configs: Sequence[EnsembleConfig]

    @abstractmethod
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        devices: Sequence[torch.device],
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor, EnsembleConfig]]:
        """Iterate over the outputs of the model.

        One for each ensemble configuration that was used to initialize the executor.

        Args:
            X: The input data to make predictions on.
            devices: The devices to run the model on.
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

        The state is first moved to CPU so the resulting file can be loaded
        on machines without a GPU. The large model weights are explicitly
        excluded to keep the file small and efficient.
        """
        state_copy = deepcopy(self)

        # Decouple the large model weights before serialization
        if hasattr(state_copy, "model"):
            state_copy.model = None
        if hasattr(state_copy, "models"):
            state_copy.models = None  # For KV cache engine

        joblib.dump(state_copy, path)

    @staticmethod
    def load_state(path: str | Path) -> InferenceEngine:
        """Load an executor saved with :meth:`save_state`."""
        return joblib.load(Path(path))


@dataclass
class InferenceEngineOnDemand(InferenceEngine):
    """Inference engine that does not cache anything, computes everything as needed.

    This is one of the slowest ways to run inference, as computation that could be
    cached is recomputed on every call. However the memory demand is lowest and
    can be more trivially parallelized across GPUs with some work.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    ensemble_configs: Sequence[EnsembleConfig]
    cat_ix: list[int]
    static_seed: int
    n_workers: int
    model: Architecture
    force_inference_dtype: torch.dtype | None

    @classmethod
    def prepare(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
        model: Architecture,
        ensemble_configs: Sequence[EnsembleConfig],
        rng: np.random.Generator,
        n_workers: int,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
    ) -> InferenceEngineOnDemand:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            rng: The random number generator.
            n_workers: The number of workers to use.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
        """
        # We save it as a static seed to be reproducible across predicts
        static_seed = rng.integers(0, int(np.iinfo(np.int32).max))
        return cls(
            X_train=X_train,
            y_train=y_train,
            ensemble_configs=ensemble_configs,
            cat_ix=cat_ix,
            model=model,
            static_seed=static_seed,
            n_workers=n_workers,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        devices: Sequence[torch.device],
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        rng = np.random.default_rng(self.static_seed)
        itr = fit_preprocessing(
            configs=self.ensemble_configs,
            X_train=self.X_train,
            y_train=self.y_train,
            random_state=rng,
            cat_ix=self.cat_ix,
            n_workers=self.n_workers,
            parallel_mode="as-ready",
        )

        if self.force_inference_dtype is not None:
            self.model.type(self.force_inference_dtype)

        model_forward_functions = (
            partial(
                self._call_model,
                X_train=X_train,
                X_test=preprocessor.transform(X).X,
                y_train=y_train,
                cat_ix=cat_ix,
                only_return_standard_out=only_return_standard_out,
                autocast=autocast,
            )
            for _, preprocessor, X_train, y_train, cat_ix in itr
        )
        outputs = parallel_execute(devices, model_forward_functions)

        for config, output in zip(self.ensemble_configs, outputs):
            yield _move_and_squeeze_output(output, devices[0]), config

        self.model.cpu()

    def _call_model(
        self,
        *,
        device: torch.device,
        is_parallel: bool,
        X_train: torch.Tensor | np.ndarray,
        X_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        cat_ix: list[int],
        autocast: bool,
        only_return_standard_out: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        # If several estimators are being run in parallel, then each thread needs its
        # own copy of the model so it can move it to its device.
        model = deepcopy(self.model) if is_parallel else self.model
        model.to(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )
        batched_cat_ix = [cat_ix]

        MemoryUsageEstimator.reset_peak_memory_if_required(
            save_peak_mem=self.save_peak_mem,
            model=model,
            X=X_full,
            cache_kv=False,
            dtype_byte_size=self.dtype_byte_size,
            device=device,
            safety_factor=1.2,
        )

        with get_autocast_context(device, enabled=autocast), torch.inference_mode():
            return model(
                X_full,
                y_train,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=batched_cat_ix,
            )


@dataclass
class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
    """Inference engine that uses preprocessed inputs, and allows batched predictions
    on several datasets at once.

    Args:
            X_trains: The training data.
            y_trains    : The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to enable torch inference mode.
    """

    X_trains: list[torch.Tensor]
    y_trains: list[torch.Tensor]
    cat_ix: list[list[list[int]]]
    model: Architecture
    ensemble_configs: Sequence[EnsembleConfig]
    force_inference_dtype: torch.dtype | None
    inference_mode: bool

    @classmethod
    def prepare(
        cls,
        X_trains: list[torch.Tensor],
        y_trains: list[torch.Tensor],
        *,
        cat_ix: list[list[list[int]]],
        model: Architecture,
        ensemble_configs: Sequence[EnsembleConfig],
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
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            inference_mode: Whether to use torch inference mode.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
        """
        # We save it as a static seed to be reproducible across predicts
        return cls(
            X_trains=X_trains,
            y_trains=y_trains,
            cat_ix=cat_ix,
            model=model,
            ensemble_configs=ensemble_configs,
            force_inference_dtype=force_inference_dtype,
            inference_mode=inference_mode,
            dtype_byte_size=dtype_byte_size,
            save_peak_mem=save_peak_mem,
        )

    @override
    def iter_outputs(
        self,
        X: list[torch.Tensor],
        *,
        devices: Sequence[torch.device],
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        # This engine currently only supports one device, so just take the first.
        device = devices[0]

        self.model = self.model.to(device)
        ensemble_size = len(self.X_trains)
        for i in range(ensemble_size):
            train_x_full = torch.cat([self.X_trains[i], X[i]], dim=-2)
            train_y_batch = self.y_trains[i]
            train_x_full = train_x_full.to(device)
            train_y_batch = train_y_batch.to(device)
            if self.force_inference_dtype is not None:
                train_x_full = train_x_full.type(self.force_inference_dtype)
                train_y_batch = train_y_batch.type(self.force_inference_dtype)  # type: ignore

            with (
                torch.autocast(device.type, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                output = self.model(
                    train_x_full.transpose(0, 1),
                    train_y_batch.transpose(0, 1),
                    only_return_standard_out=True,
                    categorical_inds=list([cat_item[i] for cat_item in self.cat_ix]),  # noqa: C411
                )

            yield output, self.ensemble_configs[i]
        if self.inference_mode:  ## if inference
            self.model = self.model.cpu()

    @override
    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
        self.inference_mode = use_inference


@dataclass
class InferenceEngineCachePreprocessing(InferenceEngine):
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
    cat_ixs: Sequence[list[int]]
    ensemble_configs: Sequence[EnsembleConfig]
    preprocessors: Sequence[SequentialFeatureTransformer]
    model: Architecture
    force_inference_dtype: torch.dtype | None
    inference_mode: bool
    no_preprocessing: bool = False

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        cat_ix: list[int],
        model: Architecture,
        ensemble_configs: Sequence[EnsembleConfig],
        n_workers: int,
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
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            n_workers: The number of workers to use.
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
            n_workers=n_workers,
            parallel_mode="block",
        )
        configs, preprocessors, X_trains, y_trains, cat_ixs = list(zip(*itr))
        return InferenceEngineCachePreprocessing(
            X_trains=X_trains,
            y_trains=y_trains,
            model=model,
            cat_ixs=cat_ixs,
            ensemble_configs=configs,
            preprocessors=preprocessors,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
            inference_mode=inference_mode,
            no_preprocessing=no_preprocessing,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        devices: Sequence[torch.device],
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        if self.force_inference_dtype is not None:
            self.model.type(self.force_inference_dtype)

        if self.no_preprocessing:
            X_tests = (X for _ in range(len(self.ensemble_configs)))
        else:
            X_tests = (
                preprocessor.transform(X).X for preprocessor in self.preprocessors
            )

        model_forward_functions = (
            partial(
                self._call_model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                cat_ix=cat_ix,
                autocast=autocast,
                only_return_standard_out=only_return_standard_out,
            )
            for X_train, X_test, y_train, cat_ix in zip(
                self.X_trains, X_tests, self.y_trains, self.cat_ixs
            )
        )
        outputs = parallel_execute(devices, model_forward_functions)

        for output, config in zip(outputs, self.ensemble_configs):
            yield _move_and_squeeze_output(output, devices[0]), config

        if self.inference_mode:
            self.model.cpu()

    def _call_model(
        self,
        *,
        device: torch.device,
        is_parallel: bool,
        X_train: torch.Tensor | np.ndarray,
        X_test: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        cat_ix: list[int],
        autocast: bool,
        only_return_standard_out: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute a model forward pass on the provided device.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        # If several estimators are being run in parallel, then each thread needs its
        # own copy of the model so it can move it to its device.
        model = deepcopy(self.model) if is_parallel else self.model
        model.to(device)

        X_full, y_train = _prepare_model_inputs(
            device, self.force_inference_dtype, X_train, X_test, y_train
        )
        batched_cat_ix = [cat_ix]

        if self.inference_mode:
            MemoryUsageEstimator.reset_peak_memory_if_required(
                save_peak_mem=self.save_peak_mem,
                model=model,
                X=X_full,
                cache_kv=False,
                device=device,
                dtype_byte_size=self.dtype_byte_size,
                safety_factor=1.2,
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
            )

    @override
    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
        self.inference_mode = use_inference


@dataclass
class InferenceEngineCacheKV(InferenceEngine):
    """Inference engine that caches the actual KV cache calculated from the context
    of the processed training data.

    This is by far the most memory intensive inference engine, as for each ensemble
    member we store the full KV cache of that model. For now this is held in CPU RAM
    (TODO(eddiebergman): verify)
    """

    preprocessors: list[SequentialFeatureTransformer]
    ensemble_configs: Sequence[EnsembleConfig]
    cat_ixs: Sequence[list[int]]
    models: list[Architecture]
    n_train_samples: list[int]
    force_inference_dtype: torch.dtype | None

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
        ensemble_configs: Sequence[EnsembleConfig],
        n_workers: int,
        model: Architecture,
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
            n_workers: The number of workers to use.
            model: The model to use.
            devices: The devices to run the model on.
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
            n_workers=n_workers,
            parallel_mode="as-ready",
        )
        models: list[Architecture] = []
        preprocessors: list[SequentialFeatureTransformer] = []
        correct_order_configs: list[EnsembleConfig] = []
        cat_ixs: Sequence[list[int]] = []
        n_train_samples: list[int] = []

        for config, preprocessor, X, y, preprocessor_cat_ix in itr:
            cat_ixs.append(preprocessor_cat_ix)
            preprocessors.append(preprocessor)
            correct_order_configs.append(config)
            n_train_samples.append(len(y))

            ens_model = deepcopy(model)
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

            if device.type != "cpu":
                ens_model = ens_model.cpu()

            models.append(ens_model)

        return InferenceEngineCacheKV(
            preprocessors=preprocessors,
            ensemble_configs=correct_order_configs,
            cat_ixs=cat_ixs,
            n_train_samples=n_train_samples,
            models=models,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        devices: Sequence[torch.device],
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        # This engine currently only supports one device, so just take the first.
        device = devices[0]

        for preprocessor, model, config, cat_ix, X_train_len in zip(
            self.preprocessors,
            self.models,
            self.ensemble_configs,
            self.cat_ixs,
            self.n_train_samples,
        ):
            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            X_test = X_test.unsqueeze(1)
            batched_cat_ix = [cat_ix]

            MemoryUsageEstimator.reset_peak_memory_if_required(
                save_peak_mem=self.save_peak_mem,
                model=model,
                X=X_test,
                cache_kv=True,
                device=device,
                dtype_byte_size=self.dtype_byte_size,
                safety_factor=1.2,  # TODO(Arjun): make customizable
                n_train_samples=X_train_len,
            )

            model = model.to(device)  # noqa: PLW2901

            if self.force_inference_dtype is not None:
                model = model.type(self.force_inference_dtype)  # noqa: PLW2901
                X_test = X_test.type(self.force_inference_dtype)
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(),
            ):
                output = model(
                    X_test,
                    y=None,
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_cat_ix,
                )

            # TODO(eddiebergman): This is not really what we want.
            # We'd rather just say unload from GPU, we already have it available on CPU.
            model = model.cpu()  # noqa: PLW2901

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, config


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
