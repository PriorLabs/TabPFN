"""Module that defines different ways to run inference with TabPFN."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import dataclasses
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import copy, deepcopy
from functools import partial
from inspect import Parameter, signature
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar
from typing_extensions import override

import joblib
import numpy as np
import torch

from tabpfn.architectures.interface import EstimatorBatchBudget
from tabpfn.architectures.kv_cache import KV_CACHE_PRECISION_DTYPES, KVCache
from tabpfn.architectures.shared.workarounds_mps_bugs import (
    maybe_replace_linears_on_mps,
)
from tabpfn.constants import DEFAULT_SAVE_PEAK_MEMORY_FACTOR, MemorySavingMode
from tabpfn.memory import should_save_peak_mem
from tabpfn.parallel_execute import parallel_execute
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable
from tabpfn.preprocessing.torch.member_rows import MemberRows, MemberRowSource
from tabpfn.settings import settings
from tabpfn.utils import as_tensor_to_read, get_autocast_context

if TYPE_CHECKING:
    from tabpfn.architectures.interface import Architecture, PerformanceOptions
    from tabpfn.architectures.kv_cache import KVCache
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
    """Wraps an iterator, accumulating wall-clock time spent in ``__next__``.

    On a single device it forces completion on that device before recording each
    step, so the timing captures real GPU compute rather than kernel-launch time.
    On multiple devices parallel_execute already waits on a per-output CUDA event,
    so no extra sync is added.
    """

    def __init__(self, inner: Iterator[_T], devices: Sequence[torch.device]) -> None:
        super().__init__()
        self._inner = inner
        self._device = devices[0]
        self._synchronize = len(devices) == 1
        self.elapsed_seconds: float = 0.0

    @override
    def __next__(self) -> _T:
        start = time.perf_counter()
        value = next(self._inner)
        if self._synchronize:
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            elif self._device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        self.elapsed_seconds += time.perf_counter() - start
        return value

    @override
    def __iter__(self) -> _TimedIterator[_T]:
        return self


def _forward_parameters(model: torch.nn.Module) -> Mapping[str, Parameter]:
    """The parameters of `model.forward`, by name."""
    return signature(model.forward).parameters


def _model_forward_kwargs(
    model: Architecture, task_type: str, estimator_ids: list[int]
) -> dict[str, str | list[int]]:
    """Return the metadata arguments accepted by the architecture."""
    parameters = _forward_parameters(model)
    kwargs: dict[str, str | list[int]] = {}
    if "task_type" in parameters:
        kwargs["task_type"] = task_type
    if "estimator_ids" in parameters:
        kwargs["estimator_ids"] = estimator_ids
    return kwargs


def _estimator_batch_budget(model: Architecture) -> EstimatorBatchBudget:
    """The model's batching budget, with the settings overriding either part.

    A model whose budget is zero does not batch estimators, whatever the settings.
    """
    budget = model.estimator_batch_budget
    if not _batches(budget):
        return budget
    rows = settings.tabpfn.max_batched_estimator_rows
    cells = settings.tabpfn.max_batched_estimator_cells
    return EstimatorBatchBudget(
        rows=budget.rows if rows is None else rows,
        cells=budget.cells if cells is None else cells,
    )


def _shared_estimator_batch_budget(
    model_caches: Sequence[_PerDeviceModelCache],
) -> EstimatorBatchBudget:
    """The tightest budget over the models, for grouping before the model is known.

    Models that do not batch contribute nothing; if none does, the budget is zero.
    """
    budgets = [_estimator_batch_budget(cache.get_any()) for cache in model_caches]
    batching = [b for b in budgets if _batches(b)]
    if not batching:
        return EstimatorBatchBudget(rows=0, cells=0)
    return EstimatorBatchBudget(
        rows=min(b.rows for b in batching), cells=min(b.cells for b in batching)
    )


def _batches(budget: EstimatorBatchBudget) -> bool:
    """Whether the budget lets any two estimators share a forward pass."""
    return budget.rows > 0 and budget.cells > 0


def _members_per_forward(
    rows_per_member: int, columns_per_member: int, budget: EstimatorBatchBudget
) -> int:
    """How many members of the given prepared shape share one forward pass.

    Bounded by the budget's rows and cells (rows times columns), summed over the
    members of the forward; either at zero runs every member alone.
    """
    by_rows = budget.rows // max(rows_per_member, 1)
    by_cells = budget.cells // max(rows_per_member * columns_per_member, 1)
    return max(1, min(by_rows, by_cells))


def _prepare_member_inputs(
    member: TabPFNEnsembleMember,
    X_test: torch.Tensor | np.ndarray | SharedColumnsTable,
    device: torch.device,
    force_inference_dtype: torch.dtype | None,
    model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """A member's train rows, then ``X_test``, as one ``(rows, 1, C)`` table, its
    targets, and categorical indices.

    The table stays in host memory where `_model_input_device` keeps it for
    ``model``, and the GPU preprocessing reads it from there.
    """
    X, y = _prepare_model_inputs(
        device,
        force_inference_dtype,
        member.X_train,
        X_test,
        member.y_train,
        x_device=_model_input_device(
            model, device, member.X_train, X_test, force_inference_dtype
        ),
    )
    # `X` is this function's own copy, so the steps write into it a block of
    # columns at a time.
    X, feature_schema = _maybe_run_gpu_preprocessing(
        X,
        gpu_preprocessor=member.gpu_preprocessor,
        num_train_rows=member.X_train.shape[0],
        feature_schema=member.feature_schema,
        consume_input=True,
        compute_device=device,
    )
    if force_inference_dtype is not None:
        # A step may compute and return float32 under a lower precision.
        X = X.type(force_inference_dtype)
    return X, y, feature_schema.indices_for(FeatureModality.CATEGORICAL)


def _member_rows(
    member: TabPFNEnsembleMember,
    X_test: torch.Tensor | np.ndarray | SharedColumnsTable,
    device: torch.device,
    force_inference_dtype: torch.dtype | None,
    model: torch.nn.Module,
) -> tuple[MemberRows, torch.Tensor, list[int]] | None:
    """A member's model input to read one block at a time, its targets and its
    categorical indices, for a model that reads row sources from host memory.

    Returns None if the input must be built whole:

    - the model reads no row source,
    - the inference dtype is forced to a dtype other than float32,
    - the input goes to the device, or
    - a step of the member's GPU preprocessing reads whole rows.
    """
    if (
        not getattr(model, "reads_row_sources", False)
        or force_inference_dtype not in (None, torch.float32)
        or _model_input_device(
            model, device, member.X_train, X_test, force_inference_dtype
        )
        == device
    ):
        return None
    rows = MemberRows.fit(
        member.X_train, X_test, member.gpu_preprocessor, member.feature_schema, device
    )
    if rows is None:
        warnings.warn(
            "A step of the member's GPU preprocessing reads whole rows, so the "
            "member's model input is built whole in host memory.",
            stacklevel=2,
        )
        return None
    return (
        rows,
        torch.as_tensor(member.y_train, dtype=torch.float32, device=device),
        rows.feature_schema.indices_for(FeatureModality.CATEGORICAL),
    )


def _cache_builds(
    positions: list[int],
    rows_per_member: int,
    columns_per_member: int,
    budget: EstimatorBatchBudget,
) -> list[list[list[int]]]:
    """The forwards that build each cache of equal-shape members.

    Returns one list of forwards per cache; each forward lists the members it
    builds together, as many as the budgets allow. A budget of zero gives every
    member a cache of its own, so predict runs them one at a time as the
    full-forward engines do.
    """
    if not _batches(budget):
        return [[[i]] for i in positions]
    per_forward = _members_per_forward(rows_per_member, columns_per_member, budget)
    return [
        [
            positions[start : start + per_forward]
            for start in range(0, len(positions), per_forward)
        ]
    ]


def _member_key(
    member: TabPFNEnsembleMember,
    index: int,
    model_caches: list[_PerDeviceModelCache],
    devices: Sequence[torch.device],
) -> tuple:
    """What a member must share with another to run in the same forward.

    Members of an architecture that does not batch ensemble members get a key of
    their own, so they always run alone. So do all members on MPS, where the
    attention kernels of recent torch builds give wrong results for a batch of
    members.
    """
    model = model_caches[member.config._model_index].get_any()
    if not _batches(model.estimator_batch_budget) or any(
        d.type == "mps" for d in devices
    ):
        return ("single", index)
    return (member.config._model_index, tuple(member.X_train.shape))


def _group_equal(keys: Sequence[tuple]) -> list[list[int]]:
    """Indices grouped by equal key, in order of each key's first appearance."""
    by_key: dict[tuple, list[int]] = {}
    for index, key in enumerate(keys):
        by_key.setdefault(key, []).append(index)
    return list(by_key.values())


def _member_groups(
    keys: Sequence[tuple], size: int, *, num_devices: int = 1
) -> list[list[int]]:
    """Member indices grouped by key, in runs of at most ``size``.

    The members of a key are first spread over the devices, so that every device
    gets a group, and only the members left per device share a forward. Groups
    follow the first member of each key, so an ensemble whose members all share a
    key yields consecutive runs in member order.
    """
    groups: list[list[int]] = []
    for indices in _group_equal(keys):
        per_device = -(-len(indices) // max(num_devices, 1))
        run = max(1, min(size, per_device))
        groups.extend(
            indices[start : start + run] for start in range(0, len(indices), run)
        )
    return groups


_ModelOutput = torch.Tensor | dict[str, torch.Tensor]
# A forward's output with the members along dimension 1, and which positions of the
# group of members it holds.
_MemberOutputs = list[tuple[list[int], _ModelOutput]]


def _member_output(output: _ModelOutput, index: int) -> _ModelOutput:
    """Member ``index`` of a batched output, its batch dimension kept."""
    if isinstance(output, dict):
        return {k: v[:, index : index + 1] for k, v in output.items()}
    return output[:, index : index + 1]


def _yield_in_member_order(
    groups: Sequence[Sequence[int]],
    group_outputs: Iterator[_MemberOutputs],
    total: int,
) -> Iterator[_ModelOutput]:
    """Unstack the groups' outputs and hand them out in member order.

    Members are yielded as soon as every earlier member is done, so groups that
    arrive in member order stream through.
    """
    outputs: list[_ModelOutput | None] = [None] * total
    next_member = 0
    # `groups` may still be growing while the outputs are produced, so it is read
    # by position rather than zipped.
    for group_index, member_outputs in enumerate(group_outputs):
        for positions, output in member_outputs:
            for column, position in enumerate(positions):
                outputs[groups[group_index][position]] = _member_output(output, column)
        while next_member < total and outputs[next_member] is not None:
            yield outputs[next_member]  # type: ignore[misc]
            outputs[next_member] = None
            next_member += 1
    assert next_member == total


def _iter_chunks(items: Iterator[_T], size: int) -> Iterator[list[_T]]:
    chunk: list[_T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _batch_member_inputs(
    xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Members' ``(rows, 1, C)`` inputs and ``(num_train,)`` targets as one batch.

    Targets given as ``(num_train, 1)`` are accepted too; the batch has them as
    ``(num_train, B)``. A single member passes through untouched, so its forward is
    the same call as without batching.
    """
    if len(xs) == 1:
        return xs[0], ys[0]
    return torch.cat(list(xs), dim=1), _batch_targets(ys)


def _batch_targets(ys: Sequence[torch.Tensor]) -> torch.Tensor:
    """Batch the members' targets as `_batch_member_inputs` does."""
    if len(ys) == 1:
        return ys[0]
    return torch.cat([y.reshape(y.shape[0], -1) for y in ys], dim=1)


def _row_weight_kwargs(
    members: Sequence[TabPFNEnsembleMember],
    sample_weight_mode: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | str]:
    """The members' per-row weights as forward arguments, batched like their rows.

    The weights take the `device` and `dtype` of the members' batched rows. Empty
    when `fit` got no per-row weights, so the forward is the same call as for a
    model that takes none.
    """
    if sample_weight_mode is None:
        return {}
    row_weights = []
    for member in members:
        assert member.row_weights is not None
        row_weights.append(
            torch.as_tensor(member.row_weights, device=device, dtype=dtype)
        )
    return {
        "row_weights_NB2": torch.stack(row_weights, dim=1),
        "sample_weight_mode": sample_weight_mode,
    }


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
        """Persist the executor state to ``path`` without the model weights."""
        joblib.dump(self._create_copy_for_pickling(), path)

    @abstractmethod
    def _create_copy_for_pickling(self) -> InferenceEngine:
        """Return a copy of the inference engine ready for pickling.

        This should remove the models, which we don't want to include in the pickled
        file. A shallow copy is safe as the copy is only pickled, never changed.
        """
        ...

    @staticmethod
    def load_state(path: str | Path, models: list[Architecture]) -> InferenceEngine:
        """Load an executor saved to disk with save_state_except_model_weights().

        The state on disk does not include the models, so these must be provided as the
        `models` parameter.
        """
        engine: InferenceEngine = joblib.load(Path(path))
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
        state_copy = copy(self)
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
        state_copy = copy(self)
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
        enable_torch_compile: bool = False,
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
            enable_torch_compile: Enable the architecture's compiled regions.
        """
        self.enable_torch_compile = enable_torch_compile
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
        # Members are preprocessed lazily, so only one chunk of them is held at a
        # time; each chunk is grouped by model and prepared shape. The members are
        # spread over the devices first; only the rest share a forward.
        num_members = len(self.ensemble_preprocessor.configs)
        # The members' prepared width is not known before they are preprocessed;
        # the feature cap of the ensemble config bounds it.
        columns = min(
            self.X_train.shape[1],
            self.ensemble_preprocessor.configs[
                0
            ].preprocess_config.max_features_per_estimator,
        )
        chunk_size = _members_per_forward(
            self.X_train.shape[0] + X.shape[0],
            columns,
            _shared_estimator_batch_budget(self.model_caches),
        )
        chunk_size = max(1, min(chunk_size, -(-num_members // len(devices))))
        groups: list[list[int]] = []
        seen = 0

        def model_forward_functions() -> Iterator[partial]:
            nonlocal seen
            for chunk in _iter_chunks(ensemble_members_iterator, chunk_size):
                keys = [
                    _member_key(em, i, self.model_caches, devices)
                    for i, em in enumerate(chunk)
                ]
                for positions in _member_groups(keys, len(chunk)):
                    members = [chunk[i] for i in positions]
                    groups.append([seen + i for i in positions])
                    yield partial(
                        self._call_model,
                        members=members,
                        X_tests=[em.transform_X_test(X) for em in members],
                        autocast=autocast,
                        only_return_standard_out=only_return_standard_out,
                        save_peak_mem=save_peak_mem,
                        task_type=task_type,
                    )
                seen += len(chunk)

        timed_outputs = _TimedIterator(
            parallel_execute(
                devices,
                model_forward_functions(),
                prewarm_lapack=self.ensemble_preprocessor.any_estimator_uses_gpu_svd(),
            ),
            devices,
        )
        for config, output in zip(
            self.ensemble_preprocessor.configs,
            _yield_in_member_order(groups, timed_outputs, num_members),
            strict=True,
        ):
            yield _move_and_squeeze_output(output, devices[0]), config

        self._speed_metrics["predict_model_forward_seconds"] = (
            timed_outputs.elapsed_seconds
        )

    def _call_model(
        self,
        *,
        device: torch.device,
        members: list[TabPFNEnsembleMember],
        X_tests: list[torch.Tensor | np.ndarray],
        autocast: bool,
        only_return_standard_out: bool,
        save_peak_mem: bool,
        task_type: str,
    ) -> _MemberOutputs:
        """Run the members' forward passes on the given device, batched when possible.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        return _call_model_on_members(
            self.model_caches,
            device=device,
            members=members,
            X_tests=X_tests,
            force_inference_dtype=self.force_inference_dtype,
            autocast=autocast,
            enable_torch_compile=self.enable_torch_compile,
            inference_mode=True,
            only_return_standard_out=only_return_standard_out,
            save_peak_mem=save_peak_mem,
            task_type=task_type,
            sample_weight_mode=self.ensemble_preprocessor.sample_weight_mode,
        )


class InferenceEngineBatchedNoPreprocessing(SingleDeviceInferenceEngine):
    """Inference engine that uses preprocessed inputs, and allows batched predictions
    on several datasets at once.
    """

    def __init__(  # noqa: PLR0913
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
        performance_options: PerformanceOptions,
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
            performance_options: Performance and memory options forwarded to
                the model on each forward call.
        """
        # Each entry of ``ensemble_configs`` is one estimator slot holding one
        # config per dataset in the batch (length == batch size). A single fused
        # forward is run per estimator over the whole dataset batch, so all
        # datasets in a batch must use the same underlying model.
        for ensemble_config in ensemble_configs:
            if len({cfg._model_index for cfg in ensemble_config}) > 1:
                raise ValueError(
                    "Batched inference requires all datasets in a batch to use the"
                    " same model; got multiple model indices in one estimator slot."
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
        self.performance_options = performance_options

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

        def _forward_outputs() -> Iterator[
            tuple[torch.Tensor | dict, list[EnsembleConfig]]
        ]:
            for i in range(batch_size):
                train_x_full = torch.cat([self.X_trains[i], X[i]], dim=-2)
                train_y_batch = self.y_trains[i]
                train_x_full = train_x_full.to(device)
                train_y_batch = train_y_batch.to(device)
                model = self.models[self.ensemble_configs[i][0]._model_index]
                if self.force_inference_dtype is not None:
                    train_x_full = train_x_full.type(self.force_inference_dtype)
                    train_y_batch = train_y_batch.type(self.force_inference_dtype)  # type: ignore
                    model.type(self.force_inference_dtype)

                kwargs = _model_forward_kwargs(
                    model,
                    task_type,
                    [config.estimator_id for config in self.ensemble_configs[i]],
                )
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
                        performance_options=self.performance_options,
                        **kwargs,
                    )
                yield output, self.ensemble_configs[i]

        timed_outputs = _TimedIterator(_forward_outputs(), [device])
        yield from timed_outputs
        self._speed_metrics["predict_model_forward_seconds"] = (
            timed_outputs.elapsed_seconds
        )

    @override
    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
        self.inference_mode = use_inference

    @override
    def _move_models_to_devices(self, devices: Sequence[torch.device]) -> None:
        # As this inference engine only supports one device, just take the first.
        device = devices[0]
        for model in self.models:
            # Apply a workaround for a nn.Linear bug on MPS. No-op if MPS not selected.
            maybe_replace_linears_on_mps(model, [device])
            model.to(device)


class InferenceEngineCachePreprocessing(MultiDeviceInferenceEngine):
    """Inference engine that caches the preprocessing for feeding as model context on
    predict.

    This will fit the preprocessors on the training data, as well as cache the
    transformed training data on RAM (not GPU RAM).

    This saves some time on each predict call, at the cost of increasing the amount
    of memory in RAM. The main functionality performed at `predict()` time is the
    forward pass through the model, run for as many ensemble members at once as
    the row and cell budgets allow: the model's ``estimator_batch_budget``, which
    the ``TABPFN_MAX_BATCHED_ESTIMATOR_ROWS`` and ``_CELLS`` settings override.
    """

    def __init__(  # noqa: PLR0913
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
        enable_torch_compile: bool = False,
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
            enable_torch_compile: Enable the architecture's compiled regions.
            inference_mode: Whether to use torch.inference mode
                (this is quicker but disables backpropagation)
            no_preprocessing: If True, skip preprocessing on test data.
                Used for differentiability.
        """
        self.enable_torch_compile = enable_torch_compile
        super().__init__(
            model_caches=[_PerDeviceModelCache(model) for model in models],
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.inference_mode = inference_mode
        self.no_preprocessing = no_preprocessing
        self.X_train_shape_before_preprocessing = X_train.shape
        self.ensemble_preprocessor = ensemble_preprocessor

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

        members = self.ensemble_members
        groups = _member_groups(
            [
                _member_key(em, i, self.model_caches, devices)
                for i, em in enumerate(members)
            ],
            _members_per_forward(
                members[0].X_train.shape[0] + X.shape[0],
                members[0].X_train.shape[1],
                _shared_estimator_batch_budget(self.model_caches),
            ),
            num_devices=len(devices),
        )
        model_forward_functions = (
            partial(
                self._call_model,
                members=[members[i] for i in group],
                X_tests=[_transform_X_test(members[i]) for i in group],
                autocast=autocast,
                only_return_standard_out=only_return_standard_out,
                save_peak_mem=save_peak_mem,
                task_type=task_type,
            )
            for group in groups
        )

        timed_outputs = _TimedIterator(
            parallel_execute(
                devices,
                model_forward_functions,
                prewarm_lapack=self.ensemble_preprocessor.any_estimator_uses_gpu_svd(),
            ),
            devices,
        )

        for output, ensemble_member in zip(
            _yield_in_member_order(groups, timed_outputs, len(members)),
            members,
            strict=True,
        ):
            yield _move_and_squeeze_output(output, devices[0]), ensemble_member.config

        self._speed_metrics["predict_model_forward_seconds"] = (
            timed_outputs.elapsed_seconds
        )

    def _call_model(
        self,
        *,
        device: torch.device,
        members: list[TabPFNEnsembleMember],
        X_tests: list[torch.Tensor | np.ndarray],
        autocast: bool,
        only_return_standard_out: bool,
        save_peak_mem: bool,
        task_type: str,
    ) -> _MemberOutputs:
        """Run the members' forward passes on the given device, batched when possible.

        Note that several instances of this function may be executed in parallel in
        different threads, one for each device in the system.
        """
        return _call_model_on_members(
            self.model_caches,
            device=device,
            members=members,
            X_tests=X_tests,
            force_inference_dtype=self.force_inference_dtype,
            autocast=autocast,
            enable_torch_compile=self.enable_torch_compile,
            inference_mode=self.inference_mode,
            only_return_standard_out=only_return_standard_out,
            save_peak_mem=save_peak_mem,
            task_type=task_type,
            sample_weight_mode=self.ensemble_preprocessor.sample_weight_mode,
        )

    @override
    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
        self.inference_mode = use_inference


def _resolve_kv_cache_precision(
    kv_cache_precision: Literal["auto", "int8", "fp8", "adaptive"] | None,
    *,
    architecture: Architecture,
    device: torch.device,
) -> Literal["auto", "int8", "fp8", "adaptive"]:
    """Resolve the KV cache dtype against ``architecture`` and ``device``.

    Raises:
        ValueError: If ``"fp8"`` is requested on MPS, which has no float8
            casts. Rejected here, before any fitting work starts; the rest of
            the fp8 path can then assume a capable device.
    """
    if kv_cache_precision == "fp8" and device.type == "mps":
        raise ValueError(
            "kv_cache_precision='fp8' is not supported on MPS: PyTorch cannot "
            "cast to float8 dtypes there. Use 'int8' (the default) or 'auto'."
        )
    supported = architecture.get_supported_kv_cache_precisions()
    if kv_cache_precision is None:
        return "int8" if "int8" in supported else "auto"
    if kv_cache_precision not in supported:
        warnings.warn(
            f"kv_cache_precision={kv_cache_precision!r} is not supported by "
            f"{type(architecture).__name__} (supported: {supported}); "
            "falling back to 'auto' (the KV cache is not quantized).",
            UserWarning,
            stacklevel=2,
        )
        return "auto"
    return kv_cache_precision


class InferenceEngineExplicitKVCache(MultiDeviceInferenceEngine):
    """Inference engine with explicit KV cache passed through forward().

    The KV cache is stored externally (not inside the model) and passed explicitly to
    the model's forward pass. This avoids deepcopying the model per ensemble member and
    keeps the model stateless.

    Each ensemble member (estimator) has its own KV cache.
    Ensemble members are dispatched across available GPUs
    via :func:`parallel_execute`.

    When ``keep_cache_on_device=True``, each per-estimator cache is kept
    on the GPU for subsequent prediction calls, avoiding CPU↔GPU transfers.

    For TabPFN-3 models, ``kv_cache_precision="int8"`` (the default) stores the KV
    cache with per-tensor symmetric quantization to save memory, dequantizing
    on-the-fly in the attention layer; ``"fp8"`` stores it as 8-bit floats
    instead (same size, float rounding semantics); ``"auto"`` keeps the
    computed dtype; ``"adaptive"`` stores each layer on the grid an attention
    backend already rounded its keys and values to (``kv_grid_dtype``), and as
    ``"int8"`` where no backend did.

    At predict, only X_test is preprocessed (CPU and GPU). The model is
    called with ``x_is_test_only=True``. ``y`` still carries the full
    train labels for the many-class decoder.
    """

    def __init__(  # noqa: PLR0913
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
        task_type: str,
        keep_cache_on_device: bool = True,
        kv_cache_precision: Literal["auto", "int8", "fp8", "adaptive"] | None = None,
        kv_cache_on_host: bool | None = None,
        enable_torch_compile: bool = False,
    ) -> None:
        """Initialize the explicit KV cache inference engine.

        Builds a KV cache per ensemble member in parallel across devices.
        Each estimator uses a different data permutation, so each cache is
        unique.

        Args:
            X_train: The training data.
            y_train: The training target.
            ensemble_preprocessor: The ensemble preprocessor to use.
            models: The models to use.
            devices: Devices to use for inference. If multiple devices are
                specified, ensemble members are parallelised across them
                during both cache build and prediction.
            dtype_byte_size: Size of the dtype in bytes.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            enable_torch_compile: Enable the architecture's compiled regions.
            autocast: Whether to use torch.autocast during cache build.
            task_type: The task type, e.g. "multiclass" or "regression". Needed
                at build time because the cached activations are task-specific,
                so a cache built for the wrong task would be silently wrong.
            keep_cache_on_device: If True (default), keep each per-estimator
                KV cache on the device where it was built.  Uses more device
                memory but avoids CPU↔GPU transfers, giving lower latency.
                When False, caches are moved to CPU after building and
                transferred to the target device on every predict call.
            kv_cache_precision: Dtype the KV cache is stored in, resolved against
                what the architecture supports (see
                :func:`_resolve_kv_cache_precision`). ``None`` (default) picks the
                architecture default (``"int8"`` when it can quantize, else
                ``"auto"``); ``"int8"`` quantizes to save memory; ``"fp8"``
                stores 8-bit floats instead; ``"auto"`` keeps the computed
                dtype; ``"adaptive"`` follows the grid an attention backend
                left the keys and values on, else ``"int8"``.
            kv_cache_on_host: Whether each cache keeps its keys and values in
                host memory, from where a predict copies to the device only what
                it reads. ``None`` (default) lets the architecture decide; an
                architecture whose cache cannot be read from host memory ignores
                it.
        """
        self.enable_torch_compile = enable_torch_compile
        super().__init__(
            model_caches=[_PerDeviceModelCache(model) for model in models],
            save_peak_mem=save_peak_mem,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

        self.keep_cache_on_device = keep_cache_on_device
        # Kept as the raw request; resolved per ensemble member in _build_cache
        # against that member's own architecture (see _resolve_kv_cache_precision).
        self.kv_cache_precision = kv_cache_precision
        self.kv_cache_on_host = kv_cache_on_host
        self.ensemble_preprocessor = ensemble_preprocessor
        self.task_type = task_type

        # Place model copies on all devices before building caches
        self.to(devices, self.force_inference_dtype, self.dtype_byte_size)

        save_peak_mem_during_build = should_save_peak_mem(
            memory_saving_mode=save_peak_mem,
            X_train_shape=tuple[int, int](X_train.shape),
            X_test_shape=(0, X_train.shape[1]),
            devices=devices,
            dtype_byte_size=dtype_byte_size,
        )
        stage_caches_on_cpu = keep_cache_on_device and save_peak_mem_during_build

        if self._builds_member_by_member(X_train, devices):
            self._build_caches_member_by_member(
                X_train,
                y_train,
                device=devices[0],
                autocast=autocast,
                save_peak_mem=save_peak_mem_during_build,
                stage_cache_on_cpu=stage_caches_on_cpu,
            )
            return

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

        # Members sharing a model and a prepared shape end up in one cache, with
        # the members along the batch dimension, so that predict runs them as one
        # batch. The members are spread over the devices first; how many of them
        # build together in one forward is bounded inside _build_cache.
        members = self.ensemble_members
        groups = _member_groups(
            [
                _member_key(em, i, self.model_caches, devices)
                for i, em in enumerate(members)
            ],
            len(members),
            num_devices=len(devices),
        )
        build_functions = (
            partial(
                self._build_cache,
                members=[members[i] for i in group],
                model_index=members[group[0]].config._model_index,
                train_shape=tuple[int, int](members[group[0]].X_train.shape),
                autocast=autocast,
                save_peak_mem=save_peak_mem_during_build,
                stage_cache_on_cpu=stage_caches_on_cpu,
            )
            for group in groups
        )
        timed_caches = _TimedIterator(
            parallel_execute(
                devices,
                build_functions,
                prewarm_lapack=self.ensemble_preprocessor.any_estimator_uses_gpu_svd(),
            ),
            devices,
        )
        built_caches = list(timed_caches)
        # One cache per group of members; `cache_groups[i]` lists the members whose
        # batch positions `kv_caches[i]` holds, in order.
        self.cache_groups: list[list[int]] = []
        self.kv_caches: list = []
        for group, (caches, device) in zip(groups, built_caches, strict=True):
            for positions, cache in caches:
                self.cache_groups.append([group[i] for i in positions])
                # A cache staged on CPU by _build_cache while the remaining groups
                # were being built moves back to its build device now.
                self.kv_caches.append(
                    cache.to(device) if stage_caches_on_cpu else cache
                )
        self._speed_metrics["fit_model_forward_seconds"] = timed_caches.elapsed_seconds

        # The caches replace the training features; predict only reads y_train.
        for ensemble_member in self.ensemble_members:
            ensemble_member.X_train = None

    def _builds_member_by_member(
        self, X_train: np.ndarray, devices: Sequence[torch.device]
    ) -> bool:
        """Whether the members are preprocessed one forward at a time for the build.

        For train rows that `_model_input_device` keeps in host memory, on one
        device and with one model: every member holds its own copy of the train
        rows, so preprocessing them all first holds one copy per member.

        Not on MPS: that build batches the members, and MPS runs every member
        alone (see `_member_key`).
        """
        if len(devices) != 1 or len(self.model_caches) != 1:
            return False
        if devices[0].type == "mps":
            return False
        model = self.model_caches[0].get_any()
        no_test_rows = np.empty((0, X_train.shape[1]), dtype=np.float32)
        x_device = _model_input_device(
            model, devices[0], X_train, no_test_rows, self.force_inference_dtype
        )
        return x_device != devices[0]

    def _build_caches_member_by_member(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
        save_peak_mem: bool,
        stage_cache_on_cpu: bool,
    ) -> None:
        """Build the caches from members that are preprocessed as they are built.

        Host memory then holds the train rows of one forward's members at a time,
        as in `InferenceEngineOnDemand`.
        """
        self.ensemble_members = []
        preprocessing_seconds = 0.0

        def members() -> Iterator[TabPFNEnsembleMember]:
            nonlocal preprocessing_seconds
            iterator = (
                self.ensemble_preprocessor.fit_transform_ensemble_members_iterator(
                    X_train=X_train, y_train=y_train, parallel_mode="in-order"
                )
            )
            while True:
                start = time.perf_counter()
                member = next(iterator, None)
                preprocessing_seconds += time.perf_counter() - start
                if member is None:
                    return
                self.ensemble_members.append(member)
                yield member

        timed_caches = _TimedIterator(
            parallel_execute(
                [device],
                [
                    partial(
                        self._build_cache,
                        members=members(),
                        model_index=0,
                        train_shape=tuple[int, int](X_train.shape),
                        autocast=autocast,
                        save_peak_mem=save_peak_mem,
                        stage_cache_on_cpu=stage_cache_on_cpu,
                    )
                ],
                prewarm_lapack=self.ensemble_preprocessor.any_estimator_uses_gpu_svd(),
            ),
            [device],
        )
        ((caches, build_device),) = list(timed_caches)
        # `_build_cache` numbers the members in the order it reads them.
        self.cache_groups = [positions for positions, _ in caches]
        self.kv_caches = [
            cache.to(build_device) if stage_cache_on_cpu else cache
            for _, cache in caches
        ]
        self._speed_metrics["fit_preprocessing_seconds"] = preprocessing_seconds
        self._speed_metrics["fit_model_forward_seconds"] = (
            timed_caches.elapsed_seconds - preprocessing_seconds
        )

    def _build_cache(  # noqa: C901
        self,
        *,
        device: torch.device,
        members: Iterable[TabPFNEnsembleMember],
        model_index: int,
        train_shape: tuple[int, int],
        autocast: bool,
        save_peak_mem: bool,
        stage_cache_on_cpu: bool,
    ) -> tuple[list[tuple[list[int], KVCache]], torch.device]:
        """Build the KV caches of a group of ensemble members on the given device.

        Members whose prepared train inputs share a shape end up in one cache with
        the members along the batch dimension. They are built in forwards of as
        many members as the batching budgets allow, and the resulting caches are
        concatenated, so the budgets bound the build but not the batch the cache
        serves at predict.

        The members share the model `model_index` and train rows of `train_shape`.
        They are read one forward's worth at a time, and a member's train rows are
        released once its model input is prepared.

        Called via :func:`parallel_execute` — may run on different devices
        in parallel threads.

        Returns:
            The caches, each with the positions in ``members`` it holds, and the
            build device, which is also the caches' destination for inference. The
            cache tensors themselves are on CPU when ``stage_cache_on_cpu`` is true
            (or when ``keep_cache_on_device`` is false); otherwise, they remain on
            the returned device.
        """
        model = self.model_caches[model_index].get(device)
        kv_cache_precision = _resolve_kv_cache_precision(
            self.kv_cache_precision, architecture=model, device=device
        )
        read: list[TabPFNEnsembleMember] = []
        if self.force_inference_dtype is not None:
            model.type(self.force_inference_dtype)

        performance_options = dataclasses.replace(
            model.get_default_performance_options(),
            enable_torch_compile=self.enable_torch_compile,
            save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
            if save_peak_mem
            else None,
            kv_cache_dtype=(
                KV_CACHE_PRECISION_DTYPES[kv_cache_precision]
                if kv_cache_precision != "auto"
                else None
            ),
            kv_cache_follows_attention_grid=kv_cache_precision == "adaptive",
            kv_cache_on_host=self.kv_cache_on_host,
        )

        def build(
            positions: list[int],
            prepared: dict[int, tuple[torch.Tensor, torch.Tensor, list[int]]],
        ) -> KVCache:
            X, y = _batch_member_inputs(
                [prepared[i][0] for i in positions],
                [prepared[i][1] for i in positions],
            )
            kwargs = _model_forward_kwargs(
                model,
                self.task_type,
                [read[i].config.estimator_id for i in positions],
            )
            # Nothing reads this forward's input after it.
            x, x_kwargs = _model_input_arguments(X, device, given_up_to=model)
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(),
            ):
                _, cache = model(
                    x,
                    y,
                    only_return_standard_out=True,
                    categorical_inds=[prepared[i][2] for i in positions],
                    performance_options=performance_options,
                    return_kv_cache=True,
                    **x_kwargs,
                    **kwargs,
                    **_row_weight_kwargs(
                        [read[i] for i in positions],
                        self.ensemble_preprocessor.sample_weight_mode,
                        X.device,
                        X.dtype,
                    ),
                )
            assert cache is not None
            if not self.keep_cache_on_device or stage_cache_on_cpu:
                cache = cache.to("cpu")
            return cache

        # The device holds the prepared train rows of one forward's members at a
        # time. Their prepared width is not known before they are preprocessed; the
        # feature cap of the ensemble config bounds it, as for a full forward.
        budget = _estimator_batch_budget(model)
        rows, columns = train_shape
        columns = min(
            columns,
            self.ensemble_preprocessor.configs[
                0
            ].preprocess_config.max_features_per_estimator,
        )
        # A model that does not batch members builds them one at a time.
        per_chunk = 1
        if _batches(budget):
            per_chunk = min(
                len(self.ensemble_preprocessor.configs),
                _members_per_forward(rows, columns, budget),
            )
        # Per prepared shape, the forwards that build its caches, in member order.
        builds: dict[tuple[int, ...], list[tuple[list[int], KVCache]]] = {}
        for chunk_members in _iter_chunks(iter(members), per_chunk):
            chunk = list(range(len(read), len(read) + len(chunk_members)))
            read.extend(chunk_members)
            prepared = {}
            for i in chunk:
                no_test_rows = np.empty((0, read[i].X_train.shape[1]), np.float32)
                prepared[i] = _prepare_member_inputs(
                    read[i], no_test_rows, device, self.force_inference_dtype, model
                )
                # The cache replaces the train rows; predict only reads `y_train`.
                read[i].X_train = None
            # GPU preprocessing may have changed the members' widths, so the chunk
            # is refined by the prepared shapes.
            shapes = [tuple(prepared[i][0].shape) for i in chunk]
            for group in _group_equal(shapes):
                positions = [chunk[j] for j in group]
                shape = tuple(prepared[positions[0]][0].shape)
                prepared_rows, _, prepared_columns = shape
                for forwards in _cache_builds(
                    positions, prepared_rows, prepared_columns, budget
                ):
                    builds.setdefault(shape, []).extend(
                        (forward, build(forward, prepared)) for forward in forwards
                    )
            del prepared

        caches: list[tuple[list[int], KVCache]] = []
        for parts in builds.values():
            if not _batches(budget):
                # Every member has a cache of its own.
                caches.extend(parts)
                continue
            built = [cache for _, cache in parts]
            cache = built[0] if len(built) == 1 else type(built[0]).concatenate(built)
            caches.append(([i for forward, _ in parts for i in forward], cache))
        return caches, device

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

        # The predict path only processes X_test (x_is_test_only=True),
        # so base the heuristic on the test shape only.
        save_peak_mem = should_save_peak_mem(
            memory_saving_mode=self.save_peak_mem,
            X_train_shape=(0, X.shape[1]),
            X_test_shape=tuple[int, int](X.shape),
            devices=devices,
            dtype_byte_size=self.dtype_byte_size,
        )

        members = self.ensemble_members
        model_forward_functions = (
            partial(
                self._call_model,
                cache_index=cache_index,
                X_tests=[members[i].transform_X_test(X) for i in group],
                autocast=autocast,
                only_return_standard_out=only_return_standard_out,
                save_peak_mem=save_peak_mem,
                task_type=task_type,
            )
            for cache_index, group in enumerate(self.cache_groups)
        )
        timed_outputs = _TimedIterator(
            parallel_execute(
                devices,
                model_forward_functions,
                prewarm_lapack=self.ensemble_preprocessor.any_estimator_uses_gpu_svd(),
            ),
            devices,
        )

        for output, ensemble_member in zip(
            _yield_in_member_order(self.cache_groups, timed_outputs, len(members)),
            members,
            strict=True,
        ):
            yield _move_and_squeeze_output(output, devices[0]), ensemble_member.config

        self._speed_metrics["predict_model_forward_seconds"] = (
            timed_outputs.elapsed_seconds
        )

    def _call_model(
        self,
        *,
        device: torch.device,
        cache_index: int,
        X_tests: list[torch.Tensor | np.ndarray],
        autocast: bool,
        only_return_standard_out: bool,
        save_peak_mem: bool,
        task_type: str,
    ) -> _MemberOutputs:
        """Predict the test rows of the members sharing ``kv_caches[cache_index]``.

        The members' test rows go through the model as one batch against their
        shared cache. When ``keep_cache_on_device`` is True, the cache is moved to
        ``device`` on the first call and kept there.

        Only X_test is uploaded and preprocessed — the model's cache fast path
        never reads train rows when ``x_is_test_only=True``.

        May be executed in parallel across threads, one per device.
        """
        members = [self.ensemble_members[i] for i in self.cache_groups[cache_index]]
        model = self.model_caches[members[0].config._model_index].get(device)
        dtype = self.force_inference_dtype or torch.float32

        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        categorical_inds: list[list[int]] = []
        for member, X_test in zip(members, X_tests, strict=True):
            X_test_tensor = as_tensor_to_read(
                X_test if isinstance(X_test, torch.Tensor) else np.asarray(X_test),
                dtype=dtype,
                device=device,
            )
            X_test_tensor = X_test_tensor.unsqueeze(1)
            y_train = torch.as_tensor(member.y_train, dtype=dtype, device=device)
            X_test_tensor, feature_schema = _maybe_run_gpu_preprocessing(
                X_test_tensor,
                gpu_preprocessor=member.gpu_preprocessor,
                num_train_rows=0,
                use_fitted_cache=True,
                feature_schema=member.feature_schema,
            )
            # Cast post-preproc tensors back to force_inference_dtype if set —
            # GPU preprocessing may emit fp32 internally (SVD/quantile for
            # numerical stability) even under a fp16 run.
            if self.force_inference_dtype is not None:
                X_test_tensor = X_test_tensor.type(self.force_inference_dtype)
                y_train = y_train.type(self.force_inference_dtype)
            xs.append(X_test_tensor)
            ys.append(y_train)
            categorical_inds.append(
                feature_schema.indices_for(FeatureModality.CATEGORICAL)
            )
        if len({tuple(x.shape) for x in xs}) != 1:
            raise RuntimeError(
                "The members sharing a KV cache produced test inputs of different "
                f"shapes: {[tuple(x.shape) for x in xs]}."
            )
        X, y = _batch_member_inputs(xs, ys)

        performance_options = dataclasses.replace(
            model.get_default_performance_options(),
            enable_torch_compile=self.enable_torch_compile,
            save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
            if save_peak_mem
            else None,
        )
        kwargs = _model_forward_kwargs(
            model, task_type, [member.config.estimator_id for member in members]
        )
        row_weight_kwargs = _row_weight_kwargs(
            members, self.ensemble_preprocessor.sample_weight_mode, X.device, X.dtype
        )

        cache_on_device = self.kv_caches[cache_index].to(device)
        if self.keep_cache_on_device:
            self.kv_caches[cache_index] = cache_on_device

        def run(x_test: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(),
            ):
                return model(
                    x_test,
                    y,
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=categorical_inds,
                    performance_options=performance_options,
                    kv_cache=cache_on_device,
                    x_is_test_only=True,
                    **kwargs,
                    **row_weight_kwargs,
                )

        n_test, batch, columns = X.shape
        chunk = _test_rows_per_forward(
            n_test, batch, columns, _estimator_batch_budget(model)
        )
        if chunk >= n_test:
            output = run(X)
        else:
            output = _concat_test_chunks(
                [run(X[start : start + chunk]) for start in range(0, n_test, chunk)]
            )
        return [(list(range(len(members))), output)]

    @override
    def _create_copy_for_pickling(self) -> InferenceEngine:
        state_copy = super()._create_copy_for_pickling()
        # Attach CPU copies of KV caches for portable serialization.
        state_copy.kv_caches = [cache.to("cpu") for cache in self.kv_caches]
        return state_copy

    @override
    def _move_models_to_devices(self, devices: Sequence[torch.device]) -> None:
        # kv_caches don't exist yet during the .to() call in __init__ (it runs
        # before the caches are built).
        has_caches = bool(getattr(self, "kv_caches", None))
        # A cache holds the batch it was built with; MPS runs members alone.
        if (
            has_caches
            and any(d.type == "mps" for d in devices)
            and any(len(group) > 1 for group in self.cache_groups)
        ):
            raise RuntimeError(
                "This KV cache was built for a batch of ensemble members and "
                "cannot be used on MPS. Fit the estimator on the MPS device instead."
            )
        super()._move_models_to_devices(devices)
        # Caches kept on device move with the models so .to() leaves nothing on
        # the old device; iter_outputs redistributes them on the next predict.
        if has_caches and self.keep_cache_on_device:
            self.kv_caches = [cache.to(devices[0]) for cache in self.kv_caches]


def _test_rows_per_forward(
    n_test: int, batch: int, columns: int, budget: EstimatorBatchBudget
) -> int:
    """Test rows one cached forward takes for a batch of members.

    The cached-inference limit and, for a batch, the row and cell budgets are all
    shared by the members of the batch.
    """
    chunk = n_test
    max_rows = settings.tabpfn.max_batched_test_rows
    if max_rows > 0:
        chunk = min(chunk, max(1, max_rows // batch))
    if batch > 1:
        if budget.rows > 0:
            chunk = min(chunk, max(1, budget.rows // batch))
        if budget.cells > 0:
            chunk = min(chunk, max(1, budget.cells // (batch * columns)))
    return chunk


def _concat_test_chunks(
    outputs: list[torch.Tensor | dict[str, torch.Tensor]],
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Reassemble the outputs of a forward that was run in test-row chunks."""
    if not isinstance(outputs[0], dict):
        return torch.cat(outputs)  # type: ignore[arg-type]
    concat_keys = {"standard", "test_embeddings"}
    shared_keys = {"train_embeddings"}
    unexpected = outputs[0].keys() - concat_keys - shared_keys
    if unexpected:
        raise RuntimeError(
            f"KV-cache test-row chunking has no reassembly rule for model "
            f"output key(s) {sorted(unexpected)}; update the chunk-merge logic "
            f"in _concat_test_chunks."
        )
    return {
        k: outputs[0][k] if k in shared_keys else torch.cat([o[k] for o in outputs])  # type: ignore[index]
        for k in outputs[0]
    }


def _call_model_on_members(  # noqa: PLR0913
    model_caches: list[_PerDeviceModelCache],
    *,
    device: torch.device,
    members: list[TabPFNEnsembleMember],
    X_tests: list[torch.Tensor | np.ndarray],
    force_inference_dtype: torch.dtype | None,
    autocast: bool,
    inference_mode: bool,
    enable_torch_compile: bool,
    only_return_standard_out: bool,
    save_peak_mem: bool,
    task_type: str,
    sample_weight_mode: str | None,
) -> _MemberOutputs:
    """Forward the members, which share a model, from their train and test rows.

    Members whose prepared inputs share a shape run as one forward with the members
    along the batch dimension. Returns each forward's output with the positions in
    ``members`` it holds.
    """
    model = model_caches[members[0].config._model_index].get(device)
    prepared: list[tuple[torch.Tensor | MemberRows, torch.Tensor, list[int]]] = []
    for member, X_test in zip(members, X_tests, strict=True):
        rows = _member_rows(member, X_test, device, force_inference_dtype, model)
        prepared.append(
            rows
            if rows is not None
            else _prepare_member_inputs(
                member, X_test, device, force_inference_dtype, model
            )
        )
    performance_options = dataclasses.replace(
        model.get_default_performance_options(),
        enable_torch_compile=enable_torch_compile,
        save_peak_memory_factor=DEFAULT_SAVE_PEAK_MEMORY_FACTOR
        if save_peak_mem
        else None,
    )
    # GPU preprocessing may have changed the members' widths, so the group is
    # refined by the prepared shapes.
    outputs: _MemberOutputs = []
    shapes = [
        (isinstance(X_full, MemberRows), tuple(X_full.shape))
        for X_full, _, _ in prepared
    ]
    for positions in _group_equal(shapes):
        inputs = [prepared[i][0] for i in positions]
        targets = [prepared[i][1] for i in positions]
        if isinstance(inputs[0], MemberRows):
            y = _batch_targets(targets)
            x, x_kwargs = None, {"x_host": MemberRowSource(inputs)}
            rows_device, rows_dtype = torch.device("cpu"), torch.float32
        else:
            X, y = _batch_member_inputs(inputs, targets)
            x, x_kwargs = _model_input_arguments(X, device)
            rows_device, rows_dtype = X.device, X.dtype
        kwargs = _model_forward_kwargs(
            model, task_type, [members[i].config.estimator_id for i in positions]
        )
        with (
            get_autocast_context(device, enabled=autocast),
            torch.inference_mode(inference_mode),
        ):
            output = model(
                x,
                y,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=[prepared[i][2] for i in positions],
                performance_options=performance_options,
                **x_kwargs,
                **kwargs,
                **_row_weight_kwargs(
                    [members[i] for i in positions],
                    sample_weight_mode,
                    rows_device,
                    rows_dtype,
                ),
            )
        outputs.append((positions, output))
    return outputs


def _prepare_model_inputs(
    device: torch.device,
    force_inference_dtype: torch.dtype | None,
    X_train: torch.Tensor | np.ndarray | SharedColumnsTable,
    X_test: torch.Tensor | np.ndarray | SharedColumnsTable,
    y_train: torch.Tensor | np.ndarray,
    *,
    x_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The train and test rows as one `(rows, 1, columns)` tensor, and the labels.

    Written into one tensor on `x_device`, by default `device`, a block of rows at a
    time, so that device holds the table once. The labels go to `device`.
    """
    dtype = force_inference_dtype if force_inference_dtype else torch.float32
    num_train, num_test = len(X_train), len(X_test)
    num_columns = X_train.shape[1] if num_train else X_test.shape[1]
    X_full = torch.empty(
        (num_train + num_test, 1, num_columns),
        dtype=dtype,
        device=device if x_device is None else x_device,
    )
    for offset, part in ((0, X_train), (num_train, X_test)):
        _chunked_copy_(part, X_full[offset : offset + len(part), 0])
    y_train = torch.as_tensor(y_train, dtype=dtype, device=device)
    return X_full, y_train


def _chunked_copy_(
    part: torch.Tensor | np.ndarray | SharedColumnsTable, out: torch.Tensor
) -> None:
    """Writes `part` into `out` in chunks, handling devices."""
    rows = max(1, _MODEL_INPUT_BLOCK_BYTES // max(1, out.shape[1] * out.element_size()))
    gathered = isinstance(part, SharedColumnsTable) and not part.is_base()
    block = None
    for start in range(0, len(part), rows):
        stop = min(start + rows, len(part))
        target = out[start:stop]
        if not gathered:
            # Rows of an array, or of a table that is its base, are views.
            target.copy_(as_tensor_to_read(part[start:stop]))
        elif target.device.type == "cpu" and target.dtype == torch.float32:
            # A table's base is float32, so its rows go straight into `out`.
            part.copy_rows_into(start, stop, target)
        else:
            if block is None:
                block = torch.from_numpy(
                    np.empty((min(rows, len(part)), part.shape[1]), part.dtype)
                )
            part.copy_rows_into(start, stop, block[: stop - start])
            target.copy_(block[: stop - start])


_MODEL_INPUT_BLOCK_BYTES = 1 << 28
"""Bytes of rows `_prepare_model_inputs` converts and copies at a time."""


def _model_input_device(
    model: torch.nn.Module,
    device: torch.device,
    X_train: torch.Tensor | np.ndarray,
    X_test: torch.Tensor | np.ndarray,
    force_inference_dtype: torch.dtype | None,
) -> torch.device:
    """Where `_prepare_model_inputs` puts the model input for `model` on `device`.

    In host memory when the model reads its input from there (its `forward` takes an
    `x_host` argument) and the input takes at least the `host_model_input_min_bytes`
    of the settings; else on `device`.
    """
    if "x_host" not in _forward_parameters(model) or device.type == "cpu":
        return device
    dtype = force_inference_dtype if force_inference_dtype else torch.float32
    num_columns = X_train.shape[1] if len(X_train) else X_test.shape[1]
    nbytes = (len(X_train) + len(X_test)) * num_columns * dtype.itemsize
    if nbytes >= settings.tabpfn.host_model_input_min_bytes:
        return torch.device("cpu")
    return device


def _model_input_arguments(
    X: torch.Tensor, device: torch.device, *, given_up_to: torch.nn.Module | None = None
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor | bool]]:
    """The model's `x` and extra keyword arguments that pass it the input `X`.

    An input that `_model_input_device` kept in host memory for a model on another
    device goes as `x_host`, with `x` None.

    With `given_up_to`, the caller reads `X` no more after the forward of that
    model. A model whose `forward` takes `free_x_host` is told so with an `x_host`,
    and may free the host memory of `X` as soon as it has read it.
    """
    if X.device.type == "cpu" and device.type != "cpu":
        kwargs: dict[str, torch.Tensor | bool] = {"x_host": X}
        if given_up_to is not None and "free_x_host" in _forward_parameters(
            given_up_to
        ):
            kwargs["free_x_host"] = True
        return None, kwargs
    return X, {}


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
    num_train_rows: int,
    use_fitted_cache: bool = False,
    consume_input: bool = False,
    compute_device: torch.device | None = None,
) -> tuple[torch.Tensor, FeatureSchema]:
    """Run GPU preprocessing if a pipeline is provided.

    Args:
        X: Input tensor.
        gpu_preprocessor: The GPU preprocessing pipeline (or None).
        feature_schema: Feature schema from CPU preprocessing.
        num_train_rows: Number of training rows for fit.
        use_fitted_cache: Reuse previously fitted state.
        consume_input: Whether the pipeline may write into `X`, which the caller
            no longer reads.
        compute_device: The device the pipeline computes on when `X` is elsewhere.

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
        consume_input=consume_input,
        compute_device=compute_device,
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
        # Apply a workaround for a nn.Linear bug on MPS. No-op if MPS not selected.
        for model in self._models.values():
            maybe_replace_linears_on_mps(model, devices)

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

    def get_any(self) -> Architecture:
        """Return the model on any of its devices, for reading its attributes."""
        return next(iter(self._models.values()))

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
