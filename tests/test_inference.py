#  Copyright (c) Prior Labs GmbH 2026.

"""Test the inference engines."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, overload
from typing_extensions import override

import pytest
import torch
from numpy.random import default_rng
from torch import Tensor

from tabpfn.architectures.interface import Architecture, PerformanceOptions
from tabpfn.architectures.kv_cache import KVCache, KVCacheEntry
from tabpfn.architectures.tabpfn_v3 import TabPFNV3Cache
from tabpfn.inference import (
    InferenceEngineCachePreprocessing,
    InferenceEngineExplicitKVCache,
    InferenceEngineOnDemand,
)
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    PreprocessorConfig,
    generate_classification_ensemble_configs,
)
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.torch import FeatureSchema


class _TestModel(Architecture):
    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.received_task_type: str | None = None
        self.seen_batch_sizes: list[int] = []

    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[True] = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[False],
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
    ) -> dict[str, Tensor]: ...

    @override
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Perform a forward pass, see doc string of `Architecture`."""
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)
        self.received_task_type = task_type
        self.seen_batch_sizes.append(x.shape[1])
        n_train_test, _, _ = x.shape
        n_train = y.shape[0]
        test_rows = n_train_test - n_train
        train_summary = x[:n_train].mean(0, keepdim=True).sum(-1, keepdim=True)
        return x[-test_rows:].sum(-1, keepdim=True) + train_summary

    @property
    @override
    def embedding_dim(self) -> int:
        return 2

    @property
    def features_per_group(self) -> int:
        return 2

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        pass


class _TestModelLegacy(Architecture):
    """A test model whose forward pass doesn't have task_type argument."""

    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
    ) -> Tensor | dict[str, Tensor]:
        del (
            only_return_standard_out,
            categorical_inds,
            performance_options,
        )
        """Perform a forward pass."""
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)
        n_train_test, _, _ = x.shape
        n_train = y.shape[0]
        test_rows = n_train_test - n_train
        train_summary = x[:n_train].mean(0, keepdim=True).sum(-1, keepdim=True)
        return x[-test_rows:].sum(-1, keepdim=True) + train_summary

    @property
    @override
    def embedding_dim(self) -> int:
        return 2

    @property
    def features_per_group(self) -> int:
        return 2

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        pass


class _TestModelWithKVCache(Architecture):
    """A test model that supports explicit KV cache forward kwargs.

    Counters track how often each path runs so tests can assert that the
    engine builds caches exactly once and reuses them on predict.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.cache_build_count = 0
        self.cache_used_count = 0

    @override
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
        return_kv_cache: bool = False,
        kv_cache: TabPFNV3Cache | None = None,
        x_is_test_only: bool = False,
    ) -> Tensor | tuple[Tensor, TabPFNV3Cache]:
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)
        n_rows = x.shape[0]
        n_train = y.shape[0]
        if x_is_test_only:
            # Test-only path: x carries only test rows
            test_rows = n_rows
            output = (
                x.sum(-2, keepdim=True).sum(-1, keepdim=True).reshape(-1, test_rows)
            )
        else:
            test_rows = n_rows - n_train
            if test_rows > 0:
                output = (
                    x.sum(-2, keepdim=True).sum(-1, keepdim=True).reshape(-1, test_rows)
                )
            else:
                # Train-only call (e.g. _build_cache) — output is discarded
                output = x.new_zeros(1, 1)

        if return_kv_cache:
            self.cache_build_count += 1
            # Build a dummy cache with a single KVCacheEntry
            dummy_kv = KVCacheEntry(
                key=torch.zeros(1, n_train, 1, 1, device=x.device),
                value=torch.zeros(1, n_train, 1, 1, device=x.device),
            )
            cache = TabPFNV3Cache(
                icl_cache=KVCache(kv={0: dummy_kv}),
                train_embeddings=torch.zeros(1, n_train, 1, device=x.device),
                train_shape=(1, n_train),
            )
            return output, cache

        if kv_cache is not None:
            self.cache_used_count += 1

        return output

    @property
    @override
    def embedding_dim(self) -> int:
        return 2

    @property
    def features_per_group(self) -> int:
        return 2

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        pass


def test__cache_preprocessing__result_equal_in_serial_and_in_parallel() -> None:
    rng = default_rng(seed=0)
    n_train = 100
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    ensemble_preprocessor = TabPFNEnsemblePreprocessor(
        configs=_create_test_ensemble_configs(
            n_configs=5,
            n_classes=3,
            num_models=1,
        ),
        n_samples=X_train.shape[0],
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        random_state=rng,
        # We want to test n_preprocessing_jobs>1 as this might mean the outputs are not
        # in the same order as the input configs, and we want to check that the parallel
        # evaluation code behaves correctly in this scenario.
        n_preprocessing_jobs=5,
    )
    engine = InferenceEngineCachePreprocessing(
        X_train,
        y_train,
        ensemble_preprocessor=ensemble_preprocessor,
        models=[_TestModel()],
        devices=[torch.device("cpu")],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem=True,
        inference_mode=True,
    )

    engine.to([torch.device("cpu")], force_inference_dtype=None, dtype_byte_size=4)
    input_kwargs = {"autocast": False, "task_type": "multiclass"}
    outputs_sequential = list(engine.iter_outputs(X_test, **input_kwargs))
    engine.to(
        [torch.device("cpu"), torch.device("cpu")],
        force_inference_dtype=None,
        dtype_byte_size=4,
    )
    outputs_parallel = list(engine.iter_outputs(X_test, **input_kwargs))

    assert len(outputs_sequential) == len(outputs_parallel)
    for par_output, par_config in outputs_parallel:
        seq_output = _find_seq_output(par_config, outputs_sequential)
        assert isinstance(seq_output, Tensor)
        assert isinstance(par_output, Tensor)
        assert torch.allclose(seq_output, par_output)


def test__cache_preprocessing__with_outlier_removal() -> None:
    def get_outputs(
        outlier_removal_std: float | None = None,
    ) -> list[tuple[torch.Tensor | dict, EnsembleConfig]]:
        rng = default_rng(seed=0)
        n_train = 50
        n_features = 4
        n_classes = 3
        X_train = rng.standard_normal(size=(n_train, n_features))
        X_train[0:10] = 500  # outliers
        y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
        X_test = rng.standard_normal(size=(2, n_features))

        num_models = 1
        models = [_TestModel() for _ in range(num_models)]
        ensemble_preprocessor = TabPFNEnsemblePreprocessor(
            configs=_create_test_ensemble_configs(
                n_configs=5,
                n_classes=3,
                num_models=num_models,
                outlier_removal_std=outlier_removal_std,
            ),
            n_samples=X_train.shape[0],
            feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
            random_state=rng,
            n_preprocessing_jobs=1,
        )
        engine = InferenceEngineOnDemand(
            X_train,
            y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=[torch.device("cpu")],
            dtype_byte_size=4,
            force_inference_dtype=None,
            save_peak_mem=True,
        )
        engine.to([torch.device("cpu")], force_inference_dtype=None, dtype_byte_size=4)
        return list(engine.iter_outputs(X_test, autocast=False, task_type="multiclass"))

    outputs_outlier_removed = get_outputs(outlier_removal_std=1.0)
    outputs_outlier_not_removed = get_outputs(outlier_removal_std=None)

    assert len(outputs_outlier_removed) == len(outputs_outlier_not_removed)
    for outlier_removed_output, outlier_not_removed_output in zip(
        outputs_outlier_removed, outputs_outlier_not_removed
    ):
        assert isinstance(outlier_removed_output[0], Tensor)
        assert isinstance(outlier_not_removed_output[0], Tensor)
        assert not torch.allclose(
            outlier_removed_output[0], outlier_not_removed_output[0]
        )


def test__cache_preprocessing__single_device_ensemble_batching() -> None:
    rng = default_rng(seed=0)
    n_train = 100
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    sequential_model = _TestModel()
    batched_model = _TestModel()
    ensemble_configs = _create_test_ensemble_configs(
        n_configs=5,
        n_classes=n_classes,
        num_models=1,
    )

    def _make_engine(
        model: _TestModel,
        *,
        ensemble_batch_size: int | None,
    ) -> InferenceEngineCachePreprocessing:
        ensemble_preprocessor = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            n_samples=X_train.shape[0],
            feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
            random_state=default_rng(seed=0),
            n_preprocessing_jobs=1,
        )
        return InferenceEngineCachePreprocessing(
            X_train,
            y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=[model],
            devices=[torch.device("cpu")],
            dtype_byte_size=4,
            force_inference_dtype=None,
            save_peak_mem=True,
            inference_mode=True,
            ensemble_batch_size=ensemble_batch_size,
        )

    sequential_engine = _make_engine(sequential_model, ensemble_batch_size=None)
    sequential_outputs = list(
        sequential_engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )

    batched_engine = _make_engine(batched_model, ensemble_batch_size=2)
    batched_outputs = list(
        batched_engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )

    assert sequential_model.seen_batch_sizes == [1, 1, 1, 1, 1]
    assert batched_model.seen_batch_sizes == [2, 2, 1]

    flattened_batched_outputs: list[tuple[Tensor, EnsembleConfig]] = []
    for output, configs in batched_outputs:
        assert isinstance(output, Tensor)
        assert isinstance(configs, list)
        output_batch = output.unsqueeze(1) if output.ndim == 2 else output
        assert output_batch.shape[1] == len(configs)
        for batch_index, config in enumerate(configs):
            flattened_batched_outputs.append((output_batch[:, batch_index], config))

    assert len(sequential_outputs) == len(flattened_batched_outputs)
    for batched_output, batched_config in flattened_batched_outputs:
        sequential_output = _find_seq_output(batched_config, sequential_outputs)
        assert isinstance(sequential_output, Tensor)
        assert torch.allclose(sequential_output, batched_output)


def test__on_demand__single_device_ensemble_batching() -> None:
    rng = default_rng(seed=0)
    n_train = 100
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    sequential_model = _TestModel()
    batched_model = _TestModel()
    ensemble_configs = _create_test_ensemble_configs(
        n_configs=5,
        n_classes=n_classes,
        num_models=1,
    )

    def _make_engine(
        model: _TestModel,
        *,
        ensemble_batch_size: int | None,
    ) -> InferenceEngineOnDemand:
        ensemble_preprocessor = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            n_samples=X_train.shape[0],
            feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
            random_state=default_rng(seed=0),
            n_preprocessing_jobs=1,
        )
        return InferenceEngineOnDemand(
            X_train,
            y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=[model],
            devices=[torch.device("cpu")],
            dtype_byte_size=4,
            force_inference_dtype=None,
            save_peak_mem=True,
            ensemble_batch_size=ensemble_batch_size,
        )

    sequential_engine = _make_engine(sequential_model, ensemble_batch_size=None)
    sequential_outputs = list(
        sequential_engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )

    batched_engine = _make_engine(batched_model, ensemble_batch_size=2)
    batched_outputs = list(
        batched_engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )

    assert sequential_model.seen_batch_sizes == [1, 1, 1, 1, 1]
    assert batched_model.seen_batch_sizes == [2, 2, 1]

    flattened_batched_outputs: list[tuple[Tensor, EnsembleConfig]] = []
    for output, configs in batched_outputs:
        assert isinstance(output, Tensor)
        assert isinstance(configs, list)
        output_batch = output.unsqueeze(1) if output.ndim == 2 else output
        assert output_batch.shape[1] == len(configs)
        for batch_index, config in enumerate(configs):
            flattened_batched_outputs.append((output_batch[:, batch_index], config))

    assert len(sequential_outputs) == len(flattened_batched_outputs)
    for batched_output, batched_config in flattened_batched_outputs:
        sequential_output = _find_seq_output(batched_config, sequential_outputs)
        assert isinstance(sequential_output, Tensor)
        assert torch.allclose(sequential_output, batched_output)


def test__cache_preprocessing__ensemble_batching_groups_alternating_configs() -> None:
    rng = default_rng(seed=0)
    n_train = 100
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    sequential_models = [_TestModel(), _TestModel()]
    batched_models = [_TestModel(), _TestModel()]
    base_config = _create_test_ensemble_configs(
        n_configs=6,
        n_classes=n_classes,
        num_models=1,
    )[0]
    ensemble_configs = [deepcopy(base_config) for _ in range(6)]
    for index, config in enumerate(ensemble_configs):
        config._model_index = index % 2
        config.feature_shift_count = index

    def _make_engine(
        models: list[_TestModel],
        *,
        ensemble_batch_size: int | None,
    ) -> InferenceEngineCachePreprocessing:
        ensemble_preprocessor = TabPFNEnsemblePreprocessor(
            configs=ensemble_configs,
            n_samples=X_train.shape[0],
            feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
            random_state=default_rng(seed=0),
            n_preprocessing_jobs=1,
        )
        return InferenceEngineCachePreprocessing(
            X_train,
            y_train,
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=[torch.device("cpu")],
            dtype_byte_size=4,
            force_inference_dtype=None,
            save_peak_mem=True,
            inference_mode=True,
            ensemble_batch_size=ensemble_batch_size,
        )

    sequential_engine = _make_engine(sequential_models, ensemble_batch_size=None)
    sequential_outputs = list(
        sequential_engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )

    batched_engine = _make_engine(batched_models, ensemble_batch_size=3)
    batched_outputs = list(
        batched_engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )

    assert [model.seen_batch_sizes for model in sequential_models] == [
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert [model.seen_batch_sizes for model in batched_models] == [[3], [3]]

    flattened_batched_outputs: list[tuple[Tensor, EnsembleConfig]] = []
    for output, configs in batched_outputs:
        assert isinstance(output, Tensor)
        assert isinstance(configs, list)
        output_batch = output.unsqueeze(1) if output.ndim == 2 else output
        assert output_batch.shape[1] == len(configs)
        for batch_index, config in enumerate(configs):
            flattened_batched_outputs.append((output_batch[:, batch_index], config))

    assert len(sequential_outputs) == len(flattened_batched_outputs)
    for batched_output, batched_config in flattened_batched_outputs:
        sequential_output = _find_seq_output(batched_config, sequential_outputs)
        assert isinstance(sequential_output, Tensor)
        assert torch.allclose(sequential_output, batched_output)


def test__on_demand__result_equal_in_serial_and_in_parallel() -> None:
    rng = default_rng(seed=0)
    n_train = 100
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    num_models = 3
    models = [_TestModel() for _ in range(num_models)]
    ensemble_preprocessor = TabPFNEnsemblePreprocessor(
        configs=_create_test_ensemble_configs(
            n_configs=5,
            n_classes=3,
            num_models=num_models,
        ),
        n_samples=X_train.shape[0],
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        random_state=rng,
        # We want to test n_preprocessing_jobs>1 as this might mean the outputs are not
        # in the same order as the input configs, and we want to check that the parallel
        # evaluation code behaves correctly in this scenario.
        n_preprocessing_jobs=5,
    )
    engine = InferenceEngineOnDemand(
        X_train,
        y_train,
        ensemble_preprocessor=ensemble_preprocessor,
        models=models,
        devices=[torch.device("cpu")],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem=True,
    )

    engine.to([torch.device("cpu")], force_inference_dtype=None, dtype_byte_size=4)
    input_kwargs = {"autocast": False, "task_type": "multiclass"}
    outputs_sequential = list(engine.iter_outputs(X_test, **input_kwargs))
    engine.to(
        [torch.device("cpu"), torch.device("cpu")],
        force_inference_dtype=None,
        dtype_byte_size=4,
    )
    outputs_parallel = list(engine.iter_outputs(X_test, **input_kwargs))

    assert len(outputs_sequential) == len(outputs_parallel)
    for par_output, par_config in outputs_parallel:
        seq_output = _find_seq_output(par_config, outputs_sequential)
        assert isinstance(seq_output, Tensor)
        assert isinstance(par_output, Tensor)
        assert torch.allclose(seq_output, par_output)


@pytest.mark.parametrize(
    ("model_cls", "task_type"),
    [
        (_TestModel, "multiclass"),
        (_TestModel, "regression"),
        (_TestModelLegacy, "multiclass"),
        (_TestModelLegacy, "regression"),
    ],
)
def test__iter_outputs__task_type_forwarded(
    model_cls: type[_TestModel | _TestModelLegacy],
    task_type: str,
) -> None:
    """task_type is forwarded to model.forward only when the model expects it."""
    rng = default_rng(seed=0)
    n_train = 50
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    model = model_cls()
    ensemble_preprocessor = TabPFNEnsemblePreprocessor(
        configs=_create_test_ensemble_configs(
            n_configs=2, n_classes=n_classes, num_models=1
        ),
        random_state=rng,
        n_samples=X_train.shape[0],
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        n_preprocessing_jobs=1,
    )
    engine = InferenceEngineOnDemand(
        X_train,
        y_train,
        ensemble_preprocessor=ensemble_preprocessor,
        models=[model],
        devices=[torch.device("cpu")],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem=True,
    )
    engine.to([torch.device("cpu")], force_inference_dtype=None, dtype_byte_size=4)
    outputs = list(engine.iter_outputs(X_test, autocast=False, task_type=task_type))
    assert len(outputs) > 0

    if isinstance(model, _TestModel):
        assert model.received_task_type == task_type
    else:
        # Models without task_type in forward should still produce outputs
        assert all(isinstance(out, Tensor) for out, _ in outputs)


def _create_test_ensemble_configs(
    n_configs: int,
    n_classes: int,
    num_models: int,
    outlier_removal_std: float | None = None,
) -> list[ClassifierEnsembleConfig]:
    preprocessor_configs = [
        PreprocessorConfig(
            "quantile_uni_coarse",
            append_original="auto",
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            "none",
            categorical_name="numeric",
            max_features_per_estimator=500,
        ),
    ]
    return generate_classification_ensemble_configs(
        num_estimators=n_configs,
        add_fingerprint_feature=True,
        polynomial_features="all",
        feature_shift_decoder="shuffle",
        preprocessor_configs=preprocessor_configs,
        class_shift_method=None,
        n_classes=n_classes,
        random_state=0,
        num_models=num_models,
        outlier_removal_std=outlier_removal_std,
    )


def _find_seq_output(
    config: EnsembleConfig,
    outputs_sequential: list[tuple[Tensor | dict, EnsembleConfig]],
) -> Tensor | dict:
    """Find the sequential output corresponding to the given config.

    The configs are not hashable, so we have to resort to this search method.
    """
    for output, trial_config in outputs_sequential:
        if trial_config == config:
            return output

    return pytest.fail(f"Parallel config was not found in sequential configs: {config}")


def test__explicit_kv_cache__produces_outputs() -> None:
    """Engine builds one cache per ensemble member and reuses each on predict."""
    rng = default_rng(seed=0)
    n_train = 50
    n_features = 4
    n_classes = 3
    n_configs = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    ensemble_preprocessor = TabPFNEnsemblePreprocessor(
        configs=_create_test_ensemble_configs(
            n_configs=n_configs,
            n_classes=n_classes,
            num_models=1,
        ),
        n_samples=X_train.shape[0],
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        random_state=rng,
        n_preprocessing_jobs=1,
    )
    model = _TestModelWithKVCache()
    engine = InferenceEngineExplicitKVCache(
        X_train,
        y_train,
        ensemble_preprocessor=ensemble_preprocessor,
        models=[model],
        devices=[torch.device("cpu")],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem="auto",
        autocast=False,
    )
    # _build_cache runs once per ensemble member during engine construction.
    assert model.cache_build_count == n_configs
    assert model.cache_used_count == 0
    assert len(engine.kv_caches) == n_configs

    outputs = list(engine.iter_outputs(X_test, autocast=False, task_type="multiclass"))
    assert len(outputs) == n_configs
    for output, _config in outputs:
        assert isinstance(output, Tensor)
    # Predict consumed each cache once and did not rebuild any of them.
    assert model.cache_build_count == n_configs
    assert model.cache_used_count == n_configs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test__explicit_kv_cache__keep_cache_on_device() -> None:
    """keep_cache_on_device=True keeps caches on GPU across predict calls."""
    rng = default_rng(seed=0)
    n_train = 50
    n_features = 4
    n_classes = 3
    n_configs = 2
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    ensemble_preprocessor = TabPFNEnsemblePreprocessor(
        configs=_create_test_ensemble_configs(
            n_configs=n_configs,
            n_classes=n_classes,
            num_models=1,
        ),
        n_samples=X_train.shape[0],
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        random_state=rng,
        n_preprocessing_jobs=1,
    )
    model = _TestModelWithKVCache()
    engine = InferenceEngineExplicitKVCache(
        X_train,
        y_train,
        ensemble_preprocessor=ensemble_preprocessor,
        models=[model],
        devices=[torch.device("cuda")],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem="auto",
        autocast=False,
        keep_cache_on_device=True,
    )
    assert model.cache_build_count == n_configs

    # First predict call — caches move to device and stay there
    outputs_first = list(
        engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )
    assert len(outputs_first) == n_configs
    # No rebuild during predict; each cache used exactly once.
    assert model.cache_build_count == n_configs
    assert model.cache_used_count == n_configs

    # Verify caches are on CUDA after the first call
    for cache in engine.kv_caches:
        for entry in cache.icl_cache.kv.values():
            assert entry.key is not None
            assert entry.key.device.type == "cuda"

    # Snapshot underlying tensor storage. With keep_cache_on_device=True the
    # second predict should reuse the on-device tensors rather than transfer
    # again — Tensor.to(device) is a no-op when already on the target device,
    # so data_ptr() should be unchanged.
    first_ptrs = [cache.icl_cache.kv[0].key.data_ptr() for cache in engine.kv_caches]

    # Second predict call — still no rebuild, caches reused.
    outputs_second = list(
        engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )
    assert len(outputs_second) == n_configs
    assert model.cache_build_count == n_configs
    assert model.cache_used_count == 2 * n_configs

    second_ptrs = [cache.icl_cache.kv[0].key.data_ptr() for cache in engine.kv_caches]
    assert first_ptrs == second_ptrs, (
        "keep_cache_on_device=True must reuse on-device cache tensors, "
        "not re-transfer them on each predict call"
    )
