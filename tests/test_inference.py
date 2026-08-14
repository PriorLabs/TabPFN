#  Copyright (c) Prior Labs GmbH 2026.

"""Test the inference engines."""

from __future__ import annotations

from typing import Literal, overload
from typing_extensions import override

import numpy as np
import pytest
import sklearn.datasets
import torch
from numpy.random import default_rng
from torch import Tensor, nn

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures import tabpfn_v2
from tabpfn.architectures.interface import Architecture, PerformanceOptions
from tabpfn.architectures.kv_cache import KVCacheEntry
from tabpfn.architectures.shared import workaround_mps_linear_bug
from tabpfn.architectures.shared.workaround_mps_linear_bug import MpsSafeLinear
from tabpfn.architectures.tabpfn_v3 import TabPFNV3Cache
from tabpfn.base import create_inference_engine, get_embeddings
from tabpfn.inference import (
    InferenceEngineCachePreprocessing,
    InferenceEngineExplicitKVCache,
    InferenceEngineOnDemand,
    MultiDeviceInferenceEngine,
    _resolve_kv_cache_precision,
)
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    PreprocessorConfig,
    generate_classification_ensemble_configs,
)
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.torch import FeatureSchema
from tabpfn.settings import settings

from .utils import get_pytest_devices, get_pytest_devices_with_mps_marked_slow


class _TestModel(Architecture):
    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.received_task_type: str | None = None

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
        n_train_test, _, _ = x.shape
        n_train, _ = y.shape
        test_rows = n_train_test - n_train
        return x.sum(-2, keepdim=True).sum(-1, keepdim=True).reshape(-1, test_rows)

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
        n_train, _ = y.shape
        test_rows = n_train_test - n_train
        return x.sum(-2, keepdim=True).sum(-1, keepdim=True).reshape(-1, test_rows)

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
                kv={0: dummy_kv},
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
        outputs_outlier_removed, outputs_outlier_not_removed, strict=True
    ):
        assert isinstance(outlier_removed_output[0], Tensor)
        assert isinstance(outlier_not_removed_output[0], Tensor)
        assert not torch.allclose(
            outlier_removed_output[0], outlier_not_removed_output[0]
        )


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
        save_peak_mem=True,
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


@pytest.mark.parametrize("device", get_pytest_devices())
def test__explicit_kv_cache__keep_on_device_reuses_tensors(device: str) -> None:
    """keep_cache_on_device=True reuses the on-device cache tensors across calls.

    Placement for the True/False modes is covered via the public API in
    ``test__public_keep_cache_on_device__controls_cache_placement``. This
    engine-level test guards the distinct optimization that True buys: repeated
    ``predict`` calls must not re-transfer the cache (``Tensor.to(device)`` is a
    no-op when already on the device), so the underlying storage is unchanged.
    """
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
        devices=[torch.device(device)],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem="auto",
        autocast=False,
        keep_cache_on_device=True,
    )
    assert model.cache_build_count == n_configs

    # First predict call — caches move to device and stay there.
    outputs_first = list(
        engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )
    assert len(outputs_first) == n_configs
    assert model.cache_build_count == n_configs
    assert model.cache_used_count == n_configs

    # Snapshot underlying tensor storage after the first predict.
    first_ptrs = [cache.kv[0].key.data_ptr() for cache in engine.kv_caches]

    # Second predict call — still no rebuild, caches reused in place.
    outputs_second = list(
        engine.iter_outputs(X_test, autocast=False, task_type="multiclass")
    )
    assert len(outputs_second) == n_configs
    assert model.cache_build_count == n_configs
    assert model.cache_used_count == 2 * n_configs

    second_ptrs = [cache.kv[0].key.data_ptr() for cache in engine.kv_caches]
    assert first_ptrs == second_ptrs, (
        "keep_cache_on_device=True must reuse on-device cache tensors, "
        "not re-transfer them on each predict call"
    )


@pytest.mark.parametrize("device", get_pytest_devices_with_mps_marked_slow())
@pytest.mark.parametrize("keep_cache_on_device", [False, True])
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__public_keep_cache_on_device__controls_cache_placement(
    estimator_cls: type, *, keep_cache_on_device: bool, device: str
) -> None:
    """The public keep_cache_on_device arg controls where KV caches are stored.

    Exercises the full public path with the real model: constructor ->
    ``fit`` -> ``create_inference_engine`` -> ``InferenceEngineExplicitKVCache``.
    With ``keep_cache_on_device=True`` each cache stays on the build device;
    with ``False`` each cache is offloaded to CPU. On the GPU CI job
    (``TABPFN_EXCLUDE_DEVICES`` drops cpu/mps) ``device="cuda"`` so the True/False
    placement is genuinely distinguished; on CPU jobs it is an end-to-end smoke
    test. ``.key`` is the (int8) tensor on both the quantized and plain cache
    entries, so the placement check holds either way.
    """
    rng = default_rng(seed=0)
    n_samples, n_features = 30, 4
    X = rng.standard_normal((n_samples, n_features))
    if estimator_cls is TabPFNClassifier:
        y = (X[:, 0] > 0).astype(int)  # guaranteed two classes
    else:
        y = rng.standard_normal(n_samples)

    est = estimator_cls(
        fit_mode="fit_with_cache",
        keep_cache_on_device=keep_cache_on_device,
        n_estimators=2,
        device=device,
    )
    est.fit(X, y)

    # The public arg reached the explicit-KV-cache engine.
    assert isinstance(est.executor_, InferenceEngineExplicitKVCache)
    assert est.executor_.keep_cache_on_device is keep_cache_on_device

    # True keeps caches on the build device; False offloads them to CPU.
    expected_device = device if keep_cache_on_device else "cpu"
    for cache in est.executor_.kv_caches:
        for entry in cache.kv.values():
            assert entry.key is not None
            assert entry.key.device.type == expected_device

    # Prediction still works end-to-end (caches moved to device on demand).
    preds = est.predict(X[:5])
    assert len(preds) == 5


class _TestModelWithLinear(_TestModelWithKVCache):
    """A test model with Linear layers that also supports the KV-cache forward."""

    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=True)
        self.linear_no_bias = nn.Linear(3, 2, bias=False)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires the MPS backend"
)
@pytest.mark.parametrize(
    "fit_mode", ["low_memory", "fit_preprocessors", "fit_with_cache"]
)
def test__init__mps_target_device__applies_mps_linear_bug_workaround(
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving an engine to MPS replaces Linears with MpsSafeLinear.

    See tabpfn.architectures.shared.workaround_mps_linear_bug.
    """
    monkeypatch.setattr(
        workaround_mps_linear_bug, "_mps_linear_bias_is_buggy", lambda: True
    )
    rng = default_rng(seed=0)
    X_train = rng.standard_normal((20, 4))
    y_train = rng.integers(0, 3, (20, 1))
    mps = torch.device("mps")
    engine = create_inference_engine(
        fit_mode=fit_mode,
        X_train=X_train,
        y_train=y_train,
        ensemble_preprocessor=TabPFNEnsemblePreprocessor(
            configs=_create_test_ensemble_configs(
                n_configs=2, n_classes=3, num_models=1
            ),
            n_samples=X_train.shape[0],
            feature_schema=FeatureSchema.from_only_categorical_indices([], 4),
            random_state=rng,
            n_preprocessing_jobs=1,
        ),
        models=[_TestModelWithLinear()],
        devices_=[mps],
        byte_size=4,
        forced_inference_dtype_=None,
        memory_saving_mode=True,
        use_autocast_=False,
        inference_mode=True,
    )

    engine.to([torch.device(mps)], force_inference_dtype=None, dtype_byte_size=4)

    assert isinstance(engine, MultiDeviceInferenceEngine)
    model = engine.model_caches[0].get(torch.device(mps))
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    bias_linears = [m for m in linears if m.bias is not None]
    no_bias_linears = [m for m in linears if m.bias is None]

    assert bias_linears, "expected at least one bias-enabled Linear"
    assert all(isinstance(m, MpsSafeLinear) for m in bias_linears)
    assert all(type(m) is nn.Linear for m in no_bias_linears)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires the MPS backend"
)
@pytest.mark.parametrize(
    "fit_mode", ["low_memory", "fit_preprocessors", "fit_with_cache"]
)
def test__to__mps_target_device__applies_mps_linear_bug_workaround(
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving an engine to MPS replaces Linears with MpsSafeLinear.

    See tabpfn.architectures.shared.workaround_mps_linear_bug.
    """
    monkeypatch.setattr(
        workaround_mps_linear_bug, "_mps_linear_bias_is_buggy", lambda: True
    )
    rng = default_rng(seed=0)
    X_train = rng.standard_normal((20, 4))
    y_train = rng.integers(0, 3, (20, 1))
    engine = create_inference_engine(
        fit_mode=fit_mode,
        X_train=X_train,
        y_train=y_train,
        ensemble_preprocessor=TabPFNEnsemblePreprocessor(
            configs=_create_test_ensemble_configs(
                n_configs=2, n_classes=3, num_models=1
            ),
            n_samples=X_train.shape[0],
            feature_schema=FeatureSchema.from_only_categorical_indices([], 4),
            random_state=rng,
            n_preprocessing_jobs=1,
        ),
        models=[_TestModelWithLinear()],
        devices_=[torch.device("cpu")],
        byte_size=4,
        forced_inference_dtype_=None,
        memory_saving_mode=True,
        use_autocast_=False,
        inference_mode=True,
    )

    mps = torch.device("mps")
    engine.to([torch.device(mps)], force_inference_dtype=None, dtype_byte_size=4)

    assert isinstance(engine, MultiDeviceInferenceEngine)
    model = engine.model_caches[0].get(torch.device(mps))
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    bias_linears = [m for m in linears if m.bias is not None]
    no_bias_linears = [m for m in linears if m.bias is None]

    assert bias_linears, "expected at least one bias-enabled Linear"
    assert all(isinstance(m, MpsSafeLinear) for m in bias_linears)
    assert all(type(m) is nn.Linear for m in no_bias_linears)


@pytest.mark.parametrize("device", get_pytest_devices())
@pytest.mark.parametrize("estimator", ["classifier", "regressor"])
def test__kv_cache_chunking__matches_unchunked(
    estimator: str,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunked cached inference must equal unchunked inference.

    We run in float64 so the only remaining
    differences are sub-tolerance floating-point rounding (in lower precisions
    GEMM rounding across differing batch sizes can accumulate beyond tolerance).
    """
    if torch.device(device).type == "mps":
        pytest.skip("float64 inference is not supported on MPS")

    # n_test must exceed the chunk size, and not be a multiple of it, so that
    # chunking actually triggers and the final (ragged) chunk is exercised.
    n_train, n_test, chunk = 32, 37, 8

    if estimator == "classifier":
        X, y = sklearn.datasets.make_classification(
            n_samples=n_train + n_test,
            n_features=5,
            n_informative=3,
            n_classes=3,
            random_state=0,
        )
        model = TabPFNClassifier(
            n_estimators=1,
            random_state=42,
            device=device,
            inference_precision=torch.float64,
            fit_mode="fit_with_cache",
        )

        def predict(m, x) -> np.ndarray:
            return m.predict_proba(x)
    else:
        X, y, _ = sklearn.datasets.make_regression(
            n_samples=n_train + n_test,
            n_features=5,
            random_state=0,
            coef=True,
        )
        model = TabPFNRegressor(
            n_estimators=1,
            random_state=42,
            device=device,
            inference_precision=torch.float64,
            fit_mode="fit_with_cache",
        )

        def predict(m, x) -> np.ndarray:
            return m.predict(x, output_type="mean")

    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train]
    model.fit(X_train, y_train)

    # Reference: chunking disabled -> a single forward pass over all test rows.
    monkeypatch.setattr(settings.tabpfn, "max_batched_test_rows", 0)
    pred_unchunked = predict(model, X_test)

    # Chunked: several forward passes over test-row chunks, then concatenated.
    monkeypatch.setattr(settings.tabpfn, "max_batched_test_rows", chunk)
    pred_chunked = predict(model, X_test)

    np.testing.assert_allclose(pred_chunked, pred_unchunked, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("device", get_pytest_devices())
def test__kv_cache_chunking__train_embeddings_not_duplicated(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunking must not inflate the dict (only_return_standard_out=False) path.

    Test embeddings are test-indexed and get concatenated across chunks, but
    train embeddings are read from the KV cache and are identical per chunk, so
    naively concatenating them would yield n_chunks * n_train rows.
    """
    if torch.device(device).type == "mps":
        pytest.skip("float64 inference is not supported on MPS")

    n_train, n_test, chunk = 32, 37, 8
    X, y = sklearn.datasets.make_classification(
        n_samples=n_train + n_test, n_features=5, n_informative=3, random_state=0
    )
    model = TabPFNClassifier(
        n_estimators=1,
        random_state=42,
        device=device,
        inference_precision=torch.float64,
        fit_mode="fit_with_cache",
    )
    model.fit(X[:n_train], y[:n_train])
    X_test = X[n_train:]

    monkeypatch.setattr(settings.tabpfn, "max_batched_test_rows", chunk)
    train_emb = get_embeddings(model, X_test, data_source="train")
    test_emb = get_embeddings(model, X_test, data_source="test")

    assert train_emb.shape[-2] == n_train
    assert test_emb.shape[-2] == n_test


def test__resolve_kv_cache_precision__warns_when_unsupported() -> None:
    """int8 on a v2 architecture (which cannot quantize) warns and uses 'auto'."""
    arch = tabpfn_v2.get_architecture(
        tabpfn_v2.TabPFNV2Config(
            max_num_classes=10,
            num_buckets=5,
            emsize=192,
            nlayers=1,
            nhead=6,
            features_per_group=2,
            seed=0,
        ),
        cache_trainset_representation=False,
    )
    with pytest.warns(UserWarning, match="not supported"):
        resolved = _resolve_kv_cache_precision(
            "int8", architecture=arch, device=torch.device("cpu")
        )
    assert resolved == "auto"
