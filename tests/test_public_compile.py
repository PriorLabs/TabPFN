#  Copyright (c) Prior Labs GmbH 2026.
"""The public compile setting reaches every inference path without changing models."""

from __future__ import annotations

import dataclasses
import functools
import pickle
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.base import clone

from tabpfn import ModelSpecs, TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures import tabpfn_v3
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
from tests import test_model_specs
from tests.test_architectures import test_aggregation_compile

multitask_specs = test_model_specs.multitask_specs
legacy_specs = test_model_specs.legacy_specs
compiled_regions = test_aggregation_compile.compiled_regions


@pytest.mark.parametrize("estimator_type", [TabPFNClassifier, TabPFNRegressor])
def test__compile_parameter__clone_and_pickle__preserve_setting(estimator_type) -> None:
    estimator = estimator_type(enable_torch_compile=True)
    assert clone(estimator).get_params()["enable_torch_compile"] is True
    restored = pickle.loads(pickle.dumps(estimator))  # noqa: S301 -- local test object
    assert restored.enable_torch_compile is True
    restored.set_params(enable_torch_compile=False)
    assert restored.enable_torch_compile is False
    assert estimator_type().enable_torch_compile is False


@pytest.mark.parametrize("estimator_type", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize(
    "fit_mode", ["low_memory", "fit_preprocessors", "fit_with_cache"]
)
@pytest.mark.parametrize("enabled", [False, True])
def test__compile_parameter__fit_and_predict__reaches_model(
    estimator_type, fit_mode, enabled, multitask_specs: ModelSpecs
) -> None:
    # Compilation itself is covered by architecture tests. Observe the options at
    # the model boundary here so every cache/engine path is tested independently.
    seen: list[tuple[bool, bool]] = []

    def observe(_model, args, kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        options = kwargs["performance_options"]
        seen.append(
            (options.enable_torch_compile, kwargs.get("return_kv_cache", False))
        )
        kwargs["performance_options"] = dataclasses.replace(
            options, enable_torch_compile=False
        )
        return args, kwargs

    hook = multitask_specs.model.register_forward_pre_hook(observe, with_kwargs=True)
    estimator = estimator_type(
        model_path=multitask_specs,
        n_estimators=1,
        device="cpu",
        random_state=0,
        fit_mode=fit_mode,
        enable_torch_compile=enabled,
        kv_cache_precision="auto",
    )
    x = np.random.default_rng(0).normal(size=(20, 3))
    y = np.arange(20) % 3
    try:
        estimator.fit(x, y)
        fit_calls = len(seen)
        estimator.predict(x[:1])
        assert len(seen) > fit_calls
        assert all(flag is enabled for flag, _ in seen)
        assert any(build for _, build in seen) == (fit_mode == "fit_with_cache")
        defaults = multitask_specs.model.get_default_performance_options()
        assert not defaults.enable_torch_compile
    finally:
        hook.remove()


@pytest.mark.parametrize("estimator_type", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("enabled", [False, True])
def test__compile_parameter__batched_prediction__reaches_model(
    estimator_type,
    enabled,
    multitask_specs: ModelSpecs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[PerformanceOptions] = []
    original = type(multitask_specs.model).forward

    def observe(self, *args, **kwargs) -> Any:
        options = kwargs["performance_options"]
        seen.append(options)
        kwargs["performance_options"] = dataclasses.replace(
            options, enable_torch_compile=False
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        type(multitask_specs.model), "forward", functools.wraps(original)(observe)
    )
    estimator = estimator_type(
        model_path=multitask_specs,
        n_estimators=1,
        device="cpu",
        random_state=0,
        enable_torch_compile=enabled,
    )
    x = np.random.default_rng(0).normal(size=(20, 3))
    y = np.arange(20) % 3
    predict = (
        estimator.predict_proba_batched
        if estimator_type is TabPFNClassifier
        else estimator.predict_batched
    )
    result = predict([x], [y], [x[:2]])
    assert len(result) == 1
    assert len(result[0]) == 2
    assert seen
    assert all(options.enable_torch_compile is enabled for options in seen)


@pytest.mark.parametrize("estimator_type", [TabPFNClassifier, TabPFNRegressor])
def test__compile_parameter__enabled_estimator__traces_without_affecting_eager_peer(
    estimator_type, multitask_specs: ModelSpecs, compiled_regions: list[str]
) -> None:
    kwargs: dict[str, Any] = {
        "model_path": multitask_specs,
        "n_estimators": 1,
        "device": "cpu",
        "random_state": 0,
        "fit_mode": "fit_preprocessors",
        "kv_cache_precision": "auto",
    }
    x = np.random.default_rng(0).normal(size=(20, 3))
    y = np.arange(20) % 3
    eager = estimator_type(**kwargs).fit(x, y)
    expected = eager.predict(x[:2])
    assert not compiled_regions
    compiled = estimator_type(**kwargs, enable_torch_compile=True).fit(x, y)
    actual = compiled.predict(x[:2])
    assert compiled_regions
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    calls = len(compiled_regions)
    eager.predict(x[:2])
    assert len(compiled_regions) == calls


def test__compile_parameter__legacy_architecture__preserves_predictions(
    legacy_specs: ModelSpecs,
) -> None:
    x = np.random.default_rng(0).normal(size=(20, 3))
    y = np.arange(20, dtype=float)
    kwargs: dict[str, Any] = {
        "model_path": legacy_specs,
        "device": "cpu",
        "n_estimators": 1,
    }
    eager = TabPFNRegressor(**kwargs).fit(x, y)
    expected = eager.predict(x[:2])
    compiled = TabPFNRegressor(**kwargs, enable_torch_compile=True).fit(x, y)
    np.testing.assert_allclose(compiled.predict(x[:2]), expected)


def test__compile_parameter__v3__uses_existing_compilation_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=3,
        embed_dim=48,
        nlayers=1,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    specs = ModelSpecs(
        tabpfn_v3.get_architecture(config).eval(),
        config,
        InferenceConfig.get_default("multiclass", ModelVersion.V2_5),
    )
    regions: list[str] = []
    real_compile = torch.compile

    def compile_region(fn, **kwargs) -> Any:
        regions.append(fn.__name__)
        return real_compile(fn, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_region)
    torch._dynamo.reset()
    x = np.random.default_rng(0).normal(size=(20, 3))
    y = np.arange(20) % 3
    kwargs: dict[str, Any] = {"model_path": specs, "device": "cpu", "n_estimators": 1}
    expected = TabPFNClassifier(**kwargs).fit(x, y).predict_proba(x[:2])
    assert not regions
    estimator = TabPFNClassifier(**kwargs, enable_torch_compile=True).fit(x, y)
    np.testing.assert_allclose(estimator.predict_proba(x[:2]), expected, atol=1e-5)
    assert set(regions) == {
        "_preprocess_and_group",
        "_process_row_chunk",
    }
