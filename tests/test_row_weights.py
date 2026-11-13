#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the `sample_weight` and `label_reliability` arguments of `fit`."""

from __future__ import annotations

import functools
from typing import Any
from typing_extensions import override

import numpy as np
import pytest
import torch
from numpy.random import default_rng
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from torch import Tensor

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.inference import (
    InferenceEngineCachePreprocessing,
    InferenceEngineExplicitKVCache,
    InferenceEngineOnDemand,
)
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.row_weights import (
    resolve_sample_weight_mode,
    row_weights_for_estimator,
    validate_row_weights,
)
from tabpfn.preprocessing.torch import FeatureSchema

from .test_inference import (
    _create_test_ensemble_configs,
    _TestModel,
    _TestModelWithKVCache,
)


class _RowWeightModel(_TestModelWithKVCache):
    """Records the per-row weights of every forward pass."""

    def __init__(self, modes: tuple[str, ...] = ("learned", "head")) -> None:
        super().__init__()
        self.modes = modes
        self.calls: list[tuple[Tensor, str, bool]] = []

    @override
    def get_supported_sample_weight_modes(self) -> tuple[str, ...]:
        return self.modes

    @override
    def forward(  # type: ignore[override]
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        row_weights: Tensor,
        sample_weight_mode: str,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Any]:
        is_cached_predict = kwargs.get("x_is_test_only", False)
        self.calls.append((row_weights, sample_weight_mode, is_cached_predict))
        return super().forward(x, y, **kwargs)


def test__validate_row_weights__stacks_both_columns() -> None:
    row_weights = validate_row_weights(
        [1.0, 3.0, 0.0, 4.0], [1.0, 0.5, 0.0, 1.0], n_samples=4
    )
    assert row_weights is not None
    np.testing.assert_allclose(row_weights[:, 0], [0.25, 0.75, 0.0, 1.0])
    np.testing.assert_allclose(row_weights[:, 1], [1.0, 0.5, 0.0, 1.0])


@pytest.mark.parametrize("scale", [1e-50, 1.0, 3e38, 1e300])
def test__row_weights__scale_to_mean_one_for_extreme_magnitudes(scale: float) -> None:
    row_weights = validate_row_weights(
        np.array([1.0, 1.0, 2.0] * 4) * scale, None, n_samples=12
    )
    assert row_weights is not None
    np.testing.assert_allclose(
        row_weights_for_estimator(row_weights, None)[:, 0],
        [0.75, 0.75, 1.5] * 4,
        rtol=1e-6,
    )


def test__row_weights_for_estimator__scales_its_own_rows_to_mean_one() -> None:
    row_weights = np.array([[1.0, 1.0], [3.0, 0.5], [0.0, 0.0], [4.0, 1.0]])
    np.testing.assert_allclose(
        row_weights_for_estimator(row_weights, None),
        [[0.5, 1.0], [1.5, 0.5], [0.0, 0.0], [2.0, 1.0]],
    )
    np.testing.assert_allclose(
        row_weights_for_estimator(row_weights, np.array([3, 1, 3])),
        [[12 / 11, 1.0], [9 / 11, 0.5], [12 / 11, 1.0]],
        rtol=1e-6,
    )


def test__row_weights_for_estimator__rejects_rows_that_all_have_weight_zero() -> None:
    row_weights = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(TabPFNValidationError, match="SUBSAMPLE_SAMPLES"):
        row_weights_for_estimator(row_weights, np.array([1, 2]))


def test__validate_row_weights__missing_argument_is_neutral() -> None:
    only_weights = validate_row_weights([2.0, 2.0], None, n_samples=2)
    only_reliability = validate_row_weights(None, [0.3, 0.9], n_samples=2)
    assert only_weights is not None
    assert only_reliability is not None
    np.testing.assert_array_equal(only_weights[:, 1], 1.0)
    np.testing.assert_array_equal(only_reliability[:, 0], 1.0)
    assert validate_row_weights(None, None, n_samples=2) is None


@pytest.mark.parametrize(
    ("sample_weight", "label_reliability", "match"),
    [
        ([0.0, 0.0, 0.0], None, "zero for every row"),
        ([1.0, -1.0, 1.0], None, "must not be negative"),
        ([1.0, np.nan, 1.0], None, "must be finite"),
        ([1.0, 1.0], None, r"shape \(3,\)"),
        ([[1.0], [1.0], [1.0]], None, r"shape \(3,\)"),
        (None, [0.5, 1.5, 1.0], r"\[0, 1\]"),
        (None, [0.5, -0.1, 1.0], r"\[0, 1\]"),
    ],
)
def test__validate_row_weights__rejects_invalid_values(
    sample_weight: list | None, label_reliability: list | None, match: str
) -> None:
    with pytest.raises(TabPFNValidationError, match=match):
        validate_row_weights(sample_weight, label_reliability, n_samples=3)


def test__validate_row_weights__rejects_class_with_zero_total_weight() -> None:
    with pytest.raises(TabPFNValidationError, match=r"class\(es\) \['b'\]"):
        validate_row_weights(
            [1.0, 0.0, 0.0, 2.0],
            None,
            n_samples=4,
            y_encoded=np.array([0, 1, 1, 2]),
            classes=np.array(["a", "b", "c"]),
        )


def test__resolve_sample_weight_mode__raises_when_config_sets_no_mode() -> None:
    with pytest.raises(TabPFNValidationError, match="does not support"):
        resolve_sample_weight_mode(
            [_RowWeightModel()],
            None,
            uses_label_reliability=False,
            task_type="multiclass",
        )


def test__resolve_sample_weight_mode__raises_when_model_lacks_mode() -> None:
    with pytest.raises(TabPFNValidationError, match="not supported by this model"):
        resolve_sample_weight_mode(
            [_TestModel()],
            "learned",
            uses_label_reliability=False,
            task_type="multiclass",
        )


def test__resolve_sample_weight_mode__head_is_classification_only() -> None:
    with pytest.raises(TabPFNValidationError, match="classification only"):
        resolve_sample_weight_mode(
            [_RowWeightModel()],
            "head",
            uses_label_reliability=False,
            task_type="regression",
        )


def test__resolve_sample_weight_mode__label_reliability_needs_learned() -> None:
    model = _RowWeightModel(modes=("head",))
    assert (
        resolve_sample_weight_mode(
            [model], "head", uses_label_reliability=False, task_type="multiclass"
        )
        == "head"
    )
    with pytest.raises(TabPFNValidationError, match="label_reliability"):
        resolve_sample_weight_mode(
            [model], "head", uses_label_reliability=True, task_type="multiclass"
        )


@pytest.mark.parametrize(
    "engine_cls",
    [
        InferenceEngineOnDemand,
        InferenceEngineCachePreprocessing,
        InferenceEngineExplicitKVCache,
    ],
)
@pytest.mark.parametrize("force_inference_dtype", [None, torch.float64])
def test__engines__forward_each_members_row_weights(
    engine_cls: type, force_inference_dtype: torch.dtype | None
) -> None:
    """Each member gets its own rows' weights, in the dtype of its inputs."""
    rng = default_rng(seed=0)
    n_train, n_features, n_classes = 20, 4, 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes, size=n_train)
    row_weights = np.stack(
        [np.arange(1, n_train + 1), np.linspace(0, 1, n_train)], axis=1
    ).astype(np.float32)
    row_indices = [rng.permutation(n_train)[:12] for _ in range(3)]

    preprocessor = TabPFNEnsemblePreprocessor(
        configs=_create_test_ensemble_configs(
            n_configs=3, n_classes=n_classes, num_models=1
        ),
        n_samples=n_train,
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        random_state=0,
        n_preprocessing_jobs=1,
        subsample_samples=row_indices,
        row_weights=row_weights,
        sample_weight_mode="head",
    )
    model = _RowWeightModel()
    common = {
        "ensemble_preprocessor": preprocessor,
        "models": [model],
        "devices": [torch.device("cpu")],
        "dtype_byte_size": 4,
        "force_inference_dtype": force_inference_dtype,
        "save_peak_mem": False,
    }
    if engine_cls is InferenceEngineExplicitKVCache:
        engine = engine_cls(
            X_train, y_train, autocast=False, task_type="multiclass", **common
        )
    elif engine_cls is InferenceEngineCachePreprocessing:
        engine = engine_cls(X_train, y_train, inference_mode=True, **common)
    else:
        engine = engine_cls(X_train, y_train, **common)
    list(
        engine.iter_outputs(
            rng.standard_normal(size=(5, n_features)),
            autocast=False,
            task_type="multiclass",
        )
    )

    assert {mode for _, mode, _ in model.calls} == {"head"}
    assert {weights.dtype for weights, _, _ in model.calls} == {
        force_inference_dtype or torch.float32
    }
    expected = sorted(
        tuple(row_weights_for_estimator(row_weights, rows).astype(np.float64).ravel())
        for rows in row_indices
    )
    for cached_predict in {is_cached for _, _, is_cached in model.calls}:
        received = [
            tuple(batch[:, b].double().numpy().ravel())
            for batch, _, is_cached in model.calls
            if is_cached == cached_predict
            for b in range(batch.shape[1])
        ]
        assert sorted(received) == expected
    if engine_cls is InferenceEngineExplicitKVCache:
        assert {is_cached for _, _, is_cached in model.calls} == {False, True}


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__raises_for_a_model_without_row_weights(estimator_cls: type) -> None:
    rng = default_rng(seed=0)
    X = rng.standard_normal((20, 3))
    y = np.arange(20) % 2
    estimator = estimator_cls(n_estimators=1, device="cpu")
    with pytest.raises(TabPFNValidationError, match="does not support"):
        estimator.fit(X, y, sample_weight=np.ones(20))
    with pytest.raises(TabPFNValidationError, match="does not support"):
        estimator.fit(X, y, label_reliability=np.ones(20))


@pytest.mark.parametrize(
    ("bagging_cls", "estimator_cls"),
    [(BaggingClassifier, TabPFNClassifier), (BaggingRegressor, TabPFNRegressor)],
)
def test__bagging__raises_for_a_model_without_row_weights(
    bagging_cls: type, estimator_cls: type
) -> None:
    """Bagging passes bootstrap counts as `sample_weight` once `fit` accepts it."""
    rng = default_rng(seed=0)
    X = rng.standard_normal((20, 3))
    y = np.arange(20) % 2
    bagging = bagging_cls(estimator_cls(n_estimators=1, device="cpu"), n_estimators=2)
    with pytest.raises(TabPFNValidationError, match="does not support"):
        bagging.fit(X, y)


def _record_row_weights(
    monkeypatch: pytest.MonkeyPatch, estimator: Any, X: np.ndarray, y: np.ndarray
) -> list[tuple[Tensor, str]]:
    """Let the estimator's model take per-row weights, and record what it gets."""
    estimator.fit(X, y)
    model_cls = type(estimator.models_[0])
    original_forward = model_cls.forward
    calls: list[tuple[Tensor, str]] = []

    @functools.wraps(original_forward)
    def forward(
        self: Any,
        *args: Any,
        row_weights: Tensor | None = None,
        sample_weight_mode: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if row_weights is not None:
            assert sample_weight_mode is not None
            calls.append((row_weights.cpu(), sample_weight_mode))
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(model_cls, "forward", forward)
    monkeypatch.setattr(
        model_cls, "get_supported_sample_weight_modes", lambda _: ("learned",)
    )
    return calls


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize(
    "fit_mode", ["low_memory", "fit_preprocessors", "fit_with_cache"]
)
def test__fit__forwards_row_weights_to_the_model(
    monkeypatch: pytest.MonkeyPatch, estimator_cls: type, fit_mode: str
) -> None:
    rng = default_rng(seed=0)
    n_train = 30
    X = rng.standard_normal((n_train, 3))
    y = (X[:, 0] > 0).astype(int)
    sample_weight = rng.uniform(0, 3, n_train)
    sample_weight[0] = 0.0
    label_reliability = rng.uniform(0, 1, n_train)
    estimator = estimator_cls(
        n_estimators=2,
        fit_mode=fit_mode,
        device="cpu",
        inference_config={"SAMPLE_WEIGHT_MODE": "learned", "SUBSAMPLE_SAMPLES": None},
    )
    calls = _record_row_weights(monkeypatch, estimator, X, y)

    estimator.fit(
        X, y, sample_weight=sample_weight, label_reliability=label_reliability
    )
    estimator.predict(X[:4])

    assert calls
    expected = np.stack(
        [sample_weight / sample_weight.mean(), label_reliability], axis=1
    )
    for row_weights, mode in calls:
        assert mode == "learned"
        assert row_weights.shape[0] == n_train
        assert row_weights.shape[2] == 2
        for b in range(row_weights.shape[1]):
            np.testing.assert_allclose(row_weights[:, b].numpy(), expected, rtol=1e-6)


@pytest.mark.parametrize("fit_mode", ["low_memory", "fit_preprocessors"])
def test__fit__raises_when_subsampling_leaves_only_weight_zero_rows(
    monkeypatch: pytest.MonkeyPatch, fit_mode: str
) -> None:
    rng = default_rng(seed=0)
    X = rng.standard_normal((30, 3))
    y = np.arange(30) % 2
    sample_weight = np.zeros(30)
    sample_weight[:2] = 1.0
    estimator = TabPFNClassifier(
        n_estimators=2,
        fit_mode=fit_mode,
        device="cpu",
        inference_config={
            "SAMPLE_WEIGHT_MODE": "learned",
            "SUBSAMPLE_SAMPLES": [np.arange(10, 20)],
        },
    )
    _record_row_weights(monkeypatch, estimator, X, y)
    with pytest.raises(TabPFNValidationError, match="SUBSAMPLE_SAMPLES"):
        estimator.fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__rejects_row_weights_with_tuning_config(
    monkeypatch: pytest.MonkeyPatch, estimator_cls: type
) -> None:
    rng = default_rng(seed=0)
    X = rng.standard_normal((30, 3))
    y = (X[:, 0] > 0).astype(int)
    estimator = estimator_cls(
        n_estimators=1,
        device="cpu",
        inference_config={"SAMPLE_WEIGHT_MODE": "learned"},
    )
    _record_row_weights(monkeypatch, estimator, X, y)
    estimator.tuning_config = {"calibrate_temperature": True}
    with pytest.raises(TabPFNValidationError, match="tuning_config"):
        estimator.fit(X, y, sample_weight=np.ones(30))
