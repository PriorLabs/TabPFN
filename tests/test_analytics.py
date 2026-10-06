#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.analytics."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Generator
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing_extensions import override

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import tabpfn
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.analytics import log_usage, set_sink
from tabpfn.analytics.events import _gpu_type, _shape, _tabpfn_version
from tabpfn.analytics.parameters import (
    _CONFIG_FIELDS,
    _EMBED_PARAMS,
    _PREDICT_PARAMS,
    config_of,
)
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor
from tabpfn.model_loading import ModelType, _get_model_source
from tabpfn.settings import settings

Event = dict[str, Any]

X_TRAIN = np.zeros((6, 3))
Y_TRAIN = np.array([0, 1, 0, 1, 0, 1])
V2_CHECKPOINT = _get_model_source(
    ModelVersion.V2, ModelType.CLASSIFIER
).default_filename
V3_CHECKPOINT = _get_model_source(
    ModelVersion.V3, ModelType.CLASSIFIER
).default_filename


class _Classifier(ClassifierMixin, BaseEstimator):
    """Mirrors how TabPFNClassifier's public methods call each other."""

    devices_: tuple[torch.device, ...]

    def __init__(
        self,
        *,
        model_path: str | Path = "auto",
        fit_mode: str = "fit_preprocessors",
        n_estimators: int = 4,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.fit_mode = fit_mode
        self.n_estimators = n_estimators

    @log_usage("fit")
    def fit(self, X: np.ndarray, y: np.ndarray) -> _Classifier:
        if len(X) == 0:
            raise ValueError("Cannot fit on an empty dataset.")
        self.n_train_samples_ = len(X)
        self.n_classes_ = len(np.unique(y))
        self.n_estimators_ = self.n_estimators
        return self

    @log_usage("predict")
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    @log_usage("predict")
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    @log_usage("embed")
    def get_embeddings(self, X: np.ndarray, data_source: str = "test") -> np.ndarray:  # noqa: ARG002
        return np.zeros((len(X), 2))

    @log_usage("predict", batched=True)
    def predict_proba_batched(
        self,
        X_train_list: list[np.ndarray],
        y_train_list: list[np.ndarray],
        X_test_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        # Like TabPFN, only accepts datasets that share one train and one test
        # shape, then fits and predicts each on a copy, leaving `self` unchanged.
        train_shapes = {X.shape for X in X_train_list}
        if len(train_shapes) > 1 or len({X.shape for X in X_test_list}) > 1:
            raise ValueError("Ragged batches are not supported.")
        worker = _Classifier(model_path=self.model_path)
        return [
            worker.fit(X_train, y_train).predict(X_test)
            for X_train, y_train, X_test in zip(
                X_train_list, y_train_list, X_test_list, strict=True
            )
        ]


class _TuningClassifier(_Classifier):
    """Fits a holdout classifier inside its own fit, like `tuning_config`."""

    @override
    @log_usage("fit")
    def fit(self, X: np.ndarray, y: np.ndarray) -> _TuningClassifier:
        half = len(X) // 2
        _Classifier().fit(X[:half], y[:half]).predict(X[half:])
        super().fit(X, y)
        return self


class _Regressor(RegressorMixin, BaseEstimator):
    def __init__(self, *, model_path: Any = "auto") -> None:
        super().__init__()
        self.model_path = model_path

    @log_usage("fit")
    def fit(self, X: np.ndarray, y: np.ndarray) -> _Regressor:  # noqa: ARG002
        self.n_train_samples_ = len(X)
        self.n_estimators_ = 8
        return self

    @log_usage("predict")
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: str = "mean",  # noqa: ARG002
        quantiles: Any = None,  # noqa: ARG002
    ) -> np.ndarray:
        return np.zeros(len(X))


@pytest.fixture
def events() -> Generator[list[Event]]:
    logged: list[Event] = []
    set_sink(logged.append)
    yield logged
    set_sink(None)


def test__log_usage__predict_runs_logged_forward__logs_only_the_outer_call(
    events: list[Event],
) -> None:
    _Classifier().fit(X_TRAIN, Y_TRAIN).predict(X_TRAIN)

    assert [(e["event"], e["method"]) for e in events] == [
        ("fit_called", "_Classifier.fit"),
        ("predict_called", "_Classifier.predict"),
    ]


def test__log_usage__fit_fits_another_estimator__logs_one_fit(
    events: list[Event],
) -> None:
    _TuningClassifier().fit(X_TRAIN, Y_TRAIN)

    assert [(e["event"], e["num_rows"]) for e in events] == [("fit_called", 6)]


def test__log_usage__each_event_has_only_the_fields_of_its_kind(
    events: list[Event],
) -> None:
    clf = _Classifier().fit(X_TRAIN, Y_TRAIN)
    clf.predict(X_TRAIN)
    clf.get_embeddings(X_TRAIN, data_source="train")

    # The analytics API rejects any field an event of its kind does not have.
    common = {
        "event",
        "event_id",
        "timestamp",
        "python_version",
        "tabpfn_version",
        "gpu_type",
        "model_path",
        "model_version",
        "task",
        "method",
        "config",
        "status",
        "duration_ms",
        "num_rows",
        "num_columns",
        "num_classes",
        "actual_estimators",
    }
    fit, predict, embed = events
    assert set(fit) == common
    assert set(predict) == common | {"fit_num_rows", "predict_params"}
    assert set(embed) == common | {"fit_num_rows", "data_source"}
    assert embed["data_source"] == "train"
    assert len({e["event_id"] for e in events}) == 3
    assert datetime.fromisoformat(fit["timestamp"]).tzinfo is not None


def test__log_usage__estimators_used_in_turn__each_event_describes_its_own_estimator(
    events: list[Event],
) -> None:
    # The earlier usage tracking kept these values per thread, so `b` overwrote `a`'s.
    a = _Classifier(model_path=f"/models/{V2_CHECKPOINT}", fit_mode="low_memory")
    b = _Classifier(model_path=V3_CHECKPOINT, fit_mode="fit_with_cache")
    a.fit(X_TRAIN, Y_TRAIN)
    b.fit(X_TRAIN[:4], Y_TRAIN[:4])
    a.predict(X_TRAIN[:2])

    assert [
        (e["model_path"], e["model_version"], e["config"]["fit_mode"], e["num_rows"])
        for e in events
    ] == [
        (V2_CHECKPOINT, "v2", "low_memory", 6),
        (V3_CHECKPOINT, "v3", "fit_with_cache", 4),
        (V2_CHECKPOINT, "v2", "low_memory", 2),
    ]
    assert events[2]["fit_num_rows"] == 6


def test__log_usage__concurrent_calls_on_threads__each_logged_from_its_own_estimator(
    events: list[Event],
) -> None:
    n_threads = 8
    all_inside_fit = threading.Barrier(n_threads)

    class _BlockingClassifier(_Classifier):
        @override
        @log_usage("fit")
        def fit(self, X: np.ndarray, y: np.ndarray) -> _BlockingClassifier:
            # Keep every thread inside its call until all of them are.
            all_inside_fit.wait(timeout=10)
            super().fit(X, y)
            return self

    def fit_and_predict(i: int) -> None:
        clf = _BlockingClassifier(n_estimators=i + 1)
        clf.fit(np.zeros((i + 2, 3)), np.arange(i + 2) % 2)
        clf.predict(np.zeros((i + 1, 3)))

    threads = [
        threading.Thread(target=fit_and_predict, args=(i,)) for i in range(n_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(events) == 2 * n_threads
    by_call = {(e["config"]["n_estimators"], e["event"]): e for e in events}
    for i in range(n_threads):
        fit = by_call[i + 1, "fit_called"]
        predict = by_call[i + 1, "predict_called"]
        assert (fit["num_rows"], fit["actual_estimators"]) == (i + 2, i + 1)
        assert (predict["num_rows"], predict["fit_num_rows"]) == (i + 1, i + 2)
        assert predict["actual_estimators"] == i + 1


def test__log_usage__calls_from_asyncio_tasks__each_logged_once(
    events: list[Event],
) -> None:
    async def fit_concurrently() -> None:
        await asyncio.gather(
            *(
                asyncio.to_thread(_Classifier(n_estimators=i + 1).fit, X_TRAIN, Y_TRAIN)
                for i in range(4)
            )
        )

    asyncio.run(fit_concurrently())

    assert sorted(e["actual_estimators"] for e in events) == [1, 2, 3, 4]


class _Metric(str, Enum):
    ROC_AUC = "roc_auc"


class _EveryKindOfParameter(_Classifier):
    """Takes the kinds of values TabPFN's estimators take, and some others."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int | str = "auto",
        softmax_temperature: float = 0.9,
        eval_metric: _Metric = _Metric.ROC_AUC,
        inference_precision: torch.dtype = torch.float16,
        device: list[torch.device] | None = None,
        n_preprocessing_jobs: int = -1,
        epochs: int = -3,
        time_limit: int | None = None,
        learning_rate: float = 1e-5,
        random_state: np.random.Generator | None = None,
        categorical_features_indices: list[int] | None = None,
        model_path: str = "/home/acme/churn.ckpt",
    ) -> None:
        super().__init__(model_path=model_path)
        self.n_estimators = n_estimators  # type: ignore[assignment]
        self.softmax_temperature = softmax_temperature
        self.eval_metric = eval_metric
        self.inference_precision = inference_precision
        self.device = device
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.epochs = epochs
        self.time_limit = time_limit
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.categorical_features_indices = categorical_features_indices


def test__log_usage__config__logs_the_settings_and_values_the_api_accepts(
    events: list[Event],
) -> None:
    _EveryKindOfParameter(
        device=[torch.device("cuda:0"), torch.device("cuda:1")],
        random_state=np.random.default_rng(0),
        categorical_features_indices=[0, 3],
    ).fit(X_TRAIN, Y_TRAIN)

    # Left out: n_estimators="auto" and epochs=-3, which the API rejects, the
    # unset time_limit, and the settings it does not accept at all.
    assert events[0]["config"] == {
        "softmax_temperature": 0.9,
        "eval_metric": "roc_auc",
        "inference_precision": "torch.float16",
        "device": ["cuda:0", "cuda:1"],
        "n_preprocessing_jobs": -1,
    }
    assert events[0]["model_path"] == "other"


def test__log_usage__config__read_from_the_constructor_as_it_is(
    events: list[Event],
) -> None:
    class _ChangedConstructor(_Classifier):
        # Adds `kv_cache_precision` and `new_option`, and drops `n_estimators`.
        def __init__(
            self,
            *,
            fit_mode: str = "low_memory",
            kv_cache_precision: str = "int8",
            new_option: int = 7,
        ) -> None:
            super().__init__(fit_mode=fit_mode)
            self.kv_cache_precision = kv_cache_precision
            self.new_option = new_option

    _ChangedConstructor().fit(X_TRAIN, Y_TRAIN)

    assert events[0]["config"] == {
        "fit_mode": "low_memory",
        "kv_cache_precision": "int8",
    }


def test__config_fields__are_parameters_of_tabpfn_estimators() -> None:
    # Were one renamed, it would silently stop being logged.
    parameters: set[str] = set()
    for estimator in (
        TabPFNClassifier,
        TabPFNRegressor,
        FinetunedTabPFNClassifier,
        FinetunedTabPFNRegressor,
    ):
        parameters |= set(inspect.signature(estimator).parameters)

    assert parameters >= set(_CONFIG_FIELDS)


@pytest.mark.parametrize(
    ("method", "params"),
    [
        (TabPFNRegressor.predict, _PREDICT_PARAMS),
        (TabPFNRegressor.predict_batched, _PREDICT_PARAMS),
        (TabPFNClassifier.get_embeddings, _EMBED_PARAMS),
        (TabPFNRegressor.get_embeddings, _EMBED_PARAMS),
    ],
)
def test__call_params__are_arguments_of_the_methods_they_are_read_from(
    method: Any, params: dict[str, Any]
) -> None:
    # Were one renamed, it would silently stop being logged.
    assert set(inspect.signature(method).parameters) >= set(params)


@pytest.mark.parametrize(
    ("estimator_class", "expected_left_out"),
    [
        (
            TabPFNClassifier,
            {
                "n_estimators",
                "softmax_temperature",
                "kv_cache_precision",
                "eval_metric",
            },
        ),
        (
            TabPFNRegressor,
            {
                "n_estimators",
                "softmax_temperature",
                "kv_cache_precision",
                "eval_metric",
            },
        ),
        (FinetunedTabPFNClassifier, {"eval_metric", "time_limit"}),
        (FinetunedTabPFNRegressor, {"eval_metric", "time_limit"}),
    ],
)
# The fine-tuning estimators warn about their own default numbers of estimators.
@pytest.mark.filterwarnings("ignore:`use_fixed_preprocessing_seed` should only be used")
def test__config_of__default_estimator__logs_each_accepted_setting_unchanged(
    estimator_class: type[BaseEstimator], expected_left_out: set[str]
) -> None:
    estimator = estimator_class()
    params = estimator.get_params(deep=False)
    config = config_of(estimator)

    # Left out: the "auto" defaults of n_estimators and softmax_temperature,
    # which the API does not accept, and the settings that default to None. A
    # default of a type the API does not accept would show up here too.
    assert set(_CONFIG_FIELDS) & set(params) - set(config) == expected_left_out
    # Logged as they are. A setting declared with the wrong type that pydantic
    # still converts, such as an int declared as a flag, shows up here.
    assert {name: (type(value), value) for name, value in config.items()} == {
        name: (type(params[name]), params[name]) for name in config
    }


@pytest.mark.parametrize(
    ("model_path", "expected_name", "expected_version"),
    [
        (f"/home/user/models/{V3_CHECKPOINT}", V3_CHECKPOINT, "v3"),
        ("/home/acme/churn-model-v3.ckpt", "other", None),
        (Path("acme-finetuned.ckpt"), "other", None),
    ],
)
def test__log_usage__model_path__logs_only_published_checkpoints(
    events: list[Event],
    model_path: str | Path,
    expected_name: str,
    expected_version: str | None,
) -> None:
    _Classifier(model_path=model_path).fit(X_TRAIN, Y_TRAIN)

    assert (events[0]["model_path"], events[0]["model_version"]) == (
        expected_name,
        expected_version,
    )


def test__log_usage__auto_model_path__logs_the_default_version(
    events: list[Event],
) -> None:
    _Classifier().fit(X_TRAIN, Y_TRAIN)

    assert (events[0]["model_path"], events[0]["model_version"]) == (
        "auto",
        settings.tabpfn.model_version.value,
    )


def test__log_usage__failed_fit__logged_as_failed_without_the_earlier_fitted_state(
    events: list[Event],
) -> None:
    clf = _Classifier().fit(X_TRAIN, Y_TRAIN)

    with pytest.raises(ValueError, match="empty dataset"):
        clf.fit(np.zeros((0, 3)), np.array([]))

    failed = events[-1]
    assert (failed["status"], failed["num_rows"]) == ("failed", 0)
    assert "num_classes" not in failed
    assert "actual_estimators" not in failed


def test__log_usage__batched_call__logged_as_one_prediction_over_its_datasets(
    events: list[Event],
) -> None:
    _Classifier().predict_proba_batched(
        [X_TRAIN, X_TRAIN], [Y_TRAIN, Y_TRAIN], [X_TRAIN[:2], X_TRAIN[:2]]
    )

    (event,) = events
    # The shape is per dataset; totals are num_datasets times these.
    assert (
        event["event"],
        event["method"],
        event["num_datasets"],
        event["num_rows"],
        event["fit_num_rows"],
        event["num_columns"],
    ) == ("predict_called", "_Classifier.predict_proba_batched", 2, 2, 6, 3)
    # Its fitted state is not read: the call leaves the estimator unchanged.
    assert "num_classes" not in event
    assert "actual_estimators" not in event


def test__log_usage__ragged_batched_call__logged_without_a_shape(
    events: list[Event],
) -> None:
    with pytest.raises(ValueError, match="Ragged"):
        _Classifier().predict_proba_batched(
            [X_TRAIN, X_TRAIN[:4]], [Y_TRAIN, Y_TRAIN[:4]], [X_TRAIN[:2], X_TRAIN[:2]]
        )

    (event,) = events
    assert (event["status"], event["num_datasets"]) == ("failed", 2)
    assert {"num_rows", "fit_num_rows", "num_columns"}.isdisjoint(event)


def test__log_usage__regressor_predict__logs_predict_params_and_regression_task(
    events: list[Event],
) -> None:
    regressor = _Regressor().fit(X_TRAIN, np.arange(6.0))
    regressor.predict(X_TRAIN)
    regressor.predict(X_TRAIN, output_type="quantiles", quantiles=[0.1, 0.9])

    assert [e.get("predict_params") for e in events] == [
        None,
        {"output_type": "mean"},
        {"output_type": "quantiles", "quantiles": [0.1, 0.9]},
    ]
    assert all(e["task"] == "regression" for e in events)
    assert all(e.get("num_classes") is None for e in events)


@pytest.mark.parametrize(("rank", "expected_events"), [("0", 1), ("1", 0)])
def test__log_usage__main_process_only__logs_only_on_rank_zero(
    events: list[Event],
    monkeypatch: pytest.MonkeyPatch,
    rank: str,
    expected_events: int,
) -> None:
    class _DistributedClassifier(_Classifier):
        @override
        @log_usage("fit", main_process_only=True)
        def fit(self, X: np.ndarray, y: np.ndarray) -> _DistributedClassifier:
            super().fit(X, y)
            return self

    monkeypatch.setenv("RANK", rank)
    _DistributedClassifier().fit(X_TRAIN, Y_TRAIN)

    assert len(events) == expected_events


def test__log_usage__sink_raises__the_call_is_unaffected() -> None:
    def broken_sink(event: Event) -> None:  # noqa: ARG001
        raise RuntimeError("The sink is broken.")

    set_sink(broken_sink)
    try:
        clf = _Classifier()
        assert clf.fit(X_TRAIN, Y_TRAIN) is clf
    finally:
        set_sink(None)


def test__log_usage__logging_starts_during_a_call__nested_calls_still_not_logged() -> (
    None
):
    logged: list[Event] = []

    class _StartsLogging(_Classifier):
        @override
        @log_usage("fit")
        def fit(self, X: np.ndarray, y: np.ndarray) -> _StartsLogging:
            set_sink(logged.append)
            super().fit(X, y)
            return self

    try:
        _StartsLogging().fit(X_TRAIN, Y_TRAIN)
    finally:
        set_sink(None)

    assert [e["method"] for e in logged] == ["fit"]


def test__log_usage__method__logged_with_the_class_that_defines_it(
    events: list[Event],
) -> None:
    class _LocalClassifier(_Classifier):
        @override
        @log_usage("fit")
        def fit(self, X: np.ndarray, y: np.ndarray) -> _LocalClassifier:
            super().fit(X, y)
            return self

    _TuningClassifier().fit(X_TRAIN, Y_TRAIN)
    _LocalClassifier().fit(X_TRAIN, Y_TRAIN)

    # A class defined in a function has "<locals>" in its qualified name, which
    # the analytics API rejects, so only the method's name is logged for it.
    assert [e["method"] for e in events] == ["_TuningClassifier.fit", "fit"]


class _FinetunedRegressor(RegressorMixin, BaseEstimator):
    """Like the fine-tuned estimators: fits an ordinary estimator at the end of
    its fit, predicts with it, and passes the options on with `**kwargs`.
    """

    def __init__(self, *, epochs: int = 3, device: str = "cpu") -> None:
        super().__init__()
        self.epochs = epochs
        self.device = device

    @property
    def finetune_model_version(self) -> ModelVersion:
        return ModelVersion.V2_6

    @log_usage("fit", main_process_only=True)
    def fit(self, X: np.ndarray, y: np.ndarray) -> _FinetunedRegressor:
        # The fine-tuned model is held in memory, not in a checkpoint file.
        regressor = _Regressor(model_path=object())
        self.finetuned_inference_regressor_ = regressor.fit(X, y)
        return self

    @log_usage("predict")
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.finetuned_inference_regressor_.predict(X, **kwargs)


def test__log_usage__fine_tuned_estimator__logged_from_the_estimator_it_fitted(
    events: list[Event],
) -> None:
    regressor = _FinetunedRegressor().fit(X_TRAIN, np.arange(6.0))
    regressor.predict(X_TRAIN[:2], output_type="quantiles", quantiles=[0.1, 0.9])

    fit, predict = events
    # The fit logs the fine-tuning settings it was asked for, and both log the
    # version that was fine-tuned.
    assert fit["config"] == {"epochs": 3, "device": "cpu"}
    assert (fit["model_version"], predict["model_version"]) == ("v2.6", "v2.6")
    # The prediction is read from the ordinary estimator that made it, and its
    # options are logged although they arrive in `**kwargs`.
    assert (
        predict["model_path"],
        predict["fit_num_rows"],
        predict["actual_estimators"],
    ) == (None, 6, 8)
    assert predict["predict_params"] == {
        "output_type": "quantiles",
        "quantiles": [0.1, 0.9],
    }


@pytest.mark.parametrize(
    ("quantiles", "expected"),
    [(np.array([0.25, 0.75]), [0.25, 0.75]), ([0.5, 1.5], None)],
)
def test__log_usage__quantiles__logged_from_arrays_and_only_between_0_and_1(
    events: list[Event], quantiles: Any, expected: list[float] | None
) -> None:
    regressor = _Regressor().fit(X_TRAIN, np.arange(6.0))
    regressor.predict(X_TRAIN, output_type="quantiles", quantiles=quantiles)

    assert events[-1]["predict_params"].get("quantiles") == expected


def test__log_usage__failed_call_of_unreadable_size__logged_only_for_predict(
    events: list[Event],
) -> None:
    clf = _Classifier()
    with pytest.raises(TypeError):
        clf.fit(None, Y_TRAIN)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        clf.predict(X_TRAIN, foo=1)  # type: ignore[call-arg]

    # The API requires the size of a fit, so a fit without one is not logged.
    assert [(e["event"], e["status"], e["num_rows"]) for e in events] == [
        ("predict_called", "failed", None)
    ]


def test__log_usage__embedding_with_a_data_source_the_api_rejects__not_logged(
    events: list[Event],
) -> None:
    clf = _Classifier().fit(X_TRAIN, Y_TRAIN)
    clf.get_embeddings(X_TRAIN, data_source="/home/acme/data.csv")

    assert [e["event"] for e in events] == ["fit_called"]


def test__log_usage__sink_calls_a_logged_method__the_call_is_logged_once() -> None:
    logged: list[Event] = []
    clf = _Classifier().fit(X_TRAIN, Y_TRAIN)

    def reentrant_sink(event: Event) -> None:
        logged.append(event)
        clf.predict(X_TRAIN)

    set_sink(reentrant_sink)
    try:
        clf.predict(X_TRAIN)
    finally:
        set_sink(None)

    assert [e["method"] for e in logged] == ["_Classifier.predict"]


def test__log_usage__tabpfn_version__logged_without_a_local_build_suffix(
    events: list[Event], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tabpfn, "__version__", "9.0.1.dev3+g8ed2398d")
    _tabpfn_version.cache_clear()
    try:
        _Classifier().fit(X_TRAIN, Y_TRAIN)
    finally:
        _tabpfn_version.cache_clear()

    assert events[0]["tabpfn_version"] == "9.0.1.dev3"


@pytest.fixture
def gpu_names(monkeypatch: pytest.MonkeyPatch) -> Generator[list[Any]]:
    """Pretend every CUDA device is an A100, recording which devices are asked."""
    asked: list[Any] = []

    def get_device_name(device: Any) -> str:
        asked.append(device)
        return "NVIDIA A100-SXM4-80GB"

    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    _gpu_type.cache_clear()
    yield asked
    _gpu_type.cache_clear()


def test__log_usage__gpu_type__name_of_the_gpu_the_estimator_ran_on(
    events: list[Event], gpu_names: list[Any]
) -> None:
    clf = _Classifier().fit(X_TRAIN, Y_TRAIN)
    clf.devices_ = (torch.device("cuda", 1), torch.device("cuda", 0))
    clf.predict(X_TRAIN)
    clf.predict(X_TRAIN)

    assert [e["gpu_type"] for e in events] == [None, *["NVIDIA A100-SXM4-80GB"] * 2]
    # Asked once, about the first device.
    assert gpu_names == [torch.device("cuda", 1)]


@pytest.mark.parametrize(
    ("device", "expected"),
    [(torch.device("cpu"), None), (torch.device("mps"), "mps")],
)
def test__gpu_type__not_cuda(
    gpu_names: list[Any], device: torch.device, expected: str | None
) -> None:
    assert _gpu_type(device) == expected
    assert gpu_names == []


@pytest.mark.usefixtures("gpu_names")
def test__gpu_type__cuda_cannot_start__none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: Any) -> str:
        raise RuntimeError("Cannot re-initialize CUDA in forked subprocess.")

    monkeypatch.setattr(torch.cuda, "get_device_name", fail)

    assert _gpu_type(torch.device("cuda", 0)) is None


@pytest.mark.usefixtures("gpu_names")
def test__gpu_type__name_the_api_does_not_accept__none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "GPU/with/slashes")

    assert _gpu_type(torch.device("cuda", 0)) is None


def test__log_usage__generator_function__rejected() -> None:
    with pytest.raises(TypeError, match="plain functions"):

        @log_usage("predict")
        def predict_in_chunks(self: Any, X: Any) -> Generator[Any]:  # noqa: ARG001
            yield X


@pytest.mark.parametrize("batched", [False, True])
def test__log_usage__method_without_the_data_arguments__rejected(
    batched: bool,
) -> None:
    with pytest.raises(TypeError, match="reads the data from"):

        @log_usage("fit", batched=batched)
        def fit_preprocessed(self: Any, X_preprocessed: Any) -> None:
            pass


def test__log_usage__decorated_method__keeps_its_name_and_signature() -> None:
    assert _Classifier.fit.__name__ == "fit"
    assert list(inspect.signature(_Classifier.fit).parameters) == ["self", "X", "y"]


class _NoConversion:
    """Reports a shape, but fails the test if anything converts its values."""

    shape = (3, 2)

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("The data was converted.")


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (pd.DataFrame({"a": [1, 2], "b": [3, 4]}), (2, 2)),
        (torch.zeros(4, 7), (4, 7)),
        (_NoConversion(), (3, 2)),
        (np.zeros(7), (7, None)),
        (["a", "b"], (2, None)),
        (None, (None, None)),
    ],
)
def test__shape__reads_rows_and_columns__none_where_unreadable(
    X: Any, expected: tuple[int | None, int | None]
) -> None:
    assert _shape(X) == expected


# --- TabPFN's estimators ------------------------------------------------------


def _any_method(self: Any, X: Any) -> None:
    """Stands in for a method, to get the wrapper that `log_usage` adds."""


# Every method that `log_usage` wraps runs this same code.
_LOGGED = log_usage("fit")(_any_method).__code__


@pytest.mark.parametrize(
    ("estimator_class", "entry_points"),
    [
        (
            TabPFNClassifier,
            {
                "fit",
                "predict",
                "predict_proba",
                "predict_logits",
                "predict_raw_logits",
                "predict_proba_batched",
                "get_embeddings",
            },
        ),
        (TabPFNRegressor, {"fit", "predict", "predict_batched", "get_embeddings"}),
        (FinetunedTabPFNClassifier, {"fit", "predict", "predict_proba"}),
        (FinetunedTabPFNRegressor, {"fit", "predict"}),
    ],
)
def test__log_usage__wraps_exactly_the_entry_points_usage_is_logged_for(
    estimator_class: type[BaseEstimator], entry_points: set[str]
) -> None:
    # Not `forward`, `fit_from_preprocessed`, `fit_with_differentiable_input` or
    # the like, which usage analytics leaves out for now.
    logged = {
        name
        for name, attribute in vars(estimator_class).items()
        if getattr(attribute, "__code__", None) is _LOGGED
    }

    assert logged == entry_points


X_REAL = np.random.default_rng(0).normal(size=(80, 4))
Y_CLASSES = np.arange(80) % 2
Y_TARGET = 2 * X_REAL[:, 0]


def test__tabpfn_classifier__logs_one_event_per_call(events: list[Event]) -> None:
    clf = TabPFNClassifier(n_estimators=1, device="cpu").fit(X_REAL, Y_CLASSES)
    clf.predict(X_REAL[:5])
    clf.predict_proba(X_REAL[:5])
    clf.predict_logits(X_REAL[:5])
    clf.predict_raw_logits(X_REAL[:5])
    clf.get_embeddings(X_REAL[:5])
    clf.predict_proba_batched(
        [X_REAL[:40], X_REAL[40:]],
        [Y_CLASSES[:40], Y_CLASSES[40:]],
        [X_REAL[:5], X_REAL[5:10]],
    )

    assert [e["method"] for e in events] == [
        "TabPFNClassifier.fit",
        "TabPFNClassifier.predict",
        "TabPFNClassifier.predict_proba",
        "TabPFNClassifier.predict_logits",
        "TabPFNClassifier.predict_raw_logits",
        "TabPFNClassifier.get_embeddings",
        "TabPFNClassifier.predict_proba_batched",
    ]
    fit, predict, *_, batched = events
    assert (fit["num_rows"], fit["num_columns"], fit["num_classes"]) == (80, 4, 2)
    assert (predict["num_rows"], predict["fit_num_rows"]) == (5, 80)
    assert (
        batched["num_datasets"],
        batched["num_rows"],
        batched["fit_num_rows"],
    ) == (2, 5, 40)


def test__tabpfn_classifier__fit_with_tuning__logs_one_fit(
    events: list[Event],
) -> None:
    TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        eval_metric="log_loss",
        tuning_config={"calibrate_temperature": True},
    ).fit(X_REAL, Y_CLASSES)

    # Not the fits and predictions on holdout data that tuning makes.
    assert [e["method"] for e in events] == ["TabPFNClassifier.fit"]


def test__tabpfn_regressor__logs_one_event_per_call(events: list[Event]) -> None:
    regressor = TabPFNRegressor(n_estimators=1, device="cpu").fit(X_REAL, Y_TARGET)
    regressor.predict(X_REAL[:5], output_type="quantiles", quantiles=[0.1, 0.9])
    regressor.get_embeddings(X_REAL[:5])
    regressor.predict_batched(
        [X_REAL[:40], X_REAL[40:]],
        [Y_TARGET[:40], Y_TARGET[40:]],
        [X_REAL[:5], X_REAL[5:10]],
    )

    assert [e["method"] for e in events] == [
        "TabPFNRegressor.fit",
        "TabPFNRegressor.predict",
        "TabPFNRegressor.get_embeddings",
        "TabPFNRegressor.predict_batched",
    ]
    assert events[1]["predict_params"] == {
        "output_type": "quantiles",
        "quantiles": [0.1, 0.9],
    }
    assert all(e["task"] == "regression" for e in events)


@pytest.mark.slow
def test__finetuned_tabpfn_classifier__logs_one_fit_for_all_of_fine_tuning(
    events: list[Event],
) -> None:
    clf = FinetunedTabPFNClassifier(
        device="cpu",
        epochs=1,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        n_finetune_ctx_plus_query_samples=50,
        early_stopping=False,
    )
    clf.fit(X_REAL, Y_CLASSES)
    clf.predict(X_REAL[:5])

    # Not the training, validation and final fits that fine-tuning makes.
    fit, predict = events
    assert (fit["method"], predict["method"]) == (
        "FinetunedTabPFNClassifier.fit",
        "FinetunedTabPFNClassifier.predict",
    )
    # Read from the ordinary estimator that fine-tuning ends with. It holds the
    # fine-tuned model in memory, so the version is the one that was fine-tuned.
    assert (fit["num_classes"], fit["model_version"]) == (
        2,
        settings.tabpfn.model_version.value,
    )
    assert (predict["num_rows"], predict["fit_num_rows"]) == (5, 80)
