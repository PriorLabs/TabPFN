"""Tests for TabPFN-owned telemetry emission behavior."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import tabpfn_common_utils.telemetry.core.decorators as telemetry_decorators
import torch

import tabpfn.classifier as classifier_module
import tabpfn.finetuning.finetuned_base as finetuned_base_module
from tabpfn import TabPFNClassifier, telemetry


def _disable_ping_file_lock(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "_telemetry_state_path", lambda: None)


def test__initialize_telemetry__captures_session_once_per_process(
    monkeypatch,
) -> None:
    events: list[str] = []
    _disable_ping_file_lock(monkeypatch)
    monkeypatch.setattr(telemetry, "_TELEMETRY_SESSION_CAPTURED", False)
    monkeypatch.setattr(telemetry, "ping", lambda: events.append("ping"))
    monkeypatch.setattr(
        telemetry,
        "capture_session",
        lambda: events.append("session"),
    )

    telemetry.initialize_telemetry()
    telemetry.initialize_telemetry()

    assert events == ["ping", "session", "ping"]


def test__initialize_telemetry__respects_suppression(
    monkeypatch,
) -> None:
    events: list[str] = []
    _disable_ping_file_lock(monkeypatch)
    monkeypatch.setattr(telemetry, "_TELEMETRY_SESSION_CAPTURED", False)
    monkeypatch.setattr(telemetry, "ping", lambda: events.append("ping"))
    monkeypatch.setattr(
        telemetry,
        "capture_session",
        lambda: events.append("session"),
    )

    with telemetry.suppress_telemetry():
        telemetry.initialize_telemetry()

    assert events == []


def test__classifier_tuning__does_not_emit_internal_telemetry(
    monkeypatch,
) -> None:
    events: list[str] = []

    _disable_ping_file_lock(monkeypatch)
    monkeypatch.setattr(telemetry, "_TELEMETRY_SESSION_CAPTURED", False)
    monkeypatch.setattr(telemetry, "ping", lambda: events.append("ping"))
    monkeypatch.setattr(
        telemetry,
        "capture_session",
        lambda: events.append("session"),
    )
    monkeypatch.setattr(
        telemetry_decorators,
        "capture_event",
        lambda event: events.append(event.name),
    )

    class DummyPreprocessor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def fake_initialize_model_variables(self: TabPFNClassifier) -> int:
        self.models_ = [object()]
        self.devices_ = (torch.device("cpu"),)
        self.use_autocast_ = False
        self.forced_inference_dtype_ = None
        self.inference_config_ = object()
        return 4

    def fake_initialize_dataset_preprocessing(
        self: TabPFNClassifier,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int | np.random.Generator,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        del random_state
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        self.n_classes_ = len(self.classes_)
        self.class_counts_ = np.array([(y == c).sum() for c in self.classes_])
        self.inferred_feature_schema_ = object()
        return [], X, y

    def fake_raw_predict(
        self: TabPFNClassifier,
        X: np.ndarray,
        *,
        return_logits: bool,
        return_raw_logits: bool = False,
    ) -> torch.Tensor:
        del return_logits, return_raw_logits
        return torch.zeros((self.n_estimators, len(X), self.n_classes_))

    def fake_create_inference_engine(**_kwargs: Any) -> object:
        return object()

    def fake_get_calibrated_softmax_temperature(
        self: TabPFNClassifier,
        holdout_raw_logits: np.ndarray,
        holdout_y_true: np.ndarray,
    ) -> float:
        del self, holdout_raw_logits, holdout_y_true
        return 1.0

    monkeypatch.setattr(
        classifier_module,
        "TabPFNEnsemblePreprocessor",
        DummyPreprocessor,
    )
    monkeypatch.setattr(
        classifier_module,
        "create_inference_engine",
        fake_create_inference_engine,
    )
    monkeypatch.setattr(
        TabPFNClassifier,
        "_initialize_model_variables",
        fake_initialize_model_variables,
    )
    monkeypatch.setattr(
        TabPFNClassifier,
        "_initialize_dataset_preprocessing",
        fake_initialize_dataset_preprocessing,
    )
    monkeypatch.setattr(TabPFNClassifier, "_raw_predict", fake_raw_predict)
    monkeypatch.setattr(
        TabPFNClassifier,
        "_get_calibrated_softmax_temperature",
        fake_get_calibrated_softmax_temperature,
    )

    classifier = TabPFNClassifier(
        n_estimators=1,
        tuning_config={
            "calibrate_temperature": True,
            "tuning_holdout_frac": 0.5,
            "tuning_n_folds": 2,
        },
    )
    X = np.zeros((600, 3))
    y = np.array([0, 1] * 300)

    classifier.fit(X, y)

    event_counts = Counter(events)
    assert event_counts["session"] == 1
    assert event_counts["fit_called"] == 1
    assert event_counts["predict_called"] == 0


def test__finetuning_training_loop_fit__does_not_emit_per_batch_fit_telemetry(
    monkeypatch,
) -> None:
    events: list[str] = []
    calls: list[str] = []
    monkeypatch.setattr(
        telemetry_decorators,
        "capture_event",
        lambda event: events.append(event.name),
    )

    def fake_fit_from_preprocessed(
        self: object,
        X_preprocessed: list[torch.Tensor],
        y_preprocessed: list[torch.Tensor],
        cat_ix: list[list[list[int]]],
        configs: list[Any],
    ) -> object:
        del X_preprocessed, y_preprocessed, cat_ix, configs
        calls.append("fit")
        return self

    fake_fit_from_preprocessed.__module__ = "tabpfn.classifier"

    class FakeEstimator:
        fit_from_preprocessed: Any

    FakeEstimator.fit_from_preprocessed = telemetry_decorators.track_model_call(
        "fit",
        param_names=["X_preprocessed", "y_preprocessed"],
    )(fake_fit_from_preprocessed)

    estimator = FakeEstimator()
    batch = SimpleNamespace(
        X_context=[torch.zeros((4, 3))],
        y_context=[torch.zeros(4)],
        cat_indices=[[]],
        configs=[[]],
    )

    estimator.fit_from_preprocessed(
        batch.X_context,
        batch.y_context,
        batch.cat_indices,
        batch.configs,
    )
    assert calls == ["fit"]
    assert events == ["fit_called"]

    calls.clear()
    events.clear()

    finetuned_base_module._fit_finetuned_estimator_from_preprocessed(
        estimator,
        batch,
    )

    assert calls == ["fit"]
    assert events == []
