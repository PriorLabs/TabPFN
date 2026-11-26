#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the `additional_model_input` argument of `fit`."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from numpy.random import default_rng
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError

_NOT_GIVEN = object()


def _record_additional_input(
    monkeypatch: pytest.MonkeyPatch, estimator: Any, X: np.ndarray, y: np.ndarray
) -> list[Any]:
    """Let the estimator's model take `additional_input`, and record what it gets."""
    estimator.fit(X, y)
    model_cls = type(estimator.models_[0])
    original_forward = model_cls.forward
    calls: list[Any] = []

    def forward(
        self: Any, *args: Any, additional_input: Any = _NOT_GIVEN, **kwargs: Any
    ) -> Any:
        calls.append(additional_input)
        return original_forward(self, *args, **kwargs)

    # The engines pass only the keyword arguments that `forward` declares.
    original_signature = inspect.signature(original_forward)
    forward.__signature__ = original_signature.replace(  # type: ignore[attr-defined]
        parameters=[
            *original_signature.parameters.values(),
            inspect.Parameter(
                "additional_input", inspect.Parameter.KEYWORD_ONLY, default=None
            ),
        ]
    )
    monkeypatch.setattr(model_cls, "forward", forward)
    return calls


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = default_rng(seed=0)
    X = rng.standard_normal((30, 3))
    return X, (X[:, 0] > 0).astype(int)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize(
    "fit_mode", ["low_memory", "fit_preprocessors", "fit_with_cache"]
)
def test__fit__forwards_additional_model_input_unchanged(
    monkeypatch: pytest.MonkeyPatch, estimator_cls: type, fit_mode: str
) -> None:
    X, y = _data()
    estimator = estimator_cls(n_estimators=2, fit_mode=fit_mode, device="cpu")
    calls = _record_additional_input(monkeypatch, estimator, X, y)
    additional_model_input = {"anything": [1, 2, 3]}

    estimator.fit(X, y, additional_model_input=additional_model_input)
    estimator.predict(X[:4])
    assert calls
    assert all(call is additional_model_input for call in calls)

    calls.clear()
    estimator.fit(X, y)
    estimator.predict(X[:4])
    assert calls
    assert all(call is _NOT_GIVEN for call in calls)


def test__pipeline__routes_additional_model_input_to_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data()
    estimator = TabPFNClassifier(n_estimators=1, device="cpu")
    calls = _record_additional_input(monkeypatch, estimator, X, y)
    additional_model_input = object()

    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(X, y, tabpfnclassifier__additional_model_input=additional_model_input)
    pipeline.predict(X[:4])

    assert calls
    assert all(call is additional_model_input for call in calls)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__raises_for_a_model_without_additional_input(
    estimator_cls: type,
) -> None:
    X, y = _data()
    estimator = estimator_cls(n_estimators=1, device="cpu")
    with pytest.raises(TabPFNValidationError, match="does not support"):
        estimator.fit(X, y, additional_model_input=object())


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__raises_for_additional_model_input_with_tuning_config(
    monkeypatch: pytest.MonkeyPatch, estimator_cls: type
) -> None:
    X, y = _data()
    estimator = estimator_cls(
        n_estimators=1, device="cpu", tuning_config={"calibrate_temperature": True}
    )
    _record_additional_input(monkeypatch, estimator, X, y)
    with pytest.raises(TabPFNValidationError, match="tuning_config"):
        estimator.fit(X, y, additional_model_input=object())
