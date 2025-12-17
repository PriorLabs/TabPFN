"""Tests that cover both the classification and regression interfaces."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tests.utils import get_pytest_devices

devices = get_pytest_devices()

device_combinations = [
    (devices[0], devices[-1]),
    # Use different cpu indicies because the same device can't appear twice. This seems
    # to work, even if there's only one cpu.
    ("auto", ["cpu:0", "cpu:1"]),
]


@pytest.mark.parametrize(("device_1", "device_2"), device_combinations)
@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
@pytest.mark.parametrize(
    "fit_mode", ["fit_preprocessors", "fit_with_cache", "low_memory"]
)
def test__to__before_fit__does_not_crash(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
    fit_mode: str,
    device_1: str,
    device_2: str,
) -> None:
    estimator = estimator_class(fit_mode=fit_mode, device=device_1, n_estimators=2)
    X_train, X_test, y_train = _get_tiny_dataset(estimator)
    estimator.to(device_2)
    estimator.fit(X_train, y_train)
    estimator.predict(X_test)


@pytest.mark.parametrize(("device_1", "device_2"), device_combinations)
@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
@pytest.mark.parametrize("fit_mode", ["fit_preprocessors", "low_memory"])
def test__to__between_fit_and_predict__does_not_crash(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
    fit_mode: str,
    device_1: str,
    device_2: str,
) -> None:
    estimator = estimator_class(fit_mode=fit_mode, device=device_1, n_estimators=2)
    X_train, X_test, y_train = _get_tiny_dataset(estimator)
    estimator.fit(X_train, y_train)
    estimator.to(device_2)
    estimator.predict(X_test)


@pytest.mark.parametrize(("device_1", "device_2"), device_combinations)
@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
@pytest.mark.parametrize("fit_mode", ["fit_preprocessors", "low_memory"])
def test__to__between_fits__outputs_equal(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
    fit_mode: str,
    device_1: str,
    device_2: str,
) -> None:
    estimator = estimator_class(fit_mode=fit_mode, device=device_1, n_estimators=2)
    X_train, X_test, y_train = _get_tiny_dataset(estimator)
    estimator.fit(X_train, y_train)
    prediction_1 = estimator.predict(X_test)
    estimator.to(device_2)
    estimator.fit(X_train, y_train)
    prediction_2 = estimator.predict(X_test)

    if isinstance(estimator, TabPFNRegressor) and (
        "mps" in device_1 or "mps" in device_2
    ):
        pytest.skip("MPS yields different predictions.")
    np.testing.assert_array_equal(prediction_1, prediction_2)


@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
def test__to__fit_with_cache_and_after_first_fit__raises_error(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
) -> None:
    estimator = estimator_class(fit_mode="fit_with_cache", n_estimators=2)
    X_train, X_test, y_train = _get_tiny_dataset(estimator)

    estimator.fit(X_train, y_train)
    with pytest.raises(NotImplementedError):
        estimator.to("cpu")

    estimator.predict(X_test)
    with pytest.raises(NotImplementedError):
        estimator.to("cpu")


def _get_tiny_dataset(
    estimator: TabPFNClassifier | TabPFNRegressor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = 4
    n_test = 2
    generator = np.random.default_rng(seed=0)
    X = generator.normal(loc=0, scale=1, size=(n_train + n_test, 3))
    if isinstance(estimator, TabPFNClassifier):
        y_train = generator.integers(0, 1, size=n_train)
    elif isinstance(estimator, TabPFNRegressor):
        y_train = generator.normal(loc=0, scale=1, size=n_train)
    return X[:n_train], X[n_train:], y_train
