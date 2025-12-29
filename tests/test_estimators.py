"""Tests that cover both the classification and regression interfaces."""

from __future__ import annotations

import platform

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
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
    estimator = estimator_class(
        fit_mode=fit_mode,
        device=device_1,
        n_estimators=2,
        # MPS doesn't support float64, so use a lower precision in that case.
        inference_precision="auto" if platform.system() == "Darwin" else torch.float64,
    )
    X_train, X_test, y_train = _get_tiny_dataset(estimator)
    estimator.fit(X_train, y_train)
    prediction_1 = estimator.predict(X_test)
    estimator.to(device_2)
    estimator.fit(X_train, y_train)
    prediction_2 = estimator.predict(X_test)

    if isinstance(estimator, TabPFNRegressor) and "mps" in devices:
        # Skip only at this point to check that calling .fit() and .to() in this order
        # doesn't cause a crash.
        pytest.skip("MPS yields different predictions.")

    np.testing.assert_array_almost_equal(
        prediction_1,
        prediction_2,
        # Use a slightly relaxed comparison as comparing between devices.
        decimal=5,
    )


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


@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
@pytest.mark.parametrize("fit_mode", ["fit_preprocessors", "low_memory"])
def test__to__after_fit__no_tensors_left_on_old_device(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
    fit_mode: str,
) -> None:
    alt_device = "cuda" if "cuda" in devices else "mps" if "mps" in devices else None
    if alt_device is None:
        pytest.skip("Test can only run when two devices are available.")

    estimator = estimator_class(fit_mode=fit_mode, device=alt_device, n_estimators=2)
    X_train, _X_test, y_train = _get_tiny_dataset(estimator)
    estimator.fit(X_train, y_train)
    estimator.to("cpu")

    tensors_not_on_cpu = _find_tensors_not_on_cpu(estimator)
    assert not tensors_not_on_cpu, f"Found tensors not on cpu: {tensors_not_on_cpu}"


@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
@pytest.mark.parametrize("fit_mode", ["fit_preprocessors", "low_memory"])
@pytest.mark.parametrize("model_version", [ModelVersion.V2, ModelVersion.V2_5])
def test__to__after_fit_and_predict__no_tensors_left_on_old_device(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
    fit_mode: str,
    model_version: ModelVersion,
) -> None:
    alt_device = "cuda" if "cuda" in devices else "mps" if "mps" in devices else None
    if alt_device is None:
        pytest.skip("Test can only run when two devices are available.")

    estimator = estimator_class.create_default_for_version(
        model_version, fit_mode=fit_mode, device=alt_device, n_estimators=2
    )
    X_train, X_test, y_train = _get_tiny_dataset(estimator)
    estimator.fit(X_train, y_train)
    estimator.predict(X_test)
    estimator.to("cpu")

    tensors_not_on_cpu = _find_tensors_not_on_cpu(estimator)
    assert not tensors_not_on_cpu, f"Found tensors not on cpu: {tensors_not_on_cpu}"


def _find_tensors_not_on_cpu(
    estimator: TabPFNClassifier | TabPFNRegressor,
    path: str = "root",
    visited: set[int] | None = None,
) -> list[str]:
    if visited is None:
        visited = set()

    obj_id = id(estimator)
    if obj_id in visited:
        return []
    visited.add(obj_id)

    results: list[str] = []

    if isinstance(estimator, torch.Tensor):
        if estimator.device.type != "cpu":
            results.append(f"{path} (device={estimator.device})")
        return results

    if hasattr(estimator, "__dict__"):
        for attr_name, attr_value in estimator.__dict__.items():
            results.extend(
                _find_tensors_not_on_cpu(attr_value, f"{path}.{attr_name}", visited)
            )

    if isinstance(estimator, dict):
        for key, value in estimator.items():
            results.extend(_find_tensors_not_on_cpu(value, f"{path}[{key!r}]", visited))
    elif isinstance(estimator, (list, tuple)):
        for i, item in enumerate(estimator):
            results.extend(_find_tensors_not_on_cpu(item, f"{path}[{i}]", visited))

    return results


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
