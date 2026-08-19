#  Copyright (c) Prior Labs GmbH 2026.

"""Tests that cover both the classification and regression interfaces."""

from __future__ import annotations

import functools
import platform
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    PerformanceOptions,
)
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs
from tabpfn.checkpoint import Checkpoint
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
from tabpfn.model_loading import download_model, resolve_model_path
from tabpfn.settings import settings
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
    "fit_mode", ["fit_preprocessors", "low_memory", "fit_with_cache"]
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
@pytest.mark.parametrize(
    "fit_mode", ["fit_preprocessors", "low_memory", "fit_with_cache"]
)
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
@pytest.mark.parametrize(
    "fit_mode", ["fit_preprocessors", "low_memory", "fit_with_cache"]
)
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
        decimal=4,
    )


@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
@pytest.mark.parametrize(
    "fit_mode", ["fit_preprocessors", "low_memory", "fit_with_cache"]
)
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
@pytest.mark.parametrize(
    "model_version",
    [ModelVersion.V2, ModelVersion.V2_5, ModelVersion.V2_6, ModelVersion.V3],
)
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


READ_ONLY_INPUT_KINDS = [
    "float_c_order",
    "float_f_order",
    "int",
    "with_nan",
    "with_inf",
]

STAND_IN_MAX_NUM_CLASSES = 10
STAND_IN_NUM_BUCKETS = 64


@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
@pytest.mark.parametrize("input_kind", READ_ONLY_INPUT_KINDS)
def test__fit_and_predict__do_not_write_into_the_callers_arrays(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
    input_kind: str,
) -> None:
    """Read-only inputs survive fit + predict.

    Handing the estimator arrays with the write flag cleared is stricter than
    comparing them before and after: numpy raises on *any* in-place write,
    including one that stores back the value it overwrote. Callers may hold
    memory-mapped or otherwise shared arrays, so neither fit nor predict may
    treat its input as scratch space.
    """
    X_train, X_test, y_train, inference_config = _get_read_only_dataset(
        estimator_class, input_kind
    )

    estimator = estimator_class(
        model_path=_get_stand_in_model_specs(estimator_class),
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config=inference_config,
    )
    estimator.fit(X_train, y_train)
    estimator.predict(X_test)

    # Guards against an estimator making its input writeable to work around the
    # error above rather than copying.
    assert not X_train.flags.writeable
    assert not y_train.flags.writeable
    assert not X_test.flags.writeable


@pytest.mark.parametrize("estimator_class", [TabPFNRegressor, TabPFNClassifier])
def test__fit_and_predict__do_not_modify_the_callers_dataframe(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
) -> None:
    """A mixed-dtype DataFrame comes back with its values and dtypes intact.

    Pandas objects cannot be frozen the way numpy arrays can (the read-only test
    above), so this compares against a deep copy instead. The frame mixes the
    dtypes whose cleaning paths differ: floats with missing values, infinities,
    integers, categoricals and strings.
    """
    X_train, X_test, y_train, inference_config = _get_dataframe_dataset(estimator_class)
    X_train_before = X_train.copy(deep=True)
    X_test_before = X_test.copy(deep=True)
    y_train_before = y_train.copy(deep=True)

    estimator = estimator_class(
        model_path=_get_stand_in_model_specs(estimator_class),
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config=inference_config,
    )
    estimator.fit(X_train, y_train)
    estimator.predict(X_test)

    pd.testing.assert_frame_equal(X_train, X_train_before)
    pd.testing.assert_frame_equal(X_test, X_test_before)
    pd.testing.assert_series_equal(y_train, y_train_before)


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


def _freeze(array: np.ndarray, *, order: str = "C") -> np.ndarray:
    """Return an owned copy of `array` in `order` that cannot be written to."""
    frozen = array.copy(order=order)
    frozen.setflags(write=False)
    return frozen


def _get_read_only_dataset(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
    input_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, bool]]:
    """Build a tiny read-only dataset of the given kind.

    Returns (X_train, X_test, y_train, inference_config).
    """
    n_train = 20
    n_test = 5
    generator = np.random.default_rng(seed=0)
    X = generator.normal(loc=0, scale=1, size=(n_train + n_test, 4))
    inference_config: dict[str, bool] = {}
    order = "C"

    if input_kind == "float_f_order":
        order = "F"
    elif input_kind == "int":
        X = (X * 10).astype(np.int64)
    elif input_kind == "with_nan":
        X[0, 0] = np.nan
        X[n_train + 1, 2] = np.nan
    elif input_kind == "with_inf":
        X[0, 0] = np.inf
        X[n_train + 1, 2] = -np.inf
        # Infinities are rejected outright unless they are read as missingness.
        inference_config = {"PASSTHROUGH_INF": True}
    elif input_kind != "float_c_order":
        raise ValueError(f"Unknown input kind {input_kind!r}")

    if issubclass(estimator_class, TabPFNClassifier):
        y_train = generator.integers(0, 3, size=n_train)
    else:
        y_train = generator.normal(loc=0, scale=1, size=n_train)

    return (
        _freeze(X[:n_train], order=order),
        _freeze(X[n_train:], order=order),
        _freeze(y_train),
        inference_config,
    )


def _get_dataframe_dataset(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, bool]]:
    """Build a tiny mixed-dtype frame.

    Returns (X_train, X_test, y_train, inference_config).
    """
    n_train = 20
    n_test = 5
    n_samples = n_train + n_test
    generator = np.random.default_rng(seed=0)

    floats = generator.normal(loc=0, scale=1, size=n_samples)
    floats[0] = np.nan
    infinities = generator.normal(loc=0, scale=1, size=n_samples)
    infinities[1] = np.inf
    infinities[n_train + 1] = -np.inf

    X = pd.DataFrame(
        {
            "float_with_nan": floats,
            "float_with_inf": infinities,
            "int": generator.integers(0, 10, size=n_samples),
            "categorical": pd.Categorical(generator.choice(["a", "b", "c"], n_samples)),
            "string": generator.choice(["x", "y"], n_samples).astype(object),
        }
    )

    if issubclass(estimator_class, TabPFNClassifier):
        y_train = pd.Series(generator.integers(0, 3, size=n_train), name="target")
    else:
        y_train = pd.Series(
            generator.normal(loc=0, scale=1, size=n_train), name="target"
        )

    return (
        X.iloc[:n_train].copy(deep=True),
        X.iloc[n_train:].copy(deep=True),
        y_train,
        {"PASSTHROUGH_INF": True},
    )


class _ConstantOutputModel(Architecture):
    """A stand-in architecture whose forward pass returns constant logits.

    The tests using it assert on their own inputs rather than on predictions, and
    everything that could touch those inputs — validation, cleaning, and the
    ensemble preprocessing — runs before the model is reached. Real weights would
    therefore only add a checkpoint download and a forward pass to each case.
    """

    def __init__(self, n_outputs: int) -> None:
        """Create a model emitting `n_outputs` logits per test row."""
        super().__init__()
        self.n_outputs = n_outputs
        # The inference engine reads the model's device off its first parameter.
        self.parameter = nn.Parameter(torch.tensor(1.0))

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
        """Return zero logits for every test row, see `Architecture.forward`."""
        del categorical_inds, performance_options, task_type
        features = x["main"] if isinstance(x, dict) else x
        n_rows, batch_size = features.shape[0], features.shape[1]
        targets = y["main"] if isinstance(y, dict) else y
        n_train_rows = 0 if targets is None else targets.shape[0]

        out = features.new_zeros((n_rows - n_train_rows, batch_size, self.n_outputs))
        return out if only_return_standard_out else {"standard": out}

    @property
    def embedding_dim(self) -> int:
        """The width of the (never produced) embeddings."""
        return 2

    @property
    def features_per_group(self) -> int:
        """The number of features the model packs into one token."""
        return 2

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """No-op: this model has no layers to configure."""


@functools.cache
def _get_shipped_inference_config(
    which: Literal["classifier", "regressor"],
) -> InferenceConfig:
    """Read the shipped model's inference config without materialising its weights.

    The preprocessing under test is the one the default model selects, and from
    v2.6 on that configuration lives inside the checkpoint rather than in
    `InferenceConfig.get_default`. Reading it back memory-mapped costs a few tens
    of milliseconds once per session and leaves the weights on disk.
    """
    version = settings.tabpfn.model_version
    (path,), _, (name,), _ = resolve_model_path(None, which, version=version.value)
    if not path.exists():
        result = download_model(path, version=version, which=which, model_name=name)
        if result != "ok":
            raise RuntimeError(f"Could not download {name}: {result}")

    checkpoint = Checkpoint(path)
    raw = (
        checkpoint.load()
        if checkpoint.is_safetensors
        else torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    )
    return InferenceConfig(**raw["inference_config"])


def _get_stand_in_model_specs(
    estimator_class: type[TabPFNClassifier] | type[TabPFNRegressor],
) -> ClassifierModelSpecs | RegressorModelSpecs:
    """Return specs that make an estimator run on `_ConstantOutputModel`.

    Passing specs as `model_path` is the supported way to supply an already
    constructed model, so the estimator's own code path stays intact: it runs the
    shipped preprocessing and only the forward pass is a stand-in.
    """
    if issubclass(estimator_class, TabPFNClassifier):
        return ClassifierModelSpecs(
            model=_ConstantOutputModel(STAND_IN_MAX_NUM_CLASSES),
            architecture_config=ArchitectureConfig(
                max_num_classes=STAND_IN_MAX_NUM_CLASSES
            ),
            inference_config=_get_shipped_inference_config("classifier"),
        )
    return RegressorModelSpecs(
        model=_ConstantOutputModel(STAND_IN_NUM_BUCKETS),
        architecture_config=ArchitectureConfig(num_buckets=STAND_IN_NUM_BUCKETS),
        inference_config=_get_shipped_inference_config("regressor"),
        norm_criterion=FullSupportBarDistribution(
            torch.linspace(-5.0, 5.0, STAND_IN_NUM_BUCKETS + 1)
        ),
    )
