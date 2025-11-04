# tests/test_save_load_fitted_model.py
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures.interface import ArchitectureConfig
from tabpfn.base import RegressorModelSpecs, initialize_tabpfn_model
from tabpfn.inference_tuning import ClassifierEvalMetrics
from tabpfn.model_loading import save_tabpfn_model

from .utils import get_pytest_devices

# filter out combinations when "mps" is exatly one device type!
# -> yields different predictions, as dtypes are partly unsupported
device_bicombination = [
    comb for comb in product(get_pytest_devices(), repeat=2) if comb.count("mps") != 1
]


# --- Fixtures for cuda availability ---
@pytest.fixture
def disable_cuda_temporarily():
    """Temporarily disable CUDA for a test."""
    original_is_available = torch.cuda.is_available  # Cache original
    yield
    torch.cuda.is_available = original_is_available  # Restore after test


# --- Fixtures for data ---
@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=40, n_features=5, random_state=42)
    return X, y


@pytest.fixture
def classification_data_with_categoricals():
    X, y = make_classification(
        n_samples=40, n_features=5, n_classes=3, n_informative=3, random_state=42
    )
    # Add a string-based categorical feature
    X_cat = X.astype(object)
    X_cat[:, 2] = np.random.choice(["A", "B", "C"], size=X.shape[0])  # noqa: NPY002
    return X_cat, y


# --- Main Test using Parametrization ---
@pytest.mark.parametrize(
    ("estimator_class", "data_fixture", "saving_device", "loading_device"),
    [
        (pred, fixture, *devs)
        for (pred, fixture), devs in product(
            [
                (TabPFNRegressor, "regression_data"),
                (TabPFNClassifier, "classification_data_with_categoricals"),
            ],
            device_bicombination,
        )
    ],
)
def test_save_load_happy_path(
    estimator_class: type[TabPFNRegressor] | type[TabPFNClassifier],
    data_fixture: str,
    saving_device: str,
    loading_device: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = request.getfixturevalue(data_fixture)

    # Simulate saving device
    if "cuda" in saving_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    elif "mps" in saving_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    elif "cpu" in saving_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    else:
        raise NotImplementedError(f"saving device: {saving_device} not found")

    model = estimator_class(device=saving_device, n_estimators=4)
    model.fit(X, y)
    path = tmp_path / "model.tabpfn_fit"

    # Save and then load the model using its class method
    model.save_fit_state(path)

    # Simulate saving device
    if "cuda" in loading_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    elif "mps" in loading_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    elif "cpu" in loading_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    else:
        raise NotImplementedError(f"saving device: {loading_device} not found")

    loaded_model = estimator_class.load_from_fit_state(path, device=loading_device)

    if loading_device == saving_device:
        # In transformer.py::add_embeddings we generate() random tensors inside a
        # fixed-seed RNG context.
        # Note: PyTorch uses different random number generators on CPU and CUDA.
        # Even with the same seed, CPU and CUDA will produce different random values.
        # This means the feature embeddings differ slightly depending on the device,
        # which in turn leads to small prediction differences between CPU and CUDA
        # models.
        # This behavior is expected and comes from the transformer architecture,
        # not a bug.

        # We cannot align the two RNG streams, so the only options are either to skip
        # the tests that compare predictions of different saving & loading devices.

        # (or use a large tolerance, which is reasonable for different random embeddings
        # but as the regressor has a difference of +-1 unit, setting such a large
        # tolerance is meaningless)

        # 1. Check that predictions are identical
        np.testing.assert_array_almost_equal(model.predict(X), loaded_model.predict(X))

        # 2. For classifiers, also check probabilities and restored classes
        if hasattr(model, "predict_proba"):
            np.testing.assert_array_almost_equal(
                model.predict_proba(X),
                loaded_model.predict_proba(X),
            )
            np.testing.assert_array_equal(model.classes_, loaded_model.classes_)

    # 3. Check that the loaded object is of the correct type
    assert isinstance(loaded_model, estimator_class)


# --- Error Handling Tests ---
def test_saving_unfitted_model_raises_error(tmp_path: Path) -> None:
    """Tests that saving an unfitted model raises a RuntimeError."""
    model = TabPFNRegressor()
    with pytest.raises(RuntimeError, match="Estimator must be fitted before saving"):
        model.save_fit_state(tmp_path / "model.tabpfn_fit")


def test_loading_mismatched_types_raises_error(regression_data, tmp_path):
    """Tests that loading a regressor as a classifier raises a TypeError."""
    X, y = regression_data
    model = TabPFNRegressor(device="cpu")
    model.fit(X, y)
    path = tmp_path / "model.tabpfn_fit"
    model.save_fit_state(path)

    with pytest.raises(
        TypeError, match="Attempting to load a 'TabPFNRegressor' as 'TabPFNClassifier'"
    ):
        TabPFNClassifier.load_from_fit_state(path)


def _init_and_save_unique_checkpoint(
    model: TabPFNRegressor | TabPFNClassifier,
    save_path: Path,
) -> tuple[torch.Tensor, ArchitectureConfig]:
    model._initialize_model_variables()
    first_param = next(model.models_[0].parameters())
    with torch.no_grad():
        first_param.copy_(torch.randn_like(first_param))
    first_model_parameter = first_param.clone()
    config_before_saving = deepcopy(model.configs_[0])
    save_tabpfn_model(model, save_path)

    return first_model_parameter, config_before_saving


def test_saving_and_loading_model_with_weights(tmp_path: Path) -> None:
    """Tests that the saving format of the `save_tabpfn_model` method is compatible with
    the loading interface of `initialize_tabpfn_model`.
    """
    # initialize a TabPFNRegressor
    regressor = TabPFNRegressor(model_path="auto", device="cpu", random_state=42)
    save_path = tmp_path / "model.ckpt"
    first_model_parameter, config_before_saving = _init_and_save_unique_checkpoint(
        model=regressor,
        save_path=save_path,
    )

    # Load the model state
    models, architecture_configs, criterion, inference_config = initialize_tabpfn_model(
        save_path, "regressor", fit_mode="low_memory"
    )
    loaded_regressor = TabPFNRegressor(
        model_path=RegressorModelSpecs(
            model=models[0],
            architecture_config=architecture_configs[0],
            norm_criterion=criterion,
            inference_config=inference_config,
        ),
        device="cpu",
    )

    # then check the model is loaded correctly
    loaded_regressor._initialize_model_variables()
    torch.testing.assert_close(
        next(loaded_regressor.models_[0].parameters()),
        first_model_parameter,
    )
    assert loaded_regressor.configs_[0] == config_before_saving


@pytest.mark.parametrize(
    ("estimator_class"),
    [TabPFNRegressor, TabPFNClassifier],
)
def test_saving_and_loading_multiple_models_with_weights(
    estimator_class: type[TabPFNRegressor] | type[TabPFNClassifier],
    tmp_path: Path,
) -> None:
    """Test that saving and loading multiple models works."""
    estimator = estimator_class(model_path="auto", device="cpu", random_state=42)
    save_path_0 = tmp_path / "model_0.ckpt"
    first_model_parameter_0, config_before_saving_0 = _init_and_save_unique_checkpoint(
        model=estimator,
        save_path=save_path_0,
    )
    estimator = estimator_class(model_path="auto", device="cpu", random_state=42)
    save_path_1 = tmp_path / "model_1.ckpt"
    first_model_parameter_1, config_before_saving_1 = _init_and_save_unique_checkpoint(
        model=estimator,
        save_path=save_path_1,
    )

    loaded_estimator = estimator_class(
        model_path=[save_path_0, save_path_1],
        device="cpu",
        random_state=42,
    )
    loaded_estimator._initialize_model_variables()

    torch.testing.assert_close(
        next(loaded_estimator.models_[0].parameters()),
        first_model_parameter_0,
    )
    torch.testing.assert_close(
        next(loaded_estimator.models_[1].parameters()),
        first_model_parameter_1,
    )
    assert loaded_estimator.configs_[0] == config_before_saving_0
    assert loaded_estimator.configs_[1] == config_before_saving_1

    with pytest.raises(ValueError, match="Your TabPFN estimator has multiple"):
        save_tabpfn_model(loaded_estimator, Path(tmp_path) / "DOES_NOT_SAVE.ckpt")

    save_tabpfn_model(
        loaded_estimator,
        [Path(tmp_path) / "0.ckpt", Path(tmp_path) / "1.ckpt"],
    )
    assert (tmp_path / "0.ckpt").exists()
    assert (tmp_path / "1.ckpt").exists()


def test_saving_and_loading_with_tuning_config(
    tmp_path: Path,
) -> None:
    """Test that saving and loading a model with a tuning config works."""
    estimator = TabPFNClassifier(
        device="cpu",
        random_state=42,
        eval_metric="f1",
        # TODO: test the case when dataclass is used
        tuning_config={
            "tune_decision_thresholds": True,
            "calibrate_temperature": True,
            "tuning_holdout_frac": 0.1,
            "tuning_n_folds": 1,
        },
    )
    X, y = make_classification(
        n_samples=50, n_features=5, n_classes=3, n_informative=3, random_state=42
    )
    path = tmp_path / "model.tabpfn_fit"
    estimator.fit(X, y)
    estimator.save_fit_state(path)
    loaded_estimator = TabPFNClassifier.load_from_fit_state(path)
    assert loaded_estimator.tuned_classification_thresholds_ is not None
    assert loaded_estimator.softmax_temperature_ is not None
    assert loaded_estimator.eval_metric_ is ClassifierEvalMetrics.F1
