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
from tabpfn.base import RegressorModelSpecs, initialize_tabpfn_model
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
    estimator_class,
    data_fixture,
    saving_device,
    loading_device,
    request,
    tmp_path,
    monkeypatch,
):
    X, y = request.getfixturevalue(data_fixture)

    # Simulate saving device
    if "cuda" in saving_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    elif "cpu" in saving_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
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
    elif "cpu" in loading_device:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
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
def test_saving_unfitted_model_raises_error(regression_data, tmp_path):
    """Tests that saving an unfitted model raises a RuntimeError."""
    X, y = regression_data
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


def test_saving_and_loading_model_with_weights(tmp_path):
    """Tests that the saving format of the `save_tabpfn_model` method is compatible with
    the loading interface of `initialize_tabpfn_model`.
    """
    # initialize a TabPFNRegressor
    regressor = TabPFNRegressor(model_path="auto", device="cpu", random_state=42)
    regressor._initialize_model_variables()

    # make sure that the model does not use the standard parameter
    first_param = next(regressor.model_.parameters())
    with torch.no_grad():
        first_param.zero_()
    first_model_parameter = first_param.clone()
    config_before_saving = deepcopy(regressor.config_)

    # Save the model state
    save_path = Path(tmp_path) / "model.ckp"
    save_tabpfn_model(regressor, save_path)

    # Load the model state
    model, config, criterion = initialize_tabpfn_model(
        save_path, "regressor", fit_mode="low_memory"
    )
    regressor = TabPFNRegressor(
        model_path=RegressorModelSpecs(
            model=model,
            config=config,
            norm_criterion=criterion,
        ),
        device="cpu",
    )

    # then check the model is loaded correctly
    regressor._initialize_model_variables()
    torch.testing.assert_close(
        next(regressor.model_.parameters()),
        first_model_parameter,
    )
    # Check that the config is the same
    assert regressor.config_ == config_before_saving
