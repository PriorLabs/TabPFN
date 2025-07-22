# tests/test_save_load_fitted_model.py
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from tabpfn import TabPFNClassifier, TabPFNRegressor


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
    ("estimator_class", "data_fixture"),
    [
        (TabPFNRegressor, "regression_data"),
        (TabPFNClassifier, "classification_data_with_categoricals"),
    ],
)
def test_save_load_happy_path(estimator_class, data_fixture, request, tmp_path):
    """Tests the standard save/load workflow, including categorical data."""
    X, y = request.getfixturevalue(data_fixture)
    model = estimator_class(device="cpu", n_estimators=4)
    model.fit(X, y)
    path = tmp_path / "model.tabpfn_fit"

    # Save and then load the model using its class method
    model.save_fit_state(path)
    loaded_model = estimator_class.load_from_fit_state(path, device="cpu")

    # 1. Check that predictions are identical
    np.testing.assert_array_almost_equal(model.predict(X), loaded_model.predict(X))

    # 2. For classifiers, also check probabilities and restored classes
    if hasattr(model, "predict_proba"):
        np.testing.assert_array_almost_equal(
            model.predict_proba(X), loaded_model.predict_proba(X)
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
