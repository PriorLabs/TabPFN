"""Test saving and loading of a fitted TabPFN classifier/regressor."""

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


def _make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    return make_regression(n_samples=40, n_features=5, random_state=42)


def _make_classification_data_with_categoricals() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=40, n_features=5, n_classes=3, n_informative=3, random_state=42
    )
    # Add a string-based categorical feature
    X_cat = X.astype(object)
    X_cat[:, 2] = np.random.choice(["A", "B", "C"], size=X.shape[0])  # noqa: NPY002
    return X_cat, y


# Exclude pairs where where "mps" is exatly one device type. MPS yields different
# predictions, as dtypes are partly unsupported.
device_pairs = [
    comb for comb in product(get_pytest_devices(), repeat=2) if comb.count("mps") != 1
]


@pytest.mark.parametrize(
    ("task_type", "saving_device", "loading_device"),
    (
        (task_type, saving_device, loading_device)
        for task_type in ["regression", "classification"]
        for (saving_device, loading_device) in device_pairs
    ),
)
def test__save_and_load_twice__predictions_equal_to_before_save(
    task_type: str,
    saving_device: str,
    loading_device: str,
    tmp_path: Path,
) -> None:
    if task_type == "regression":
        estimator_class = TabPFNRegressor
        X, y = _make_regression_data()
    elif task_type == "classification":
        estimator_class = TabPFNClassifier
        X, y = _make_classification_data_with_categoricals()
    else:
        raise ValueError

    original_model = estimator_class(device=saving_device, n_estimators=4)
    original_model.fit(X, y)

    path_1 = tmp_path / "model_1.tabpfn_fit"
    original_model.save_fit_state(path_1)
    loaded_model_1 = estimator_class.load_from_fit_state(path_1, device=loading_device)
    path_2 = tmp_path / "model_2.tabpfn_fit"
    loaded_model_1.save_fit_state(path_2)
    loaded_model_2 = estimator_class.load_from_fit_state(path_2, device=loading_device)

    assert isinstance(loaded_model_1, estimator_class)
    assert isinstance(loaded_model_2, estimator_class)

    original_preds = original_model.predict(X)
    np.testing.assert_array_almost_equal(original_preds, loaded_model_1.predict(X))
    np.testing.assert_array_almost_equal(original_preds, loaded_model_2.predict(X))

    if isinstance(original_model, TabPFNClassifier):
        original_probas = original_model.predict_proba(X)
        np.testing.assert_array_almost_equal(
            original_probas, loaded_model_1.predict_proba(X)
        )
        np.testing.assert_array_almost_equal(
            original_probas, loaded_model_2.predict_proba(X)
        )
        np.testing.assert_array_equal(original_model.classes_, loaded_model_1.classes_)
        np.testing.assert_array_equal(original_model.classes_, loaded_model_2.classes_)


# --- Error Handling Tests ---
def test_saving_unfitted_model_raises_error(tmp_path: Path) -> None:
    """Tests that saving an unfitted model raises a RuntimeError."""
    model = TabPFNRegressor()
    with pytest.raises(RuntimeError, match="Estimator must be fitted before saving"):
        model.save_fit_state(tmp_path / "model.tabpfn_fit")


def test__load_regressor_state_in_classifier__raises_error(tmp_path: Path) -> None:
    X, y = _make_regression_data()
    model = TabPFNRegressor(device="cpu")
    model.fit(X, y)
    path = tmp_path / "model.tabpfn_fit"
    model.save_fit_state(path)

    with pytest.raises(
        TypeError, match="Attempting to load a 'TabPFNRegressor' as 'TabPFNClassifier'"
    ):
        TabPFNClassifier.load_from_fit_state(path)


def test__load_classifier_state_in_regressor__raises_error(tmp_path: Path) -> None:
    X, y = _make_classification_data_with_categoricals()
    model = TabPFNClassifier(device="cpu")
    model.fit(X, y)
    path = tmp_path / "model.tabpfn_fit"
    model.save_fit_state(path)

    with pytest.raises(
        TypeError, match="Attempting to load a 'TabPFNClassifier' as 'TabPFNRegressor'"
    ):
        TabPFNRegressor.load_from_fit_state(path)


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
