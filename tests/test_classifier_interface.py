from __future__ import annotations

import io
import os
import sys
import typing
from itertools import product
from typing import Callable, Literal

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import torch
from sklearn import config_context
from sklearn.base import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from torch import nn

from tabpfn import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

feature_shift_decoders = ["shuffle", "rotate"]
multiclass_decoders = ["shuffle", "rotate"]
fit_modes = [
    "low_memory",
    "fit_preprocessors",
    "fit_with_cache",
]
inference_precision_methods = ["auto", "autocast", torch.float64]
remove_remove_outliers_stds = [None, 12]
estimators = [1, 2]

all_combinations = list(
    product(
        estimators,
        devices,
        feature_shift_decoders,
        multiclass_decoders,
        fit_modes,
        inference_precision_methods,
        remove_remove_outliers_stds,
    ),
)


# Wrap in fixture so it's only loaded in if a test using it is run
@pytest.fixture(scope="module")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X, y = X[:40], y[:40]
    return X, y  # type: ignore


@pytest.mark.parametrize(
    (
        "n_estimators",
        "device",
        "feature_shift_decoder",
        "multiclass_decoder",
        "fit_mode",
        "inference_precision",
        "remove_outliers_std",
    ),
    all_combinations,
)
def test_fit(
    n_estimators: int,
    device: Literal["cuda", "cpu"],
    feature_shift_decoder: Literal["shuffle", "rotate"],
    multiclass_decoder: Literal["shuffle", "rotate"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    remove_outliers_std: int | None,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    if device == "cpu" and inference_precision == "autocast":
        pytest.skip("Only GPU supports inference_precision")

    model = TabPFNClassifier(
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        inference_config={
            "OUTLIER_REMOVAL_STD": remove_outliers_std,
            "CLASS_SHIFT_METHOD": multiclass_decoder,
            "FEATURE_SHIFT_METHOD": feature_shift_decoder,
        },
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (
        X.shape[0],
        len(np.unique(y)),
    ), "Probabilities shape is incorrect"

    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect!"


# TODO(eddiebergman): Should probably run a larger suite with different configurations
@parametrize_with_checks(
    [
        TabPFNClassifier(
            n_estimators=2, inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True}
        ),
    ],
)
def test_sklearn_compatible_estimator(
    estimator: TabPFNClassifier,
    check: Callable[[TabPFNClassifier], None],
) -> None:
    if check.func.__name__ in (  # type: ignore
        "check_methods_subset_invariance",
        "check_methods_sample_order_invariance",
    ):
        estimator.inference_precision = torch.float64

    check(estimator)


def test_balanced_probabilities(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that balance_probabilities=True works correctly."""
    X, y = X_y

    model = TabPFNClassifier(
        balance_probabilities=True,
    )

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    # Check that probabilities sum to 1 for each prediction
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    # Check that the mean probability for each class is roughly equal
    mean_probs = probabilities.mean(axis=0)
    expected_mean = 1.0 / len(np.unique(y))
    assert np.allclose(
        mean_probs,
        expected_mean,
        rtol=0.1,
    ), "Class probabilities are not properly balanced"


def test_classifier_in_pipeline(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that TabPFNClassifier works correctly within a sklearn pipeline."""
    X, y = X_y

    # Create a simple preprocessing pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                TabPFNClassifier(
                    n_estimators=2,  # Fewer estimators for faster testing
                ),
            ),
        ],
    )

    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)

    # Check that probabilities sum to 1 for each prediction
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    # Check that the mean probability for each class is roughly equal
    mean_probs = probabilities.mean(axis=0)
    expected_mean = 1.0 / len(np.unique(y))
    assert np.allclose(
        mean_probs,
        expected_mean,
        rtol=0.1,
    ), "Class probabilities are not properly balanced in pipeline"


def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that dict configs behave identically to PreprocessorConfig objects."""
    X, y = X_y

    # Define same config as both dict and object
    dict_config = {
        "name": "quantile_uni_coarse",
        "append_original": False,  # changed from default
        "categorical_name": "ordinal_very_common_categories_shuffled",
        "global_transformer_name": "svd",
        "subsample_features": -1,
    }

    object_config = PreprocessorConfig(
        name="quantile_uni_coarse",
        append_original=False,  # changed from default
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd",
        subsample_features=-1,
    )

    # Create two models with same random state
    model_dict = TabPFNClassifier(
        inference_config={"PREPROCESS_TRANSFORMS": [dict_config]},
        n_estimators=2,
        random_state=42,
    )

    model_obj = TabPFNClassifier(
        inference_config={"PREPROCESS_TRANSFORMS": [object_config]},
        n_estimators=2,
        random_state=42,
    )

    # Fit both models
    model_dict.fit(X, y)
    model_obj.fit(X, y)

    # Compare predictions
    pred_dict = model_dict.predict(X)
    pred_obj = model_obj.predict(X)
    np.testing.assert_array_equal(pred_dict, pred_obj)

    # Compare probabilities
    prob_dict = model_dict.predict_proba(X)
    prob_obj = model_obj.predict_proba(X)
    np.testing.assert_array_almost_equal(prob_dict, prob_obj)


class ModelWrapper(nn.Module):
    def __init__(self, original_model):  # noqa: D107
        super().__init__()
        self.model = original_model

    def forward(
        self,
        X,
        y,
        single_eval_pos,
        only_return_standard_out,
        categorical_inds,
    ):
        return self.model(
            None,
            X,
            y,
            single_eval_pos=single_eval_pos,
            only_return_standard_out=only_return_standard_out,
            categorical_inds=categorical_inds,
        )


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    if os.name == "nt":
        pytest.skip("onnx export is not tested on windows")
    if sys.version_info >= (3, 13):
        pytest.xfail("onnx is not yet supported on Python 3.13")
    X, y = X_y
    with torch.no_grad():
        classifier = TabPFNClassifier(n_estimators=1, device="cpu", random_state=42)
        # load the model so we can access it via classifier.model_
        classifier.fit(X, y)
        # this is necessary if cuda is available
        classifier.predict(X)
        # replicate the above call with random tensors of same shape
        X = torch.randn(
            (X.shape[0] * 2, 1, X.shape[1] + 1),
            generator=torch.Generator().manual_seed(42),
        )
        y = (
            torch.rand(y.shape, generator=torch.Generator().manual_seed(42))
            .round()
            .to(torch.float32)
        )
        dynamic_axes = {
            "X": {0: "num_datapoints", 1: "batch_size", 2: "num_features"},
            "y": {0: "num_labels"},
        }
        torch.onnx.export(
            ModelWrapper(classifier.model_).eval(),
            (X, y, y.shape[0], True, []),
            io.BytesIO(),
            input_names=[
                "X",
                "y",
                "single_eval_pos",
                "only_return_standard_out",
                "categorical_inds",
            ],
            output_names=["output"],
            opset_version=17,  # using 17 since we use torch>=2.1
            dynamic_axes=dynamic_axes,
        )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_get_embeddings(X_y: tuple[np.ndarray, np.ndarray], data_source: str) -> None:
    """Test that get_embeddings returns valid embeddings for a fitted model."""
    X, y = X_y
    n_estimators = 3

    model = TabPFNClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)

    # Cast to Literal type for mypy
    valid_data_source = typing.cast(Literal["train", "test"], data_source)
    embeddings = model.get_embeddings(X, valid_data_source)

    # Need to access the model through the executor
    model_instance = typing.cast(typing.Any, model.executor_).model
    encoder_shape = next(
        m.out_features
        for m in model_instance.encoder.modules()
        if isinstance(m, nn.Linear)
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == n_estimators
    assert embeddings.shape[1] == X.shape[0]
    assert embeddings.shape[2] == encoder_shape


def test_pandas_output_config():
    """Test compatibility with sklearn's output configuration settings."""
    # Generate synthetic classification data
    X, y = sklearn.datasets.make_classification(
        n_samples=50,
        n_features=10,
        random_state=19,
    )

    # Initialize TabPFN
    model = TabPFNClassifier(n_estimators=1, random_state=42)

    # Get default predictions
    model.fit(X, y)
    default_pred = model.predict(X)
    default_proba = model.predict_proba(X)

    # Test with pandas output
    with config_context(transform_output="pandas"):
        model.fit(X, y)
        pandas_pred = model.predict(X)
        pandas_proba = model.predict_proba(X)
        np.testing.assert_array_equal(default_pred, pandas_pred)
        np.testing.assert_array_almost_equal(default_proba, pandas_proba)

    # Test with polars output
    with config_context(transform_output="polars"):
        model.fit(X, y)
        polars_pred = model.predict(X)
        polars_proba = model.predict_proba(X)
        np.testing.assert_array_equal(default_pred, polars_pred)
        np.testing.assert_array_almost_equal(default_proba, polars_proba)


def test_constant_feature_handling(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that constant features are properly handled and don't affect predictions."""
    X, y = X_y

    # Create a TabPFNClassifier with fixed random state for reproducibility
    model = TabPFNClassifier(n_estimators=2, random_state=42)
    model.fit(X, y)

    # Get predictions on original data
    original_predictions = model.predict(X)
    original_probabilities = model.predict_proba(X)

    # Create a new dataset with added constant features
    X_with_constants = np.hstack(
        [
            X,
            np.zeros((X.shape[0], 3)),  # Add 3 constant zero features
            np.ones((X.shape[0], 2)),  # Add 2 constant one features
            np.full((X.shape[0], 1), 5.0),  # Add 1 constant with value 5.0
        ],
    )

    # Create and fit a new model with the same random state
    model_with_constants = TabPFNClassifier(n_estimators=2, random_state=42)
    model_with_constants.fit(X_with_constants, y)

    # Get predictions on data with constant features
    constant_predictions = model_with_constants.predict(X_with_constants)
    constant_probabilities = model_with_constants.predict_proba(X_with_constants)

    # Verify predictions are the same
    np.testing.assert_array_equal(
        original_predictions,
        constant_predictions,
        err_msg="Predictions changed after adding constant features",
    )

    # Verify probabilities are the same (within numerical precision)
    np.testing.assert_array_almost_equal(
        original_probabilities,
        constant_probabilities,
        decimal=5,
        err_msg="Prediction probabilities changed after adding constant features",
    )


def test_classifier_with_text_and_na() -> None:
    """Test that TabPFNClassifier correctly handles text columns with NA values."""
    # Create a DataFrame with text and NA values
    # Create test data with text and NA values
    data = {
        "text_feature": [
            "good product",
            "bad service",
            None,
            "excellent",
            "average",
            None,
        ],
        "numeric_feature": [10, 5, 8, 15, 7, 12],
        "all_na_column": [None, None, None, None, None, None],  # Column with all NaNs
        "target": [1, 0, 1, 1, 0, 0],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Split into X and y
    X = df[["text_feature", "numeric_feature", "all_na_column"]]
    y = df["target"]

    # Initialize and fit TabPFN on data with text+NA and a column with all NAs
    classifier = TabPFNClassifier(device="cpu", n_estimators=2)

    # This should now work without raising errors
    classifier.fit(X, y)

    # Verify we can predict
    probabilities = classifier.predict_proba(X)
    predictions = classifier.predict(X)

    # Check output shapes
    assert probabilities.shape == (X.shape[0], len(np.unique(y)))
    assert predictions.shape == (X.shape[0],)
