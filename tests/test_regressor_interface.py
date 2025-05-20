from __future__ import annotations

import io
import os
import sys
import typing
from itertools import product
from typing import Callable, Literal

import numpy as np
import pytest
import sklearn.datasets
import torch
from sklearn import config_context
from sklearn.base import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from torch import nn

from tabpfn import TabPFNRegressor
from tabpfn.preprocessing import PreprocessorConfig

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

feature_shift_decoders = ["shuffle", "rotate"]
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
        fit_modes,
        inference_precision_methods,
        remove_remove_outliers_stds,
    ),
)


# Wrap in fixture so it's only loaded in if a test using it is run
@pytest.fixture(scope="module")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    X, y = X[:40], y[:40]
    return X, y  # type: ignore


@pytest.mark.parametrize(
    (
        "n_estimators",
        "device",
        "feature_shift_decoder",
        "fit_mode",
        "inference_precision",
        "remove_outliers_std",
    ),
    all_combinations,
)
def test_regressor(
    n_estimators: int,
    device: Literal["cuda", "cpu"],
    feature_shift_decoder: Literal["shuffle", "rotate"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    remove_outliers_std: int | None,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    if device == "cpu" and inference_precision == "autocast":
        pytest.skip("Only GPU supports inference_precision")

    model = TabPFNRegressor(
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        inference_config={
            "OUTLIER_REMOVAL_STD": remove_outliers_std,
            "FEATURE_SHIFT_METHOD": feature_shift_decoder,
        },
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    # Should not fail prediction
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect"

    # check different modes
    predictions = model.predict(X, output_type="median")
    assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect"
    predictions = model.predict(X, output_type="mode")
    assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect"
    quantiles = model.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
    assert isinstance(quantiles, list)
    assert len(quantiles) == 2
    assert quantiles[0].shape == (X.shape[0],), "Predictions shape is incorrect"


# TODO: Should probably run a larger suite with different configurations
@parametrize_with_checks([TabPFNRegressor(n_estimators=2)])
def test_sklearn_compatible_estimator(
    estimator: TabPFNRegressor,
    check: Callable[[TabPFNRegressor], None],
) -> None:
    if check.func.__name__ in (  # type: ignore
        "check_methods_subset_invariance",
        "check_methods_sample_order_invariance",
    ):
        estimator.inference_precision = torch.float64
        pytest.xfail("We're not at 1e-7 difference yet")
    check(estimator)


def test_regressor_in_pipeline(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that TabPFNRegressor works correctly within a sklearn pipeline."""
    X, y = X_y

    # Create a simple preprocessing pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                TabPFNRegressor(
                    n_estimators=2,  # Fewer estimators for faster testing
                ),
            ),
        ],
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    # Check predictions shape
    assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect"

    # Test different prediction modes through the pipeline
    predictions_median = pipeline.predict(X, output_type="median")
    assert predictions_median.shape == (
        X.shape[0],
    ), "Median predictions shape is incorrect"

    predictions_mode = pipeline.predict(X, output_type="mode")
    assert predictions_mode.shape == (
        X.shape[0],
    ), "Mode predictions shape is incorrect"

    quantiles = pipeline.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
    assert isinstance(quantiles, list)
    assert len(quantiles) == 2
    assert quantiles[0].shape == (
        X.shape[0],
    ), "Quantile predictions shape is incorrect"


def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that dict configs behave identically to PreprocessorConfig objects."""
    X, y = X_y

    # Define same config as both dict and object
    dict_config = {
        "name": "quantile_uni",
        "append_original": False,  # changed from default
        "categorical_name": "ordinal_very_common_categories_shuffled",
        "global_transformer_name": "svd",
        "subsample_features": -1,
    }

    object_config = PreprocessorConfig(
        name="quantile_uni",
        append_original=False,  # changed from default
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd",
        subsample_features=-1,
    )

    # Create two models with same random state
    model_dict = TabPFNRegressor(
        inference_config={"PREPROCESS_TRANSFORMS": [dict_config]},
        n_estimators=2,
        random_state=42,
    )

    model_obj = TabPFNRegressor(
        inference_config={"PREPROCESS_TRANSFORMS": [object_config]},
        n_estimators=2,
        random_state=42,
    )

    # Fit both models
    model_dict.fit(X, y)
    model_obj.fit(X, y)

    # Compare predictions for different output types
    for output_type in ["mean", "median", "mode"]:
        # Cast output_type to a valid literal type for mypy
        valid_output_type = typing.cast(
            typing.Literal["mean", "median", "mode"],
            output_type,
        )
        pred_dict = model_dict.predict(X, output_type=valid_output_type)
        pred_obj = model_obj.predict(X, output_type=valid_output_type)
        np.testing.assert_array_almost_equal(
            pred_dict,
            pred_obj,
            err_msg=f"Predictions differ for output_type={output_type}",
        )

    # Compare quantile predictions
    quantiles = [0.1, 0.5, 0.9]
    quant_dict = model_dict.predict(X, output_type="quantiles", quantiles=quantiles)
    quant_obj = model_obj.predict(X, output_type="quantiles", quantiles=quantiles)

    for q_dict, q_obj in zip(quant_dict, quant_obj):
        np.testing.assert_array_almost_equal(
            q_dict,
            q_obj,
            err_msg="Quantile predictions differ",
        )


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


# WARNING: unstable for scipy<1.11.0
@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    if os.name == "nt":
        pytest.skip("onnx export is not tested on windows")
    if sys.version_info >= (3, 13):
        pytest.xfail("onnx is not yet supported on Python 3.13")
    X, y = X_y
    with torch.no_grad():
        regressor = TabPFNRegressor(n_estimators=1, device="cpu", random_state=43)
        # load the model so we can access it via classifier.model_
        regressor.fit(X, y)
        # this is necessary if cuda is available
        regressor.predict(X)
        # replicate the above call with random tensors of same shape
        X = torch.randn(
            (X.shape[0] * 2, 1, X.shape[1] + 1),
            generator=torch.Generator().manual_seed(42),
        )
        y = (torch.randn(y.shape, generator=torch.Generator().manual_seed(42)) > 0).to(
            torch.float32,
        )
        dynamic_axes = {
            "X": {0: "num_datapoints", 1: "batch_size", 2: "num_features"},
            "y": {0: "num_labels"},
        }
        torch.onnx.export(
            ModelWrapper(regressor.model_).eval(),
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

    model = TabPFNRegressor(n_estimators=n_estimators)
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


def test_overflow():
    """Test which fails for scipy<1.11.0."""
    # Fetch a small sample of the California housing dataset
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    X, y = X[:20], y[:20]

    # Create and fit the regressor
    regressor = TabPFNRegressor(n_estimators=1, device="cpu", random_state=42)

    regressor.fit(X, y)

    predictions = regressor.predict(X)
    assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect"


def test_cpu_large_dataset_warning():
    """Test that a warning is raised when using CPU with large datasets."""
    # Create a CPU model
    model = TabPFNRegressor(device="cpu")

    # Create synthetic data slightly above the warning threshold
    rng = np.random.default_rng(seed=42)
    X_large = rng.random((201, 10))
    y_large = rng.random(201)

    # Check that a warning is raised
    with pytest.warns(
        UserWarning, match="Running on CPU with more than 200 samples may be slow"
    ):
        model.fit(X_large, y_large)


def test_cpu_large_dataset_warning_override():
    """Test that runtime error is raised when using CPU with large datasets
    and that we can disable the error with ignore_pretraining_limits.
    """
    rng = np.random.default_rng(seed=42)
    X_large = rng.random((1001, 10))
    y_large = rng.random(1001)

    model = TabPFNRegressor(device="cpu")
    with pytest.raises(
        RuntimeError, match="Running on CPU with more than 1000 samples is not"
    ):
        model.fit(X_large, y_large)

    # -- Test overrides
    model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    model.fit(X_large, y_large)

    # Set environment variable to allow large datasets to avoid RuntimeError
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    try:
        model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=False)
        model.fit(X_large, y_large)
    finally:
        # Clean up environment variable
        os.environ.pop("TABPFN_ALLOW_CPU_LARGE_DATASET")


def test_cpu_large_dataset_error():
    """Test that an error is raised when using CPU with very large datasets."""
    # Create a CPU model
    model = TabPFNRegressor(device="cpu")

    # Create synthetic data above the error threshold
    rng = np.random.default_rng(seed=42)
    X_large = rng.random((1501, 10))
    y_large = rng.random(1501)

    # Check that a RuntimeError is raised
    with pytest.raises(
        RuntimeError, match="Running on CPU with more than 1000 samples is not"
    ):
        model.fit(X_large, y_large)


def test_pandas_output_config():
    """Test compatibility with sklearn's output configuration settings."""
    # Generate synthetic regression data
    X, y = sklearn.datasets.make_regression(
        n_samples=50,
        n_features=10,
        random_state=19,
    )

    # Initialize TabPFN
    model = TabPFNRegressor(n_estimators=1, random_state=42)

    # Get default predictions
    model.fit(X, y)
    default_pred = model.predict(X)

    # Test with pandas output
    with config_context(transform_output="pandas"):
        model.fit(X, y)
        pandas_pred = model.predict(X)
        np.testing.assert_array_almost_equal(default_pred, pandas_pred)

    # Test with polars output
    with config_context(transform_output="polars"):
        model.fit(X, y)
        polars_pred = model.predict(X)
        np.testing.assert_array_almost_equal(default_pred, polars_pred)


def test_constant_feature_handling(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that constant features are properly handled and don't affect predictions."""
    X, y = X_y

    # Create a TabPFNRegressor with fixed random state for reproducibility
    model = TabPFNRegressor(n_estimators=2, random_state=42)
    model.fit(X, y)

    # Get predictions on original data
    original_predictions = model.predict(X)

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
    model_with_constants = TabPFNRegressor(n_estimators=2, random_state=42)
    model_with_constants.fit(X_with_constants, y)

    # Get predictions on data with constant features
    constant_predictions = model_with_constants.predict(X_with_constants)

    # Verify predictions are the same (within numerical precision)
    np.testing.assert_array_almost_equal(
        original_predictions,
        constant_predictions,
        decimal=5,  # Allow small numerical differences
        err_msg="Predictions changed after adding constant features",
    )
