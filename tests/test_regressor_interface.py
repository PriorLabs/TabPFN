from __future__ import annotations

import io
import os
import typing
from itertools import product
from typing import Callable, Literal
from unittest import mock

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
from tabpfn.base import RegressorModelSpecs, initialize_tabpfn_model
from tabpfn.model_loading import ModelSource
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.utils import infer_devices

from .utils import check_cpu_float16_support, get_pytest_devices

devices = get_pytest_devices()

# --- Environment-Aware Check for CPU Float16 Support ---
is_cpu_float16_supported = check_cpu_float16_support()

# --- Define parameter combinations ---
# These are the parameters we want to test in our grid search
feature_shift_decoders = ["shuffle", "rotate"]
fit_modes = [
    "low_memory",
    "fit_preprocessors",
    "fit_with_cache",
]
inference_precision_methods = ["auto", "autocast", torch.float64, torch.float16]
remove_outliers_stds = [None, 12]
estimators = [1, 2]

model_paths = ModelSource.get_regressor_v2().filenames
primary_model = ModelSource.get_regressor_v2().default_filename
other_models = [model_path for model_path in model_paths if model_path != primary_model]

# --- Build parameter combinations ---
# Full grid for the first (primary) model path
_full_grid = product(
    estimators,
    devices,  # device
    feature_shift_decoders,
    fit_modes,
    inference_precision_methods,
    remove_outliers_stds,
    [primary_model],  # only the first entry
)

# Minimal "smoke" grid for all remaining model paths (one combo per path)
_smoke_grid = product(
    [1],  # n_estimators
    ["cpu"],  # device (fast & universally available)
    ["shuffle"],  # feature_shift_decoder
    ["fit_preprocessors"],  # fit_mode
    ["auto"],  # inference_precision
    [remove_outliers_stds[0]],  # remove_outliers_std
    other_models,  # every non-first model path
)

all_combinations = list(_full_grid) + list(_smoke_grid)


# Wrap in fixture so it's only loaded in if a test using it is run
@pytest.fixture(scope="module")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    X, y, _ = sklearn.datasets.make_regression(
        n_samples=50, n_features=9, random_state=0, coef=True
    )
    return X, y


@pytest.mark.parametrize(
    (
        "n_estimators",
        "device",
        "feature_shift_decoder",
        "fit_mode",
        "inference_precision",
        "remove_outliers_std",
        "model_path",
    ),
    all_combinations,
)
def test_regressor(
    n_estimators: int,
    device: Literal["cuda", "mps", "cpu"],
    feature_shift_decoder: Literal["shuffle", "rotate"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    remove_outliers_std: int | None,
    model_path: str,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    if inference_precision == "autocast":
        if torch.device(device).type == "cpu":
            pytest.skip("CPU device does not support 'autocast' inference.")
        if torch.device(device).type == "mps" and torch.__version__ < "2.5":
            pytest.skip("MPS does not support mixed precision before PyTorch 2.5")

    # Use the environment-aware check to skip only if necessary
    if (
        torch.device(device).type == "cpu"
        and inference_precision == torch.float16
        and not is_cpu_float16_supported
    ):
        pytest.skip("CPU float16 matmul not supported in this PyTorch version.")
    if torch.device(device).type == "mps" and inference_precision == torch.float64:
        pytest.skip("MPS does not support float64, which is required for this check.")

    model = TabPFNRegressor(
        model_path=model_path,
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


# The different fitting modes manage the random state differently.
@pytest.mark.skip(
    reason="The prediction is actually different depending on the fitting mode."
)
def test_fit_modes_all_return_equal_results(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    kwargs = {
        "n_estimators": 10,
        "device": "cpu",
        "inference_precision": torch.float32,
        "random_state": 0,
    }
    X, y = X_y

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor(fit_mode="fit_preprocessors", **kwargs)
    tabpfn.fit(X, y)
    preds = tabpfn.predict(X)

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor(fit_mode="fit_with_cache", **kwargs)
    tabpfn.fit(X, y)
    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X))

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor(fit_mode="low_memory", **kwargs)
    tabpfn.fit(X, y)
    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X))


# TODO: Should probably run a larger suite with different configurations
@parametrize_with_checks([TabPFNRegressor(n_estimators=2)])
def test_sklearn_compatible_estimator(
    estimator: TabPFNRegressor,
    check: Callable[[TabPFNRegressor], None],
) -> None:
    _auto_devices = infer_devices(devices="auto")
    if any(device.type == "mps" for device in _auto_devices):
        pytest.skip("MPS does not support float64, which is required for this check.")

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
    assert predictions_median.shape == (X.shape[0],), (
        "Median predictions shape is incorrect"
    )

    predictions_mode = pipeline.predict(X, output_type="mode")
    assert predictions_mode.shape == (X.shape[0],), (
        "Mode predictions shape is incorrect"
    )

    quantiles = pipeline.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
    assert isinstance(quantiles, list)
    assert len(quantiles) == 2
    assert quantiles[0].shape == (X.shape[0],), (
        "Quantile predictions shape is incorrect"
    )


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
        only_return_standard_out,
        categorical_inds,
    ):
        return self.model(
            X,
            y,
            only_return_standard_out=only_return_standard_out,
            categorical_inds=categorical_inds,
        )


# WARNING: unstable for scipy<1.11.0
@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    if os.name == "nt":
        pytest.skip("onnx export is not tested on windows")
    X, y = X_y
    with torch.no_grad():
        regressor = TabPFNRegressor(n_estimators=1, device="cpu", random_state=43)
        # load the model so we can access it via classifier.models_
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

        # From 2.9 PyTorch changed the default export mode from TorchScript to
        # Dynamo. We don't support Dynamo, so disable it. The `dynamo` flag is only
        # available in newer PyTorch versions, hence we don't always include it.
        export_kwargs = {"dynamo": False} if torch.__version__ >= "2.9" else {}
        torch.onnx.export(
            ModelWrapper(regressor.models_[0]).eval(),
            (X, y, True, [[]]),
            io.BytesIO(),
            input_names=[
                "X",
                "y",
                "only_return_standard_out",
                "categorical_inds",
            ],
            output_names=["output"],
            opset_version=17,  # using 17 since we use torch>=2.1
            dynamic_axes=dynamic_axes,
            **export_kwargs,
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
    model_instances = typing.cast(typing.Any, model.executor_).models
    encoder_shape = next(
        m.out_features
        for m in model_instances[0].encoder.modules()
        if isinstance(m, nn.Linear)
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == n_estimators
    assert embeddings.shape[1] == X.shape[0]
    assert embeddings.shape[2] == encoder_shape


def test_overflow_bug_does_not_occur():
    """Test that an overflow does not occur in the preprocessing.

    This can occur if scipy<1.11.0, see
    https://github.com/PriorLabs/TabPFN/issues/175 .

    It no longer appears to happen with the current preprocessing configuration, but
    test just in case.
    """
    rng = np.random.default_rng(seed=0)
    # This is a specially crafted dataset with nearly constant features that has been
    # found to trigger the bug. The California housing dataset will also trigger it.
    n = 20
    X = 100.0 + rng.normal(loc=0.0, scale=0.0001, size=(n, 9))
    y = rng.normal(loc=0.0, scale=1.0, size=(n,))

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

    # Mock the settings to allow large datasets to avoid RuntimeError
    with mock.patch("tabpfn.base.settings.tabpfn.allow_cpu_large_dataset", new=True):
        model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=False)
        model.fit(X_large, y_large)


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


def test_constant_target(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that TabPFNRegressor predicts a constant
    value when the target y is constant.
    """
    X, _ = X_y
    y_constant = np.full(X.shape[0], 5.0)  # Create a constant target array

    model = TabPFNRegressor(n_estimators=2, random_state=42)
    model.fit(X, y_constant)

    predictions = model.predict(X)
    assert np.all(predictions == 5.0), "Predictions are not constant as expected"

    # Test different output types
    predictions_median = model.predict(X, output_type="median")
    assert np.all(predictions_median == 5.0), (
        "Median predictions are not constant as expected"
    )

    predictions_mode = model.predict(X, output_type="mode")
    assert np.all(predictions_mode == 5.0), (
        "Mode predictions are not constant as expected"
    )

    quantiles = model.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
    for quantile_prediction in quantiles:
        assert np.all(quantile_prediction == 5.0), (
            "Quantile predictions are not constant as expected"
        )

    full_output = model.predict(X, output_type="full")
    assert np.all(full_output["mean"] == 5.0), (
        "Mean predictions are not constant as expected for full output"
    )
    assert np.all(full_output["median"] == 5.0), (
        "Median predictions are not constant as expected for full output"
    )
    assert np.all(full_output["mode"] == 5.0), (
        "Mode predictions are not constant as expected for full output"
    )
    for quantile_prediction in full_output["quantiles"]:
        assert np.all(quantile_prediction == 5.0), (
            "Quantile predictions are not constant as expected for full output"
        )


def test_initialize_model_variables_regressor_sets_required_attributes() -> None:
    # 1) Standalone initializer
    model, config, norm_criterion = initialize_tabpfn_model(
        model_path="auto",
        which="regressor",
        fit_mode="low_memory",
    )
    assert model is not None, "model should be initialized for regressor"
    assert config is not None, "config should be initialized for regressor"
    assert norm_criterion is not None, (
        "norm_criterion should be initialized for regressor"
    )

    # 2) Test the sklearn-style wrapper on TabPFNRegressor
    regressor = TabPFNRegressor(device="cpu", random_state=42)
    regressor._initialize_model_variables()

    assert hasattr(regressor, "models_")
    assert regressor.models_ is not None

    assert hasattr(regressor, "configs_")
    assert regressor.configs_ is not None

    assert hasattr(regressor, "znorm_space_bardist_")
    assert regressor.znorm_space_bardist_ is not None

    # 3) Reuse via RegressorModelSpecs
    spec = RegressorModelSpecs(
        model=regressor.models_[0],
        config=regressor.configs_[0],
        norm_criterion=regressor.znorm_space_bardist_,
    )
    reg2 = TabPFNRegressor(model_path=spec)
    reg2._initialize_model_variables()

    assert hasattr(reg2, "models_")
    assert reg2.models_ is not None

    assert hasattr(reg2, "configs_")
    assert reg2.configs_ is not None

    assert hasattr(reg2, "znorm_space_bardist_")
    assert reg2.znorm_space_bardist_ is not None
