from __future__ import annotations

import io
import itertools
import os
import typing
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
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import ModelSource, prepend_cache_path
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.utils import infer_devices

from .utils import (
    get_pytest_devices,
    is_cpu_float16_supported,
    mark_mps_configs_as_slow,
    patch_layernorm_no_affine,
)

devices = get_pytest_devices()


model_sources = [ModelSource.get_regressor_v2(), ModelSource.get_regressor_v2_5()]
fit_modes = ["low_memory", "fit_preprocessors", "fit_with_cache"]


@pytest.fixture(scope="module")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    X, y, _ = sklearn.datasets.make_regression(
        n_samples=9, n_features=3, random_state=0, coef=True
    )
    return X, y


@pytest.mark.parametrize(
    ("device", "n_estimators", "fit_mode", "inference_precision"),
    mark_mps_configs_as_slow(
        itertools.product(
            devices,
            [1, 2],  # n_estimators
            fit_modes,
            ["auto", "autocast", torch.float64, torch.float16],  # inference_precision
        )
    ),
)
def test__fit_predict__passes_sklearn_check_and_outputs_correct_shape(
    device: str,
    n_estimators: int,
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    if inference_precision == "autocast":
        if torch.device(device).type == "cpu":
            pytest.skip("CPU device does not support 'autocast' inference.")
        if torch.device(device).type == "mps" and torch.__version__ < "2.5":
            pytest.skip("MPS does not support mixed precision before PyTorch 2.5")
    if (
        torch.device(device).type == "cpu"
        and inference_precision == torch.float16
        and not is_cpu_float16_supported()
    ):
        pytest.skip("CPU float16 matmul not supported in this PyTorch version.")
    if torch.device(device).type == "mps" and inference_precision == torch.float64:
        pytest.skip("MPS does not support float64, which is required for this check.")

    model = TabPFNRegressor(
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        random_state=42,
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


@pytest.mark.parametrize("device", [d for d in devices if d == "mps"])
def test__fit_predict__mps_smoke_test__outputs_correct_shape(
    device: str, X_y: tuple[np.ndarray, np.ndarray]
) -> None:
    """Basic test of fit+predict on MPS.

    The other MPS tests in this file are disabled in PRs because they are slow, so this
    test provides some coverage.
    """
    model = TabPFNRegressor(n_estimators=2, device=device, random_state=42)
    X, y = X_y
    model.fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)


non_default_model_paths = list(
    itertools.chain.from_iterable(
        (
            model_path
            for model_path in model_source.filenames
            if model_path != model_source.default_filename
        )
        for model_source in model_sources
    )
)


@pytest.mark.parametrize(
    ("device", "model_path"),
    mark_mps_configs_as_slow(itertools.product(devices, non_default_model_paths)),
)
def test__fit_predict__alternative_model_paths__outputs_correct_shape(
    device: str,
    model_path: str | list[str],
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    model = TabPFNRegressor(
        model_path=prepend_cache_path(model_path),
        n_estimators=1,
        device=device,
        random_state=42,
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize(
    ("device", "fit_mode"),
    mark_mps_configs_as_slow(itertools.product(devices, fit_modes)),
)
def test__fit_predict__multiple_model_paths__outputs_correct_shape(
    device: str,
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    model = TabPFNRegressor(
        model_path=prepend_cache_path(model_sources[0].filenames[:2]),
        n_estimators=1,
        device=device,
        fit_mode=fit_mode,
        random_state=42,
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize(
    ("device", "feature_shift_decoder", "remove_outliers_std"),
    mark_mps_configs_as_slow(
        itertools.chain.from_iterable(
            [
                (device, "shuffle", None),
                (device, "rotate", 12),
                (device, "shuffle", 12),
            ]
            for device in devices
        )
    ),
)
def test__fit_predict__specify_inference_config__outputs_correct_shape(
    device: str,
    feature_shift_decoder: Literal["shuffle", "rotate"],
    remove_outliers_std: int | None,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    model = TabPFNRegressor(
        n_estimators=1,
        device=device,
        inference_config={
            "OUTLIER_REMOVAL_STD": remove_outliers_std,
            "FEATURE_SHIFT_METHOD": feature_shift_decoder,
        },
        random_state=42,
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    assert model.predict(X).shape == (X.shape[0],)


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


def test_multiple_models_predict_different_results(
    X_y: tuple[np.ndarray, np.ndarray],
):
    X, y = X_y

    model_a = model_sources[0].filenames[0]
    model_b = model_sources[0].filenames[1]
    two_identical_models = [model_a, model_a]
    two_different_models = [model_a, model_b]

    def get_prediction(model_paths: list[str]) -> np.ndarray:
        regressor = TabPFNRegressor(
            n_estimators=2,
            random_state=42,
            model_path=prepend_cache_path(model_paths),
        )
        regressor.fit(X, y)
        return regressor.predict(X)

    single_model_pred = get_prediction(model_paths=[model_a])
    two_identical_models_pred = get_prediction(model_paths=two_identical_models)
    two_different_models_pred = get_prediction(model_paths=two_different_models)

    assert not np.all(single_model_pred == single_model_pred[0:1]), (
        "Predictions are identical for all samples, indicating trivial output"
    )
    assert np.all(single_model_pred == two_identical_models_pred)
    assert not np.all(single_model_pred == two_different_models_pred)


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
        "max_features_per_estimator": 500,
    }

    object_config = PreprocessorConfig(
        name="quantile_uni",
        append_original=False,  # changed from default
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd",
        max_features_per_estimator=500,
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
        regressor = TabPFNRegressor(
            n_estimators=1, device="cpu", random_state=43, memory_saving_mode=True
        )
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
        patch_layernorm_no_affine(regressor.models_[0])

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
def test_get_embeddings(
    X_y: tuple[np.ndarray, np.ndarray], data_source: Literal["train", "test"]
) -> None:
    """Test that get_embeddings returns valid embeddings for a fitted model."""
    X, y = X_y
    n_estimators = 3

    model = TabPFNRegressor(n_estimators=n_estimators)
    model.fit(X, y)

    embeddings = model.get_embeddings(X, data_source)

    # Need to access the model through the executor
    model_instance = next(iter(model.executor_.model_caches[0]._models.values()))
    hidden_size = model_instance.ninp

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == n_estimators
    assert embeddings.shape[1] == X.shape[0]
    assert embeddings.shape[2] == hidden_size


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


@pytest.mark.parametrize("constant_value", [0.0, 1.0, -1.0, 1e-5, -1e-5, 1e5, -1e5])
def test_constant_target(
    X_y: tuple[np.ndarray, np.ndarray], constant_value: float
) -> None:
    """Test that TabPFNRegressor predicts a constant
    value when the target y is constant, for both small and large values.
    """
    X, _ = X_y

    y_constant = np.full(X.shape[0], constant_value)

    model = TabPFNRegressor(n_estimators=2, random_state=42)
    model.fit(X, y_constant)

    predictions = model.predict(X)
    assert np.all(predictions == constant_value), (
        f"Predictions are not constant as expected for value {constant_value}"
    )

    # Test different output types
    predictions_median = model.predict(X, output_type="median")
    assert np.all(predictions_median == constant_value), (
        f"Median predictions are not constant as expected for value {constant_value}"
    )

    predictions_mode = model.predict(X, output_type="mode")
    assert np.all(predictions_mode == constant_value), (
        f"Mode predictions are not constant as expected for value {constant_value}"
    )

    quantiles = model.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
    for quantile_prediction in quantiles:
        assert np.all(quantile_prediction == constant_value), (
            f"Quantile predictions are not constant as expected for"
            f" value {constant_value}"
        )

    full_output = model.predict(X, output_type="full")
    assert np.all(full_output["mean"] == constant_value), (
        f"Mean predictions are not constant as expected for full output for"
        f" value {constant_value}"
    )
    assert np.all(full_output["median"] == constant_value), (
        f"Median predictions are not constant as expected for full output for"
        f" value {constant_value}"
    )
    assert np.all(full_output["mode"] == constant_value), (
        f"Mode predictions are not constant as expected for full output for"
        f" value {constant_value}"
    )
    for quantile_prediction in full_output["quantiles"]:
        assert np.all(quantile_prediction == constant_value), (
            f"Quantile predictions are not constant as expected for full output for"
            f" value {constant_value}"
        )


def test_initialize_model_variables_regressor_sets_required_attributes() -> None:
    # 1) Standalone initializer
    model, architecture_configs, norm_criterion, inference_config = (
        initialize_tabpfn_model(
            model_path="auto",
            which="regressor",
            fit_mode="low_memory",
        )
    )
    assert model is not None, "model should be initialized for regressor"
    assert architecture_configs is not None, (
        "config should be initialized for regressor"
    )
    assert norm_criterion is not None, (
        "norm_criterion should be initialized for regressor"
    )
    assert inference_config is not None

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
        architecture_config=regressor.configs_[0],
        norm_criterion=regressor.znorm_space_bardist_,
        inference_config=regressor.inference_config_,
    )
    reg2 = TabPFNRegressor(model_path=spec)
    reg2._initialize_model_variables()

    assert hasattr(reg2, "models_")
    assert reg2.models_ is not None

    assert hasattr(reg2, "configs_")
    assert reg2.configs_ is not None

    assert hasattr(reg2, "znorm_space_bardist_")
    assert reg2.znorm_space_bardist_ is not None


@pytest.mark.parametrize("n_features", [1, 2])
def test__TabPFNRegressor__few_features__works(n_features: int) -> None:
    """Test that TabPFNRegressor works correctly with 1 or 2 features."""
    n_samples = 50

    X, y, _ = sklearn.datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42,
        coef=True,
    )

    model = TabPFNRegressor(
        n_estimators=2,
        random_state=42,
    )

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],), (
        f"Predictions shape is incorrect for {n_features} features"
    )
    assert not np.isnan(predictions).any(), "Predictions contain NaN values"
    assert not np.isinf(predictions).any(), "Predictions contain infinite values"

    predictions_median = model.predict(X, output_type="median")
    assert predictions_median.shape == (X.shape[0],), (
        f"Median predictions shape is incorrect for {n_features} features"
    )

    predictions_mode = model.predict(X, output_type="mode")
    assert predictions_mode.shape == (X.shape[0],), (
        f"Mode predictions shape is incorrect for {n_features} features"
    )

    quantiles = model.predict(X, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    assert isinstance(quantiles, list), "Quantiles should be returned as a list"
    assert len(quantiles) == 3, "Should return 3 quantiles"
    for i, q in enumerate(quantiles):
        assert q.shape == (X.shape[0],), (
            f"Quantile {i} shape is incorrect for {n_features} features"
        )


def test__create_default_for_version__v2__uses_correct_defaults() -> None:
    estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V2)

    assert isinstance(estimator, TabPFNRegressor)
    assert estimator.n_estimators == 8
    assert estimator.softmax_temperature == 0.9
    assert isinstance(estimator.model_path, str)
    assert "regressor" in estimator.model_path
    assert "-v2-" in estimator.model_path


def test__create_default_for_version__v2_5__uses_correct_defaults() -> None:
    estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)

    assert isinstance(estimator, TabPFNRegressor)
    assert estimator.n_estimators == 8
    assert estimator.softmax_temperature == 0.9
    assert isinstance(estimator.model_path, str)
    assert "regressor" in estimator.model_path
    assert "-v2.5-" in estimator.model_path


def test__create_default_for_version__passes_through_overrides() -> None:
    estimator = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_5, n_estimators=16
    )

    assert estimator.n_estimators == 16
    assert estimator.softmax_temperature == 0.9
