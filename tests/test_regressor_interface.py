#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import io
import itertools
import os
import typing
from collections.abc import Callable
from typing import Any, Literal
from unittest import mock

import numpy as np
import pytest
import sklearn.base
import sklearn.datasets
import torch
from sklearn import config_context
from sklearn.base import check_is_fitted
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from torch import nn

import tabpfn.regressor as regressor_module
from tabpfn import TabPFNRegressor
from tabpfn.architectures import tabpfn_v2_5
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.base import RegressorModelSpecs, initialize_tabpfn_model
from tabpfn.constants import ModelVersion
from tabpfn.inference import InferenceEngineBatchedNoPreprocessing
from tabpfn.inference_config import InferenceConfig
from tabpfn.inference_tuning import (
    MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING,
    RegressorEvalMetrics,
    RegressorTuningConfig,
    get_tuning_temperatures,
)
from tabpfn.model_loading import ModelSource, prepend_cache_path
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.settings import settings
from tabpfn.utils import infer_devices
from tabpfn.validation import ensure_compatible_predict_input_sklearn

from .utils import (
    get_pytest_devices,
    is_cpu_float16_supported,
    mark_mps_configs_as_slow,
    patch_layernorm_no_affine,
)

devices = get_pytest_devices()


model_sources = [
    ModelSource.get_regressor_v2(),
    ModelSource.get_regressor_v2_5(),
    ModelSource.get_regressor_v3(),
]
fit_modes = ["low_memory", "fit_preprocessors"]


def test__show_progress_bar__is_configurable() -> None:
    model = TabPFNRegressor(show_progress_bar=True)
    assert model.show_progress_bar is True
    assert model.get_params()["show_progress_bar"] is True

    default_model = TabPFNRegressor()
    assert default_model.show_progress_bar is False
    assert default_model.get_params()["show_progress_bar"] is False


def test__predict__show_progress_bar_true__tiny_dataset_does_not_crash() -> None:
    model = TabPFNRegressor(n_estimators=1, show_progress_bar=True, random_state=42)
    X, y = sklearn.datasets.make_regression(
        n_samples=9,
        n_features=3,
        random_state=0,
        coef=False,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (X.shape[0],)


def test__forward__passes_progress_bar_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TabPFNRegressor(n_estimators=1, show_progress_bar=True, random_state=42)
    X, y = sklearn.datasets.make_regression(
        n_samples=9, n_features=3, random_state=0, coef=False
    )
    model.fit(X, y)

    captured_kwargs: dict[str, object] = {}

    def fake_tqdm(iterable, **kwargs) -> typing.Iterable[object]:
        captured_kwargs.update(kwargs)
        return iterable

    monkeypatch.setattr(regressor_module, "tqdm", fake_tqdm)

    averaged_logits, outputs, borders = model.forward(X, use_inference_mode=True)

    assert averaged_logits is not None
    assert outputs
    assert borders
    assert captured_kwargs["disable"] is False


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
    fit_mode: Literal["low_memory", "fit_preprocessors"],
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


@pytest.mark.slow
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
    fit_mode: Literal["low_memory", "fit_preprocessors"],
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


@pytest.mark.parametrize(
    "model_version",
    [ModelVersion.V2, ModelVersion.V2_5, ModelVersion.V2_6, ModelVersion.V3],
)
# Disable MPS as it doesn't support float64.
@pytest.mark.parametrize("device", [d for d in get_pytest_devices() if d != "mps"])
def test__fit_preprocessors_and_with_cache_produce_equal_results(
    X_y: tuple[np.ndarray, np.ndarray],
    model_version: ModelVersion,
    device: str,
) -> None:
    kwargs = {
        "version": model_version,
        "n_estimators": 2,
        "inference_precision": torch.float64,
        "random_state": 0,
        "device": device,
    }
    X, y = X_y

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor.create_default_for_version(
        fit_mode="fit_preprocessors", **kwargs
    )
    tabpfn.fit(X, y)
    preds = tabpfn.predict(X)

    # kv_cache_precision="auto" keeps the cache un-quantized, so the cached path
    # matches the non-cached path (int8 quantization would perturb it).
    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor.create_default_for_version(
        fit_mode="fit_with_cache", kv_cache_precision="auto", **kwargs
    )
    tabpfn.fit(X, y)
    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X), decimal=2)


@pytest.mark.parametrize("device", get_pytest_devices())
def test__fit_preprocessors_and_with_cache_with_quantized_kv_cache__v3(
    X_y: tuple[np.ndarray, np.ndarray], device: str
) -> None:
    kwargs = {
        "version": ModelVersion.V3,
        "n_estimators": 2,
        "inference_precision": torch.float32,
        "random_state": 0,
        "device": device,
    }
    X, y = X_y

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor.create_default_for_version(
        fit_mode="fit_preprocessors", **kwargs
    )
    tabpfn.fit(X, y)
    preds = tabpfn.predict(X)

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor.create_default_for_version(
        fit_mode="fit_with_cache", **kwargs
    )
    tabpfn.fit(X, y)
    np.testing.assert_allclose(preds, tabpfn.predict(X), rtol=0.25)


@pytest.mark.parametrize("model_version", list(ModelVersion))
# Disable MPS as it doesn't support float64.
@pytest.mark.parametrize("device", [d for d in get_pytest_devices() if d != "mps"])
def test__fit_preprocessors_and_low_memory_produce_equal_results(
    X_y: tuple[np.ndarray, np.ndarray], model_version: ModelVersion, device: str
) -> None:
    kwargs = {
        "version": model_version,
        "n_estimators": 2,
        "inference_precision": torch.float64,
        "random_state": 0,
        "device": device,
    }
    X, y = X_y

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor.create_default_for_version(
        fit_mode="fit_preprocessors", **kwargs
    )
    tabpfn.fit(X, y)
    preds = tabpfn.predict(X)

    torch.random.manual_seed(0)
    tabpfn = TabPFNRegressor.create_default_for_version(fit_mode="low_memory", **kwargs)
    tabpfn.fit(X, y)
    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X))


@pytest.mark.parametrize("model_version", list(ModelVersion))
def test__fit_and_predict__on_demo_dataset__r2_reasonable(
    model_version: ModelVersion,
) -> None:
    X, y = sklearn.datasets.make_friedman1(n_samples=200, noise=0.1, random_state=0)
    model = TabPFNRegressor.create_default_for_version(
        version=model_version, random_state=0
    )
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    assert r2 > 0.95


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

    for q_dict, q_obj in zip(quant_dict, quant_obj, strict=True):
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
        regressor = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5,
            n_estimators=1,
            device="cpu",
            random_state=43,
            memory_saving_mode=True,
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
    hidden_size = model_instance.embedding_dim

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

    # Create synthetic data slightly above the warning threshold (1000 for v3)
    rng = np.random.default_rng(seed=42)
    X_large = rng.random((1001, 10))
    y_large = rng.random(1001)

    # Check that a warning is raised
    with pytest.warns(
        UserWarning, match="Running on CPU with more than 1000 samples may be slow"
    ):
        model.fit(X_large, y_large)


def test_cpu_large_dataset_warning_override():
    """Test that runtime error is raised when using CPU with large datasets
    and that we can disable the error with ignore_pretraining_limits.
    """
    rng = np.random.default_rng(seed=42)
    X_large = rng.random((5001, 10))
    y_large = rng.random(5001)

    model = TabPFNRegressor(device="cpu")
    with pytest.raises(
        RuntimeError, match="Running on CPU with more than 5000 samples is not"
    ):
        model.fit(X_large, y_large)

    # -- Test overrides
    model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    model.fit(X_large, y_large)

    # Mock the settings to allow large datasets to avoid RuntimeError
    with mock.patch.object(settings.tabpfn, "allow_cpu_large_dataset", new=True):
        model = TabPFNRegressor(device="cpu", ignore_pretraining_limits=False)
        model.fit(X_large, y_large)


def test_cpu_large_dataset_error():
    """Test that an error is raised when using CPU with very large datasets."""
    # Create a CPU model
    model = TabPFNRegressor(device="cpu")

    # Create synthetic data above the error threshold
    rng = np.random.default_rng(seed=42)
    X_large = rng.random((5501, 10))
    y_large = rng.random(5501)

    # Check that a RuntimeError is raised
    with pytest.raises(
        RuntimeError, match="Running on CPU with more than 5000 samples is not"
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
    model = TabPFNRegressor(
        n_estimators=1,
        random_state=42,
        inference_config={
            "POLYNOMIAL_FEATURES": "no",
            # Set to False to avoid different fingerprint features for tiny
            # numerical variations e.g. from non-deterministic algorithms in
            # preprocessing steps before adding the fingerprint feature.
            "FINGERPRINT_FEATURE": False,
        },
    )
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
    model_with_constants = TabPFNRegressor(
        n_estimators=1,
        random_state=42,
        inference_config={
            "POLYNOMIAL_FEATURES": "no",
            "FINGERPRINT_FEATURE": False,  # see above for why
        },
    )
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
    assert estimator.n_estimators == "auto"
    assert estimator.softmax_temperature == "auto"
    assert isinstance(estimator.model_path, str)
    assert "regressor" in estimator.model_path
    assert "-v2-" in estimator.model_path


def test__create_default_for_version__v2_5__uses_correct_defaults() -> None:
    estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)

    assert isinstance(estimator, TabPFNRegressor)
    assert estimator.n_estimators == "auto"
    assert estimator.softmax_temperature == "auto"
    assert isinstance(estimator.model_path, str)
    assert "regressor" in estimator.model_path
    assert "-v2.5-" in estimator.model_path


def test__create_default_for_version__v2_6__uses_correct_defaults() -> None:
    estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V2_6)

    assert isinstance(estimator, TabPFNRegressor)
    assert estimator.n_estimators == "auto"
    assert estimator.softmax_temperature == "auto"
    assert isinstance(estimator.model_path, str)
    assert "regressor" in estimator.model_path
    assert "-v2.6-" in estimator.model_path


def test__create_default_for_version__v3__uses_correct_defaults() -> None:
    estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V3)

    assert isinstance(estimator, TabPFNRegressor)
    assert estimator.n_estimators == "auto"
    assert estimator.softmax_temperature == "auto"
    assert isinstance(estimator.model_path, str)
    assert "regressor" in estimator.model_path
    assert "-v3-" in estimator.model_path


def test__create_default_for_version__passes_through_overrides() -> None:
    estimator = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_5, n_estimators=16
    )

    assert estimator.n_estimators == 16
    assert estimator.softmax_temperature == "auto"


# ---------------------------------------------------------------------------
# differentiable_input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
def test__fit_with_differentiable_input__grad_flows_to_upstream_module(
    device: str,
) -> None:
    """End-to-end: a loss computed from forward(use_inference_mode=True) after
    fit_with_differentiable_input must produce a non-zero, finite gradient on
    an upstream torch module's weights.
    """
    torch.manual_seed(0)
    D, N_train, N_test = 8, 30, 10
    linear = nn.Linear(D, D).to(device)

    X_train = linear(torch.randn(N_train, D, device=device))
    X_test = linear(torch.randn(N_test, D, device=device))
    y_train = torch.randn(N_train, device=device)
    y_test = torch.randn(N_test, device=device)

    reg = TabPFNRegressor(
        n_estimators=1,
        ignore_pretraining_limits=True,
        device=device,
        differentiable_input=True,
    )
    reg.fit_with_differentiable_input(X_train, y_train)

    averaged_logits, _outputs, borders = reg.forward(X_test, use_inference_mode=True)

    # averaged_logits is [N_borders, N_samples] after the transpose in
    # forward(); reduce to a scalar per sample via softmax over bin centers.
    per_sample_logits = averaged_logits.transpose(0, 1)  # [N_test, N_borders]
    border_t = torch.as_tensor(
        borders[0],
        device=per_sample_logits.device,
        dtype=per_sample_logits.dtype,
    )
    n_logits = per_sample_logits.shape[-1]
    if border_t.numel() == n_logits + 1:
        bin_centers = (border_t[:-1] + border_t[1:]) / 2.0
    else:
        bin_centers = border_t
    probs = torch.softmax(per_sample_logits.float(), dim=-1)
    pred_z = (probs * bin_centers).sum(dim=-1)
    pred = pred_z * float(reg.y_train_std_) + float(reg.y_train_mean_)

    loss = torch.nn.functional.mse_loss(pred.float(), y_test.float())
    assert loss.requires_grad
    loss.backward()

    grad = linear.weight.grad
    assert grad is not None, "gradient did not reach upstream nn.Linear"
    assert torch.isfinite(grad).all(), "gradient contained NaN/Inf"
    assert grad.norm().item() > 0, "gradient norm is zero — graph was detached"


def test__fit__differentiable_input_true__raises_helpful_error() -> None:
    """Calling .fit() (instead of fit_with_differentiable_input) when
    differentiable_input=True must raise a clear error pointing users to the
    correct API rather than silently running a non-differentiable path.
    """
    reg = TabPFNRegressor(
        n_estimators=1,
        ignore_pretraining_limits=True,
        device="cpu",
        differentiable_input=True,
    )
    X = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
    y = np.random.default_rng(0).standard_normal(20).astype(np.float32)
    with pytest.raises(ValueError, match="fit_with_differentiable_input"):
        reg.fit(X, y)


@pytest.mark.parametrize(
    ("case_id", "extra_kwargs", "X", "y", "match"),
    [
        # The differentiable path uses an identity preprocessor and has no
        # ordinal-encoding step, so categorical columns have no valid handling.
        (
            "categorical_features",
            {"categorical_features_indices": [0]},
            torch.randn(20, 4),
            torch.randn(20),
            "Categorical features",
        ),
        # Constant target collapses the bardist borders to a single point.
        (
            "constant_target",
            {},
            torch.randn(5, 4),
            torch.full((5,), 3.14),
            "Constant or near-constant target",
        ),
        # torch.std defaults to correction=1 and returns NaN for N=1; our path
        # uses correction=0 so std collapses to 0 and trips the constant-target
        # guard instead of a downstream NaN.
        (
            "single_sample",
            {},
            torch.randn(1, 4),
            torch.tensor([2.0]),
            "Constant or near-constant target",
        ),
    ],
)
def test__fit_with_differentiable_input__bad_input_raises_value_error(
    case_id: str,
    extra_kwargs: dict[str, object],
    X: torch.Tensor,
    y: torch.Tensor,
    match: str,
) -> None:
    """Bad inputs to the differentiable fit path must raise ValueError with a
    clear message rather than producing NaNs or crashing downstream.
    """
    del case_id  # Only used for parametrize ids.
    reg = TabPFNRegressor(
        n_estimators=1,
        ignore_pretraining_limits=True,
        device="cpu",
        differentiable_input=True,
        **extra_kwargs,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match=match):
        reg.fit_with_differentiable_input(X, y)


def test__fit_with_differentiable_input__std_matches_population_definition() -> None:
    """The differentiable path's y_train_std_ should match np.std (population
    std, ddof=0), not torch's default sample std (correction=1), so it lines
    up with the standard fit() path.
    """
    reg = TabPFNRegressor(
        n_estimators=1,
        ignore_pretraining_limits=True,
        device="cpu",
        differentiable_input=True,
    )
    X = torch.randn(20, 4)
    y_np = np.random.default_rng(0).standard_normal(20).astype(np.float32)
    y = torch.from_numpy(y_np)
    reg.fit_with_differentiable_input(X, y)
    expected = float(np.std(y_np))  # ddof=0
    assert abs(reg.y_train_std_ - expected) < 1e-5, (
        f"y_train_std_ should equal np.std(y) (population std); "
        f"got {reg.y_train_std_}, expected {expected}"
    )


def test__fit_with_differentiable_input__feature_schema_cols_independent() -> None:
    """Each column's Feature must be a distinct instance — list multiplication
    `[Feature(...)] * n` would alias all columns to one mutable dataclass.
    """
    reg = TabPFNRegressor(
        n_estimators=1,
        ignore_pretraining_limits=True,
        device="cpu",
        differentiable_input=True,
    )
    X = torch.randn(10, 4)
    y = torch.randn(10)
    reg.fit_with_differentiable_input(X, y)
    feats = reg.inferred_feature_schema_.features
    assert len(feats) == 4
    # Distinct instances, not aliases.
    ids = {id(f) for f in feats}
    assert len(ids) == 4, "feature columns share the same Feature instance"


def test__fit_with_differentiable_input__second_call_refreshes_target_stats() -> None:
    """A second call with different y must update y_train_mean_/std_ and the
    raw_space_bardist_; only the model load and ensemble configs are cached.
    """
    torch.manual_seed(0)
    reg = TabPFNRegressor(
        n_estimators=1,
        ignore_pretraining_limits=True,
        device="cpu",
        differentiable_input=True,
    )
    X1 = torch.randn(20, 4)
    y1 = torch.randn(20) * 10.0 + 100.0  # mean ~100, std ~10
    reg.fit_with_differentiable_input(X1, y1)
    mean1, std1 = reg.y_train_mean_, reg.y_train_std_
    bardist_borders1 = reg.raw_space_bardist_.borders.clone()

    X2 = torch.randn(20, 4)
    y2 = torch.randn(20) * 0.5 - 5.0  # mean ~-5, std ~0.5
    reg.fit_with_differentiable_input(X2, y2)
    mean2, std2 = reg.y_train_mean_, reg.y_train_std_

    assert abs(mean2 - mean1) > 1.0, (
        f"y_train_mean_ should reflect new y; got {mean1} -> {mean2}"
    )
    assert abs(std2 - std1) > 1.0, (
        f"y_train_std_ should reflect new y; got {std1} -> {std2}"
    )
    # raw_space_bardist_ borders are derived from y stats; they must move.
    assert not torch.allclose(reg.raw_space_bardist_.borders, bardist_borders1), (
        "raw_space_bardist_ must be rebuilt to the new target scale"
    )


# =============================================================================
# predict_batched: batched multi-dataset inference (public API)
# =============================================================================


def _mk_reg_dataset(
    seed: int, n: int = 60, f: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.RandomState(seed)
    X = r.randn(n, f).astype(np.float32)
    y = (X[:, 0] * 2.0 + 0.5 * X[:, 1] + 0.1 * r.randn(n)).astype(np.float32)
    return X, y


@pytest.mark.parametrize("device", devices)
def test__predict_batched__matches_per_dataset(device: str) -> None:
    """predict_batched matches per-dataset fit+predict, in input order."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    data = [_mk_reg_dataset(s) for s in range(3)]
    X_list = [d[0] for d in data]
    y_list = [d[1] for d in data]
    X_tests = [_mk_reg_dataset(s + 100, n=8)[0] for s in range(3)]

    reg = TabPFNRegressor(
        n_estimators=4,
        device=device,
        random_state=42,
        inference_precision=torch.float32,
    )
    batched_mean = reg.predict_batched(X_list, y_list, X_tests, output_type="mean")
    batched_q = reg.predict_batched(X_list, y_list, X_tests, output_type="quantiles")

    assert len(batched_mean) == 3
    assert batched_mean[0].shape == (8,)
    assert len(batched_q[0]) == 9  # default quantiles

    atol = 1e-4 if torch.device(device).type == "cpu" else 1e-2
    for i, (X, y) in enumerate(data):
        ref = TabPFNRegressor(
            n_estimators=4,
            device=device,
            random_state=42,
            inference_precision=torch.float32,
        )
        ref.fit(X, y)
        np.testing.assert_allclose(
            batched_mean[i], ref.predict(X_tests[i], output_type="mean"), atol=atol
        )
        ref_q = ref.predict(X_tests[i], output_type="quantiles")
        for bq, rq in zip(batched_q[i], ref_q, strict=True):
            np.testing.assert_allclose(bq, rq, atol=atol)


def test__predict_batched__full_output_uses_per_dataset_criterion() -> None:
    """output_type="full" returns each dataset's own raw-space bar distribution.

    The raw-space bardist is rescaled by the dataset's own y mean and std, so
    datasets on different target scales must not share the first one's criterion.
    """
    X_a, y_a = _mk_reg_dataset(0)
    X_b, y_b = _mk_reg_dataset(1)
    y_b = y_b * 1000.0 + 5000.0  # same features, very different target scale

    reg = TabPFNRegressor(
        n_estimators=2, device="cpu", random_state=42, inference_precision=torch.float32
    )
    out = reg.predict_batched(
        [X_a, X_b], [y_a, y_b], [X_a[:5], X_b[:5]], output_type="full"
    )

    assert len(out) == 2
    borders_a = out[0]["criterion"].borders
    borders_b = out[1]["criterion"].borders
    assert not torch.allclose(borders_a, borders_b)
    # And the point estimates land on each dataset's own scale.
    assert abs(float(np.mean(out[0]["mean"]))) < 100.0
    assert float(np.mean(out[1]["mean"])) > 1000.0

    for i, (X, y) in enumerate([(X_a, y_a), (X_b, y_b)]):
        ref = TabPFNRegressor(
            n_estimators=2,
            device="cpu",
            random_state=42,
            inference_precision=torch.float32,
        )
        ref.fit(X, y)
        ref_full = ref.predict(X[:5], output_type="full")
        # Relative: dataset B lives on a ~5000 scale, where float32 decode noise
        # is a few ulp rather than an absolute 1e-4.
        np.testing.assert_allclose(out[i]["mean"], ref_full["mean"], rtol=1e-4)
        torch.testing.assert_close(
            out[i]["criterion"].borders, ref_full["criterion"].borders
        )


def test__predict_batched__uses_fitted_target_transforms() -> None:
    """The border mapping must read the members' configs, not ensemble_configs_.

    With ``n_preprocessing_jobs > 1`` ``target_transform`` is fitted in a joblib
    worker, so ``ensemble_configs_`` keeps an unfitted copy and decoding with it
    would silently produce wrong borders.
    """
    data = [_mk_reg_dataset(s) for s in range(2)]
    X_list = [d[0] for d in data]
    y_list = [d[1] for d in data]
    X_tests = [d[0][:5] for d in data]

    kwargs = {
        "n_estimators": 4,
        "device": "cpu",
        "random_state": 42,
        "n_preprocessing_jobs": 2,
        "inference_precision": torch.float32,
    }
    batched = TabPFNRegressor(**kwargs).predict_batched(X_list, y_list, X_tests)

    # Without a target transform in play the test would pass vacuously.
    probe = TabPFNRegressor(**kwargs)
    probe.fit(X_list[0], y_list[0])
    assert any(
        m.config.target_transform is not None for m in probe.executor_.ensemble_members
    )

    for i, (X, y) in enumerate(data):
        ref = TabPFNRegressor(**kwargs)
        ref.fit(X, y)
        np.testing.assert_allclose(batched[i], ref.predict(X_tests[i]), atol=1e-4)


@pytest.mark.parametrize("device", devices)
def test__predict_batched__matches_per_dataset_dataframe(device: str) -> None:
    """Batched prediction matches per-dataset on non-numeric DataFrame inputs.

    predict_batched must clean X_test as predict does. The frames mix a
    categorical column, an object column, and NaNs placed within the first 5
    rows so X_test exercises them too.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    pd = pytest.importorskip("pandas")

    def mkdf(seed: int, n: int = 60) -> tuple[typing.Any, np.ndarray]:
        r = np.random.RandomState(seed)
        num = r.randn(n, 3).astype(np.float32)
        df = pd.DataFrame(
            {
                "f0": num[:, 0],
                "f1": num[:, 1],
                "f2": num[:, 2],
                "cat": pd.Categorical(r.choice(["a", "b", "c"], size=n)),
                "txt": r.choice(["red", "green", "blue"], size=n).astype(object),
            }
        )
        df.loc[[1, 4, 17], "f1"] = np.nan
        df.loc[[2, 4, 23], "txt"] = None
        y = (num[:, 0] * 2.0 + 0.1 * r.randn(n)).astype(np.float32)
        return df, y

    data = [mkdf(s) for s in range(3)]
    X_tests = [d[0].iloc[:5] for d in data]

    reg = TabPFNRegressor(
        n_estimators=2,
        device=device,
        random_state=42,
        inference_precision=torch.float32,
    )
    batched = reg.predict_batched(
        [d[0] for d in data], [d[1] for d in data], X_tests, output_type="mean"
    )

    atol = 1e-4 if torch.device(device).type == "cpu" else 1e-2
    for i, (X, y) in enumerate(data):
        ref = TabPFNRegressor(
            n_estimators=2,
            device=device,
            random_state=42,
            inference_precision=torch.float32,
        )
        ref.fit(X, y)
        np.testing.assert_allclose(batched[i], ref.predict(X_tests[i]), atol=atol)


def test__predict_batched__handles_constant_target_dataset() -> None:
    """A constant-target dataset is answered analytically and stitched back.

    It takes no part in the fused forward, and must not disturb the others.
    """
    X_a, y_a = _mk_reg_dataset(0)
    X_b, _ = _mk_reg_dataset(1)
    y_b = np.full(X_b.shape[0], 7.5, dtype=np.float32)
    X_c, y_c = _mk_reg_dataset(2)

    reg = TabPFNRegressor(
        n_estimators=2, device="cpu", random_state=42, inference_precision=torch.float32
    )
    out = reg.predict_batched(
        [X_a, X_b, X_c], [y_a, y_b, y_c], [X_a[:5], X_b[:5], X_c[:5]]
    )

    assert len(out) == 3
    np.testing.assert_allclose(out[1], np.full(5, 7.5), atol=1e-6)
    for i, (X, y) in enumerate([(X_a, y_a), (X_b, y_b), (X_c, y_c)]):
        ref = TabPFNRegressor(
            n_estimators=2,
            device="cpu",
            random_state=42,
            inference_precision=torch.float32,
        )
        ref.fit(X, y)
        np.testing.assert_allclose(out[i], ref.predict(X[:5]), atol=1e-4)


@pytest.mark.parametrize("device", devices)
def test__predict_batched__fp16_matches_fp32(device: str) -> None:
    """predict_batched works with inference_precision=float16.

    Guards the regressor path against the fp16 dtype mismatch fixed for the
    classifier. Older torch lacks CPU fp16 matmul, so skip CPU there.
    """
    if torch.device(device).type == "cpu" and not is_cpu_float16_supported():
        pytest.skip("CPU float16 matmul not supported in this PyTorch version.")

    data = [_mk_reg_dataset(s) for s in range(3)]
    X_list = [d[0] for d in data]
    y_list = [d[1] for d in data]
    X_tests = [d[0][:5] for d in data]

    out16 = TabPFNRegressor(
        n_estimators=2,
        device=device,
        random_state=42,
        inference_precision=torch.float16,
    ).predict_batched(X_list, y_list, X_tests)
    out32 = TabPFNRegressor(
        n_estimators=2,
        device=device,
        random_state=42,
        inference_precision=torch.float32,
    ).predict_batched(X_list, y_list, X_tests)

    assert len(out16) == 3
    for a, b in zip(out16, out32, strict=True):
        assert a.shape == (5,)
        # fp16 accumulates differently; only coarse agreement is expected.
        assert np.abs(a - b).mean() < 0.5 * float(np.std(y_list[0]))


def test__predict_batched__rejects_ragged() -> None:
    """Ragged batches are rejected (padding would corrupt the fused forward)."""
    X_a, y_a = _mk_reg_dataset(0, n=40)
    X_b, y_b = _mk_reg_dataset(1, n=30)  # different number of training rows

    reg = TabPFNRegressor(n_estimators=2, device="cpu", random_state=42)
    with pytest.raises(ValueError, match=r"ragged"):
        reg.predict_batched([X_a, X_b], [y_a, y_b], [X_a[:3], X_b[:3]])


def test__predict_batched__rejects_float64_precision() -> None:
    """float64 must raise rather than silently compute at float32.

    The fused batch is built at float32 and predict_batched bypasses the
    float64 assert in _iter_forward_executor, so without this guard a float64
    request returns float32-precision numbers while claiming predict parity.
    """
    X, y = _mk_reg_dataset(0, n=40)
    reg = TabPFNRegressor(
        n_estimators=2, device="cpu", random_state=42, inference_precision=torch.float64
    )
    with pytest.raises(NotImplementedError, match=r"float64"):
        reg.predict_batched([X, X], [y, y], [X[:3], X[:3]])


def test__predict_batched__rejects_tuning_config() -> None:
    """tuning_config calibrates a per-dataset temperature → unsupported, raises.

    `ensemble_softmax_temperature_` is fitted on each dataset's own holdout, so
    a shared batch has no single temperature to apply. Without this guard the
    fused path silently returned *uncalibrated* predictions while claiming
    parity with `predict`, and paid for a discarded calibration sweep per
    dataset on top. Mirrors the classifier's `predict_proba_batched` guard.
    """
    X, y = _mk_reg_dataset(0, n=40)
    reg = TabPFNRegressor(
        n_estimators=2,
        device="cpu",
        random_state=42,
        model_path=_create_dummy_regressor_model_specs(),
        tuning_config={"calibrate_temperature": True},
    )
    with pytest.raises(NotImplementedError, match=r"tuning_config"):
        reg.predict_batched([X, X], [y, y], [X[:3], X[:3]])


def test__predict_batched__shares_its_logit_reduction_with_predict() -> None:
    """The two paths must reduce accumulated logits through the same code.

    `predict_batched` once carried its own copy of the averaging tail and so
    silently dropped the `ensemble_softmax_temperature_` step that `predict`
    applies. Pinning both to `_reduce_accumulated_logits` is what stops them
    drifting apart again, so assert the batched path really routes through it.
    """
    reg = TabPFNRegressor(
        n_estimators=2,
        device="cpu",
        random_state=42,
        inference_precision=torch.float32,
        model_path=_create_dummy_regressor_model_specs(),
    )
    datasets = [_mk_reg_dataset(s, n=40) for s in (0, 1)]

    calls = []
    original = TabPFNRegressor._reduce_accumulated_logits

    def spy(self, accumulated_logits, n_estimators):  # noqa: ANN202
        calls.append(n_estimators)
        return original(self, accumulated_logits, n_estimators)

    with mock.patch.object(TabPFNRegressor, "_reduce_accumulated_logits", spy):
        reg.predict_batched(
            [d[0] for d in datasets],
            [d[1] for d in datasets],
            [d[0][:5] for d in datasets],
        )

    # One reduction per dataset, each over the full ensemble.
    assert calls == [2, 2]


def test__predict_batched__float64_inputs_match_per_dataset() -> None:
    """float64 *arrays* stay supported; only float64 precision is rejected.

    Real feature pipelines hand over float64 (np.asarray of a DataFrame, for
    one), so this must behave exactly like the float32 case.
    """
    data = [_mk_reg_dataset(s) for s in range(2)]
    data = [(X.astype(np.float64), y.astype(np.float64)) for X, y in data]
    X_list = [d[0] for d in data]
    y_list = [d[1] for d in data]
    X_tests = [d[0][:5] for d in data]

    kwargs = {
        "n_estimators": 2,
        "device": "cpu",
        "random_state": 42,
        "inference_precision": torch.float32,
    }
    batched = TabPFNRegressor(**kwargs).predict_batched(X_list, y_list, X_tests)
    for i, (X, y) in enumerate(data):
        ref = TabPFNRegressor(**kwargs)
        ref.fit(X, y)
        np.testing.assert_allclose(batched[i], ref.predict(X_tests[i]), rtol=1e-4)


def test__predict_batched__rejects_bad_arguments() -> None:
    """Length, emptiness, output_type and quantile validation mirror predict."""
    X, y = _mk_reg_dataset(0, n=40)
    reg = TabPFNRegressor(n_estimators=2, device="cpu", random_state=42)

    with pytest.raises(ValueError, match=r"equal length"):
        reg.predict_batched([X, X], [y], [X[:3], X[:3]])
    with pytest.raises(ValueError, match=r"empty dataset list"):
        reg.predict_batched([], [], [])
    with pytest.raises(regressor_module.TabPFNValidationError):
        reg.predict_batched([X], [y], [X[:3]], output_type="nonsense")  # type: ignore[arg-type]
    with pytest.raises(regressor_module.TabPFNValidationError):
        reg.predict_batched([X], [y], [X[:3]], output_type="quantiles", quantiles=[1.5])


def test__predict_batched__folds_estimators_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each estimator output is folded in before the next one is produced.

    Collecting the fused outputs first would keep every estimator's
    (n_test, n_datasets, n_buckets) tensor alive at once.
    """
    rng = np.random.default_rng(0)
    constant = rng.normal(size=(60, 12)).astype(np.float32)
    constant[:, :5] = 0
    dense = rng.normal(size=(60, 12)).astype(np.float32)
    X_list = [constant, dense, constant.copy()]
    y_list = [X[:, 5] - X[:, 6] for X in X_list]
    X_tests = [X[:5] for X in X_list]

    n_folded = 0
    group_sizes: list[int] = []
    translate = TabPFNRegressor._translate_batched_logits

    def counting_translate(self: TabPFNRegressor, **kwargs: typing.Any) -> torch.Tensor:
        nonlocal n_folded
        n_folded += 1
        return translate(self, **kwargs)

    iter_outputs = InferenceEngineBatchedNoPreprocessing.iter_outputs

    def checking_iter_outputs(
        self: InferenceEngineBatchedNoPreprocessing,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Iterator[typing.Any]:
        folded_before_group = n_folded
        group_size = self.X_trains[0].shape[0]
        group_sizes.append(group_size)
        for estimator, item in enumerate(iter_outputs(self, *args, **kwargs)):
            assert n_folded == folded_before_group + estimator * group_size, (
                "outputs are being materialized instead of folded in as they arrive"
            )
            yield item

    monkeypatch.setattr(
        TabPFNRegressor, "_translate_batched_logits", counting_translate
    )
    monkeypatch.setattr(
        InferenceEngineBatchedNoPreprocessing, "iter_outputs", checking_iter_outputs
    )

    reg = TabPFNRegressor(
        n_estimators=3, device="cpu", random_state=42, inference_precision=torch.float32
    )
    reg.predict_batched(X_list, y_list, X_tests)
    assert group_sizes == [2, 1]
    assert n_folded == 3 * len(X_list)


def test__predict_batched__does_not_mutate_estimator() -> None:
    """predict_batched runs on an internal clone, leaving self untouched."""
    data = [_mk_reg_dataset(s) for s in range(3)]
    X_list = [d[0] for d in data]
    y_list = [d[1] for d in data]
    X_tests = [d[0][:3] for d in data]

    # An unfitted estimator stays unfitted (no executor_ left behind).
    reg = TabPFNRegressor(n_estimators=2, device="cpu", random_state=42)
    reg.predict_batched(X_list, y_list, X_tests)
    assert not hasattr(reg, "executor_")
    assert reg.fit_mode != "batched"

    # A prior fit is preserved exactly across a batched call.
    a_x, a_y = _mk_reg_dataset(99, n=80)
    fitted = TabPFNRegressor(
        n_estimators=2, device="cpu", random_state=42, inference_precision=torch.float32
    )
    fitted.fit(a_x, a_y)
    before = fitted.predict(a_x[:5])
    fitted.predict_batched(X_list, y_list, X_tests)
    after = fitted.predict(a_x[:5])
    np.testing.assert_array_equal(before, after)


def _create_dummy_regressor_model_specs() -> RegressorModelSpecs:
    """A tiny in-memory model, so tuning tests need no checkpoint download."""
    # The regression head is sized by `max_num_classes`, so it has to match
    # `num_buckets` or the bucket logits do not fit.
    num_buckets = 100
    minimal_config = tabpfn_v2_5.TabPFNV2p5Config(
        emsize=8,
        features_per_group=1,
        max_num_classes=num_buckets,
        nhead=2,
        nlayers=2,
        num_buckets=num_buckets,
    )
    return RegressorModelSpecs(
        model=tabpfn_v2_5.get_architecture(
            config=minimal_config,
            cache_trainset_representation=False,
        ),
        architecture_config=minimal_config,
        inference_config=InferenceConfig.get_default(
            task_type="regression",
            model_version=ModelVersion.V2_5,
        ),
        norm_criterion=FullSupportBarDistribution(
            torch.linspace(-3, 3, num_buckets + 1)
        ),
    )


def test__compute_holdout_validation_data__returns_self_consistent_triples() -> None:
    X, y = sklearn.datasets.make_regression(
        n_samples=120, n_features=4, noise=10.0, random_state=0
    )
    # Shift the target well away from zero mean and unit variance, so that raw
    # units are clearly distinguishable from z-normalised ones.
    y = y * 3.0 + 100.0
    regressor = TabPFNRegressor(
        n_estimators=1,
        random_state=0,
        device="cpu",
        model_path=_create_dummy_regressor_model_specs(),
    )
    # The tuning regressors take their categorical indices from this attribute,
    # which `fit` sets before it reaches this method.
    regressor.categorical_features_indices_ = None

    folds = regressor._compute_holdout_validation_data(
        X=X, y=y, holdout_frac=0.5, n_folds=2
    )

    assert len(folds) == 2
    for logits, raw_space_bardist, y_holdout in folds:
        n_holdout = len(X) // 2
        assert logits.shape == (n_holdout, raw_space_bardist.num_bars)
        assert y_holdout.shape == (n_holdout,)
        # The bar distribution scores these targets directly, so all three must
        # agree on device and the targets must match the borders' dtype.
        assert y_holdout.device == logits.device
        assert y_holdout.dtype == raw_space_bardist.borders.dtype
        # Targets are in the units of the original column, not z-normalised.
        assert float(y_holdout.mean()) == pytest.approx(y.mean(), abs=5 * y.std())
        # The borders are in the target's units too. They need not bracket every
        # target -- the outermost buckets carry half-normal tails -- but their
        # span tracks the target's spread rather than a z-normalised one.
        borders = raw_space_bardist.borders
        assert borders.min() < np.median(y) < borders.max()
        assert float(borders.max() - borders.min()) > y.std()

    # Each fold normalises with its own statistics, which is exactly why the
    # folds cannot be concatenated and scored under one bar distribution.
    assert not torch.equal(folds[0][1].borders, folds[1][1].borders)


def test__compute_holdout_validation_data__skips_constant_training_targets() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    # One single distinct target value: with two folds it lands in exactly one
    # fold's holdout, leaving that fold's *training* half constant.
    y = np.full(40, 7.0)
    y[3] = 9.0
    regressor = TabPFNRegressor(
        n_estimators=1,
        random_state=0,
        device="cpu",
        model_path=_create_dummy_regressor_model_specs(),
    )
    # The tuning regressors take their categorical indices from this attribute,
    # which `fit` sets before it reaches this method.
    regressor.categorical_features_indices_ = None

    folds = regressor._compute_holdout_validation_data(
        X=X, y=y, holdout_frac=0.5, n_folds=2
    )

    # The fold whose training half is constant has a degenerate two-border
    # distribution and no executor, so it is dropped rather than scored.
    assert len(folds) == 1

    # With no variation at all, every fold is dropped and calibration has nothing
    # to work with -- `find_regression_optimal_temperature` falls back gracefully.
    assert (
        regressor._compute_holdout_validation_data(
            X=X, y=np.full(40, 7.0), holdout_frac=0.5, n_folds=2
        )
        == []
    )


def _tuning_regressor_kwargs(**overrides: Any) -> dict[str, Any]:
    """Constructor arguments for a cheap, download-free calibration test."""
    return {
        "n_estimators": 1,
        "random_state": 0,
        "device": "cpu",
        "inference_precision": torch.float32,
        "model_path": _create_dummy_regressor_model_specs(),
        **overrides,
    }


def _make_tuning_sized_regression_data(
    n_samples: int = MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING + 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Regression data big enough that tuning does not warn about the sample count."""
    return sklearn.datasets.make_regression(
        n_samples=n_samples, n_features=4, noise=10.0, random_state=0
    )


@pytest.mark.parametrize(
    ("eval_metric", "expected"),
    [
        (None, RegressorEvalMetrics.NLL),
        ("nll", RegressorEvalMetrics.NLL),
        ("crps", RegressorEvalMetrics.CRPS),
        (RegressorEvalMetrics.CRPS, RegressorEvalMetrics.CRPS),
    ],
)
def test__fit__with_calibrate_temperature__honours_each_eval_metric(
    eval_metric: str | RegressorEvalMetrics | None,
    expected: RegressorEvalMetrics,
) -> None:
    X, y = _make_tuning_sized_regression_data()
    regressor = TabPFNRegressor(
        eval_metric=eval_metric,
        tuning_config={
            "calibrate_temperature": True,
            "tuning_holdout_frac": 0.5,
            "tuning_n_folds": 1,
        },
        **_tuning_regressor_kwargs(),
    )
    regressor.fit(X, y)

    assert regressor.eval_metric_ is expected
    # Every metric drives the same search, so the winner is always a grid point.
    assert regressor.ensemble_softmax_temperature_ in set(get_tuning_temperatures())
    assert regressor.predict(X[:8]).shape == (8,)


def test__fit__with_calibrate_temperature__sets_temperature_and_predicts() -> None:
    X, y = _make_tuning_sized_regression_data()
    regressor = TabPFNRegressor(
        tuning_config={
            "calibrate_temperature": True,
            "tuning_holdout_frac": 0.5,
            "tuning_n_folds": 1,
        },
        **_tuning_regressor_kwargs(),
    )
    regressor.fit(X, y)

    assert regressor.eval_metric_ is RegressorEvalMetrics.NLL
    assert isinstance(regressor.ensemble_softmax_temperature_, float)
    assert np.isfinite(regressor.ensemble_softmax_temperature_)
    # The search only ever returns a point of the shared grid, so a value outside
    # it would mean the fallback fired and no calibration happened.
    assert regressor.ensemble_softmax_temperature_ in set(get_tuning_temperatures())

    # Every output type still has the shape it had before calibration existed: the
    # temperature is applied to the aggregated logits, which every output reads.
    X_test = X[:20]
    n_test = len(X_test)
    for output_type in ("mean", "median", "mode"):
        prediction = regressor.predict(X_test, output_type=output_type)
        assert prediction.shape == (n_test,)
        assert np.isfinite(prediction).all()

    quantiles = [0.1, 0.5, 0.9]
    quantile_predictions = regressor.predict(
        X_test, output_type="quantiles", quantiles=quantiles
    )
    assert len(quantile_predictions) == len(quantiles)
    assert all(q.shape == (n_test,) for q in quantile_predictions)

    main_output = regressor.predict(X_test, output_type="main")
    assert set(main_output) == {"mean", "median", "mode", "quantiles"}

    full_output = regressor.predict(X_test, output_type="full")
    assert full_output["logits"].shape == (
        n_test,
        regressor.raw_space_bardist_.num_bars,
    )
    assert torch.isfinite(full_output["logits"]).all()


def test__fit__without_tuning_config__leaves_the_temperature_a_no_op() -> None:
    X, y = _make_tuning_sized_regression_data(n_samples=100)
    regressor = TabPFNRegressor(**_tuning_regressor_kwargs())
    regressor.fit(X, y)

    assert regressor.ensemble_softmax_temperature_ == 1.0

    # A model pickled before this attribute existed has to predict identically, and
    # the `temperature != 1.0` guard has to make the division vanish rather than
    # apply a floating-point round trip. Both are what keeps an uncalibrated fit
    # bit-identical to one from before calibration was added.
    logits_with_attribute = regressor._compute_aggregated_logits(
        ensure_compatible_predict_input_sklearn(X[:20], regressor)
    )
    del regressor.ensemble_softmax_temperature_
    logits_without_attribute = regressor._compute_aggregated_logits(
        ensure_compatible_predict_input_sklearn(X[:20], regressor)
    )
    assert torch.equal(logits_with_attribute, logits_without_attribute)


def test__fit__constant_target_with_tuning_config__skips_calibration() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = np.full(40, 7.0)

    regressor = TabPFNRegressor(
        tuning_config={
            "calibrate_temperature": True,
            "tuning_holdout_frac": 0.5,
            "tuning_n_folds": 1,
        },
        **_tuning_regressor_kwargs(),
    )
    # No tuning warning is expected: a constant target returns from `fit` before the
    # tuning configuration is ever resolved.
    regressor.fit(X, y)

    # A constant target returns from `fit` before calibration: there is no executor
    # to produce holdout logits and nothing to calibrate. The attribute still has
    # to exist, because every fitted estimator exposes it.
    assert regressor.is_constant_target_
    assert regressor.ensemble_softmax_temperature_ == 1.0
    assert np.all(regressor.predict(X) == 7.0)


def test__fit__tuning_config_without_features_enabled__warns_and_is_ignored() -> None:
    X, y = _make_tuning_sized_regression_data()
    regressor = TabPFNRegressor(
        tuning_config={"calibrate_temperature": False},
        **_tuning_regressor_kwargs(),
    )

    with pytest.warns(
        UserWarning,
        match=r".*no tuning features were enabled.*`calibrate_temperature=True`.*",
    ):
        regressor.fit(X, y)

    assert regressor.ensemble_softmax_temperature_ == 1.0


def test__fit__small_dataset_with_tuning__warns_but_succeeds() -> None:
    X, y = _make_tuning_sized_regression_data(
        n_samples=MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING - 1
    )
    regressor = TabPFNRegressor(
        tuning_config={
            "calibrate_temperature": True,
            "tuning_holdout_frac": 0.5,
            "tuning_n_folds": 1,
        },
        **_tuning_regressor_kwargs(),
    )

    with pytest.warns(
        UserWarning,
        match=r".*We recommend tuning only for datasets with more than.*",
    ):
        regressor.fit(X, y)

    # The warning is advice, not a veto: calibration still runs.
    assert regressor.ensemble_softmax_temperature_ in set(get_tuning_temperatures())
    assert regressor.predict(X[:5]).shape == (5,)


def test__fit__invalid_eval_metric__raises() -> None:
    regressor = TabPFNRegressor(eval_metric="mae", **_tuning_regressor_kwargs())
    X, y = _make_tuning_sized_regression_data(n_samples=40)

    with pytest.raises(ValueError, match=r"Invalid eval_metric: `mae`"):
        regressor.fit(X, y)


@pytest.mark.parametrize(
    ("eval_metric", "tuning_config"),
    [
        (None, None),
        ("nll", {"calibrate_temperature": True}),
        (RegressorEvalMetrics.NLL, RegressorTuningConfig(calibrate_temperature=True)),
    ],
)
def test__eval_metric_and_tuning_config__round_trip_through_sklearn_clone(
    eval_metric: str | RegressorEvalMetrics | None,
    tuning_config: dict | RegressorTuningConfig | None,
) -> None:
    regressor = TabPFNRegressor(eval_metric=eval_metric, tuning_config=tuning_config)

    params = regressor.get_params()
    assert params["eval_metric"] == eval_metric
    assert params["tuning_config"] == tuning_config

    # sklearn requires `__init__` to store its arguments unmodified: `clone` raises a
    # RuntimeError if the constructed object's params are not the very objects it was
    # given. Any normalisation of `eval_metric` or `tuning_config` in `__init__` --
    # which is why both are validated in `fit` instead -- would fail here.
    cloned = sklearn.base.clone(regressor)
    assert cloned.eval_metric == eval_metric
    assert cloned.tuning_config == tuning_config

    round_tripped = TabPFNRegressor().set_params(**params)
    assert round_tripped.eval_metric == eval_metric
    assert round_tripped.tuning_config == tuning_config
