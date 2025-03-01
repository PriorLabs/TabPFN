from __future__ import annotations

import io
import os
from itertools import product
from typing import Callable, Literal

import numpy as np
import pytest
import sklearn.datasets
import torch
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

all_combinations = list(
    product(
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
    return X, y  # type: ignore


@pytest.mark.parametrize(
    (
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
    [TabPFNClassifier(inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True})],
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

    embeddings = model.get_embeddings(X, data_source)

    encoder_shape = next(
        m.out_features
        for m in model.executor_.model.encoder.modules()
        if isinstance(m, nn.Linear)
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == n_estimators
    assert embeddings.shape[1] == X.shape[0]
    assert embeddings.shape[2] == encoder_shape


def test_classifier_with_pandas_dataframes() -> None:
    """Test that TabPFNClassifier works with pandas DataFrames and numpy arrays.

    This test verifies that TabPFN handles both pandas and numpy data types
    correctly and consistently.
    """
    import numpy as np
    import pandas as pd

    # Create synthetic classification data
    rng = np.random.default_rng(42)
    X_np = rng.random((100, 10))
    y_np = rng.choice([0, 1], size=100)

    # Convert to pandas for testing
    X_pd = pd.DataFrame(X_np, columns=[f"feature_{i}" for i in range(10)])
    y_pd = pd.Series(y_np, name="target")

    # Test 1: Train with pandas, predict with pandas
    model1 = TabPFNClassifier(n_estimators=2, random_state=42)
    model1.fit(X_pd, y_pd)
    pd_pred1 = model1.predict(X_pd)
    pd_proba1 = model1.predict_proba(X_pd)

    # Test 2: Train with pandas, predict with numpy
    np_pred1 = model1.predict(X_np)
    np_proba1 = model1.predict_proba(X_np)

    # Test 3: Train with numpy, predict with numpy
    model2 = TabPFNClassifier(n_estimators=2, random_state=42)
    model2.fit(X_np, y_np)
    np_pred2 = model2.predict(X_np)
    np_proba2 = model2.predict_proba(X_np)

    # Test 4: Train with numpy, predict with pandas
    pd_pred2 = model2.predict(X_pd)
    pd_proba2 = model2.predict_proba(X_pd)

    # Check that predictions have the expected shape
    assert pd_pred1.shape == (X_pd.shape[0],)
    assert np_pred1.shape == (X_np.shape[0],)
    assert np_pred2.shape == (X_np.shape[0],)
    assert pd_pred2.shape == (X_pd.shape[0],)

    # Check that predictions are the same regardless of input type
    np.testing.assert_array_equal(pd_pred1, np_pred1)
    np.testing.assert_array_equal(pd_pred2, np_pred2)

    # Check that probabilities are the same regardless of input type
    np.testing.assert_array_almost_equal(pd_proba1, np_proba1)
    np.testing.assert_array_almost_equal(pd_proba2, np_proba2)

    # Test with pipeline including a StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", TabPFNClassifier(n_estimators=2, random_state=42)),
        ]
    )

    # Test pipeline with pandas input
    pipe.fit(X_pd, y_pd)
    pipe_pd_pred = pipe.predict(X_pd)

    # Test pipeline with numpy input
    pipe.fit(X_np, y_np)
    pipe_np_pred = pipe.predict(X_np)

    # Check that predictions have the expected shape
    assert pipe_pd_pred.shape == (X_pd.shape[0],)
    assert pipe_np_pred.shape == (X_np.shape[0],)
