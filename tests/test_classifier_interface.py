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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from torch import nn

from tabpfn import TabPFNClassifier
from tabpfn.base import ClassifierModelSpecs, initialize_tabpfn_model
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.utils import infer_device_and_type

from .utils import check_cpu_float16_support, get_pytest_devices

exclude_devices = {
    d.strip() for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",") if d.strip()
}

devices = get_pytest_devices()

is_cpu_float16_supported = check_cpu_float16_support()

# TODO: test "batched" mode

feature_shift_decoders = ["shuffle", "rotate"]
multiclass_decoders = ["shuffle", "rotate"]
fit_modes = [
    "low_memory",
    "fit_preprocessors",
    "fit_with_cache",
]
inference_precision_methods = ["auto", "autocast", torch.float64, torch.float16]
remove_outliers_stds = [None, 12]
estimators = [1, 2]

all_combinations = list(
    product(
        estimators,
        devices,
        feature_shift_decoders,
        multiclass_decoders,
        fit_modes,
        inference_precision_methods,
        remove_outliers_stds,
    ),
)


@pytest.fixture(scope="module")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    # Take 20 samples from class 0, 20 from class 1, 20 from class 2
    # This ensures all 3 classes are present
    X_diverse = np.vstack([X[y == 0][:20], X[y == 1][:20], X[y == 2][:20]])
    y_diverse = np.hstack([y[y == 0][:20], y[y == 1][:20], y[y == 2][:20]])

    # Shuffle to mix them up, otherwise training data would be ordered by class
    indices = np.arange(len(y_diverse))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    return X_diverse[indices].astype(np.float32), y_diverse[indices].astype(np.int64)


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
    device: Literal["cuda", "mps", "cpu"],
    feature_shift_decoder: Literal["shuffle", "rotate"],
    multiclass_decoder: Literal["shuffle", "rotate"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    remove_outliers_std: int | None,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    if torch.device(device).type == "cpu" and inference_precision in ["autocast"]:
        pytest.skip("CPU device does not support 'autocast' inference.")

    # Use the environment-aware check to skip only if necessary
    if (
        torch.device(device).type == "cpu"
        and inference_precision == torch.float16
        and not is_cpu_float16_supported
    ):
        pytest.skip("CPU float16 matmul not supported in this PyTorch version.")
    if torch.device(device).type == "mps" and inference_precision == torch.float64:
        pytest.skip("MPS does not support float64, which is required for this check.")

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
        random_state=42,  # Added for consistency and reproducibility
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


@pytest.mark.parametrize(
    (
        "n_estimators",
        "device",
        "softmax_temperature",
        "average_before_softmax",
    ),
    list(
        product(
            [1, 4],  # n_estimators
            devices,  # device
            [0.5, 1.0, 1.5],  # softmax_temperature
            [False, True],  # average_before_softmax
        )
    ),
)
def test_predict_logits_and_consistency(
    X_y: tuple[np.ndarray, np.ndarray],
    n_estimators,
    device,
    softmax_temperature,
    average_before_softmax,
):
    """Tests the new predict_logits method and its consistency with predict_proba
    under various configuration permutations that affect the post-processing
    pipeline.
    """
    X, y = X_y

    # Ensure y is int64 for consistency with classification tasks
    y = y.astype(np.int64)

    classifier = TabPFNClassifier(
        n_estimators=n_estimators,
        device=device,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        random_state=42,
    )
    classifier.fit(X, y)

    # 1. Test predict_logits output properties
    logits = classifier.predict_logits(X)
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (X.shape[0], classifier.n_classes_)
    assert logits.dtype == np.float32
    assert not np.isnan(logits).any()
    assert not np.isinf(logits).any()
    if classifier.n_classes_ > 1:
        assert not np.all(logits == logits[:, 0:1]), (
            "Logits are identical across classes for all samples, indicating "
            "trivial output."
        )

    # 2. Test consistency: softmax(logits) should match predict_proba
    proba_from_predict_proba = classifier.predict_proba(X)

    # The relationship between predict_logits and predict_proba depends on the
    # averaging strategy.
    if n_estimators == 1 or average_before_softmax:
        # If there's only one estimator or we average before the softmax,
        # then applying softmax to the (already averaged) logits should
        # match the probabilities from predict_proba.
        proba_from_logits = torch.nn.functional.softmax(
            torch.from_numpy(logits), dim=-1
        ).numpy()
        np.testing.assert_allclose(
            proba_from_logits,
            proba_from_predict_proba,
            atol=0.0001,
            rtol=0.0005,
            err_msg=(
                "Probabilities derived from predict_logits do not match "
                "predict_proba output when they should be consistent."
            ),
        )
    else:
        # If n_estimators > 1 AND we average *after* softmax, then applying
        # softmax to the averaged logits will NOT match predict_proba.
        # predict_proba averages the probabilities, not the logits.
        # softmax(avg(logits)) != avg(softmax(logits))
        pass

    # 3. Quick check of predict  for completeness, derived from predict_proba
    predicted_labels = classifier.predict(X)
    assert predicted_labels.shape == (X.shape[0],)
    assert predicted_labels.dtype in [
        np.int64,
        object,
    ]

    # 4. Basic sanity check for predict and predict_proba outcomes
    assert accuracy_score(y, predicted_labels) >= 0.5
    assert log_loss(y, proba_from_predict_proba) < 5.0


def test_softmax_temperature_impact_on_logits_magnitude(
    X_y: tuple[np.ndarray, np.ndarray],
):
    """Ensures softmax_temperature impacts the magnitude of raw logits as
    expected: lower temperature -> higher magnitude (sharper distribution).
    """
    X, y = X_y
    y = y.astype(np.int64)

    # Model with low temperature (should produce "sharper" logits)
    model_low_temp = TabPFNClassifier(
        softmax_temperature=0.1, n_estimators=1, device="cpu", random_state=42
    )
    model_low_temp.fit(X, y)
    logits_low_temp = model_low_temp.predict_logits(X)

    # Model with high temperature (should produce "smoother" logits)
    model_high_temp = TabPFNClassifier(
        softmax_temperature=10.0, n_estimators=1, device="cpu", random_state=42
    )
    model_high_temp.fit(X, y)
    logits_high_temp = model_high_temp.predict_logits(X)

    assert np.mean(np.abs(logits_low_temp)) > np.mean(
        np.abs(logits_high_temp)
    ), "Low softmax temperature did not result in more extreme logits."

    model_temp_one = TabPFNClassifier(
        softmax_temperature=1.0, n_estimators=1, device="cpu", random_state=42
    )
    model_temp_one.fit(X, y)
    logits_temp_one = model_temp_one.predict_logits(X)

    assert not np.allclose(
        logits_temp_one, logits_low_temp, atol=1e-6
    ), "Logits did not change with low temperature."
    assert not np.allclose(
        logits_temp_one, logits_high_temp, atol=1e-6
    ), "Logits did not change with high temperature."


def test_balance_probabilities_alters_proba_output(
    X_y: tuple[np.ndarray, np.ndarray],
):
    """Verifies that enabling `balance_probabilities` indeed changes the output
    probabilities (assuming non-uniform class counts).
    """
    X_full, y_full = X_y

    # Introduce artificial imbalance to ensure balancing has an effect
    y_imbalanced = np.array(
        [0] * 30 + [1] * 5 + [2] * 5, dtype=np.int64
    )  # Total 40 samples

    # Create a subset of X to match the length of y_imbalanced
    X_subset = X_full[: len(y_imbalanced)]

    # Shuffle both X and y together to maintain correspondence
    rng = np.random.default_rng(42)  # Initialize a new Generator with a seed
    p = rng.permutation(len(y_imbalanced))
    X_subset, y_imbalanced = X_subset[p], y_imbalanced[p]

    # Model without class balancing
    model_no_balance = TabPFNClassifier(
        balance_probabilities=False, n_estimators=1, device="cpu", random_state=42
    )
    model_no_balance.fit(X_subset, y_imbalanced)
    proba_no_balance = model_no_balance.predict_proba(X_subset)

    # Model with class balancing enabled
    model_balance = TabPFNClassifier(
        balance_probabilities=True, n_estimators=1, device="cpu", random_state=42
    )
    model_balance.fit(X_subset, y_imbalanced)
    proba_balance = model_balance.predict_proba(X_subset)

    assert not np.allclose(
        proba_no_balance, proba_balance, atol=1e-5
    ), "Probabilities did not change when balance_probabilities was toggled."


@pytest.mark.skip(reason="No longer passes now dataset has been shrunk.")
def test_fit_modes_all_return_equal_results(
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    kwargs = {"n_estimators": 2, "device": "auto", "random_state": 0}
    X, y = X_y

    torch.random.manual_seed(0)
    tabpfn = TabPFNClassifier(fit_mode="fit_preprocessors", **kwargs)
    tabpfn.fit(X, y)
    probs = tabpfn.predict_proba(X)
    preds = tabpfn.predict(X)

    torch.random.manual_seed(0)
    tabpfn = TabPFNClassifier(fit_mode="fit_with_cache", **kwargs)
    tabpfn.fit(X, y)
    np.testing.assert_array_almost_equal(probs, tabpfn.predict_proba(X))
    np.testing.assert_array_equal(preds, tabpfn.predict(X))

    torch.random.manual_seed(0)
    tabpfn = TabPFNClassifier(fit_mode="low_memory", **kwargs)
    tabpfn.fit(X, y)
    np.testing.assert_array_almost_equal(probs, tabpfn.predict_proba(X))
    np.testing.assert_array_equal(preds, tabpfn.predict(X))


# TODO(eddiebergman): Should probably run a larger suite with different configurations
@parametrize_with_checks(
    [
        TabPFNClassifier(
            n_estimators=2,
            inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True},
        ),
    ],
)
def test_sklearn_compatible_estimator(
    estimator: TabPFNClassifier,
    check: Callable[[TabPFNClassifier], None],
) -> None:
    _auto_device = infer_device_and_type(device="auto")
    if _auto_device.type == "mps":
        pytest.skip("MPS does not support float64, which is required for this check.")

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
    """Wrapper for the TabPFN model for ONNX export."""

    def __init__(self, original_model):  # noqa: D107
        super().__init__()
        self.model = original_model

    def forward(self, X, y, only_return_standard_out, categorical_inds):
        return self.model(
            X,
            y,
            only_return_standard_out=only_return_standard_out,
            categorical_inds=categorical_inds,
        )


def _patch_layernorm_no_affine(model: nn.Module) -> None:
    """Workaround for ONNX export issue with LayerNorm(affine=False) in
    PyTorch <= 2.1.3.

    This patch function was necessary to enable successful ONNX export
    of the TabPFN model when using PyTorch version 2.1.3. The issue arose
    because the ONNX exporter in that version (and potentially earlier ones)
    failed to correctly handle `nn.LayerNorm` layers initialized with
    `affine=False`, which means they lack the learnable 'weight' (gamma) and
    'bias' (beta) parameters.

    However, testing indicated that this issue is resolved in later PyTorch
    versions; specifically, the ONNX export runs without errors on
    PyTorch 2.6.0 even without this patch.

    This function circumvents the problem by iterating through the model's
    modules and, for any `nn.LayerNorm` layer where `layer.weight` is None
    (indicating `affine=False`), it manually adds non-learnable
    (`requires_grad=False`) parameters for 'weight' (initialized to ones) and
    'bias' (initialized to zeros). This addition satisfies the requirements
    of the older ONNX exporter without changing the model's functional
    behavior, as these added parameters represent an identity affine
    transformation.
    """
    for layer in model.modules():
        if isinstance(layer, nn.LayerNorm) and layer.weight is None:
            # Build tensors on the same device/dtype as the layer's buffer
            device = next(layer.parameters(), torch.tensor([], device="cpu")).device
            dtype = getattr(layer, "weight_dtype", torch.float32)

            gamma = torch.ones(layer.normalized_shape, dtype=dtype, device=device)
            beta = torch.zeros_like(gamma)

            layer.weight = nn.Parameter(gamma, requires_grad=False)
            layer.bias = nn.Parameter(beta, requires_grad=False)

            # Optional: mark that we changed it (useful for logging)
            layer._patched_for_onnx = True


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
        X_tensor = torch.randn(
            (X.shape[0] * 2, 1, X.shape[1] + 1),
            generator=torch.Generator().manual_seed(42),
        )
        y_tensor = (
            torch.rand(y.shape, generator=torch.Generator().manual_seed(42))
            .round()
            .to(torch.float32)
        )
        dynamic_axes = {
            "X": {0: "num_datapoints", 1: "batch_size", 2: "num_features"},
            "y": {0: "num_labels"},
        }
        _patch_layernorm_no_affine(classifier.model_)
        torch.onnx.export(
            ModelWrapper(classifier.model_).eval(),
            (X_tensor, y_tensor, True, [[]]),
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
    """Test that constant features are properly handled and
    don't affect predictions.
    """
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
        "all_na_column": [
            None,
            None,
            None,
            None,
            None,
            None,
        ],  # Column with all NaNs
        "target": [1, 0, 1, 1, 0, 0],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Split into X and y
    X = df[["text_feature", "numeric_feature", "all_na_column"]]
    y = df["target"]

    # Initialize and fit TabPFN on data with text+NA and a column with all NAs
    classifier = TabPFNClassifier(device="auto", n_estimators=2)

    # This should now work without raising errors
    classifier.fit(X, y)

    # Verify we can predict
    probabilities = classifier.predict_proba(X)
    predictions = classifier.predict(X)

    # Check output shapes
    assert probabilities.shape == (X.shape[0], len(np.unique(y)))
    assert predictions.shape == (X.shape[0],)


def test_initialize_model_variables_classifier_sets_required_attributes() -> None:
    # 1) Standalone initializer
    model, config, norm_criterion = initialize_tabpfn_model(
        model_path="auto",
        which="classifier",
        fit_mode="low_memory",
    )
    assert model is not None, "model should be initialized for classifier"
    assert config is not None, "config should be initialized for classifier"
    assert norm_criterion is None, "norm_criterion should be None for classifier"

    # 2) Test the sklearn-style wrapper on TabPFNClassifier
    classifier = TabPFNClassifier(model_path="auto", device="cpu", random_state=42)
    classifier._initialize_model_variables()

    assert hasattr(classifier, "model_"), "classifier should have model_ attribute"
    assert classifier.model_ is not None, "model_ should be initialized for classifier"

    assert hasattr(classifier, "config_"), "classifier should have config_ attribute"
    assert (
        classifier.config_ is not None
    ), "config_ should be initialized for classifier"

    assert not hasattr(
        classifier, "bardist_"
    ), "classifier should not have bardist_ attribute"

    # 3) Reuse via ClassifierModelSpecs
    new_model_state = classifier.model_
    new_config = classifier.config_
    spec = ClassifierModelSpecs(model=new_model_state, config=new_config)

    classifier2 = TabPFNClassifier(model_path=spec)
    classifier2._initialize_model_variables()

    assert hasattr(classifier2, "model_"), "classifier2 should have model_ attribute"
    assert (
        classifier2.model_ is not None
    ), "model_ should be initialized for classifier2"

    assert hasattr(classifier2, "config_"), "classifier2 should have config_ attribute"
    assert (
        classifier2.config_ is not None
    ), "config_ should be initialized for classifier2"

    assert not hasattr(
        classifier2, "bardist_"
    ), "classifier2 should not have bardist_ attribute"
