from __future__ import annotations

import io
import itertools
import os
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
from tabpfn.architectures import base
from tabpfn.architectures.base.config import ModelConfig
from tabpfn.base import ClassifierModelSpecs, initialize_tabpfn_model
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
from tabpfn.inference_tuning import (
    MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING,
    ClassifierEvalMetrics,
    ClassifierTuningConfig,
)
from tabpfn.model_loading import ModelSource, prepend_cache_path
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.utils import infer_devices

from .utils import (
    get_pytest_devices,
    is_cpu_float16_supported,
    mark_mps_configs_as_slow,
    patch_layernorm_no_affine,
)

exclude_devices = {
    d.strip() for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",") if d.strip()
}

devices = get_pytest_devices()


@pytest.fixture(scope="module")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    n_classes = 3
    return sklearn.datasets.make_classification(
        n_samples=3 * n_classes,
        n_classes=n_classes,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )


model_sources = [ModelSource.get_classifier_v2(), ModelSource.get_classifier_v2_5()]
fit_modes = ["low_memory", "fit_preprocessors", "fit_with_cache"]


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

    model = TabPFNClassifier(
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

    assert model.predict_proba(X).shape == (X.shape[0], len(np.unique(y)))
    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize("device", [d for d in devices if d == "mps"])
def test__fit_predict__mps_smoke_test__outputs_correct_shape(
    device: str, X_y: tuple[np.ndarray, np.ndarray]
) -> None:
    """Basic test of fit+predict on MPS.

    The other MPS tests in this file are disabled in PRs because they are slow, so this
    test provides some coverage.
    """
    model = TabPFNClassifier(n_estimators=2, device=device, random_state=42)
    X, y = X_y
    model.fit(X, y)
    assert model.predict_proba(X).shape == (X.shape[0], len(np.unique(y)))
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
    model = TabPFNClassifier(
        model_path=prepend_cache_path(model_path),
        n_estimators=1,
        device=device,
        random_state=42,
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    assert model.predict_proba(X).shape == (X.shape[0], len(np.unique(y)))
    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize(
    ("device", "fit_mode"),
    mark_mps_configs_as_slow(itertools.product(devices, fit_modes)),
)
def test__fit_predict__multiple_model_paths__outputs_correct_shape(
    device: str,
    fit_mode: str,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    model = TabPFNClassifier(
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

    assert model.predict_proba(X).shape == (X.shape[0], len(np.unique(y)))
    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize(
    ("device", "feature_shift_decoder", "multiclass_decoder", "remove_outliers_std"),
    mark_mps_configs_as_slow(
        itertools.chain.from_iterable(
            [
                (device, "shuffle", "rotate", None),
                (device, "rotate", "shuffle", 12),
                (device, "shuffle", "rotate", 12),
            ]
            for device in devices
        )
    ),
)
def test__fit_predict__specify_inference_config__outputs_correct_shape(
    device: str,
    feature_shift_decoder: Literal["shuffle", "rotate"],
    multiclass_decoder: Literal["shuffle", "rotate"],
    remove_outliers_std: int | None,
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    model = TabPFNClassifier(
        n_estimators=1,
        device=device,
        inference_config={
            "OUTLIER_REMOVAL_STD": remove_outliers_std,
            "CLASS_SHIFT_METHOD": multiclass_decoder,
            "FEATURE_SHIFT_METHOD": feature_shift_decoder,
        },
        random_state=42,
    )

    X, y = X_y

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    assert model.predict_proba(X).shape == (X.shape[0], len(np.unique(y)))
    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize(
    (
        "device",
        "n_estimators",
        "softmax_temperature",
        "average_before_softmax",
    ),
    mark_mps_configs_as_slow(
        product(
            devices,  # device
            [1, 2],  # n_estimators
            [0.5, 1.0, 1.5],  # softmax_temperature
            [False, True],  # average_before_softmax
        )
    ),
)
def test__predict_logits__output_has_correct_properties_and_consistent_with_proba(
    device: str,
    n_estimators: int,
    softmax_temperature: float,
    average_before_softmax: bool,
) -> None:
    """Test predict_logits() and its consistency with predict_proba().

    Consider configuration permutations that affect the post-processing pipeline.
    """
    X, y = sklearn.datasets.make_classification(
        n_samples=40,
        n_classes=3,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

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


@pytest.mark.parametrize(("n_estimators"), [1, 2])
def test_predict_raw_logits(
    X_y: tuple[np.ndarray, np.ndarray],
    n_estimators: int,
):
    """Tests the predict_raw_logits method."""
    X, y = X_y

    # Ensure y is int64 for consistency with classification tasks
    y = y.astype(np.int64)

    classifier = TabPFNClassifier(
        n_estimators=n_estimators,
        random_state=42,
    )
    classifier.fit(X, y)

    logits = classifier.predict_raw_logits(X)
    assert logits.shape[0] == n_estimators
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (n_estimators, X.shape[0], classifier.n_classes_)
    assert logits.dtype == np.float32
    assert not np.isnan(logits).any()
    assert not np.isinf(logits).any()
    if classifier.n_classes_ > 1:
        assert not np.all(logits == logits[:, 0:1]), (
            "Logits are identical across classes for all samples, indicating "
            "trivial output."
        )


def test_multiple_models_predict_different_logits(X_y: tuple[np.ndarray, np.ndarray]):
    """Tests the predict_raw_logits method."""
    X, y = X_y

    model_a = model_sources[0].filenames[0]
    model_b = model_sources[0].filenames[1]
    two_identical_models = [model_a, model_a]
    two_different_models = [model_a, model_b]

    # Ensure y is int64 for consistency with classification tasks
    y = y.astype(np.int64)

    def get_averaged_logits(model_paths: list[str]) -> np.ndarray:
        classifier = TabPFNClassifier(
            n_estimators=2,
            random_state=42,
            model_path=prepend_cache_path(model_paths),
        )
        classifier.fit(X, y)
        # shape: E=estimators, R=rows, C=columns
        logits_ERC = classifier.predict_raw_logits(X)
        return logits_ERC.mean(axis=0)

    single_model_logits = get_averaged_logits(model_paths=[model_a])
    two_identical_models_logits = get_averaged_logits(model_paths=two_identical_models)
    two_different_models_logits = get_averaged_logits(model_paths=two_different_models)

    assert not np.all(single_model_logits == single_model_logits[:, 0:1]), (
        "Logits are identical across classes for all samples, indicating trivial output"
    )
    assert np.all(single_model_logits == two_identical_models_logits)
    assert not np.all(single_model_logits == two_different_models_logits)


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

    assert np.mean(np.abs(logits_low_temp)) > np.mean(np.abs(logits_high_temp)), (
        "Low softmax temperature did not result in more extreme logits."
    )

    model_temp_one = TabPFNClassifier(
        softmax_temperature=1.0, n_estimators=1, device="cpu", random_state=42
    )
    model_temp_one.fit(X, y)
    logits_temp_one = model_temp_one.predict_logits(X)

    assert not np.allclose(logits_temp_one, logits_low_temp, atol=1e-6), (
        "Logits did not change with low temperature."
    )
    assert not np.allclose(logits_temp_one, logits_high_temp, atol=1e-6), (
        "Logits did not change with high temperature."
    )


def test_balance_probabilities_alters_proba_output() -> None:
    """Verifies that enabling `balance_probabilities` indeed changes the output
    probabilities (assuming non-uniform class counts).
    """
    n_classes = 5
    X_full, _y_full = sklearn.datasets.make_classification(
        n_samples=30 * n_classes,
        n_classes=n_classes,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=0,
    )

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

    assert not np.allclose(proba_no_balance, proba_balance, atol=1e-5), (
        "Probabilities did not change when balance_probabilities was toggled."
    )


@pytest.mark.skip(
    reason="The result is actually different depending on the fitting mode."
)
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
    _auto_devices = infer_devices(devices="auto")
    if any(device.type == "mps" for device in _auto_devices):
        pytest.skip("MPS does not support float64, which is required for this check.")

    if (
        check.func.__name__ == "check_classifiers_train"  # type: ignore
        and _auto_devices[0].type == "cpu"
    ):
        pytest.skip(
            "We currently skip this check on CPU because CPU inference with "
            "float64 is brokwn for datasets with small number of features."
        )

    if check.func.__name__ in (  # type: ignore
        "check_methods_subset_invariance",
        "check_methods_sample_order_invariance",
    ):
        estimator.inference_precision = torch.float64

    check(estimator)


@pytest.mark.skip(reason="This test is flaky and needs to be fixed.")
def test_balanced_probabilities() -> None:
    """Test that balance_probabilities=True works correctly."""
    n_classes = 3
    n_features = 3

    # Create an IMBALANCED dataset
    X, y = sklearn.datasets.make_classification(
        n_samples=60,
        n_classes=n_classes,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        weights=[0.8, 0.1, 0.1],
        random_state=42,
    )

    model_unbalanced = TabPFNClassifier(
        balance_probabilities=False,
        random_state=42,
        n_estimators=2,
    )
    model_unbalanced.fit(X, y)
    proba_unbalanced = model_unbalanced.predict_proba(X)

    model_balanced = TabPFNClassifier(
        balance_probabilities=True,
        random_state=42,
        n_estimators=2,
    )
    model_balanced.fit(X, y)
    proba_balanced = model_balanced.predict_proba(X)

    mean_proba_unbalanced = proba_unbalanced.mean(axis=0)
    mean_proba_balanced = proba_balanced.mean(axis=0)

    # Balanced should be MORE uniform than unbalanced
    balanced_deviation = np.std(mean_proba_balanced)
    unbalanced_deviation = np.std(mean_proba_unbalanced)
    assert balanced_deviation < unbalanced_deviation, (
        "Balancing did not make probabilities more uniform"
    )


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
    assert probabilities.shape == (X.shape[0], len(np.unique(y)))


def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that dict configs behave identically to PreprocessorConfig objects."""
    X, y = X_y

    # Define same config as both dict and object
    dict_config = {
        "name": "quantile_uni_coarse",
        "append_original": False,  # changed from default
        "categorical_name": "ordinal_very_common_categories_shuffled",
        "global_transformer_name": "svd",
        "max_features_per_estimator": 500,
    }

    object_config = PreprocessorConfig(
        name="quantile_uni_coarse",
        append_original=False,  # changed from default
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd",
        max_features_per_estimator=500,
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


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    if os.name == "nt":
        pytest.skip("onnx export is not tested on windows")
    X, y = X_y
    with torch.no_grad():
        classifier = TabPFNClassifier(n_estimators=1, device="cpu", random_state=42)
        # load the model so we can access it via classifier.models_
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
        patch_layernorm_no_affine(classifier.models_[0])

        # From 2.9 PyTorch changed the default export mode from TorchScript to
        # Dynamo. We don't support Dynamo, so disable it. The `dynamo` flag is only
        # available in newer PyTorch versions, hence we don't always include it.
        export_kwargs = {"dynamo": False} if torch.__version__ >= "2.9" else {}
        torch.onnx.export(
            ModelWrapper(classifier.models_[0]).eval(),
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
            **export_kwargs,
        )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_get_embeddings(
    X_y: tuple[np.ndarray, np.ndarray], data_source: Literal["train", "test"]
) -> None:
    """Test that get_embeddings returns valid embeddings for a fitted model."""
    X, y = X_y
    n_estimators = 3

    model = TabPFNClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)

    embeddings = model.get_embeddings(X, data_source)

    # Need to access the model through the executor
    model_instance = next(iter(model.executor_.model_caches[0]._models.values()))
    encoder_shape = next(
        m.out_features
        for m in model_instance.encoder.modules()
        if isinstance(m, nn.Linear)
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == n_estimators
    assert embeddings.shape[1] == X.shape[0]
    assert embeddings.shape[2] == encoder_shape


def test_pandas_output_config(X_y: tuple[np.ndarray, np.ndarray]):
    """Test compatibility with sklearn's output configuration settings."""
    X, y = X_y

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
    models, architecture_configs, norm_criterion, inference_config = (
        initialize_tabpfn_model(
            model_path="auto",
            which="classifier",
            fit_mode="low_memory",
        )
    )
    assert models is not None, "model should be initialized for classifier"
    assert architecture_configs is not None, (
        "config should be initialized for classifier"
    )
    assert norm_criterion is None, "norm_criterion should be None for classifier"
    assert inference_config is not None

    # 2) Test the sklearn-style wrapper on TabPFNClassifier
    classifier = TabPFNClassifier(device="cpu", random_state=42)
    classifier._initialize_model_variables()

    assert hasattr(classifier, "models_")
    assert classifier.models_ is not None

    assert hasattr(classifier, "configs_")
    assert classifier.configs_ is not None

    assert not hasattr(classifier, "znorm_space_bardist_")

    # 3) Reuse via ClassifierModelSpecs
    spec = ClassifierModelSpecs(
        model=classifier.models_[0],
        architecture_config=classifier.configs_[0],
        inference_config=classifier.inference_config_,
    )

    classifier2 = TabPFNClassifier(model_path=spec)
    classifier2._initialize_model_variables()

    assert hasattr(classifier2, "models_")
    assert classifier2.models_ is not None

    assert hasattr(classifier2, "configs_")
    assert classifier2.configs_ is not None

    assert not hasattr(classifier2, "znorm_space_bardist_")


@pytest.mark.parametrize("n_features", [1, 2])
def test__TabPFNClassifier__few_features__works(n_features: int) -> None:
    """Test that TabPFNClassifier works correctly with 1 or 2 features."""
    n_classes = 2
    n_samples = 20 * n_classes

    X, y = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    model = TabPFNClassifier(
        n_estimators=2,
        random_state=42,
    )

    returned_model = model.fit(X, y)
    assert returned_model is model, "Returned model is not the same as the model"
    check_is_fitted(returned_model)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (
        X.shape[0],
        n_classes,
    ), f"Probabilities shape is incorrect for {n_features} features"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities do not sum to 1"

    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],), (
        f"Predictions shape is incorrect for {n_features} features"
    )
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.3, f"Accuracy too low with {n_features} features: {accuracy}"


@pytest.mark.parametrize(
    (
        "eval_metric",
        "tuning_holdout_pct",
        "tuning_holdout_n_splits",
        "tune_decision_thresholds",
        "calibrate_temperature",
        "expected_equal",
    ),
    [
        (ClassifierEvalMetrics.F1, 0.1, 1, False, True, False),
        (ClassifierEvalMetrics.ACCURACY, 0.2, 1, False, False, True),
        (ClassifierEvalMetrics.ACCURACY, 0.7, 1, True, False, False),
        (ClassifierEvalMetrics.F1, 0.05, 2, True, False, False),
        (ClassifierEvalMetrics.F1, 0.2, 1, False, True, False),
        (ClassifierEvalMetrics.BALANCED_ACCURACY, 0.1, 1, False, False, True),
    ],
)
def test__fit_with_tuning_config__works_with_different_eval_metrics(
    eval_metric: ClassifierEvalMetrics,
    tuning_holdout_pct: float,
    tuning_holdout_n_splits: int,
    tune_decision_thresholds: bool,
    calibrate_temperature: bool,
    expected_equal: bool,
) -> None:
    X, y = sklearn.datasets.make_classification(
        n_samples=MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING + 1,
        n_classes=2,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    max_num_classes = len(np.unique(y))

    if eval_metric is ClassifierEvalMetrics.ACCURACY:
        tuning_config = ClassifierTuningConfig(
            calibrate_temperature=calibrate_temperature,
            tune_decision_thresholds=tune_decision_thresholds,
            tuning_holdout_frac=tuning_holdout_pct,
            tuning_n_folds=tuning_holdout_n_splits,
        )
    else:
        # Also check parsing tuning config as dict.
        tuning_config = {
            "calibrate_temperature": calibrate_temperature,
            "tune_decision_thresholds": tune_decision_thresholds,
            "tuning_holdout_frac": tuning_holdout_pct,
            "tuning_n_folds": tuning_holdout_n_splits,
        }

    kwargs = {
        "fit_mode": "fit_preprocessors",
        "eval_metric": eval_metric,
        "n_estimators": 1,
        "device": "cpu",
        "inference_precision": torch.float32,
        "random_state": 0,
        "model_path": _create_dummy_classifier_model_specs(
            max_num_classes=max_num_classes
        ),
    }

    torch.random.manual_seed(0)
    tabpfn_with_tuning = TabPFNClassifier(
        tuning_config=tuning_config,
        **kwargs,
    )
    tabpfn_with_tuning.fit(X, y)
    preds_with_tuning = tabpfn_with_tuning.predict_proba(X[0 : X.shape[0] // 4])

    assert len(preds_with_tuning) == X.shape[0] // 4

    torch.random.manual_seed(0)
    tabpfn_no_tuning = TabPFNClassifier(**kwargs)
    tabpfn_no_tuning.fit(X, y)
    preds_no_tuning = tabpfn_no_tuning.predict_proba(X[0 : X.shape[0] // 4])

    assert np.allclose(preds_with_tuning, preds_no_tuning, atol=1e-5) == expected_equal

    if calibrate_temperature:
        assert (
            tabpfn_with_tuning.softmax_temperature_
            != tabpfn_no_tuning.softmax_temperature_
        )
    else:
        assert (
            tabpfn_with_tuning.softmax_temperature_
            == tabpfn_no_tuning.softmax_temperature_
            == tabpfn_with_tuning.softmax_temperature
        )


def test__logits_to_probabilities__same_as_predict_proba(
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = X_y
    max_num_classes = len(np.unique(y))

    model = TabPFNClassifier(
        n_estimators=1,
        random_state=42,
        model_path=_create_dummy_classifier_model_specs(
            max_num_classes=max_num_classes
        ),
    )
    model.fit(X, y)

    raw_logits = model.predict_raw_logits(X)
    probas = model.logits_to_probabilities(raw_logits)

    expected_probas = model.predict_proba(X)
    assert np.allclose(probas, expected_probas, atol=1e-4, rtol=1e-3)


def test__fit_with_f1_metric_without_tuning_config__warns(
    X_y: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that warning is issued when F1 metric used without tuning config."""
    X, y = X_y

    clf = TabPFNClassifier(
        eval_metric="f1",
        tuning_config=None,
        n_estimators=1,
        model_path=_create_dummy_classifier_model_specs(
            max_num_classes=len(np.unique(y))
        ),
    )

    with pytest.warns(
        UserWarning,
        match=r".*haven't specified any tuning configuration.*",
    ):
        clf.fit(X, y)


def test__fit_with_small_dataset_and_tuning__warns() -> None:
    default_rng = np.random.default_rng(seed=42)
    X = default_rng.random((MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING - 1, 10))
    y = default_rng.integers(0, 2, MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING - 1)

    clf = TabPFNClassifier(
        eval_metric="f1",
        tuning_config={
            "tune_decision_thresholds": True,
        },
        n_estimators=1,
        model_path=_create_dummy_classifier_model_specs(
            max_num_classes=len(np.unique(y))
        ),
    )

    with pytest.warns(
        UserWarning,
        match=r".*We recommend tuning only for datasets with more than.*",
    ):
        clf.fit(X, y)


def test__fit_with_roc_auc_metric_with_threshold_tuning__warns() -> None:
    """Test that warning is issued when ROC AUC metric used with threshold tuning."""
    n_classes = 2
    X, y = sklearn.datasets.make_classification(
        n_samples=30 * n_classes,
        n_classes=n_classes,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )

    clf = TabPFNClassifier(
        eval_metric="roc_auc",
        tuning_config={
            "tune_decision_thresholds": True,
            "calibrate_temperature": False,
            "tuning_holdout_frac": 0.1,
            "tuning_n_folds": 1,
        },
        n_estimators=1,
        device="cpu",
        model_path=_create_dummy_classifier_model_specs(
            max_num_classes=len(np.unique(y))
        ),
        random_state=0,
    )

    with pytest.warns(
        UserWarning,
        match=(
            r".*with threshold tuning or temperature calibration "
            r"enabled.*is independent of these tunings.*"
        ),
    ):
        clf.fit(X, y)


def _create_dummy_classifier_model_specs(
    max_num_classes: int = 10,
) -> ClassifierModelSpecs:
    minimal_config = ModelConfig(
        emsize=8,
        features_per_group=1,
        max_num_classes=max_num_classes,
        nhead=2,
        nlayers=2,
        remove_duplicate_features=True,
        num_buckets=100,
    )
    model = base.get_architecture(
        config=minimal_config,
        n_out=max_num_classes,
        cache_trainset_representation=False,
    )
    inference_config = InferenceConfig.get_default(
        task_type="multiclass",
        model_version=ModelVersion.V2_5,
    )
    return ClassifierModelSpecs(
        model=model,
        architecture_config=minimal_config,
        inference_config=inference_config,
    )


def test__create_default_for_version__v2__uses_correct_defaults() -> None:
    estimator = TabPFNClassifier.create_default_for_version(ModelVersion.V2)

    assert isinstance(estimator, TabPFNClassifier)
    assert estimator.n_estimators == 8
    assert estimator.softmax_temperature == 0.9
    assert isinstance(estimator.model_path, str)
    assert "classifier" in estimator.model_path
    assert "-v2-" in estimator.model_path


def test__create_default_for_version__v2_5__uses_correct_defaults() -> None:
    estimator = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)

    assert isinstance(estimator, TabPFNClassifier)
    assert estimator.n_estimators == 8
    assert estimator.softmax_temperature == 0.9
    assert isinstance(estimator.model_path, str)
    assert "classifier" in estimator.model_path
    assert "-v2.5-" in estimator.model_path


def test__create_default_for_version__passes_through_overrides() -> None:
    estimator = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, n_estimators=16
    )

    assert estimator.n_estimators == 16
    assert estimator.softmax_temperature == 0.9
