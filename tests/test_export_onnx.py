from __future__ import annotations

import os
import sys
from typing import Literal

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor


@pytest.fixture(autouse=True, scope="module")
def check_onnx_compatible():
    if os.name == "nt":
        pytest.skip("ONNX export is not tested on Windows")
    if sys.version_info >= (3, 13):
        pytest.xfail("ONNX is not yet supported on Python 3.13")
    if sys.version_info < (3, 10):
        pytest.skip("our onnx export doesn't work on python 3.9")
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("ONNX or ONNX Runtime not available")


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_missing_model_error():
    """Test that appropriate error is raised when trying to
    use ONNX with a missing model. Here we specify a model path
    that does not exist to simulate the case where the model
    has not been compiled.
    """
    # Generate synthetic data
    rng = np.random.default_rng()
    X = rng.standard_normal((50, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=50)

    # Try to use ONNX backend when model doesn't exist
    classifier = TabPFNClassifier(
        device="cpu", use_onnx=True, model_path="/fake_dir/tabpfn_classifier_v2.ckpt"
    )

    # Expect a FileNotFoundError with a specific message
    with pytest.raises(
        FileNotFoundError,
        match=(
            r"ONNX model not found at:.*please compile the model by "
            r"running.*compile_onnx_models\(\)"
        ),
    ):
        classifier.fit(X, y)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_export_and_inference():
    """Test that TabPFN models can be exported to ONNX
    and produce correct predictions.
    """
    from tabpfn.misc.compile_to_onnx import compile_onnx_models

    # Compile the model to ONNX format (using default output directory)
    compile_onnx_models(skip_test=True)

    # Generate synthetic data for testing
    n_samples = 100
    n_features = 10
    rng = np.random.default_rng()
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples)

    # Split into train/test
    train_size = 80
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    # Test with PyTorch backend
    classifier_torch = TabPFNClassifier(device="cpu", use_onnx=False)
    classifier_torch.fit(X_train, y_train)

    # Get predictions with PyTorch backend
    torch_probs = classifier_torch.predict_proba(X_test)
    torch_preds = classifier_torch.predict(X_test)

    # Test with ONNX backend
    classifier_onnx = TabPFNClassifier(device="cpu", use_onnx=True)
    classifier_onnx.fit(X_train, y_train)

    # Get predictions with ONNX backend
    onnx_probs = classifier_onnx.predict_proba(X_test)
    onnx_preds = classifier_onnx.predict(X_test)

    # Check that the predictions roughly match
    np.testing.assert_allclose(torch_probs, onnx_probs, rtol=1e-2, atol=1e-2)

    # same for regressor
    regressor_torch = TabPFNRegressor(device="cpu", use_onnx=False)
    regressor_torch.fit(X_train, y_train)
    regressor_onnx = TabPFNRegressor(device="cpu", use_onnx=True)
    regressor_onnx.fit(X_train, y_train)

    torch_preds = regressor_torch.predict(X_test)
    onnx_preds = regressor_onnx.predict(X_test)

    # Check that the predictions roughly match
    np.testing.assert_allclose(torch_preds, onnx_preds, rtol=1e-2, atol=1e-2)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.parametrize("which", ["classifier", "regressor"])
def test_onnx_session_reuse(which: Literal["classifier", "regressor"]):
    """Test that the ONNX session is reused when fitting a model multiple times
    with the same model path and device.
    """
    # Generate synthetic data
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((50, 10)).astype(np.float32)
    y1 = rng.integers(0, 2, size=50)

    X2 = rng.standard_normal((40, 10)).astype(np.float32)
    y2 = rng.integers(0, 2, size=40)

    # Create a classifier with ONNX backend
    if which == "classifier":
        sklearn_model = TabPFNClassifier(device="cpu", use_onnx=True)
    else:
        sklearn_model = TabPFNRegressor(device="cpu", use_onnx=True)

    # First fit
    sklearn_model.fit(X1, y1)

    # Get reference to the first model
    first_model = sklearn_model.model_

    # Mock print function to check if message is displayed
    import builtins

    original_print = builtins.print
    printed_messages = []

    def mock_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        printed_messages.append(message)
        original_print(*args, **kwargs)

    # Replace print with our mock
    builtins.print = mock_print

    try:
        # Second fit with same configuration
        sklearn_model.fit(X2, y2)

        # Assert that the model object is the same (session reused)
        assert sklearn_model.model_ is first_model

        # Check that the print message appears
        assert any(
            "Using same ONNX session as last fit call" in msg
            for msg in printed_messages
        )

        # Now test with a different device to force new session
        if torch.cuda.is_available():
            # Change device to force new session
            sklearn_model.device = "cuda"
            sklearn_model.fit(X1, y1)

            # Should be a different model object now
            assert sklearn_model.model_ is not first_model

            # Restore device
            sklearn_model.device = "cpu"
            sklearn_model.fit(X1, y1)

            # Should be a new model again
            assert sklearn_model.model_ is not first_model
    finally:
        # Restore original print function
        builtins.print = original_print


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.parametrize("which", ["classifier", "regressor"])
def test_onnx_deterministic(which: Literal["classifier", "regressor"]):
    """Test that TabPFN models using ONNX are deterministic when using the same seed."""
    from tabpfn.misc.compile_to_onnx import compile_onnx_models

    # Compile the model to ONNX format if needed
    compile_onnx_models(skip_test=True)

    # Generate synthetic data
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((50, 10)).astype(np.float32)

    if which == "classifier":
        y_train = rng.integers(0, 3, size=50)  # 3 classes
        X_test = rng.standard_normal((20, 10)).astype(np.float32)

        # First model with fixed seed
        model1 = TabPFNClassifier(device="cpu", use_onnx=True, random_state=123)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)
        proba1 = model1.predict_proba(X_test)

        # Second model with same seed
        model2 = TabPFNClassifier(device="cpu", use_onnx=True, random_state=123)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)
        proba2 = model2.predict_proba(X_test)

        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(proba1, proba2)

        # Third model with different seed
        model3 = TabPFNClassifier(device="cpu", use_onnx=True, random_state=456)
        model3.fit(X_train, y_train)
        pred3 = model3.predict(X_test)
        proba3 = model3.predict_proba(X_test)

        # Predictions should be different (with high probability)
        # We use assert_raises to verify they're different
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(proba1, proba3)

    else:  # regressor
        y_train = rng.standard_normal(50)
        X_test = rng.standard_normal((20, 10)).astype(np.float32)

        # First model with fixed seed
        model1 = TabPFNRegressor(device="cpu", use_onnx=True, random_state=123)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)

        # Second model with same seed
        model2 = TabPFNRegressor(device="cpu", use_onnx=True, random_state=123)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)

        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)

        # Third model with different seed
        model3 = TabPFNRegressor(device="cpu", use_onnx=True, random_state=456)
        model3.fit(X_train, y_train)
        pred3 = model3.predict(X_test)

        # Predictions should be different (with high probability)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(pred1, pred3)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.parametrize("model_class", [TabPFNClassifier, TabPFNRegressor])
def test_cuda_provider_missing_error(model_class):
    """Test that TabPFN models raise the correct error when trying to use CUDA
    without CUDAExecutionProvider available in ONNX Runtime.
    """
    import onnxruntime as ort

    # Generate synthetic data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5)).astype(np.float32)
    y = (
        rng.integers(0, 2, size=20)
        if model_class == TabPFNClassifier
        else rng.standard_normal(20)
    )

    # Mock ort.get_available_providers to return only CPUExecutionProvider
    original_get_providers = ort.get_available_providers

    try:
        # Replace providers with only CPU
        ort.get_available_providers = lambda: ["CPUExecutionProvider"]

        # Create model with CUDA device and ONNX enabled
        model = model_class(device="cuda", use_onnx=True)

        # The error should be raised during fit
        with pytest.raises(
            ValueError,
            match="Device is cuda but CUDAExecutionProvider is not available in ONNX",
        ):
            model.fit(X, y)
    finally:
        # Restore original function
        ort.get_available_providers = original_get_providers
