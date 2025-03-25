# ruff: noqa: T201
"""Module providing wrappers to use ONNX models with a PyTorch-like interface.

This module defines wrappers for ONNX models as well as helper functions to export
and validate ONNX models derived from TabPFN models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import sklearn.datasets
import torch
from torch import nn

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.model.loading import resolve_model_path


def _check_cuda_provider(device: torch.device) -> None:
    if (
        device.type == "cuda"
        and "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        raise ValueError(
            "Device is cuda but CUDAExecutionProvider is not available in ONNX. "
            "Check that you installed onnxruntime-gpu and have a GPU."
        )


def _check_onnx_setup() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "ONNX is not installed. " "Please install it using `pip install onnx`."
        ) from None
    if sys.version_info < (3, 10):
        raise ValueError(
            "TabPFN ONNX export is not yet supported on Python 3.9. "
            "Please upgrade to Python 3.10 or higher."
        ) from None


class ONNXModelWrapper:
    """Wrap ONNX model to match the PyTorch model interface."""

    def __init__(self, model_path: Path, device: torch.device):
        """Initialize the ONNX model wrapper.

        Args:
            model_path: Path to the ONNX model file.
            device: The device to run the model on.
        """
        self.model_path = model_path
        self.device = device
        _check_cuda_provider(self.device)
        if device.type == "cuda":
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device.type == "cpu":
            self.providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Invalid device: {device}")
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers,
        )

    def to(
        self,
        device: torch.device,
    ) -> ONNXModelWrapper:
        """Moves the model to the specified device.

        Args:
            device: The target device (cuda or cpu).

        Returns:
            self
        """
        # Only recreate session if device type has changed
        _check_cuda_provider(device)
        if device.type != self.device.type:
            if device.type == "cuda":
                cuda_provider = "CUDAExecutionProvider"
                self.providers = [cuda_provider, "CPUExecutionProvider"]
                # Reinitialize session with CUDA provider
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=self.providers,
                )
            else:
                self.providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=self.providers,
                )
            # Update the device
            self.device = device
        return self

    def type(
        self,
        dtype: torch.dtype,  # noqa: ARG002
    ) -> ONNXModelWrapper:
        """Changes the model data type.

        The ONNX runtime handles dtype conversion internally; this method does nothing.

        Args:
            dtype: The target data type (unused).

        Returns:
            self
        """
        return self

    def cpu(self) -> ONNXModelWrapper:
        """Moves the model to CPU.

        This is a no-op for the ONNX model wrapper.

        Returns:
            self
        """
        return self

    def eval(self) -> ONNXModelWrapper:
        """Sets the model to evaluation mode.

        For the ONNX model wrapper, this does nothing and simply returns self.

        Returns:
            self
        """
        return self

    def __call__(
        self,
        style: torch.Tensor | None,  # noqa: ARG002
        X: torch.Tensor,
        y: torch.Tensor | None,
        *,
        single_eval_pos: int | None = None,
        only_return_standard_out: bool = False,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        """Run inference using the ONNX model.

        Args:
            style: Unused tensor placeholder.
            X: Input tensor.
            y: Target tensor.
            single_eval_pos: Position to evaluate at. Defaults to -1 if not provided.
            only_return_standard_out: Flag to return only the standard output.

        Returns:
            A torch tensor containing the model output.
        """
        # Convert inputs to numpy
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) and y is not None else y

        # Prepare ONNX inputs
        onnx_inputs = {
            "X": X_np,
            "y": y_np if y_np is not None else np.zeros((0,), dtype=np.float32),
            "single_eval_pos": np.array(
                single_eval_pos if single_eval_pos is not None else -1,
                dtype=np.int64,
            ),
        }

        # Run inference
        outputs = self.session.run(None, onnx_inputs)

        # Convert back to torch tensor and move to the appropriate device
        output_tensor = torch.from_numpy(outputs[0])
        if "CUDAExecutionProvider" in self.providers:
            output_tensor = output_tensor.cuda()
        return output_tensor

    def forward(
        self,
        style: torch.Tensor | None,
        X: torch.Tensor,
        y: torch.Tensor | None,
        *,
        single_eval_pos: int | None = None,
        only_return_standard_out: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass that delegates to __call__.

        Args:
            style: Unused tensor placeholder.
            X: Input tensor.
            y: Target tensor.
            single_eval_pos: Position to evaluate at. Defaults to -1 if not provided.
            only_return_standard_out: Flag to return only the standard output.

        Returns:
            A torch tensor containing the model output.
        """
        return self.__call__(
            style,
            X,
            y,
            single_eval_pos=single_eval_pos,
            only_return_standard_out=only_return_standard_out,
        )


class ModelWrapper(nn.Module):
    """A wrapper class to embed an ONNX model within the PyTorch nn.Module interface.
    Only used for exporting the model to ONNX format.
    """

    def __init__(self, original_model: ONNXModelWrapper):
        """Initialize the ModelWrapper.

        Args:
            original_model: The original model object to wrap.
        """
        super().__init__()
        self.model = original_model

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        single_eval_pos: torch.Tensor,
        only_return_standard_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Perform a forward pass.

        Args:
            X: Input tensor.
            y: Target tensor.
            single_eval_pos: Position for evaluation.
            only_return_standard_out: Whether to return only standard outputs.

        Returns:
            The output tensor from the model.
        """
        return self.model(
            None,
            X,
            y,
            single_eval_pos=single_eval_pos,
            only_return_standard_out=only_return_standard_out,
        )


def export_model(
    output_path: Path,
    model_type: str = "classifier",
) -> None:
    """Export the TabPFN model to the ONNX format.

    This function creates a sample model based on the specified
    model_type ('classifier' or 'regressor'), trains it on a small dataset,
    and exports the model to ONNX format with dynamic axes.

    Args:
        output_path: The file path where the ONNX model should be saved.
        model_type: The type of model to export ('classifier' or 'regressor').
    """
    # Load sample dataset for initialization
    if model_type == "classifier":
        X, y = sklearn.datasets.load_iris(return_X_y=True)
    else:  # regressor
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)

    with torch.no_grad():
        # Initialize and fit the model
        if model_type == "classifier":
            model = TabPFNClassifier(n_estimators=1, device="cpu", random_state=42)
        else:
            model = TabPFNRegressor(n_estimators=1, device="cpu", random_state=42)

        model.fit(X, y)
        # NOTE: Calling model.predict(X) at this point would break the export process.

        # Create sample input tensors
        X = torch.randn(
            (X.shape[0] * 2, 1, X.shape[1] + 1),
            generator=torch.Generator().manual_seed(42),
        )
        # make the first feature categorical
        X[:, 0, 0] = torch.randint(0, 10, (X.shape[0],))

        if model_type == "classifier":
            y = (
                torch.rand(y.shape, generator=torch.Generator().manual_seed(42))
                .round()
                .to(torch.float32)
            )
        else:
            y = torch.rand(y.shape, generator=torch.Generator().manual_seed(42))

        single_eval_pos = torch.tensor(
            y.shape[0],
            dtype=torch.int64,
        )  # Convert to tensor

        only_return_standard_out = torch.tensor(
            data=True,
            dtype=torch.bool,
        )  # Convert to tensor

        # Define dynamic axes for variable input sizes
        dynamic_axes = {
            "X": {0: "num_datapoints", 2: "num_features"},
            "y": {0: "num_datapoints"},
            "single_eval_pos": {},
            "only_return_standard_out": {},
        }

        # Export the model
        torch.onnx.export(
            ModelWrapper(model.model_).eval(),
            (X, y, single_eval_pos, only_return_standard_out),
            output_path,
            input_names=[
                "X",
                "y",
                "single_eval_pos",
                "only_return_standard_out",
            ],
            output_names=["output"],
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )


def check_onnx_model(model_path: Path) -> None:
    """Validate the ONNX model.

    Loads the ONNX model and runs a checker to ensure that the model is valid.

    Args:
        model_path: The path to the ONNX model file.
    """
    onnx_model = onnx.load(model_path)  # Load the ONNX model
    onnx.checker.check_model(onnx_model)  # Check if the model is valid


def check_input_names(model_path: Path) -> None:
    """Load the ONNX model to check its input names.

    Args:
        model_path: The path to the ONNX model file.
    """
    onnx.load(model_path)
    # Print output names


def test_models() -> None:
    """Test both TabPFNClassifier and TabPFNRegressor with and without ONNX.

    This function validates that both the original PyTorch models and the
    exported ONNX models work correctly on simple datasets.

    Args:
        model_path_classifier: Path to the exported ONNX classifier model.
        model_path_regressor: Path to the exported ONNX regressor model.
    """
    from sklearn.datasets import load_diabetes, load_iris
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split

    from tabpfn import TabPFNClassifier, TabPFNRegressor

    # Test classifier
    def _test_classifier(*, use_onnx: bool = False) -> float:
        # Load dataset
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Create and fit model
        if use_onnx:
            model = TabPFNClassifier(n_estimators=2, use_onnx=True)
        else:
            model = TabPFNClassifier(n_estimators=2, use_onnx=False)

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    # Test regressor
    def _test_regressor(*, use_onnx: bool = False) -> float:
        # Load dataset
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Create and fit model
        if use_onnx:
            model = TabPFNRegressor(n_estimators=2, use_onnx=True)
        else:
            model = TabPFNRegressor(n_estimators=2, use_onnx=False)

        model.fit(X_train, y_train)

        # Make predictions (mean)
        y_pred_mean = model.predict(X_test)
        y_pred_full = model.predict(X_test, output_type="full")
        assert len(y_pred_full.keys()) > 2
        return mean_squared_error(y_test, y_pred_mean)

    # Test with PyTorch backend
    clf_acc_torch = _test_classifier(use_onnx=False)
    reg_mse_torch = _test_regressor(use_onnx=False)

    # Test with ONNX backend
    clf_acc_onnx = _test_classifier(use_onnx=True)
    reg_mse_onnx = _test_regressor(use_onnx=True)

    # Compare results

    # Check if results are similar
    accuracy_diff = abs(clf_acc_torch - clf_acc_onnx)
    mse_ratio = reg_mse_torch / max(reg_mse_onnx, 1e-10)

    if accuracy_diff > 0.1 or mse_ratio < 0.5 or mse_ratio > 2.0:
        raise ValueError(
            "FAILED: the performance of the ONNX model is not "
            "similar to the PyTorch model. \n"
            f"Accuracy PyTorch: {clf_acc_torch}, Accuracy ONNX: {clf_acc_onnx}, \n"
            f"MSE PyTorch: {reg_mse_torch}, MSE ONNX: {reg_mse_onnx}"
        )
    print(
        "SUCCESS: the performance of the ONNX model is "
        "similar to the PyTorch model. \n"
        f"Accuracy PyTorch: {clf_acc_torch}, Accuracy ONNX: {clf_acc_onnx}, \n"
        f"MSE PyTorch: {reg_mse_torch}, MSE ONNX: {reg_mse_onnx}"
    )


def compile_onnx_models(suffix: str = "", *, skip_test: bool = False) -> None:
    """Compile the ONNX models.

    Args:
        suffix: The suffix to append to the file names of the ONNX models.
        skip_test: Whether to skip the performance test of the ONNX models.
    """
    _check_onnx_setup()

    classifier_path, _, _ = resolve_model_path(None, "classifier", "v2", use_onnx=True)
    regressor_path, _, _ = resolve_model_path(None, "regressor", "v2", use_onnx=True)

    # Add suffix before the .onnx extension
    stem = classifier_path.stem
    classifier_path = classifier_path.with_name(f"{stem}{suffix}").with_suffix(".onnx")
    stem = regressor_path.stem
    regressor_path = regressor_path.with_name(f"{stem}{suffix}").with_suffix(".onnx")

    export_model(classifier_path, "classifier")
    check_onnx_model(classifier_path)
    check_input_names(classifier_path)

    export_model(regressor_path, "regressor")
    check_onnx_model(regressor_path)
    check_input_names(regressor_path)

    if not len(suffix) and not skip_test:
        test_models()
    elif not skip_test:
        print("model name suffix is not empty, skipping test")
