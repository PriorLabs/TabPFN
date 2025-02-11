"""Module providing wrappers to use ONNX models with a PyTorch-like interface.

This module defines wrappers for ONNX models as well as helper functions to export
and validate ONNX models derived from TabPFN models.
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import sklearn.datasets
import torch
from torch import nn

from tabpfn import TabPFNClassifier, TabPFNRegressor


class ONNXModelWrapper:
    """Wrap ONNX model to match the PyTorch model interface."""

    def __init__(self, model_path: str):
        """Initialize the ONNX model wrapper.

        Args:
            model_path: Path to the ONNX model file.
        """
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],  # TODO: Add GPU support
        )

    def to(
        self,
        device: torch.device,  # noqa: ARG002
    ) -> ONNXModelWrapper:
        """Moves the model to the specified device.

        This is a no-op for the ONNX model wrapper. GPU support is not implemented.

        Args:
            device: The target device (unused).

        Returns:
            self
        """
        # TODO: Add GPU support by changing provider
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
    ) -> torch.Tensor:
        """Run inference using the ONNX model.

        Args:
            style: Unused tensor placeholder.
            X: Input tensor.
            y: Target tensor.
            single_eval_pos: Position to evaluate at. Defaults to -1 if not provided.
            only_return_standard_out: Flag to return only the standard output.

        Returns:
            A torch tensor containing the model output.

        Note that only_return_standard_out is not used in the ONNX runtime.
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

        # Convert back to a torch tensor
        return torch.from_numpy(outputs[0])


class ModelWrapper(nn.Module):
    """A wrapper class to embed an ONNX model within the PyTorch nn.Module interface."""

    def __init__(self, original_model):
        """Initialize the ModelWrapper.

        Args:
            original_model: The original model object to wrap.
        """
        super().__init__()
        self.model = original_model

    def forward(self, X, y, single_eval_pos, only_return_standard_out):
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
    output_path: str,
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
        model.predict(X)

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
            "X": {0: "num_datapoints", 1: "batch_size", 2: "num_features"},
            "y": {0: "num_labels"},
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


def check_onnx_model(model_path: str) -> None:
    """Validate the ONNX model.

    Loads the ONNX model and runs a checker to ensure that the model is valid.

    Args:
        model_path: The path to the ONNX model file.
    """
    onnx_model = onnx.load(model_path)  # Load the ONNX model
    onnx.checker.check_model(onnx_model)  # Check if the model is valid


def check_input_names(model_path: str) -> None:
    """Load the ONNX model to check its input names.

    Args:
        model_path: The path to the ONNX model file.
    """
    onnx.load(model_path)
    # get input names from graph
    graph = onnx.load(model_path).graph
    [input_node.name for input_node in graph.input]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TabPFN models to ONNX format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model",
        help=(
            "Base output path for the ONNX models (will append _classifier.onnx and "
            "_regressor.onnx)"
        ),
    )

    args = parser.parse_args()

    # Export both models with appropriate suffixes
    classifier_path = f"{args.output}_classifier.onnx"
    regressor_path = f"{args.output}_regressor.onnx"

    export_model(classifier_path, "classifier")
    check_onnx_model(classifier_path)
    check_input_names(classifier_path)

    export_model(regressor_path, "regressor")
    check_onnx_model(regressor_path)
    check_input_names(regressor_path)
