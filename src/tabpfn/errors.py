"""Custom exception classes for TabPFN."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tabpfn.constants import XType


class TabPFNError(Exception):
    """Base class for all TabPFN-specific exceptions."""


class TabPFNUserError(TabPFNError):
    """Base class for errors caused by invalid user input (safe to map to HTTP 400)."""


class TabPFNValidationError(ValueError, TabPFNUserError):
    """User provided invalid data (shape, NaNs, categories, etc.)."""


class TabPFNLicenseError(TabPFNError):
    """Error raised when the user has not accepted the TabPFN license."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "TabPFN requires license acceptance before downloading.\n\n"
                "To accept the license, run your script in an interactive terminal\n"
                "so a browser window can open for login, or set the TABPFN_TOKEN\n"
                "environment variable with a valid token obtained from\n"
                "https://ux.priorlabs.ai"
            )
        super().__init__(message)


class TabPFNHuggingFaceGatedRepoError(TabPFNError):
    """Error raised when a model is gated and requires user to accept terms."""

    def __init__(self, repo_id: str):
        message = (
            f"HuggingFace authentication error downloading from '{repo_id}'.\n"
            "This model is gated and requires you to accept its terms.\n\n"
            "Please follow these steps:\n"
            f"1. Visit https://huggingface.co/{repo_id} in your browser and"
            f" accept the terms of use.\n"
            "2. Log in to your Hugging Face account via"
            " the command line by running:\n"
            "   hf auth login\n"
            "   (Alternatively, you can set the HF_TOKEN environment variable"
            "   with a read token.)\n\n"
            "For detailed instructions, see "
            "https://docs.priorlabs.ai/how-to-access-gated-models"
        )
        super().__init__(message)


class TabPFNOutOfMemoryError(TabPFNError):
    """Base class for GPU out-of-memory errors during prediction.

    This error provides guidance on how to handle large test sets that exceed
    available GPU memory.
    """

    device_name: str = "GPU"  # Override in subclasses

    def __init__(
        self,
        original_error: Exception | None = None,
        *,
        n_train_samples: int | None = None,
        n_test_samples: int | None = None,
        n_features: int | None = None,
        model_type: str = "classifier",
    ):
        predict_method = "predict_proba" if model_type == "classifier" else "predict"

        size_info = f" with {n_test_samples:,} test samples" if n_test_samples else ""

        size_line = ""
        if n_train_samples is not None and n_test_samples is not None:
            size_line = (
                f"Your sizes: {n_train_samples:,} train / "
                f"{n_test_samples:,} test samples"
            )
            if n_features is not None:
                size_line += f", {n_features} features"
            size_line += ".\n"

        message = (
            f"{self.device_name} out of memory{size_info}.\n\n"
            f"This is issue is usually caused by one of the following two reasons:\n\n"
            f"1) Large test set — split into batches:\n\n"
            f"    predictions = []\n"
            f"    for i in range(0, len(X_test), 100):\n"
            f"        pred = model.{predict_method}("
            f"X_test[i:i + 100])\n"
            f"        predictions.append(pred)\n"
            f"    predictions = np.vstack(predictions)\n\n"
            f"2) Large training set — batching won't help.\n"
            f"   You need subsampling or ensembling, see:\n"
            f"   https://github.com/PriorLabs/tabpfn-extensions/"
            f"blob/main/examples/large_datasets/"
            f"large_datasets_example.py\n\n"
            f"{size_line}"
            f"Not sure which? If model.{predict_method}(X_test[:1]) "
            f"also fails, it's (2)."
        )
        if original_error is not None:
            message += f"\n\nOriginal error: {original_error}"
        super().__init__(message)
        self.original_error = original_error


class TabPFNCUDAOutOfMemoryError(TabPFNOutOfMemoryError):
    """Error raised when CUDA GPU runs out of memory during prediction."""

    device_name = "CUDA"


class TabPFNMPSOutOfMemoryError(TabPFNOutOfMemoryError):
    """Error raised when MPS (Apple Silicon) runs out of memory during prediction."""

    device_name = "MPS"


@contextmanager
def handle_oom_errors(
    devices: tuple[torch.device, ...],
    X: XType,
    model_type: str,
    n_train_samples: int | None = None,
    n_features: int | None = None,
) -> Generator[None, None, None]:
    """Context manager to catch OOM errors and raise helpful TabPFN exceptions.

    Args:
        devices: The devices the model is running on.
        X: The input data (used to get n_samples for the error message).
        model_type: Either "classifier" or "regressor".
        n_train_samples: Number of training samples (for the error message).
        n_features: Number of features (for the error message).

    Raises:
        TabPFNCUDAOutOfMemoryError: If a CUDA OOM error occurs.
        TabPFNMPSOutOfMemoryError: If an MPS OOM error occurs.
    """
    try:
        yield
    except torch.OutOfMemoryError as e:
        n_test_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        raise TabPFNCUDAOutOfMemoryError(
            e,
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            n_features=n_features,
            model_type=model_type,
        ) from None
    except RuntimeError as e:
        is_mps = any(d.type == "mps" for d in devices)
        is_oom = "out of memory" in str(e).lower()
        if is_mps and is_oom:
            n_test_samples = X.shape[0] if hasattr(X, "shape") else len(X)
            raise TabPFNMPSOutOfMemoryError(
                e,
                n_train_samples=n_train_samples,
                n_test_samples=n_test_samples,
                n_features=n_features,
                model_type=model_type,
            ) from None
        raise
