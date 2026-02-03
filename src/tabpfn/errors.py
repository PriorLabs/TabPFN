"""Custom exception classes for TabPFN."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations


class TabPFNError(Exception):
    """Base class for all TabPFN-specific exceptions."""


class TabPFNUserError(TabPFNError):
    """Base class for errors caused by invalid user input (safe to map to HTTP 400)."""


class TabPFNValidationError(ValueError, TabPFNUserError):
    """User provided invalid data (shape, NaNs, categories, etc.)."""


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
        n_test_samples: int | None = None,
        model_type: str = "classifier",
    ):
        if model_type == "classifier":
            predict_method = "predict_proba"
        else:
            predict_method = "predict"

        size_info = f" with {n_test_samples:,} samples" if n_test_samples else ""

        message = (
            f"{self.device_name} out of memory{size_info}.\n\n"
            f"Solution: Split your test data into smaller batches:\n\n"
            f"    batch_size = 1000\n"
            f"    predictions = []\n"
            f"    for i in range(0, len(X_test), batch_size):\n"
            f"        predictions.append(model.{predict_method}(X_test[i:i + batch_size]))\n"
            f"    predictions = np.vstack(predictions)"
        )
        if original_error is not None:
            message += f"\n\nOriginal error: {original_error}"
        super().__init__(message)
        self.original_error = original_error


class TabPFNCUDAOutOfMemoryError(TabPFNOutOfMemoryError):
    """Error raised when CUDA GPU runs out of memory during prediction."""

    device_name = "CUDA"


class TabPFNMPSOutOfMemoryError(TabPFNOutOfMemoryError):
    """Error raised when MPS (Apple Silicon) GPU runs out of memory during prediction."""

    device_name = "MPS"
