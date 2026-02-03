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


class TabPFNCUDAOutOfMemoryError(TabPFNError):
    """Error raised when CUDA runs out of memory during prediction.

    This error provides guidance on how to handle large test sets that exceed
    available GPU memory.
    """

    def __init__(self, original_error: Exception | None = None):
        message = (
            "CUDA out of memory during prediction.\n\n"
            "The test set is too large to process in a single batch. "
            "To resolve this, split your test data into smaller batches:\n\n"
            "    # Split predictions into batches\n"
            "    batch_size = 10000  # Adjust based on your GPU memory\n"
            "    predictions = []\n"
            "    for i in range(0, len(X_test), batch_size):\n"
            "        batch = X_test[i:i + batch_size]\n"
            "        predictions.append(clf.predict_proba(batch))\n"
            "    predictions = np.concatenate(predictions, axis=0)\n\n"
            "Alternatively, reduce the number of test samples or use a GPU "
            "with more memory."
        )
        if original_error is not None:
            message += f"\n\nOriginal error: {original_error}"
        super().__init__(message)
