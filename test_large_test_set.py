"""Minimal test script to reproduce CUDA error with large test sets.

This script demonstrates the CUDA "invalid configuration argument" error
that occurs when predicting on large test sets with TabPFN.

The error occurs in scaled_dot_product_attention when the batch size
is too large with flash attention / memory-efficient SDPA enabled.

Requirements:
- A CUDA-enabled GPU
- torch >= 2.4
- tabpfn

Usage:
    python test_large_test_set.py
"""

import numpy as np
from sklearn.datasets import make_classification

from tabpfn import TabPFNClassifier


def test_large_test_set_prediction():
    """Test that predicting on large test sets triggers memory issues.

    This test creates a dataset with a large number of test samples
    to reproduce the CUDA error reported in the issue.
    """
    # Create a small training set (TabPFN limit is ~10K samples)
    n_train_samples = 1000
    n_features = 20
    n_classes = 5

    X_train, y_train = make_classification(
        n_samples=n_train_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        random_state=42,
    )

    # Create a LARGE test set - this is what triggers the error
    # The threshold mentioned in the discussion is around 20,000 samples
    n_test_samples = 25_000  # Adjust this to trigger the error on your hardware

    X_test, _ = make_classification(
        n_samples=n_test_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        random_state=123,
    )

    print(f"Training samples: {n_train_samples}")
    print(f"Test samples: {n_test_samples}")
    print(f"Features: {n_features}")
    print(f"Classes: {n_classes}")

    # Initialize and fit the classifier
    clf = TabPFNClassifier(device="mps")  # Use CUDA to trigger the error

    print("\nFitting classifier...")
    clf.fit(X_train, y_train)

    print(f"\nPredicting on {n_test_samples} test samples...")
    print("This may trigger a CUDA error on some hardware configurations.")

    try:
        # This is where the error occurs with large test sets
        predictions = clf.predict_proba(X_test)
        print(f"Prediction successful! Shape: {predictions.shape}")
    except RuntimeError as e:
        if "CUDA error" in str(e) or "invalid configuration argument" in str(e):
            print(f"\n*** CUDA ERROR REPRODUCED ***")
            print(f"Error: {e}")
            print("\nThis error occurs when the test set is too large for the")
            print("GPU memory with flash attention / memory-efficient SDPA enabled.")
            print("\nSuggested fix: Batch the predictions into smaller chunks.")
        else:
            raise


def test_batched_prediction_workaround():
    """Demonstrate a manual workaround by batching predictions."""
    import warnings

    n_train_samples = 1000
    n_features = 20
    n_classes = 5

    X_train, y_train = make_classification(
        n_samples=n_train_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        random_state=42,
    )

    n_test_samples = 25_000
    X_test, _ = make_classification(
        n_samples=n_test_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        random_state=123,
    )

    clf = TabPFNClassifier(device="mps")
    clf.fit(X_train, y_train)

    # Manual batching workaround
    test_batch_size = 5000  # Adjust based on GPU memory
    n_batches = (n_test_samples + test_batch_size - 1) // test_batch_size

    print(f"\nUsing batched prediction with batch_size={test_batch_size}")
    print(f"Number of batches: {n_batches}")

    predictions_list = []
    for i in range(n_batches):
        start_idx = i * test_batch_size
        end_idx = min((i + 1) * test_batch_size, n_test_samples)
        X_batch = X_test[start_idx:end_idx]

        print(f"  Batch {i+1}/{n_batches}: samples {start_idx} to {end_idx}")
        batch_preds = clf.predict_proba(X_batch)
        predictions_list.append(batch_preds)

    predictions = np.vstack(predictions_list)
    print(f"\nBatched prediction successful! Shape: {predictions.shape}")
    return predictions


if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("TabPFN Large Test Set CUDA Error Reproduction Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA is not available. This test requires a CUDA GPU.")
        print("The error may not reproduce on CPU (but will be slower).")
        print("Running anyway with device='cpu'...")

    print("\n--- Test 1: Direct prediction (may fail) ---")
    test_large_test_set_prediction()

    print("\n" + "=" * 60)
    print("\n--- Test 2: Batched prediction workaround ---")
    test_batched_prediction_workaround()
