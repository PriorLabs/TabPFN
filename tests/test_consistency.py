"""Model consistency tests for TabPFN.

These tests verify that TabPFN models produce consistent predictions across code
changes. If predictions change, developers must explicitly acknowledge and verify
the improvement.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from sklearn.utils import check_random_state

# mypy: ignore-errors
from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore


class TestExactPredictions:
    """Test exact prediction consistency for very small datasets.

    Instead of comparing statistical summaries, these tests compare
    actual prediction values. This provides a more precise check for
    small datasets.
    """

    # Reference predictions directory
    REFERENCE_DIR = pathlib.Path(__file__).parent / "reference_predictions"

    @pytest.fixture(autouse=True)
    def ensure_reference_dir(self):
        """Ensure the reference predictions directory exists."""
        self.REFERENCE_DIR.mkdir(exist_ok=True)

    def get_reference_path(self, dataset: str, model_type: str) -> pathlib.Path:
        """Get the path to the reference prediction file."""
        return self.REFERENCE_DIR / f"{dataset}_{model_type}_predictions.json"

    def save_reference(self, predictions: np.ndarray, path: pathlib.Path) -> None:
        """Save reference predictions to a file."""
        with path.open("w") as f:
            # Convert to list for JSON serialization and maintain precision
            json.dump(predictions.tolist(), f, indent=2)

    def load_reference(self, path: pathlib.Path) -> np.ndarray:
        """Load reference predictions from a file."""
        if not path.exists():
            return None

        with path.open("r") as f:
            return np.array(json.load(f))

    def test_tiny_classifier_consistency(self):
        """Test exact prediction consistency on a very small classification dataset."""
        # Create a tiny dataset
        random_state = check_random_state(42)
        X = random_state.rand(10, 5)  # 10 samples, 5 features
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Binary classification

        # Split into train/test
        X_train, X_test = X[:7], X[7:]
        y_train, _y_test = y[:7], y[7:]

        # Create classifier with fixed settings
        clf = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
        clf.fit(X_train, y_train)

        # Get predictions
        predictions = clf.predict_proba(X_test)

        # Reference file path
        ref_path = self.get_reference_path("tiny", "classifier")

        # Load reference predictions or create them
        reference = self.load_reference(ref_path)
        if reference is None:
            # Save current predictions as reference
            self.save_reference(predictions, ref_path)
            pytest.skip(f"Created new reference predictions at {ref_path}")

        # Compare with exact precision
        np.testing.assert_array_equal(
            predictions,
            reference,
            err_msg=(
                f"TabPFNClassifier tiny dataset predictions have changed.\n"
                f"Expected: {reference}\n"
                f"Actual: {predictions}\n\n"
                f"If this change is intentional:\n"
                f"1. Verify the changes improve model performance\n"
                f"2. Delete the reference file at {ref_path} to update it\n"
                f"3. Document the improvement in your PR description\n"
            ),
        )

    def test_tiny_regressor_consistency(self):
        """Test exact prediction consistency on a very small regression dataset."""
        # Create a tiny dataset
        random_state = check_random_state(42)
        X = random_state.rand(10, 5)  # 10 samples, 5 features
        y = random_state.rand(10) * 10  # Continuous target

        # Split into train/test
        X_train, X_test = X[:7], X[7:]
        y_train, _y_test = y[:7], y[7:]

        # Create regressor with fixed settings
        reg = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
        reg.fit(X_train, y_train)

        # Get predictions
        predictions = reg.predict(X_test)

        # Reference file path
        ref_path = self.get_reference_path("tiny", "regressor")

        # Load reference predictions or create them
        reference = self.load_reference(ref_path)
        if reference is None:
            # Save current predictions as reference
            self.save_reference(predictions, ref_path)
            pytest.skip(f"Created new reference predictions at {ref_path}")

        # Compare with high precision (but not exact due to potential float differences)
        np.testing.assert_allclose(
            predictions,
            reference,
            rtol=1e-5,
            atol=1e-5,
            err_msg=(
                f"TabPFNRegressor tiny dataset predictions have changed.\n"
                f"Expected: {reference}\n"
                f"Actual: {predictions}\n\n"
                f"If this change is intentional:\n"
                f"1. Verify the changes improve model performance\n"
                f"2. Delete the reference file at {ref_path} to update it\n"
                f"3. Document the improvement in your PR description\n"
            ),
        )


# Helper function to update reference predictions
def update_all_reference_predictions():
    """Generate and save reference predictions for all test datasets.

    Run this function manually when intentionally updating reference predictions:
    ```
    python -c "from tests.test_consistency import update_all_reference_predictions; \
    update_all_reference_predictions()"
    ```
    """
    test_instance = TestExactPredictions()
    # Create the directory manually instead of using the fixture
    test_instance.REFERENCE_DIR.mkdir(exist_ok=True)

    # Force recreation of reference files by deleting existing ones
    for path in test_instance.REFERENCE_DIR.glob("*_predictions.json"):
        path.unlink()

    # Create helper function to generate predictions and save them directly
    # This avoids using pytest.skip() which is meant for test environments
    def generate_and_save_reference(dataset_name, model_type, generate_func):
        try:
            predictions = generate_func()
            ref_path = test_instance.get_reference_path(dataset_name, model_type)
            test_instance.save_reference(predictions, ref_path)
            return True
        except (ValueError, TypeError, RuntimeError):
            # Specific exceptions we might expect during generation
            return False

    # Generate tiny classifier predictions
    def generate_tiny_classifier():
        random_state = check_random_state(42)
        X = random_state.rand(10, 5)  # 10 samples, 5 features
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Binary classification

        # Split into train/test
        X_train, X_test = X[:7], X[7:]
        y_train, _ = y[:7], y[7:]

        # Create classifier with fixed settings
        clf = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
        clf.fit(X_train, y_train)

        # Get predictions
        return clf.predict_proba(X_test)

    # Generate tiny regressor predictions
    def generate_tiny_regressor():
        random_state = check_random_state(42)
        X = random_state.rand(10, 5)  # 10 samples, 5 features
        y = random_state.rand(10) * 10  # Continuous target

        # Split into train/test
        X_train, X_test = X[:7], X[7:]
        y_train, _ = y[:7], y[7:]

        # Create regressor with fixed settings
        reg = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
        reg.fit(X_train, y_train)

        # Get predictions
        return reg.predict(X_test)

    # Generate all reference predictions
    success_count = 0
    total_count = 0

    total_count += 1
    if generate_and_save_reference("tiny", "classifier", generate_tiny_classifier):
        success_count += 1

    total_count += 1
    if generate_and_save_reference("tiny", "regressor", generate_tiny_regressor):
        success_count += 1


if __name__ == "__main__":
    # This makes it easier to run the prediction reference updates
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "predictions":
        update_all_reference_predictions()
