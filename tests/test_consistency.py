"""Model consistency tests for TabPFN.

These tests verify that TabPFN models produce consistent predictions across code
changes. If predictions change significantly, tests will fail, prompting the developer
to verify that the changes are legitimate improvements.

This approach uses small, fixed datasets with reproducible random seeds to ensure
consistency across code changes. Each test case maintains its own reference file.

Platform Considerations:
-----------------------
Reference predictions are platform-specific since floating-point calculations may
vary slightly across different operating systems, Python versions, and hardware.

Platform Metadata:
This system automatically stores reference platform information in:
  /reference_predictions/platform_metadata.json

By default, consistency tests only run on matching platforms:
- Same operating system (from platform metadata)
- Same Python major.minor version (e.g., 3.10, ignoring patch version)

To force tests to run on any platform:
- Set FORCE_CONSISTENCY_TESTS=1 environment variable

CI Configuration:
- In CI environments, reference values should be generated on a consistent platform
- Platform metadata is automatically updated when generating references
- Test runs on different platforms should set FORCE_CONSISTENCY_TESTS=1

If you need to update reference values:
1. Run: python tests/test_consistency.py
2. Include the updated reference files in your PR
3. Document the reason for the update in your PR description

How It Works:
------------
Each test is a subclass of ConsistencyTest that defines:
1. The dataset to use (via get_test_data)
2. The model configuration (via get_model)
3. How to extract predictions (via get_prediction_func)

When run, each test:
1. Creates a model
2. Fits it to a fixed dataset
3. Compares the predictions to previously saved references
4. Fails if predictions differ beyond tolerance thresholds

This design ensures tight coupling between test implementation and reference
generation, making it easier to maintain and extend the test suite.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import pathlib
import platform
import sys

import numpy as np
import pytest
from sklearn.utils import check_random_state

# mypy: ignore-errors
from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore

# Test configuration parameters
DEFAULT_N_ESTIMATORS = 2  # Small number for quick tests
TEST_TOLERANCE_RTOL = 1e-3  # 0.1% relative tolerance
TEST_TOLERANCE_ATOL = 1e-3  # 0.001 absolute tolerance

# Sample configuration
# Fixed seeds and indices make tests more stable and predictable
FIXED_RANDOM_SEED = 42  # Always use the same random seed for reproducibility

# Reference platform info is loaded from metadata file
# No hardcoded defaults needed anymore

# CI platform configurations - platforms supported in our CI system
# This is used to check compatibility of reference values
CI_PLATFORMS = [
    # (OS system name, python major.minor)
    ("Linux", "3.9"),
    ("Darwin", "3.9"),
    ("Windows", "3.9"),
    ("Linux", "3.13"),
    ("Darwin", "3.13"),
    ("Windows", "3.13"),
]


# Platform-specific helper functions
def is_ci_compatible_platform(os_name, python_version):
    """Check if a platform is CI-compatible.

    Verifies that the given OS and Python version combination is used in CI.

    Args:
        os_name: The OS name (from platform.system())
        python_version: The Python version (e.g., "3.9.1")

    Returns:
        bool: True if the platform is CI-compatible, False otherwise
    """
    # Extract major.minor version for comparison
    if python_version and "." in python_version:
        python_major_minor = ".".join(python_version.split(".")[:2])
    else:
        python_major_minor = python_version

    # Check against CI platforms list
    return (os_name, python_major_minor) in CI_PLATFORMS


def is_reference_platform():
    """Check if the current platform matches the reference platform.

    Performs a relaxed Python version check (major.minor only) but exact OS check.
    Returns False if metadata file doesn't exist or can't be read.
    """
    metadata_file = (
        pathlib.Path(__file__).parent
        / "reference_predictions"
        / "platform_metadata.json"
    )

    if not metadata_file.exists():
        return False

    try:
        with metadata_file.open("r") as f:
            metadata = json.load(f)

        # Extract reference info and current info
        ref_os = metadata.get("os")
        ref_python = metadata.get("python_version", "")

        # Only compare major.minor versions (e.g., 3.9 instead of 3.9.1)
        ref_python_major_minor = (
            ".".join(ref_python.split(".")[:2]) if ref_python else ""
        )
        current_python_major_minor = ".".join(platform.python_version().split(".")[:2])

        # Match OS exactly, but only match Python major.minor version
        return (
            platform.system() == ref_os
            and current_python_major_minor == ref_python_major_minor
        )
    except (OSError, json.JSONDecodeError):
        return False


def should_run_consistency_tests():
    """Determine if consistency tests should run on this platform.

    Tests run if:
    1. We're on the reference platform, or
    2. FORCE_CONSISTENCY_TESTS=1 in environment
    """
    # Always run if explicitly forced
    if os.environ.get("FORCE_CONSISTENCY_TESTS", "0") == "1":
        return True

    # Run if we're on the reference platform
    if is_reference_platform():
        return True

    # Special handling for CI - print warning but just skip instead of failing
    if os.environ.get("CI", "false").lower() in ("true", "1", "yes"):
        import logging

        # Get reference platform metadata
        metadata_file = (
            pathlib.Path(__file__).parent
            / "reference_predictions"
            / "platform_metadata.json"
        )
        if metadata_file.exists():
            try:
                with metadata_file.open("r") as f:
                    metadata = json.load(f)

                # Get reference platform info
                ref_os = metadata.get("os", "Unknown")
                ref_python = metadata.get("python_version", "Unknown")
                ".".join(ref_python.split(".")[:2]) if ref_python else "Unknown"

                # Check if reference platform is CI-compatible
                if not is_ci_compatible_platform(ref_os, ref_python):
                    platform_str = f"{ref_os}, Python {ref_python}"
                    logging.warning(
                        f"WARNING: Reference platform ({platform_str})"
                        f" is not compatible with any CI configuration."
                        f" Regenerate reference values on a CI platform."
                    )

                # Create platform descriptors for logging
                ref_plat = f"{ref_os} with Python {ref_python}"
                curr_sys = platform.system()
                curr_ver = platform.python_version()
                curr_plat = f"{curr_sys} with Python {curr_ver}"

                # Log a message about why tests are being skipped
                logging.warning(
                    f"Skipping consistency tests in CI due to platform mismatch!\n"
                    f"Reference platform: {ref_plat}\n"
                    f"Current platform: {curr_plat}\n"
                    f"Tests need matching platform or FORCE_CONSISTENCY_TESTS=1"
                )

                # Return False to skip test
                return False
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logging.warning(f"Error reading metadata in CI: {e}")
                return False

    # Not forced and not reference platform, so skip
    return False


# Platform-specific test decorator
platform_specific = pytest.mark.skipif(
    not should_run_consistency_tests(),
    reason=(
        "Tests require matching platform (same OS and Python version) "
        "or FORCE_CONSISTENCY_TESTS=1 environment variable"
    ),
)


# Test data generators for reproducible datasets
def get_tiny_classification_data(*, seed_modifier=0):
    """Get a tiny fixed classification dataset for testing.

    Args:
        seed_modifier: Optional modifier to the random seed, used in tests
                      to create intentionally different data
    """
    random_state = check_random_state(FIXED_RANDOM_SEED + seed_modifier)
    X = random_state.rand(10, 5)  # 10 samples, 5 features
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Binary classification

    # Split into train/test
    X_train, X_test = X[:7], X[7:]
    y_train = y[:7]

    return X_train, y_train, X_test


def get_tiny_regression_data():
    """Get a tiny fixed regression dataset for testing."""
    random_state = check_random_state(FIXED_RANDOM_SEED)
    X = random_state.rand(10, 5)  # 10 samples, 5 features
    y = random_state.rand(10) * 10  # Continuous target

    # Split into train/test
    X_train, X_test = X[:7], X[7:]
    y_train = y[:7]

    return X_train, y_train, X_test


def get_iris_multiclass_data():
    """Get a small subset of iris data for multiclass testing."""
    from sklearn.datasets import load_iris

    # Load iris dataset with 3 well-separated classes
    X, y = load_iris(return_X_y=True)

    # Use fixed test samples (first sample of each class)
    test_indices = [0, 50, 100]  # First sample of each class

    # Use fixed training samples (samples 1-6 from each class)
    train_indices = []
    for base in [0, 50, 100]:  # Start of each class
        train_indices.extend([base + i for i in range(1, 7)])  # 6 samples per class

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, _y_test = y[train_indices], y[test_indices]

    return X_train, y_train, X_test


def get_ensemble_data():
    """Get a fixed dataset for ensemble testing."""
    # Use the tiny classification dataset but with a larger ensemble
    return get_tiny_classification_data()


class ConsistencyTest:
    """Base class for model consistency tests.

    Each test case should subclass this and implement:
    1. get_dataset_name() - returns the unique name for the test case
    2. get_test_data() - returns (X_train, y_train, X_test) tuple
    3. get_model() - returns the TabPFN model to test with
    4. get_prediction_func() - returns function to get predictions

    This design ensures tight coupling between test generation and execution.
    """

    # Reference predictions directory
    REFERENCE_DIR = pathlib.Path(__file__).parent / "reference_predictions"

    # Platform metadata file
    PLATFORM_METADATA_FILE = REFERENCE_DIR / "platform_metadata.json"

    @classmethod
    def setup_class(cls):
        """Ensure the reference predictions directory exists."""
        cls.REFERENCE_DIR.mkdir(exist_ok=True)

    @classmethod
    def save_platform_metadata(cls):
        """Save current platform information to metadata file."""
        metadata = {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "generated_at": f"{datetime.datetime.now().isoformat()}",
        }

        with cls.PLATFORM_METADATA_FILE.open("w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_platform_metadata(cls):
        """Load platform metadata from file.

        Returns empty dict if file doesn't exist or can't be read.
        """
        if not cls.PLATFORM_METADATA_FILE.exists():
            return {}

        try:
            with cls.PLATFORM_METADATA_FILE.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # More specific exceptions for file reading issues
            return {}

    def get_dataset_name(self):
        """Get the unique name for this test case."""
        raise NotImplementedError("Subclasses must implement get_dataset_name()")

    def get_test_data(self):
        """Get the test data tuple (X_train, y_train, X_test)."""
        raise NotImplementedError("Subclasses must implement get_test_data()")

    def get_model(self):
        """Get the model instance to test."""
        raise NotImplementedError("Subclasses must implement get_model()")

    def get_prediction_func(self):
        """Get the function to extract predictions from the model."""
        raise NotImplementedError("Subclasses must implement get_prediction_func()")

    def get_reference_path(self):
        """Get the path to the reference prediction file for this test."""
        return self.REFERENCE_DIR / f"{self.get_dataset_name()}_predictions.json"

    def save_reference(self, predictions):
        """Save predictions as reference for this test."""
        path = self.get_reference_path()
        with path.open("w") as f:
            json.dump(predictions.tolist(), f, indent=2)

    def load_reference(self):
        """Load reference predictions for this test."""
        path = self.get_reference_path()
        if not path.exists():
            return None
        with path.open("r") as f:
            return np.array(json.load(f))

    def run_test(self, *, override=False):
        """Run the consistency test.

        Args:
            override: If True, force update reference predictions

        Returns:
            predictions: The model predictions
        """
        # Get test data and model
        X_train, y_train, X_test = self.get_test_data()
        model = self.get_model()

        # Fit model
        model.fit(X_train, y_train)

        # Get predictions
        prediction_func = self.get_prediction_func()
        predictions = prediction_func(model, X_test)

        # Handle reference
        ref_path = self.get_reference_path()
        if override:
            self.save_reference(predictions)
            return predictions

        # Compare with reference
        reference = self.load_reference()
        if reference is None:
            # Save current predictions as reference
            self.save_reference(predictions)
            pytest.skip(f"Created new reference predictions at {ref_path}")
            return predictions

        # Compare with consistent tolerance settings
        np.testing.assert_allclose(
            predictions,
            reference,
            rtol=TEST_TOLERANCE_RTOL,
            atol=TEST_TOLERANCE_ATOL,
            err_msg=(
                f"Predictions for {self.get_dataset_name()} have changed.\n"
                f"If this change is intentional:\n"
                f"1. Verify the changes improve model performance\n"
                f"2. Delete the reference file at {ref_path} to update it\n"
                f"3. Document the improvement in your PR description\n"
            ),
        )
        return predictions


class TestTinyClassifier(ConsistencyTest):
    """Test prediction consistency for a tiny binary classifier."""

    def get_dataset_name(self):
        return "tiny_classifier"

    def get_test_data(self):
        return get_tiny_classification_data()

    def get_model(self):
        return TabPFNClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=FIXED_RANDOM_SEED,
            device="cpu",
        )

    def get_prediction_func(self):
        return lambda model, X: model.predict_proba(X)

    @platform_specific
    def test_consistency(self):
        """Test prediction consistency on a very small classification dataset."""
        self.run_test()


class TestTinyRegressor(ConsistencyTest):
    """Test prediction consistency for a tiny regressor."""

    def get_dataset_name(self):
        return "tiny_regressor"

    def get_test_data(self):
        return get_tiny_regression_data()

    def get_model(self):
        return TabPFNRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=FIXED_RANDOM_SEED,
            device="cpu",
        )

    def get_prediction_func(self):
        return lambda model, X: model.predict(X)

    @platform_specific
    def test_consistency(self):
        """Test prediction consistency on a very small regression dataset."""
        self.run_test()


class TestMulticlassClassifier(ConsistencyTest):
    """Test prediction consistency for a multiclass classifier."""

    def get_dataset_name(self):
        return "iris_multiclass"

    def get_test_data(self):
        return get_iris_multiclass_data()

    def get_model(self):
        return TabPFNClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=FIXED_RANDOM_SEED,
            device="cpu",
        )

    def get_prediction_func(self):
        return lambda model, X: model.predict_proba(X)

    @platform_specific
    def test_consistency(self):
        """Test prediction consistency on iris multiclass dataset."""
        self.run_test()


class TestEnsembleClassifier(ConsistencyTest):
    """Test prediction consistency for a classifier with larger ensemble."""

    def get_dataset_name(self):
        return "ensemble_classifier"

    def get_test_data(self):
        return get_ensemble_data()

    def get_model(self):
        return TabPFNClassifier(
            n_estimators=5,  # Larger ensemble for this test
            random_state=FIXED_RANDOM_SEED,
            device="cpu",
        )

    def get_prediction_func(self):
        return lambda model, X: model.predict_proba(X)

    @platform_specific
    def test_consistency(self):
        """Test prediction consistency with larger ensemble."""
        self.run_test()


def update_reference_predictions():
    """Generate and save reference predictions for all test cases.

    Uses the same test classes as the actual tests to ensure consistency.
    Warns if reference values are being generated on a non-CI platform.
    """
    current_os = platform.system()
    current_python = platform.python_version()

    # Check if we're on a CI-compatible platform
    is_ci_platform = is_ci_compatible_platform(current_os, current_python)

    # Ensure reference dir exists
    ref_dir = ConsistencyTest.REFERENCE_DIR
    ref_dir.mkdir(exist_ok=True)

    # Clear existing reference files
    for path in ref_dir.glob("*_predictions.json"):
        path.unlink()

    # Save and display platform information
    ConsistencyTest.save_platform_metadata()

    if not is_ci_platform:
        for _os_name, _py_ver in CI_PLATFORMS:
            pass

        # Ask for confirmation before proceeding
        proceed = input("\nProceed anyway? (y/n): ")
        if proceed.lower() not in ["y", "yes"]:
            return
    else:
        pass

    # Create and update references for each test case
    test_cases = [
        TestTinyClassifier(),
        TestTinyRegressor(),
        TestMulticlassClassifier(),
        TestEnsembleClassifier(),
    ]

    for test_case in test_cases:
        with contextlib.suppress(Exception):
            test_case.run_test(override=True)


class TestInconsistencyDetection(ConsistencyTest):
    """Test that our consistency checks correctly detect inconsistencies.

    This is a meta-test that verifies our consistency test framework works
    by purposely creating a mismatch between reference and actual predictions.
    """

    def get_dataset_name(self):
        return "inconsistency_test"

    def get_test_data(self, seed_modifier=0):
        """Get test data with an optional seed modifier to create inconsistency."""
        return get_tiny_classification_data(seed_modifier=seed_modifier)

    def get_model(self):
        return TabPFNClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=FIXED_RANDOM_SEED,
            device="cpu",
        )

    def get_prediction_func(self):
        return lambda model, X: model.predict_proba(X)

    def test_consistency_detection(self):
        """Test that our consistency test framework correctly detects inconsistencies.

        This test:
        1. First creates reference predictions with default data
        2. Then tries to run the test with different data (modified seed)
        3. Expects the test to fail with an AssertionError due to changed predictions
        """
        # Setup phase - create reference with one dataset
        X_train, y_train, X_test = self.get_test_data(seed_modifier=0)
        model = self.get_model()
        model.fit(X_train, y_train)
        predictions = self.get_prediction_func()(model, X_test)

        # Save as reference
        self.save_reference(predictions)

        # Test phase - try with different data
        X_train_mod, y_train_mod, X_test_mod = self.get_test_data(seed_modifier=10)
        model_mod = self.get_model()
        model_mod.fit(X_train_mod, y_train_mod)
        predictions_mod = self.get_prediction_func()(model_mod, X_test_mod)

        # This should fail because predictions will differ
        with pytest.raises(AssertionError) as excinfo:
            np.testing.assert_allclose(
                predictions_mod,
                self.load_reference(),
                rtol=TEST_TOLERANCE_RTOL,
                atol=TEST_TOLERANCE_ATOL,
            )

        # Verify the error message mentions predictions have changed
        assert "have changed" in str(excinfo.value) or "Not equal" in str(excinfo.value)


class TestCIPlatformValidation:
    """Tests that reference platform metadata is valid for CI environments."""

    def test_reference_platform_ci_compatibility(self):
        """Verifies reference platform matches a supported CI configuration."""
        metadata_file = ConsistencyTest.PLATFORM_METADATA_FILE

        # Skip if no metadata exists
        if not metadata_file.exists():
            pytest.skip("No platform metadata file exists")

        # Load metadata
        try:
            with metadata_file.open("r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            pytest.fail(f"Failed to read platform metadata: {e}")

        # Extract platform information and check compatibility
        ref_os = metadata.get("os")
        ref_python = metadata.get("python_version", "")
        is_compatible = is_ci_compatible_platform(ref_os, ref_python)

        # Assert compatibility with detailed error message
        allowed_platforms = ", ".join(f"{os}/{py}" for os, py in CI_PLATFORMS)
        message = (
            f"Reference platform ({ref_os}, Python {ref_python}) is not a CI platform. "
            f"Allowed: {allowed_platforms}. Regenerate on a CI platform."
        )

        assert is_compatible, message


def print_platform_info():
    """Print platform information relevant for consistency tests.

    Shows current platform, CI compatibility status, and reference platform details.
    """
    # Current platform information
    current_os = platform.system()
    current_python = platform.python_version()
    ".".join(current_python.split(".")[:2])

    # Check if we're on a CI-compatible platform
    is_ci_platform = is_ci_compatible_platform(current_os, current_python)

    if not is_ci_platform:
        for _os_name, _py_ver in CI_PLATFORMS:
            pass

    # Load and print reference platform info
    metadata_file = ConsistencyTest.PLATFORM_METADATA_FILE
    if metadata_file.exists():
        try:
            metadata = ConsistencyTest.load_platform_metadata()

            if metadata:
                ref_os = metadata.get("os", "N/A")
                ref_python = metadata.get("python_version", "N/A")
                ".".join(ref_python.split(".")[:2]) if ref_python else "N/A"

                # Check if reference platform matches CI
                ref_is_ci = is_ci_compatible_platform(ref_os, ref_python)

                # Check if current platform matches reference platform
                is_reference_platform()

                if not ref_is_ci:
                    pass

        except (json.JSONDecodeError, OSError, KeyError):
            pass
    else:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TabPFN Consistency Test Utilities")
    parser.add_argument(
        "--update-reference",
        action="store_true",
        help="Update reference predictions for the current platform",
    )
    parser.add_argument(
        "--print-platform",
        action="store_true",
        help="Print information about the current platform",
    )

    args = parser.parse_args()

    if args.update_reference:
        # Update reference predictions for the current platform
        update_reference_predictions()
    elif args.print_platform or len(sys.argv) == 1:
        # If no args or explicit request to print platform info
        print_platform_info()
    else:
        # Show help if unknown args
        parser.print_help()
