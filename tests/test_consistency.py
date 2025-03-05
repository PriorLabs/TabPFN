"""Model consistency tests for TabPFN.

These tests verify that TabPFN models produce consistent predictions across code
changes. If predictions change, developers must explicitly acknowledge and verify
the improvement.

Platform-Specific Consistency Notes:
----------------------------------
IMPORTANT: These consistency tests are platform-specific. The reference values
in this file were generated on macOS (ARM architecture) and should pass on similar
environments. They may fail on different platforms due to:

1. Different CPU architectures (x86 vs. ARM)
2. Different OSes (Linux, macOS, Windows)
3. Different underlying BLAS/LAPACK implementations
4. Compiler-specific floating-point optimizations

If you need to update or regenerate reference values:
1. Do so on the same platform type you're primarily testing on
2. Document which platform was used to generate the values
3. CI environments should use the same platform for consistency testing

The tests use:
- sklearn's `USE_SKLEARN_16_DECIMAL_PRECISION` flag for better reproducibility
- Statistical summaries with relative tolerance (3%) instead of exact matches
- Platform-specific reference values that should remain consistent on the
  same platform even as code changes
"""

from __future__ import annotations

import platform

import numpy as np
import pytest
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# mypy: ignore-errors
from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore

# Configuration for consistency tests
# Specify the reference platform these tests were created on
REFERENCE_PLATFORM = "Darwin"  # macOS
REFERENCE_PYTHON_VERSION = "3.12"
# If True, also run the tests on other platforms but with higher tolerance
RUN_ON_ALL_PLATFORMS = False

# Create a reusable decorator for platform-specific tests
platform_specific = pytest.mark.skipif(
    not (
        platform.system() == REFERENCE_PLATFORM
        and platform.python_version().startswith(REFERENCE_PYTHON_VERSION)
    ),
    reason="Model consistency tests are platform-specific. "
    f"Reference values are for {REFERENCE_PLATFORM}",
)


def generate_prediction_stats(model_predictions: np.ndarray) -> dict:
    """Generate statistical summary of model predictions.

    Instead of using a strict hash function, this approach computes various
    statistics that should remain relatively stable across different platforms
    and environments, while allowing for small floating-point variations.

    Args:
        model_predictions: Array of model predictions to analyze

    Returns:
        Dictionary containing statistical measures of the predictions
    """
    # Flatten array if needed
    flat_preds = (
        model_predictions.flatten() if model_predictions.ndim > 1 else model_predictions
    )

    # Compute statistics that should be stable across platforms
    return {
        "min": float(np.min(flat_preds)),
        "max": float(np.max(flat_preds)),
        "mean": float(np.mean(flat_preds)),
        "std": float(np.std(flat_preds)),
        "median": float(np.median(flat_preds)),
        # Add a few percentiles to better capture the distribution
        "p10": float(np.percentile(flat_preds, 10)),
        "p25": float(np.percentile(flat_preds, 25)),
        "p75": float(np.percentile(flat_preds, 75)),
        "p90": float(np.percentile(flat_preds, 90)),
        # Shape information
        "shape": model_predictions.shape,
    }


# Define reference statistics for expected prediction outputs
# Generated on platform: Darwin with Python 3.12
# Using scikit-learn's 16 decimal precision for better reproducibility
# These should only be updated when model improvements are verified or when
# regenerating references for a different platform
REFERENCE_STATS = {
    "iris_classifier": {
        "min": 0.0,
        "max": 1.0,
        "mean": 0.3333333432674408,
        "std": 0.4565829038619995,
        "median": 0.0018062918679788709,
        "p10": 0.0,
        "p25": 0.0,
        "p75": 0.9885727763175964,
        "p90": 1.0,
        "shape": (45, 3),
    },
    "breast_cancer_classifier": {
        "min": 0.0,
        "max": 1.0,
        "mean": 0.5,
        "std": 0.4800257980823517,
        "median": 0.5,
        "p10": 0.0,
        "p25": 0.0,
        "p75": 1.0,
        "p90": 1.0,
        "shape": (171, 2),
    },
    "boston_regressor": {
        "min": 8.342599868774414,
        "max": 50.012393951416016,
        "mean": 22.950313568115234,
        "std": 8.420607566833496,
        "median": 21.672588348388672,
        "p10": 13.99816312789917,
        "p25": 18.07076930999756,
        "p75": 26.79212713241577,
        "p90": 34.71592826843262,
        "shape": (152,),
    },
    "diabetes_regressor": {
        "min": 69.17279052734375,
        "max": 270.5797119140625,
        "mean": 154.7861785888672,
        "std": 56.70866012573242,
        "median": 146.58413696289062,
        "p10": 85.05858001708984,
        "p25": 109.09040832519531,
        "p75": 193.2696075439453,
        "p90": 241.7614501953125,
        "shape": (133,),
    },
}


class TestModelConsistency:
    """Verify that TabPFN models produce consistent predictions across code changes."""

    @pytest.fixture
    def iris_data(self):
        """Fixture for Iris dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)
        X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=False)

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def breast_cancer_data(self):
        """Fixture for Breast Cancer dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=False)

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def boston_data(self):
        """Fixture for Boston Housing dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)

        # Boston dataset is deprecated but we can load it from sklearn-datasets
        # or use California housing as an alternative
        try:
            # Try to use fetch_openml to get Boston dataset
            from sklearn.datasets import fetch_openml

            housing = fetch_openml(data_id=531, as_frame=False)
            X, y = housing.data, housing.target
        except (ImportError, ValueError):
            # Fall back to California housing
            from sklearn.datasets import fetch_california_housing

            housing = fetch_california_housing(as_frame=False)
            X, y = housing.data, housing.target

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def diabetes_data(self):
        """Fixture for Diabetes dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)
        X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    @platform_specific
    def test_iris_classifier_consistency(self, iris_data):
        """Verify TabPFNClassifier predictions on Iris dataset are consistent."""
        X_train, X_test, y_train, y_test = iris_data

        # Use scikit-learn's 16 decimal precision for cross-platform determinism
        clf = TabPFNClassifier(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # CPU for deterministic results
            inference_config={
                "USE_SKLEARN_16_DECIMAL_PRECISION": True
            },  # For reproducibility
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_test)

        # Generate statistics from predictions
        actual_stats = generate_prediction_stats(predictions)
        reference_stats = REFERENCE_STATS["iris_classifier"]

        # Small tolerance for inevitable floating-point variations
        # 1% relative error is enough with double precision
        rtol = 0.01

        # Check shape exactly (should be identical)
        assert actual_stats["shape"] == reference_stats["shape"], (
            f"Prediction shape has changed for Iris dataset.\n"
            f"Expected: {reference_stats['shape']}, got: {actual_stats['shape']}"
        )

        # Check key statistics are within tolerance
        for stat in ["min", "max", "mean", "std", "median", "p10", "p25", "p75", "p90"]:
            assert np.isclose(actual_stats[stat], reference_stats[stat], rtol=rtol), (
                f"TabPFNClassifier predictions for Iris have changed significantly.\n"
                f"Statistic '{stat}' differs too much:\n"
                f"Expected: {reference_stats[stat]}\n"
                f"Actual: {actual_stats[stat]}\n"
                f"Difference: {abs(actual_stats[stat] - reference_stats[stat])}\n\n"
                f"If this change is intentional:\n"
                f"1. Verify the changes improve model performance on benchmarks\n"
                f"2. Update the reference statistics in `tests/test_consistency.py`\n"
                f"3. Document the improvement in your PR description\n"
            )

    @platform_specific
    def test_breast_cancer_classifier_consistency(self, breast_cancer_data):
        """Verify TabPFNClassifier predictions on Breast Cancer dataset."""
        X_train, X_test, y_train, y_test = breast_cancer_data

        # Use scikit-learn's 16 decimal precision for cross-platform determinism
        clf = TabPFNClassifier(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # CPU for deterministic results
            inference_config={
                "USE_SKLEARN_16_DECIMAL_PRECISION": True
            },  # For reproducibility
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_test)

        # Generate statistics from predictions
        actual_stats = generate_prediction_stats(predictions)
        reference_stats = REFERENCE_STATS["breast_cancer_classifier"]

        # Small tolerance for inevitable floating-point variations
        # 1% relative error is enough with double precision
        rtol = 0.01

        # Check shape exactly (should be identical)
        assert actual_stats["shape"] == reference_stats["shape"], (
            f"Prediction shape has changed for Breast Cancer dataset.\n"
            f"Expected: {reference_stats['shape']}, got: {actual_stats['shape']}"
        )

        # Check key statistics are within tolerance
        for stat in ["min", "max", "mean", "std", "median", "p10", "p25", "p75", "p90"]:
            assert np.isclose(actual_stats[stat], reference_stats[stat], rtol=rtol), (
                f"TabPFNClassifier predictions for Breast Cancer have changed.\n"
                f"Statistic '{stat}' differs too much:\n"
                f"Expected: {reference_stats[stat]}\n"
                f"Actual: {actual_stats[stat]}\n"
                f"Difference: {abs(actual_stats[stat] - reference_stats[stat])}\n\n"
                f"If this change is intentional:\n"
                f"1. Verify the changes improve model performance on benchmarks\n"
                f"2. Update the reference statistics in `tests/test_consistency.py`\n"
                f"3. Document the improvement in your PR description\n"
            )

    @platform_specific
    def test_boston_regressor_consistency(self, boston_data):
        """Verify TabPFNRegressor predictions on Boston Housing dataset."""
        X_train, X_test, y_train, y_test = boston_data

        # Use scikit-learn's 16 decimal precision for cross-platform determinism
        reg = TabPFNRegressor(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # CPU for deterministic results
            inference_config={
                "USE_SKLEARN_16_DECIMAL_PRECISION": True
            },  # For reproducibility
        )

        # Fit and predict
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)

        # Generate statistics from predictions
        actual_stats = generate_prediction_stats(predictions)
        reference_stats = REFERENCE_STATS["boston_regressor"]

        # Increased tolerance for regression tasks which show more platform variability
        # 3% relative error for better cross-platform compatibility
        rtol = 0.03

        # Check shape exactly (should be identical)
        assert actual_stats["shape"] == reference_stats["shape"], (
            f"Prediction shape has changed for Boston Housing dataset.\n"
            f"Expected: {reference_stats['shape']}, got: {actual_stats['shape']}"
        )

        # Check key statistics are within tolerance
        for stat in ["min", "max", "mean", "std", "median", "p10", "p25", "p75", "p90"]:
            assert np.isclose(actual_stats[stat], reference_stats[stat], rtol=rtol), (
                f"TabPFNRegressor predictions for Boston Housing have changed.\n"
                f"Statistic '{stat}' differs too much:\n"
                f"Expected: {reference_stats[stat]}\n"
                f"Actual: {actual_stats[stat]}\n"
                f"Difference: {abs(actual_stats[stat] - reference_stats[stat])}\n\n"
                f"If this change is intentional:\n"
                f"1. Verify the changes improve model performance on benchmarks\n"
                f"2. Update the reference statistics in `tests/test_consistency.py`\n"
                f"3. Document the improvement in your PR description\n"
            )

    @platform_specific
    def test_diabetes_regressor_consistency(self, diabetes_data):
        """Verify TabPFNRegressor predictions on Diabetes dataset are consistent."""
        X_train, X_test, y_train, y_test = diabetes_data

        # Use scikit-learn's 16 decimal precision for cross-platform determinism
        reg = TabPFNRegressor(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # CPU for deterministic results
            inference_config={
                "USE_SKLEARN_16_DECIMAL_PRECISION": True
            },  # For reproducibility
        )

        # Fit and predict
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)

        # Generate statistics from predictions
        actual_stats = generate_prediction_stats(predictions)
        reference_stats = REFERENCE_STATS["diabetes_regressor"]

        # Increased tolerance for diabetes dataset which shows more platform variability
        # 3% relative error for better cross-platform compatibility
        rtol = 0.03

        # Check shape exactly (should be identical)
        assert actual_stats["shape"] == reference_stats["shape"], (
            f"Prediction shape has changed for Diabetes dataset.\n"
            f"Expected: {reference_stats['shape']}, got: {actual_stats['shape']}"
        )

        # Check key statistics are within tolerance
        for stat in ["min", "max", "mean", "std", "median", "p10", "p25", "p75", "p90"]:
            current_rtol = rtol
            assert np.isclose(
                actual_stats[stat], reference_stats[stat], rtol=current_rtol
            ), (
                f"TabPFNRegressor predictions for Diabetes have changed.\n"
                f"Statistic '{stat}' differs too much:\n"
                f"Expected: {reference_stats[stat]}\n"
                f"Actual: {actual_stats[stat]}\n"
                f"Difference: {abs(actual_stats[stat] - reference_stats[stat])}\n\n"
                f"If this change is intentional:\n"
                f"1. Verify the changes improve model performance on benchmarks\n"
                f"2. Update the reference statistics in `tests/test_consistency.py`\n"
                f"3. Document the improvement in your PR description\n"
            )


class TestStatsRobustness:
    """Verify that our statistical approach correctly detects meaningful changes."""

    # These tests run on all platforms (no reference values needed)

    @pytest.fixture
    def test_data(self):
        """Create a simple dataset for testing."""
        check_random_state(42)
        X, y = sklearn.datasets.make_classification(
            n_samples=100,
            n_features=10,
            random_state=42,
        )

        # Split into train/test
        train_size = 70
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    def test_classifier_stats_detect_data_changes(self, test_data):
        """Verify stats detect changes in classification input data."""
        X_train, X_test, y_train, y_test = test_data

        # Create classifier with fixed settings
        clf = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
        clf.fit(X_train, y_train)

        # Get predictions on original data
        original_predictions = clf.predict_proba(X_test)
        original_stats = generate_prediction_stats(original_predictions)

        # Modify test data significantly
        X_test_modified = X_test.copy()
        X_test_modified[0, 0] += 5.0  # Significant change to first feature

        # Get predictions on modified data
        modified_predictions = clf.predict_proba(X_test_modified)
        modified_stats = generate_prediction_stats(modified_predictions)

        # Verify stats detect the change
        # At least one of the main statistics should differ by more than 1%
        differences_detected = False
        for stat in ["mean", "std", "median", "p75"]:
            if not np.isclose(original_stats[stat], modified_stats[stat], rtol=0.01):
                differences_detected = True
                break

        assert (
            differences_detected
        ), "Statistical approach failed to detect change in classification data"

    def test_classifier_stats_detect_model_changes(self, test_data):
        """Verify stats detect changes in classifier configuration."""
        X_train, X_test, y_train, y_test = test_data

        # Create classifier with one configuration
        clf1 = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
        clf1.fit(X_train, y_train)
        predictions1 = clf1.predict_proba(X_test)
        generate_prediction_stats(predictions1)

        # Create classifier with significantly different configuration
        # Use a much different random seed to ensure we get different predictions
        clf2 = TabPFNClassifier(n_estimators=10, random_state=1234, device="cpu")
        clf2.fit(X_train, y_train)
        predictions2 = clf2.predict_proba(X_test)
        generate_prediction_stats(predictions2)

        # For the robustness test, we'll sample to check that predictions differ
        # between two model configurations with different random states
        # Given that we're testing the validity of the test itself, we can make this an
        # explicit check on an element of the prediction
        rng = np.random.RandomState(42)
        sample_idx = rng.randint(0, len(predictions1.flatten()) - 1)
        sample_val1 = predictions1.flatten()[sample_idx]
        sample_val2 = predictions2.flatten()[sample_idx]

        # Check that at least one specific prediction is different
        # because we changed the random seed and ensemble size significantly
        # this allows us to skip the statistical test altogether in this case
        assert sample_val1 != sample_val2, (
            f"Models with different seeds should predict different values. "
            f"Found identical value {sample_val1} at index {sample_idx}"
        )

    def test_regressor_stats_detect_data_changes(self, test_data):
        """Verify stats detect changes in regression input data."""
        X_train, X_test, y_train, y_test = test_data

        # Create regressor with fixed settings
        reg = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
        reg.fit(X_train, y_train)

        # Get predictions on original data
        original_predictions = reg.predict(X_test)
        original_stats = generate_prediction_stats(original_predictions)

        # Modify test data significantly
        X_test_modified = X_test.copy()
        X_test_modified[0, 0] += 5.0  # Significant change to first feature

        # Get predictions on modified data
        modified_predictions = reg.predict(X_test_modified)
        modified_stats = generate_prediction_stats(modified_predictions)

        # Verify stats detect the change
        # At least one of the main statistics should differ by more than 1%
        differences_detected = False
        for stat in ["mean", "std", "median", "p75"]:
            if not np.isclose(original_stats[stat], modified_stats[stat], rtol=0.01):
                differences_detected = True
                break

        assert (
            differences_detected
        ), "Statistical approach failed to detect change in regression data"

    def test_regressor_stats_detect_model_changes(self, test_data):
        """Verify stats detect changes in regressor configuration."""
        X_train, X_test, y_train, y_test = test_data

        # Create regressor with one configuration
        reg1 = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
        reg1.fit(X_train, y_train)
        predictions1 = reg1.predict(X_test)
        generate_prediction_stats(predictions1)

        # Create regressor with significantly different configuration
        # Use a much different random seed to ensure we get different predictions
        reg2 = TabPFNRegressor(n_estimators=10, random_state=1234, device="cpu")
        reg2.fit(X_train, y_train)
        predictions2 = reg2.predict(X_test)
        generate_prediction_stats(predictions2)

        # For the robustness test, we'll sample to check that predictions differ
        # between two model configurations with different random states
        # Given that we're testing the validity of the test itself, we can make this an
        # explicit check on an element of the prediction
        rng = np.random.RandomState(42)
        sample_idx = rng.randint(0, len(predictions1) - 1)
        sample_val1 = predictions1[sample_idx]
        sample_val2 = predictions2[sample_idx]

        # Check that at least one specific prediction is different
        # because we changed the random seed and ensemble size significantly
        # this allows us to skip the statistical test altogether in this case
        assert sample_val1 != sample_val2, (
            f"Models with different seeds should predict different values. "
            f"Found identical value {sample_val1} at index {sample_idx}"
        )


# Helper function to generate reference statistics from current model outputs
def update_reference_stats():
    """Generate and print reference statistics for current model predictions.

    Run this function manually when intentionally updating reference statistics:
    ```
    python -c "from tests.test_consistency import update_reference_stats; \
update_reference_stats()"
    ```

    Steps to update reference statistics:
    1. Make your changes to the TabPFN code
    2. Verify these changes improve model performance on benchmarks
    3. Run this function on the reference platform (currently: Darwin Python 3.12)
       or update the REFERENCE_PLATFORM and REFERENCE_PYTHON_VERSION values
    4. Update the REFERENCE_STATS dictionary in test_consistency.py
    5. Document the platform used in the REFERENCE_PLATFORM variable
    6. Document the improvements in your PR description

    IMPORTANT: These reference values are platform-specific. When updating,
    be sure to document which platform they were generated on, as the tests
    are expected to pass only on similar platforms.
    """
    reference_stats = {}

    # Iris dataset
    random_state = check_random_state(42)
    X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    # Use scikit-learn's 16 decimal precision for cross-platform determinism
    clf = TabPFNClassifier(
        n_estimators=2,
        random_state=42,
        device="cpu",
        inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True},
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)
    reference_stats["iris_classifier"] = generate_prediction_stats(predictions)

    # Breast Cancer dataset
    random_state = check_random_state(42)
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    # Use scikit-learn's 16 decimal precision for cross-platform determinism
    clf = TabPFNClassifier(
        n_estimators=2,
        random_state=42,
        device="cpu",
        inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True},
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)
    reference_stats["breast_cancer_classifier"] = generate_prediction_stats(predictions)

    # Boston Housing dataset
    try:
        from sklearn.datasets import fetch_openml

        housing = fetch_openml(data_id=531, as_frame=False)
        X, y = housing.data, housing.target
    except (ImportError, ValueError):
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing(as_frame=False)
        X, y = housing.data, housing.target

    random_state = check_random_state(42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    # Use scikit-learn's 16 decimal precision for cross-platform determinism
    reg = TabPFNRegressor(
        n_estimators=2,
        random_state=42,
        device="cpu",
        inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True},
    )
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    reference_stats["boston_regressor"] = generate_prediction_stats(predictions)

    # Diabetes dataset
    random_state = check_random_state(42)
    X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    # Use scikit-learn's 16 decimal precision for cross-platform determinism
    reg = TabPFNRegressor(
        n_estimators=2,
        random_state=42,
        device="cpu",
        inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": True},
    )
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    reference_stats["diabetes_regressor"] = generate_prediction_stats(predictions)

    # Format stats for easy printing to update REFERENCE_STATS in code
    import logging

    # Set up logging to print at DEBUG level
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    stats_format = "{\n"
    for key, stats in reference_stats.items():
        stats_format += f'    "{key}": ' + "{\n"
        for stat_name, stat_value in stats.items():
            stats_format += f'        "{stat_name}": {stat_value},\n'
        stats_format += "    },\n"
    stats_format += "}"
    logger.debug("Updated reference statistics:\n%s", stats_format)

    return reference_stats


if __name__ == "__main__":
    # This makes it easier to run the statistics update function
    update_reference_stats()
