"""Tests for validation.ensure_compatible_fit_inputs function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.initialization import tag_features_and_sanitize_data
from tabpfn.validation import ensure_compatible_fit_inputs


@pytest.fixture
def classifier() -> TabPFNClassifier:
    return TabPFNClassifier(n_estimators=1)


@pytest.fixture
def regressor() -> TabPFNRegressor:
    return TabPFNRegressor(n_estimators=1)


@pytest.fixture
def cpu_devices() -> tuple[torch.device, ...]:
    return (torch.device("cpu"),)


class TestEnsureCompatibleFitInputsBasic:
    """Tests for basic input handling."""

    def test__ensure_compatible_fit_inputs__numpy_arrays(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that numpy arrays are accepted and converted correctly."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        X, y, feature_names, n_features, original_y_name = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=classifier,
            max_num_samples=10_000,
            max_num_features=500,
            ignore_pretraining_limits=False,
            devices=cpu_devices,
        )

        assert X.shape == (3, 2)
        assert len(y) == 3
        assert n_features == 2
        assert feature_names is None
        assert original_y_name is None

    def test__ensure_compatible_fit_inputs__pandas_dataframe(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that pandas DataFrames preserve column names."""
        X = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0], "feature_b": [4.0, 5.0, 6.0]})
        y = np.array([0, 1, 0])

        _, _, feature_names, _, _ = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=classifier,
            max_num_samples=10_000,
            max_num_features=500,
            ignore_pretraining_limits=False,
            devices=cpu_devices,
        )

        assert list(feature_names) == ["feature_a", "feature_b"]  # type: ignore

    def test__ensure_compatible_fit_inputs__pandas_series_y(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that pandas Series y preserves its name."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = pd.Series([0, 1, 0], name="target_column")

        _, _, _, _, original_y_name = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=classifier,
            max_num_samples=10_000,
            max_num_features=500,
            ignore_pretraining_limits=False,
            devices=cpu_devices,
        )

        assert original_y_name == "target_column"


class TestEnsureCompatibleFitInputsValidation:
    """Tests for input validation and error handling."""

    def test__ensure_compatible_fit_inputs__too_many_features(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that exceeding max features raises an error."""
        X = np.random.default_rng(42).random((5, 10))
        y = np.array([0, 1, 0, 1, 0])

        with pytest.raises(TabPFNValidationError, match="Number of features"):
            ensure_compatible_fit_inputs(
                X,
                y,
                estimator=classifier,
                max_num_samples=10_000,
                max_num_features=5,  # Less than 10 features
                ignore_pretraining_limits=False,
                devices=cpu_devices,
            )

    def test__ensure_compatible_fit_inputs__too_many_samples(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that exceeding max samples raises an error."""
        X = np.random.default_rng(42).random((100, 2))
        y = np.array([0, 1] * 50)

        with pytest.raises(TabPFNValidationError, match="Number of samples"):
            ensure_compatible_fit_inputs(
                X,
                y,
                estimator=classifier,
                max_num_samples=50,  # Less than 100 samples
                max_num_features=500,
                ignore_pretraining_limits=False,
                devices=cpu_devices,
            )

    def test__ensure_compatible_fit_inputs__ignore_limits(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that ignore_pretraining_limits bypasses size checks."""
        X = np.random.default_rng(42).random((100, 10))
        y = np.array([0, 1] * 50)

        # Should not raise even though limits are exceeded
        X, *_ = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=classifier,
            max_num_samples=50,
            max_num_features=5,
            ignore_pretraining_limits=True,
            devices=cpu_devices,
        )

        assert X.shape == (100, 10)

    def test__ensure_compatible_fit_inputs__too_few_samples(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that providing only one sample raises an error."""
        X = np.array([[1.0, 2.0]])
        y = np.array([0])

        with pytest.raises(TabPFNValidationError, match="sample"):
            ensure_compatible_fit_inputs(
                X,
                y,
                estimator=classifier,
                max_num_samples=10_000,
                max_num_features=500,
                ignore_pretraining_limits=False,
                devices=cpu_devices,
            )

    def test__ensure_compatible_fit_inputs__mismatched_lengths(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that mismatched X and y lengths raise an error."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1])  # Only 2 elements, X has 3 rows

        with pytest.raises(TabPFNValidationError):
            ensure_compatible_fit_inputs(
                X,
                y,
                estimator=classifier,
                max_num_samples=10_000,
                max_num_features=500,
                ignore_pretraining_limits=False,
                devices=cpu_devices,
            )

    def test__ensure_compatible_fit_inputs__no_features(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that providing zero features raises an error."""
        X = np.array([[], [], []])
        y = np.array([0, 1, 0])

        with pytest.raises(TabPFNValidationError, match="feature"):
            ensure_compatible_fit_inputs(
                X,
                y,
                estimator=classifier,
                max_num_samples=10_000,
                max_num_features=500,
                ignore_pretraining_limits=False,
                devices=cpu_devices,
            )


class TestEnsureCompatibleFitInputsWithNaN:
    """Tests for handling NaN values."""

    def test__ensure_compatible_fit_inputs__nan_in_features(
        self, classifier: TabPFNClassifier, cpu_devices: tuple[torch.device, ...]
    ) -> None:
        """Test that NaN values in X are accepted."""
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        X, *_ = ensure_compatible_fit_inputs(
            X,
            y,
            estimator=classifier,
            max_num_samples=10_000,
            max_num_features=500,
            ignore_pretraining_limits=False,
            devices=cpu_devices,
        )

        assert np.isnan(X[0, 1])


class TestTagFeaturesAndSanitizeData:
    """Tests for tag_features_and_sanitize_data function with different input types."""

    # Note: min_samples_for_inference controls when auto-inference of categorical
    # features from numeric columns kicks in. We use 2, so tests need > 2 samples.
    MIN_SAMPLES_FOR_INFERENCE = 2
    MAX_UNIQUE_FOR_CATEGORY = 10
    MIN_UNIQUE_FOR_NUMERICAL = 4

    @pytest.mark.parametrize(
        ("input_data", "expected_modalities"),
        [
            pytest.param(
                np.array([[1.5, 2.3], [3.1, 4.7], [5.2, 6.8], [7.4, 8.1]]),
                {FeatureModality.NUMERICAL: [0, 1], FeatureModality.CATEGORICAL: []},
                id="float_array_all_numerical",
            ),
            pytest.param(
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                {FeatureModality.NUMERICAL: [0, 1], FeatureModality.CATEGORICAL: []},
                id="int_array_high_unique_numerical",
            ),
            pytest.param(
                np.array([[0, 1], [1, 0], [0, 1], [1, 0]]),
                {FeatureModality.NUMERICAL: [], FeatureModality.CATEGORICAL: [0, 1]},
                id="int_array_low_unique_categorical",
            ),
            pytest.param(
                np.array([[1.5, 0], [3.1, 1], [5.2, 0], [7.4, 1]]),
                {FeatureModality.NUMERICAL: [0], FeatureModality.CATEGORICAL: [1]},
                id="mixed_float_numerical_int_categorical",
            ),
            pytest.param(
                np.array(
                    [["a", "x"], ["b", "y"], ["a", "x"], ["b", "y"]], dtype=object
                ),
                {FeatureModality.NUMERICAL: [], FeatureModality.CATEGORICAL: [0, 1]},
                id="string_array_categorical",
            ),
            pytest.param(
                np.array(
                    [[1.5, "a"], [3.1, "b"], [5.2, "a"], [7.4, "b"]], dtype=object
                ),
                {FeatureModality.NUMERICAL: [0], FeatureModality.CATEGORICAL: [1]},
                id="mixed_numeric_string_object_array",
            ),
            pytest.param(
                np.array([[1.5, np.nan], [3.1, 4.7], [np.nan, 6.8], [7.4, 8.1]]),
                {FeatureModality.NUMERICAL: [0, 1], FeatureModality.CATEGORICAL: []},
                id="float_array_with_nan_numerical",
            ),
            pytest.param(
                np.array([[True, False], [False, True], [True, False], [False, True]]),
                {FeatureModality.NUMERICAL: [], FeatureModality.CATEGORICAL: [0, 1]},
                id="boolean_array_categorical",
            ),
        ],
    )
    def test__tag_features_and_sanitize_data__input_types(
        self,
        input_data: np.ndarray,
        expected_modalities: dict[FeatureModality, list[int]],
    ) -> None:
        """Test that different input types are correctly tagged and sanitized."""
        feature_names = [f"feature_{i}" for i in range(input_data.shape[1])]
        X_out, ord_encoder, feature_metadata = tag_features_and_sanitize_data(
            X=input_data,
            feature_names=feature_names,
            min_samples_for_inference=self.MIN_SAMPLES_FOR_INFERENCE,
            max_unique_for_category=self.MAX_UNIQUE_FOR_CATEGORY,
            min_unique_for_numerical=self.MIN_UNIQUE_FOR_NUMERICAL,
            provided_categorical_indices=None,
        )
        assert feature_metadata.feature_names == feature_names
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == input_data.shape
        assert X_out.dtype == np.float64
        assert ord_encoder is not None

        assert (
            feature_metadata.indices_for(FeatureModality.NUMERICAL)
            == expected_modalities[FeatureModality.NUMERICAL]
        )
        assert (
            feature_metadata.indices_for(FeatureModality.CATEGORICAL)
            == expected_modalities[FeatureModality.CATEGORICAL]
        )

    @pytest.mark.parametrize(
        ("input_data", "expected_modalities"),
        [
            pytest.param(
                pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}),
                {FeatureModality.NUMERICAL: [0], FeatureModality.CATEGORICAL: []},
                id="pandas_single_numerical_column",
            ),
            pytest.param(
                pd.DataFrame(
                    {"num": [1.5, 3.1, 5.2, 7.4], "cat": ["a", "b", "a", "b"]}
                ),
                {FeatureModality.NUMERICAL: [0], FeatureModality.CATEGORICAL: [1]},
                id="pandas_mixed_numerical_and_string",
            ),
            pytest.param(
                pd.DataFrame(
                    {
                        "cat1": pd.Categorical(["a", "b", "a", "b"]),
                        "cat2": pd.Categorical(["x", "y", "x", "y"]),
                    }
                ),
                {FeatureModality.NUMERICAL: [], FeatureModality.CATEGORICAL: [0, 1]},
                id="pandas_categorical_dtype",
            ),
            pytest.param(
                pd.DataFrame({"bin": [0, 1, 0, 1], "num": [1.5, 3.1, 5.2, 7.4]}),
                {FeatureModality.NUMERICAL: [1], FeatureModality.CATEGORICAL: [0]},
                id="pandas_binary_int_and_float",
            ),
        ],
    )
    def test__tag_features_and_sanitize_data__pandas_input(
        self,
        input_data: pd.DataFrame,
        expected_modalities: dict[FeatureModality, list[int]],
    ) -> None:
        """Test that pandas DataFrames are correctly processed and tagged."""
        X_out, ord_encoder, feature_metadata = tag_features_and_sanitize_data(
            X=input_data.values,
            feature_names=None,
            min_samples_for_inference=self.MIN_SAMPLES_FOR_INFERENCE,
            max_unique_for_category=self.MAX_UNIQUE_FOR_CATEGORY,
            min_unique_for_numerical=self.MIN_UNIQUE_FOR_NUMERICAL,
            provided_categorical_indices=None,
        )

        assert feature_metadata.feature_names == [None] * input_data.shape[1]
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == input_data.shape
        assert X_out.dtype == np.float64

        assert ord_encoder is not None

        assert (
            feature_metadata.indices_for(FeatureModality.NUMERICAL)
            == expected_modalities[FeatureModality.NUMERICAL]
        )
        assert (
            feature_metadata.indices_for(FeatureModality.CATEGORICAL)
            == expected_modalities[FeatureModality.CATEGORICAL]
        )

    # This test currently fails because the standard ColumnTransformer used inside
    # the ordinal encoder inside `tag_features_and_sanitize_data` is not preserving the
    # column order. We need to switch to the OrderPreservingColumnTransformer
    # to fix this. For now, we skipt this test and activate it once we switch
    # to the OrderPreservingColumnTransformer.
    @pytest.mark.skip
    def test__tag_features_and_sanitize_data__preserves_column_order(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "ratio": [0.4, 0.5, 0.6],
                "risk": ["High", None, "Low"],
                "height": ["Low", "Low", "Low"],
                "amount": [10.2, 20.4, 20.5],
                "type": ["guest", "member", pd.NA],
            }
        )
        X_out_first, _, feature_metadata_first = tag_features_and_sanitize_data(
            X=df.values,
            feature_names=None,
            min_samples_for_inference=self.MIN_SAMPLES_FOR_INFERENCE,
            max_unique_for_category=self.MAX_UNIQUE_FOR_CATEGORY,
            min_unique_for_numerical=self.MIN_UNIQUE_FOR_NUMERICAL,
            provided_categorical_indices=None,
        )
        X_out_second, _, feature_metadata_second = tag_features_and_sanitize_data(
            X=X_out_first,
            feature_names=None,
            min_samples_for_inference=self.MIN_SAMPLES_FOR_INFERENCE,
            max_unique_for_category=self.MAX_UNIQUE_FOR_CATEGORY,
            min_unique_for_numerical=self.MIN_UNIQUE_FOR_NUMERICAL,
            provided_categorical_indices=None,
        )

        # If the column order is preserved, the data should be the same.
        # If not, this test will fail.
        np.testing.assert_equal(X_out_first, X_out_second)

        # Note that depending on the settings for max_unique_for_category and
        # min_unique_for_numerical, the modalities may be different if
        # auto-detecting them on an ordinally encoded data frame.
        assert feature_metadata_first.features == feature_metadata_second.features
