#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for validation.ensure_compatible_fit_inputs function."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import OrdinalEncoder

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing import (
    clean as clean_module,
    clean_data,
)
from tabpfn.preprocessing.clean import (
    _is_single_float_block,
    _owned_float64_values,
    expand_date_and_text_features,
    fix_dtypes,
    process_text_na_dataframe,
    stringify_datetime_columns,
)
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder
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


def _get_schema(
    n_numerical_features: int = 0,
    n_categorical_features: int = 0,
    n_text_features: int = 0,
    n_constant_features: int = 0,
) -> FeatureSchema:
    features = []
    for i in range(n_numerical_features):
        features.append(
            Feature(name=f"feature_{i}", modality=FeatureModality.NUMERICAL)
        )
    for i in range(n_categorical_features):
        features.append(
            Feature(name=f"feature_{i}", modality=FeatureModality.CATEGORICAL)
        )
    for i in range(n_text_features):
        features.append(Feature(name=f"feature_{i}", modality=FeatureModality.TEXT))
    for i in range(n_constant_features):
        features.append(Feature(name=f"feature_{i}", modality=FeatureModality.CONSTANT))
    return FeatureSchema(features=features)


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
        ("input_data", "modalities"),
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
        modalities: dict[FeatureModality, list[int]],
    ) -> None:
        """Test that different input types are correctly tagged and sanitized."""
        schema = _get_schema(
            n_numerical_features=len(modalities[FeatureModality.NUMERICAL]),
            n_categorical_features=len(modalities[FeatureModality.CATEGORICAL]),
        )
        X_out, ord_encoder, _, _ = clean_data(
            X=input_data,
            feature_schema=schema,
        )
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == input_data.shape
        assert X_out.dtype == np.float64
        assert ord_encoder is not None

    @pytest.mark.parametrize(
        ("input_data", "modalities"),
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
        modalities: dict[FeatureModality, list[int]],
    ) -> None:
        """Test that pandas DataFrames are correctly processed and tagged."""
        schema = _get_schema(
            n_numerical_features=len(modalities[FeatureModality.NUMERICAL]),
            n_categorical_features=len(modalities[FeatureModality.CATEGORICAL]),
        )
        X_out, ord_encoder, _, _ = clean_data(
            X=input_data.values,
            feature_schema=schema,
        )

        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == input_data.shape
        assert X_out.dtype == np.float64

        assert ord_encoder is not None

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
                "amount": [10.2, 20.4, 20.5],
                "type": ["guest", "member", pd.NA],
            }
        )
        schema = FeatureSchema(
            features=[
                Feature(name="ratio", modality=FeatureModality.NUMERICAL),
                Feature(name="risk", modality=FeatureModality.TEXT),
                Feature(name="amount", modality=FeatureModality.NUMERICAL),
                Feature(name="type", modality=FeatureModality.CATEGORICAL),
            ]
        )
        X_out_first, _, feature_schema_first, _ = clean_data(
            X=df.values,
            feature_schema=schema,
        )
        X_out_second, _, feature_schema_second, _ = clean_data(
            X=X_out_first,
            feature_schema=schema,
        )

        # If the column order is preserved, the data should be the same.
        # If not, this test will fail.
        np.testing.assert_equal(X_out_first, X_out_second)

        # Note that depending on the settings for max_unique_for_category and
        # min_unique_for_numerical, the modalities may be different if
        # auto-detecting them on an ordinally encoded data frame.
        assert feature_schema_first.features == feature_schema_second.features


def test__classifier_fit_predict__all_missing_categorical_then_strings() -> None:
    """End-to-end regression for the ordinal-encoder fit/predict dtype asymmetry.

    A categorical column that is all-missing at fit but has real string values at
    predict used to crash with `could not convert string to float`.
    """
    rng = np.random.default_rng(0)
    n = 30
    X_train = pd.DataFrame(
        {
            "num0": rng.normal(size=n),
            "num1": rng.normal(size=n),
            "cat": pd.Series([None] * n, dtype="object"),  # all-missing at fit
        }
    )
    y = (X_train["num0"] > 0).astype(int).to_numpy()

    X_test = pd.DataFrame(
        {
            "num0": rng.normal(size=10),
            "num1": rng.normal(size=10),
            "cat": rng.choice(["normal", "high", "low"], size=10),  # strings at predict
        }
    )

    model = TabPFNClassifier(n_estimators=2, random_state=42)
    model.fit(X_train, y)

    assert model.predict(X_test).shape == (X_test.shape[0],)
    assert model.predict_proba(X_test).shape == (X_test.shape[0], len(np.unique(y)))


def test__classifier_fit_predict__all_missing_declared_categorical_then_strings() -> (
    None
):
    """End-to-end regression for a *declared* categorical column that is all-missing."""
    rng = np.random.RandomState(0)
    n = 40
    X_fit = pd.DataFrame(
        {
            "a": rng.rand(n),
            "c": pd.Series([np.nan] * n, dtype="category"),  # all-missing at fit
        }
    )
    y = pd.Series((rng.rand(n) > 0.5).astype(int))

    X_pred = pd.DataFrame(
        {
            "a": rng.rand(6),
            "c": pd.Series(
                ["normal", "abn", "normal", "abn", "normal", "abn"], dtype="category"
            ),
        }
    )

    clf = TabPFNClassifier(
        device="cpu",
        n_estimators=1,
        ignore_pretraining_limits=True,
        categorical_features_indices=[1],  # "c" explicitly declared categorical
        random_state=0,
    )
    clf.fit(X_fit, y)

    proba = clf.predict_proba(X_pred)
    assert proba.shape == (len(X_pred), 2)
    assert np.isfinite(proba).all()


def test__classifier_fit__string_category_plus_nullable_dtype() -> None:
    """A string category alongside a pandas nullable dtype must not crash at fit.

    Regression for a fit-time crash (seen on the `home_credit` dataset): a pandas
    *nullable* extension dtype column (``Float64``/``Int64``/``boolean``) makes
    sklearn's ``check_array`` perform an early whole-frame ``astype`` during
    ``validate_data(..., dtype=None)``. That cast then tries to push a string-valued
    category column (e.g. the hash-like ``'0e63c0f0'``) to float64 and raises
    ``could not convert string to float``, surfaced as a ``TabPFNValidationError``.
    A plain string-category column on its own is fine; it only breaks once a
    nullable column forces the early conversion.
    """
    n = 20
    X = pd.DataFrame(
        {
            # pandas *nullable* extension dtype -> triggers the early whole-frame
            # conversion in sklearn check_array (Int64 / boolean reproduce this too).
            "nullable_num": pd.array(np.arange(n, dtype=float), dtype="Float64"),
            # string-valued category, like the hash-ish '0e63c0f0' in the data.
            "str_cat": pd.Categorical(["0e63c0f0", "xx"] * (n // 2)),
        }
    )
    y = np.array([0, 1] * (n // 2))

    clf = TabPFNClassifier(device="cpu", n_estimators=1, random_state=0)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()


def test__classifier_fit__string_category_plus_numpy_bool() -> None:
    """A string cat alongside a plain numpy ``bool`` column must not crash at fit."""
    n = 20
    X = pd.DataFrame(
        {
            "bool_feat": np.array([True, False] * (n // 2), dtype=bool),
            "str_cat": pd.Categorical(["0e63c0f0", "xx"] * (n // 2)),
        }
    )
    assert X["bool_feat"].dtype == np.dtype("bool")
    y = np.array([0, 1] * (n // 2))

    clf = TabPFNClassifier(device="cpu", n_estimators=1, random_state=0)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()


def test__classifier_fit__string_dtype_plus_numpy_bool() -> None:
    """A pandas ``string`` column alongside a numpy ``bool`` must not crash at fit."""
    n = 20
    X = pd.DataFrame(
        {
            "bool_feat": np.array([True, False] * (n // 2), dtype=bool),
            "txt": pd.array(["0e63c0f0", "xx"] * (n // 2), dtype="string"),
        }
    )
    assert X["bool_feat"].dtype == np.dtype("bool")
    assert isinstance(X["txt"].dtype, pd.StringDtype)
    y = np.array([0, 1] * (n // 2))

    clf = TabPFNClassifier(device="cpu", n_estimators=1, random_state=0)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()


def test__classifier_predict__numeric_against_string_fit_categories() -> None:
    """A column that is string at fit but numeric at predict must not crash.

    Regression for a predict-time crash (seen on the `anes_voting` dataset). TabPFN
    builds an ``OrdinalEncoder`` at fit time and *freezes* the columns it applies to.
    If a column is string-typed at fit, its ``categories_`` are stored as a
    string/object array. When the *same* column position is numeric (float) at
    predict, ``OrdinalEncoder._transform`` used to compare float values against string
    categories and raise a ``TypeError`` (``'<' not supported between 'float' and
    'str'`` / ``ufunc 'isnan' not supported``).

    With `USE_TEXT` on and more distinct values than
    `MAX_UNIQUE_FOR_CATEGORICAL_FEATURES`, the column is detected as `TEXT` and
    encoded into numeric features before the ordinal encoder ever sees it. The
    predict-time floats are stringified and scored
    against the vocabulary fitted on train, which shares nothing with them, so
    nothing reaches the fit/predict drift check as a mismatch and no warning is
    raised. With `USE_TEXT` off (the default), the column instead reaches the
    ordinal encoder, which is
    `test__process_text_na_dataframe__numeric_against_string_fit_categories` below.
    """
    n, n_unique = 120, 60
    y = np.array([0, 1] * (n // 2))

    # Fit: 'code' is string-typed -> encoder.categories_ become strings.
    X_fit = pd.DataFrame(
        {
            "num": np.arange(n, dtype="float64"),
            "code": pd.array([f"s{i % n_unique}" for i in range(n)], dtype="string"),
        }
    )
    # Predict: the *same* column is now numeric float (a later time-split fold
    # codes the same field as numbers) -> float values vs string categories_.
    X_pred = pd.DataFrame(
        {
            "num": np.arange(n, dtype="float64"),
            "code": np.array([float(i % n_unique) for i in range(n)], dtype="float64"),
        }
    )

    clf = TabPFNClassifier(
        device="cpu",
        n_estimators=1,
        random_state=0,
        inference_config={"USE_TEXT": True},
    )
    clf.fit(X_fit, y)

    proba = clf.predict_proba(X_pred)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()
    # The column was encoded as text, so it reaches the model as numeric features
    # rather than as a category the frozen encoder has to recognise.
    assert {kind for kind, _, _ in clf.expansion_encoders_.values()} == {"text"}


def test__classifier_predict__numpy_array_against_string_fit_categories() -> None:
    """Predicting with a numpy array (no column names) after a named-DataFrame fit.

    ``validate_data`` converts both fit and predict inputs to numpy before
    ``fix_dtypes`` wraps them into an integer-column DataFrame, so the frozen encoder's
    columns are integer positions and match a numpy-array predict input. This guards
    against a ``KeyError`` when aligning predict-time dtypes against the fit categories.

    The array is given the fit-time column names before conversion, so it survives the
    text column having been expanded into several features.
    """
    n, n_unique = 120, 60
    y = np.array([0, 1] * (n // 2))

    # Fit with *named* columns; 'code' is string-categorical.
    X_fit = pd.DataFrame(
        {
            "num": np.arange(n, dtype="float64"),
            "code": pd.array([f"s{i % n_unique}" for i in range(n)], dtype="string"),
        }
    )
    # Predict with a bare numpy array; the 'code' position is now numeric.
    X_pred = np.column_stack(
        [
            np.arange(n, dtype="float64"),
            np.array([float(i % n_unique) for i in range(n)], dtype="float64"),
        ]
    )

    clf = TabPFNClassifier(device="cpu", n_estimators=1, random_state=0)
    clf.fit(X_fit, y)

    proba = clf.predict_proba(X_pred)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()


def test__process_text_na_dataframe__numeric_against_string_fit_categories() -> None:
    """The predict-time fix isolated to ``clean.process_text_na_dataframe`` (no model).

    Same root cause as
    ``test__classifier_predict__numeric_against_string_fit_categories``, narrowed to the
    component that owns predict-time cleaning: fit the ordinal encoder on string
    categories, then transform a frame whose ``code`` column arrives numeric. The
    mismatched column must be coerced to string (with a warning) and its unseen values
    encoded as the unknown code (``-1``), instead of crashing.
    """
    n, n_unique = 120, 60
    X_fit = pd.DataFrame(
        {
            "num": np.arange(n, dtype="float64"),
            "code": pd.array([f"s{i % n_unique}" for i in range(n)], dtype="string"),
        }
    )
    X_pred = pd.DataFrame(
        {
            "num": np.arange(n, dtype="float64"),
            "code": np.array([float(i % n_unique) for i in range(n)], dtype="float64"),
        }
    )

    encoder = get_ordinal_encoder()
    # Fit freezes the 'code' column -> string categories_.
    process_text_na_dataframe(X_fit.copy(), ord_encoder=encoder, fit_encoder=True)

    with pytest.warns(UserWarning, match="differs.*from fit time"):
        out = process_text_na_dataframe(X_pred.copy(), ord_encoder=encoder)

    # Column order is preserved: out[:, 1] is 'code', all unseen -> unknown code -1.
    assert out.shape == (n, 2)
    assert (out[:, 1] == -1).all()


def test__process_text_na_dataframe__string_against_numeric_fit_categories() -> None:
    """Reverse direction: numeric at fit, string at predict -> coerce to fit (numeric).

    A column that was numeric-categorical at fit is interpreted as that fit dtype at
    predict: numeric-looking strings (``"1"``) now match their fit category (previously
    they were all treated as unseen). A non-numeric string (``"abc"``) is coerced to
    ``NaN``; for a column with no missing values at fit that encodes to the unknown
    code, as before. A warning is emitted for the dtype change.
    """
    n, n_unique = 120, 4
    X_fit = pd.DataFrame(
        {
            "num": np.arange(n, dtype="float64"),
            # numeric categorical -> encoder.categories_ are numeric (int64).
            "code": pd.Categorical([i % n_unique for i in range(n)]),
        }
    )
    # Predict: same field now arrives as strings; index 0 is non-numeric.
    codes = [str(i % n_unique) for i in range(n)]
    codes[0] = "abc"
    X_pred = pd.DataFrame(
        {
            "num": np.arange(n, dtype="float64"),
            "code": pd.array(codes, dtype="string"),
        }
    )

    encoder = get_ordinal_encoder()
    process_text_na_dataframe(X_fit.copy(), ord_encoder=encoder, fit_encoder=True)

    with pytest.warns(UserWarning, match="differs.*from fit time"):
        out = process_text_na_dataframe(X_pred.copy(), ord_encoder=encoder)

    assert out.shape == (n, 2)
    # 'code' is column index 1. Numeric-looking strings now match a fit category and
    # encode to a valid (non-negative) code; the non-numeric "abc" -> NaN -> unknown.
    assert out[0, 1] == -1
    assert (out[1:, 1] >= 0).all()


# --- all-numeric fast path ------------------------------------------------------
#
# A numeric ndarray with no categorical columns has nothing to ordinally encode, so
# `clean_data` converts it straight to float64 instead of building the intermediate
# DataFrame and copying it back out. These pin the properties that shortcut has to
# keep: same values, same layout, an owned array, and an encoder that is fitted
# exactly as the general path leaves it.


def _numeric_schema(n_cols: int, cat_indices: tuple[int, ...] = ()) -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(
                name=f"f{i}",
                modality=FeatureModality.CATEGORICAL
                if i in cat_indices
                else FeatureModality.NUMERICAL,
            )
            for i in range(n_cols)
        ]
    )


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "bool", "float16"])
@pytest.mark.parametrize("passthrough_inf", [False, True])
def test__clean_data__numeric_array_converts_without_intermediate_copy(
    dtype: str, *, passthrough_inf: bool
) -> None:
    """The shortcut returns exactly the float64 cast of its input."""
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((20, 4)) * 3).astype(dtype)

    out, encoder, schema, _ = clean_data(
        X=X, feature_schema=_numeric_schema(4), passthrough_inf=passthrough_inf
    )

    np.testing.assert_array_equal(out, X.astype(np.float64))
    assert out.dtype == np.float64
    assert schema.num_columns == 4
    # Owned and writeable: callers mutate the cleaned array in place downstream.
    assert not np.shares_memory(out, X)
    assert out.flags.writeable
    # Nothing was selected for encoding, so the encoder learned no categories.
    assert not hasattr(encoder.named_transformers_["encoder"], "categories_")
    assert encoder.n_features_in_ == 4


@pytest.mark.parametrize("passthrough_inf", [False, True])
def test__clean_data__non_finite_survive_the_numeric_shortcut(
    *, passthrough_inf: bool
) -> None:
    """NaN and +/-inf come through the shortcut at their original positions."""
    X = np.arange(12, dtype="float32").reshape(4, 3)
    X[0, 1] = np.nan
    X[1, 2] = np.inf
    X[3, 0] = -np.inf

    out, _, _, _ = clean_data(
        X=X, feature_schema=_numeric_schema(3), passthrough_inf=passthrough_inf
    )

    np.testing.assert_array_equal(out, X.astype(np.float64))


def test__clean_data__numeric_shortcut_matches_the_encoder_path() -> None:
    """The shortcut and the general path agree, value for value.

    A declared categorical column forces the same data down the general path, so
    the only difference between the two calls is the route taken.
    """
    rng = np.random.default_rng(0)
    X = (rng.integers(0, 4, size=(50, 3))).astype("float64")

    shortcut, _, _, _ = clean_data(X=X, feature_schema=_numeric_schema(3))
    general, encoder, _, _ = clean_data(
        X=X, feature_schema=_numeric_schema(3, cat_indices=(0,))
    )

    # Column 0 is ordinally encoded in the general call; its codes happen to equal
    # the original small integers, so the two agree everywhere.
    assert encoder.named_transformers_["encoder"].categories_[0].size == 4
    np.testing.assert_array_equal(shortcut, general)
    assert shortcut.dtype == general.dtype
    assert shortcut.flags.f_contiguous == general.flags.f_contiguous


def test__clean_data__numeric_string_column__is_parsed_not_alphabetized() -> None:
    """Regression for the bug found this session: a column already correctly
    detected `NUMERICAL` was, until now, never actually parsed -- it fell through
    to the ordinal encoder and was encoded by alphabetical rank, discarding its
    real numeric order (`"10"` sorts before `"2"`).
    """
    values = ["1", "2", "3", "10", "20", "30", "100", "200"]
    X = np.array([[v] for v in values], dtype=object)
    schema = FeatureSchema(
        features=[Feature(name="f0", modality=FeatureModality.NUMERICAL)]
    )

    out, _, _, _ = clean_data(X=X, feature_schema=schema)

    np.testing.assert_array_equal(
        out.ravel(), [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0]
    )


def test__clean_data__numeric_string_column__unparseable_values_become_nan() -> None:
    values = ["1", "2", "not a number", "4"]
    X = np.array([[v] for v in values], dtype=object)
    schema = FeatureSchema(
        features=[Feature(name="f0", modality=FeatureModality.NUMERICAL)]
    )

    out, _, _, _ = clean_data(X=X, feature_schema=schema)

    np.testing.assert_array_equal(out.ravel()[:2], [1.0, 2.0])
    assert np.isnan(out.ravel()[2])
    assert out.ravel()[3] == 4.0


class TestExpandDateAndTextFeatures:
    """`expand_date_and_text_features`, the mechanism replacing `InputTypeConverter`."""

    def _schema(self, modality: FeatureModality) -> FeatureSchema:
        return FeatureSchema(
            features=[
                Feature(name="num", modality=FeatureModality.NUMERICAL),
                Feature(name="col", modality=modality),
            ]
        )

    def test__no_date_or_text_columns__is_a_no_op(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        schema = FeatureSchema(
            features=[Feature(name="a", modality=FeatureModality.NUMERICAL)] * 2
        )
        out, out_schema, fitted = expand_date_and_text_features(X, schema)
        assert out is X
        assert out_schema is schema
        assert fitted == {}

    def test__date_column__use_text_irrelevant__always_expands(self) -> None:
        """A `DATE` feature only exists when `USE_DATES` already decided to expand
        it (see `detect_feature_modalities`), so it is unconditional here.
        """
        n = 60
        X = np.array(
            [
                [float(i), d]
                for i, d in enumerate(
                    pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
                )
            ],
            dtype=object,
        )
        out, schema, fitted = expand_date_and_text_features(
            X, self._schema(FeatureModality.DATE), use_text=False
        )
        assert out.shape[1] > 2
        assert schema.num_columns == out.shape[1]
        assert all(f.modality is FeatureModality.NUMERICAL for f in schema.features)
        assert list(fitted.keys()) == [1]
        assert fitted[1][0] == "date"

    def test__text_column__use_text_off__is_unchanged(self) -> None:
        n = 60
        X = np.array(
            [[float(i), f"free text number {i} words words"] for i in range(n)],
            dtype=object,
        )
        out, schema, fitted = expand_date_and_text_features(
            X, self._schema(FeatureModality.TEXT), use_text=False
        )
        assert out is X
        assert schema.num_columns == 2
        assert fitted == {}

    def test__text_column__use_text_on__expands(self) -> None:
        n = 60
        X = np.array(
            [[float(i), f"free text number {i} words words"] for i in range(n)],
            dtype=object,
        )
        out, schema, fitted = expand_date_and_text_features(
            X, self._schema(FeatureModality.TEXT), use_text=True
        )
        assert out.shape[1] > 2
        assert schema.num_columns == out.shape[1]
        assert list(fitted.keys()) == [1]
        assert fitted[1][0] == "text"

    def test__predict_time__reuses_fitted_encoder_without_a_schema(self) -> None:
        """The predict path passes `feature_schema=None`; only `fitted` matters."""
        n = 60
        dates = pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
        X_fit = np.array([[float(i), d] for i, d in enumerate(dates)], dtype=object)
        _, _, fitted = expand_date_and_text_features(
            X_fit, self._schema(FeatureModality.DATE), use_text=False
        )

        X_predict = np.array([[0.0, dates[0]], [1.0, dates[1]]], dtype=object)
        out, schema, fitted_out = expand_date_and_text_features(
            X_predict, None, fitted=fitted
        )
        assert out.shape[0] == 2
        assert schema is None
        assert fitted_out is fitted


def test__stringify_datetime_columns__no_datetime_columns__is_a_no_op() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    out = stringify_datetime_columns(X)
    assert out is X


def test__stringify_datetime_columns__formats_and_preserves_missing() -> None:
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "signed_on": pd.to_datetime(["2020-01-01", None, "2020-01-03"]),
        }
    )
    out = stringify_datetime_columns(X)
    assert out["signed_on"].tolist() == [
        "2020-01-01 00:00:00",
        np.nan,
        "2020-01-03 00:00:00",
    ]
    assert out["num"].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("use_dates", [False, True])
def test__classifier_fit_predict__native_datetime_column__does_not_crash(
    use_dates: bool,
) -> None:
    """Regression: mixing a native `datetime64` column with any other dtype used
    to crash with a numpy dtype-promotion error inside `validate_data`, regardless
    of `USE_DATES` -- this is the crash the original skrub-based PR set out to fix.
    """
    n = 50
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"num": rng.normal(size=n), "signed_on": pd.date_range("2020-01-01", periods=n)}
    )
    y = rng.integers(0, 2, n)

    clf = TabPFNClassifier(
        n_estimators=1, device="cpu", inference_config={"USE_DATES": use_dates}
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()


def test__fix_dtypes__numeric_array_stays_consolidated() -> None:
    """A numeric ndarray needs no per-column cast, so its frame stays one block.

    A fragmented frame has to be re-materialised by every later `to_numpy`, which
    is a full extra copy of the data.
    """
    frame = fix_dtypes(np.random.default_rng(0).standard_normal((8, 5)), None)

    assert _is_single_float_block(frame)


def test__fix_dtypes__mixed_numeric_dtypes_are_still_cast() -> None:
    """Skipping the no-op cast must not skip the casts that do something."""
    raw = pd.DataFrame(
        {
            "a": pd.array([1, 2, None], dtype="Int64"),
            "b": np.array([1.5, 2.5, 3.5], dtype="float32"),
            "c": [1.0, 2.0, 3.0],
        }
    )

    frame = fix_dtypes(raw, cat_indices=None)

    assert list(frame.dtypes) == [np.dtype("float64")] * 3
    assert np.isnan(frame["a"].to_numpy()[2])


# --- ownership of the identity path's output -------------------------------------
#
# The identity path hands its array straight to the caller, who writes into it, so it
# has to own it. How many copies that takes depends on the frame's block layout,
# which is what these two pin.


def _fragmented_float_frame(values: np.ndarray) -> pd.DataFrame:
    """A float frame held as one block per column, as a recast frame is left."""
    frame = pd.DataFrame(values)
    frame[frame.columns] = frame[frame.columns].astype(values.dtype)
    return frame


def test__owned_float64_values__assembles_a_multi_block_frame_in_one_buffer() -> None:
    """A frame of many blocks is assembled directly, never routed through `to_numpy`.

    Asserted through the frame rather than the result, because what the result looks
    like depends on the pandas: `to_numpy` consolidates the frame in place before
    pandas 3, so a run that went through it would cost a second full-size buffer and
    leave the frame holding one block instead of many.
    """
    values = np.random.default_rng(0).standard_normal((8, 5))
    frame = _fragmented_float_frame(values)
    assert not _is_single_float_block(frame)

    out = _owned_float64_values(frame)

    np.testing.assert_array_equal(out, values)
    assert out.flags.writeable
    assert out.flags.f_contiguous
    assert not np.shares_memory(out, values)
    # Still one block per column: nothing consolidated it on the way.
    assert not _is_single_float_block(frame)
    # And the buffer is the caller's alone -- writing into it cannot reach the frame.
    assert not any(
        np.shares_memory(out, frame.iloc[:, position].to_numpy(copy=False))
        for position in range(frame.shape[1])
    )


def test__owned_float64_values__copies_a_single_block_frame() -> None:
    """A single-block frame is handed out as a view of the array it was built from."""
    values = np.random.default_rng(0).standard_normal((8, 5))
    frame = pd.DataFrame(values, copy=False)
    assert _is_single_float_block(frame)
    assert np.shares_memory(frame.to_numpy(dtype=np.float64, copy=False), values)

    out = _owned_float64_values(frame)

    np.testing.assert_array_equal(out, values)
    assert out.flags.writeable
    assert out.flags.f_contiguous
    # Writing into the result must not reach the caller's array. Under copy-on-write
    # the view is read-only, but on pandas 2 it is writeable and aliases `values`.
    assert not np.shares_memory(out, values)
    out[0, 0] = 12345.0
    assert values[0, 0] != 12345.0


# --- the frozen shortcut's preconditions -----------------------------------------
#
# At predict time the encoding step is skipped when a fitted encoder selected no
# columns. Skipping `transform` skips its checks too, so the shortcut is only taken
# for a frame `transform` would have handled the same way.


def _float_frame(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame({name: np.array(values) for name, values in columns.items()})


def test__encoding_is_identity__an_unfitted_encoder_is_not_an_empty_selection() -> None:
    """An encoder that never selected anything because it was never fitted.

    It has no selection to read, which must not be mistaken for having selected no
    columns -- that would hand the frame back unencoded instead of failing.
    """
    with pytest.raises(AttributeError):
        clean_module._encoding_is_identity(
            _float_frame(a=[1.0, 2.0], b=[3.0, 4.0]),
            get_ordinal_encoder(),
            fit_encoder=False,
        )


def test__process_text_na_dataframe__frozen_shortcut_rejects_a_narrower_frame() -> None:
    """A width sklearn would refuse has to keep failing."""
    encoder = get_ordinal_encoder()
    encoder.fit(_float_frame(a=[1.0, 2.0], b=[3.0, 4.0], c=[5.0, 6.0]))

    with pytest.raises((ValueError, KeyError), match="c"):
        process_text_na_dataframe(
            _float_frame(a=[1.0, 2.0], b=[3.0, 4.0]), ord_encoder=encoder
        )


def test__process_text_na_dataframe__frozen_shortcut_reorders_like_sklearn() -> None:
    """Reordered columns come back in fit order, as `transform` returns them.

    `ColumnTransformer` selects by name, so it lines a reordered frame back up with
    the fit-time order. The shortcut returns the frame's own order, which for this
    frame is a different array -- so it must not be taken here.
    """
    fitted_on = _float_frame(a=[1.0, 2.0], b=[3.0, 4.0], c=[5.0, 6.0])
    encoder = get_ordinal_encoder()
    encoder.fit(fitted_on)

    out = process_text_na_dataframe(fitted_on[["a", "c", "b"]], ord_encoder=encoder)

    np.testing.assert_array_equal(out, fitted_on.to_numpy())


def test__process_text_na_dataframe__frozen_shortcut_taken_when_lined_up() -> None:
    """The shortcut still applies to the frame the encoder was fitted on."""
    frame = _float_frame(a=[1.0, 2.0], b=[3.0, 4.0])
    encoder = get_ordinal_encoder()
    encoder.fit(frame)

    with mock.patch.object(
        clean_module, "_owned_float64_values", wraps=clean_module._owned_float64_values
    ) as shortcut:
        out = process_text_na_dataframe(frame, ord_encoder=encoder)

    assert shortcut.call_count == 1
    np.testing.assert_array_equal(out, frame.to_numpy())


# --- mixed-column assembly -------------------------------------------------------
#
# With categorical columns present, `clean_data` writes the encoded columns straight
# into the output array rather than letting `ColumnTransformer` stack its
# transformers' outputs and then reorder them. These pin what that has to preserve:
# the fitted encoder sklearn would have produced, and the values in the right places.


def _mixed_frame_inputs(
    n: int = 200, levels: int = 6
) -> tuple[np.ndarray, FeatureSchema]:
    """An object array of alternating numeric and low-cardinality string columns."""
    rng = np.random.default_rng(0)
    cols = 6
    X = np.empty((n, cols), dtype=object)
    numeric_cols, string_cols = range(0, cols, 2), range(1, cols, 2)
    X[:, numeric_cols] = rng.standard_normal((n, len(numeric_cols)))
    names = np.array([f"lvl_{i:02d}" for i in range(levels)], dtype=object)
    X[:, string_cols] = names[rng.integers(0, levels, (n, len(string_cols)))]
    return X, _numeric_schema(cols, cat_indices=tuple(string_cols))


def test__clean_data__mixed_columns_take_the_assembly_path() -> None:
    """The assembly is used, and its result is placed by column, not by transformer."""
    X, schema = _mixed_frame_inputs()

    with mock.patch(
        "tabpfn.preprocessing.clean._encode_into_preallocated",
        wraps=clean_module._encode_into_preallocated,
    ) as assembled:
        out, _encoder, _, _ = clean_data(X=X, feature_schema=schema)

    assert assembled.call_count == 1, "fell back to ColumnTransformer.fit_transform"
    # Numeric columns keep their own values, in their own positions -- the thing the
    # reorder used to be responsible for.
    for index in range(0, X.shape[1], 2):
        np.testing.assert_array_equal(out[:, index], X[:, index].astype(np.float64))
    # Encoded columns hold codes within the learned range, at their own positions.
    for index in range(1, X.shape[1], 2):
        assert set(np.unique(out[:, index])) <= set(range(6))
    assert out.dtype == np.float64
    assert out.flags.writeable


def test__clean_data__assembly_fits_the_encoder_sklearn_would_have() -> None:
    """The one-row structural fit plus a full refit must leave sklearn's own state.

    The encoder is stored on the estimator and drives predict, so any divergence here
    outlives the call that produced it.
    """
    X, schema = _mixed_frame_inputs()
    frame = fix_dtypes(
        X.copy(), cat_indices=schema.indices_for(FeatureModality.CATEGORICAL)
    )

    _, assembled, _, _ = clean_data(X=X, feature_schema=schema)
    reference = get_ordinal_encoder()
    reference.fit_transform(frame)

    assert reference.n_features_in_ == assembled.n_features_in_
    assert reference.output_indices_ == assembled.output_indices_
    assert reference.sparse_output_ == assembled.sparse_output_
    assert [(name, cols) for name, _, cols in reference.transformers_] == [
        (name, cols) for name, _, cols in assembled.transformers_
    ]
    for expected, got in zip(
        reference.named_transformers_["encoder"].categories_,
        assembled.named_transformers_["encoder"].categories_,
        strict=True,
    ):
        np.testing.assert_array_equal(expected, got)


def test__clean_data__assembly_matches_the_column_transformer_exactly() -> None:
    """Same values as routing the same frame through the encoder itself."""
    X, schema = _mixed_frame_inputs()
    frame = fix_dtypes(
        X.copy(), cat_indices=schema.indices_for(FeatureModality.CATEGORICAL)
    )

    assembled, _, _, _ = clean_data(X=X, feature_schema=schema)
    expected = get_ordinal_encoder().fit_transform(frame).astype(np.float64)

    np.testing.assert_array_equal(assembled, expected)


def test__clean_data__object_passthrough_falls_back_to_the_encoder() -> None:
    """A column the encoder skips but that is not float64 must not be assembled.

    `object` is not in the encoder's dtype selection, so it reaches the output through
    the passthrough; writing it into a float64 array is not the conversion the
    encoder's path would have done, so the assembly has to decline.
    """
    frame = pd.DataFrame(
        {
            "num": np.arange(6, dtype="float64"),
            "cat": pd.Categorical(["a", "b"] * 3),
            "obj": np.array([1, 2, 3, 4, 5, 6], dtype=object),
        }
    )
    encoder = get_ordinal_encoder()

    assert not clean_module._can_write_encoded_columns(
        frame, clean_module._encoder_selection(encoder.fit(frame.iloc[:1]))
    )
    # The real call still works, through sklearn, and learns the categories exactly
    # once. Declining the assembly *after* fitting them would pay for a second pass
    # over the data and then discard it, since `fit_transform` fits again itself.
    unpatched_fit = OrdinalEncoder.fit
    with mock.patch.object(
        OrdinalEncoder, "fit", autospec=True, side_effect=unpatched_fit
    ) as fitted:
        out = process_text_na_dataframe(
            frame, ord_encoder=get_ordinal_encoder(), fit_encoder=True
        )
    rows_per_fit = [len(call.args[1]) for call in fitted.call_args_list]
    assert rows_per_fit.count(len(frame)) == 1, (
        f"expected one fit over all {len(frame)} rows, got fits over {rows_per_fit}"
    )
    assert out.shape == (6, 3)
    assert out.dtype == np.float64


def test__cast_columns__a_shared_block_is_never_assigned_into() -> None:
    """A frame whose columns share a block must not be cast one `__setitem__` at a time.

    That spelling deletes each column out of the block it sits in, rebuilding the
    whole block every time, so the cast is quadratic in column count -- the reason
    the mixed mixes ran for minutes on the lowest supported pandas. It is the layout
    `fix_dtypes` hands the categorical cast: a freshly built object array.
    """
    values = np.empty((4, 6), dtype=object)
    values[:, :] = "lvl"
    frame = pd.DataFrame(values, copy=True)
    columns = frame.columns[1::2]
    if not clean_module._cast_columns_share_a_block(frame, columns):
        pytest.skip("this pandas gives an object array a block per column")

    with mock.patch.object(
        pd.DataFrame, "__setitem__", autospec=True, side_effect=AssertionError
    ):
        out = clean_module._cast_columns(frame, columns, "category")

    assert list(out.dtypes[1::2].map(str)) == ["category"] * 3
    assert list(out.dtypes[0::2].map(str)) == ["object"] * 3


def test__cast_columns__a_block_per_column_frame_keeps_its_own_columns() -> None:
    """The assignment route is taken there, and still must not reach the caller.

    `astype` is the wrong tool once every column has its own block: the delete it
    avoids is free, and building every cast column before assembling them holds a
    second full copy of the frame, which cost 47.7% more transient RSS on the
    `numeric-object` mix.
    """
    frame = _fragmented_float_frame(np.random.default_rng(0).standard_normal((8, 5)))
    assert len(frame._mgr.blocks) >= frame.shape[1]
    before = list(frame.dtypes)

    out = clean_module._cast_columns(frame, frame.columns[:3], "float32")

    assert [str(dtype) for dtype in out.dtypes] == ["float32"] * 3 + ["float64"] * 2
    assert list(frame.dtypes) == before


def test__cast_columns__a_shared_block_it_does_not_touch_does_not_count() -> None:
    """Only the blocks the cast columns sit in decide the route.

    A frame can hold a shared block and still cast cheaply, which is the shape
    pandas 3 hands the categorical cast: every string column in a block of its own,
    all the numeric ones consolidated into one. Routing that to `astype` would copy
    the consolidated block for nothing.
    """
    frame = pd.DataFrame(np.random.default_rng(0).standard_normal((8, 4)))
    frame["a"] = pd.array(range(8), dtype="Int64")
    frame["b"] = pd.array(range(8), dtype="Int64")
    assert clean_module._cast_columns_share_a_block(frame, frame.columns) is True
    assert clean_module._cast_columns_share_a_block(frame, ["a", "b"]) is False

    assignments = []
    original = pd.DataFrame.__setitem__

    def recording(self, key, value):  # noqa: ANN202
        assignments.append(key)
        return original(self, key, value)

    with mock.patch.object(pd.DataFrame, "__setitem__", recording):
        out = clean_module._cast_columns(frame, ["a", "b"], "float64")

    assert assignments, "the private columns should have been assigned, not astyped"
    assert [str(dtype) for dtype in out.dtypes[-2:]] == ["float64", "float64"]
    assert [str(dtype) for dtype in frame.dtypes[-2:]] == ["Int64", "Int64"]


def test__fix_dtypes__leaves_a_caller_s_frame_as_it_found_it() -> None:
    """Casting returns a new frame rather than writing into the one handed in."""
    X, schema = _mixed_frame_inputs()
    caller_frame = pd.DataFrame(X)
    before = list(caller_frame.dtypes)

    fix_dtypes(
        caller_frame, cat_indices=schema.indices_for(FeatureModality.CATEGORICAL)
    )

    assert list(caller_frame.dtypes) == before


def test__coerce_nullable_dtypes_to_numpy__converts_without_touching_the_input() -> (
    None
):
    """Nullable and boolean columns become float64; the caller's frame is untouched."""
    frame = pd.DataFrame(
        {
            "nullable": pd.array([1, None, 3], dtype="Int64"),
            "boolean": pd.array([True, False, None], dtype="boolean"),
            "text": pd.array(["a", "b", "c"], dtype="string"),
        }
    )

    out = clean_module.coerce_nullable_dtypes_to_numpy(frame)

    assert [str(dtype) for dtype in out.dtypes] == ["float64", "float64", "string"]
    np.testing.assert_array_equal(out["nullable"], [1.0, np.nan, 3.0])
    np.testing.assert_array_equal(out["boolean"], [1.0, 0.0, np.nan])
    assert [str(dtype) for dtype in frame.dtypes] == ["Int64", "boolean", "string"]


def test__fix_dtypes__duplicate_column_names_are_all_cast() -> None:
    """Duplicate labels collapse to one entry in the mapping, and both columns cast.

    Columns of different dtypes under one label remain unsupported -- selecting them
    by name takes both, so the cast that suits one is applied to the other and
    raises, before this change as after it.
    """
    frame = pd.DataFrame(np.array([[1, 2], [3, 4]], dtype=object))
    frame.columns = ["dup", "dup"]

    out = fix_dtypes(frame, cat_indices=None)

    assert [str(dtype) for dtype in out.dtypes] == ["float64", "float64"]
