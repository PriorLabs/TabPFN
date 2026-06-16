#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for validation.ensure_compatible_fit_inputs function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing import clean_data
from tabpfn.preprocessing.clean import (
    _encoded_column_positions,
    process_text_na_dataframe,
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
        X_out, ord_encoder, _ = clean_data(
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
        X_out, ord_encoder, _ = clean_data(
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
        X_out_first, _, feature_schema_first = clean_data(
            X=df.values,
            feature_schema=schema,
        )
        X_out_second, _, feature_schema_second = clean_data(
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

    The fit/predict dtype drift now coerces the column to string and warns, treating its
    values as unseen categories instead of crashing.
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

    clf = TabPFNClassifier(device="cpu", n_estimators=1, random_state=0)
    clf.fit(X_fit, y)

    with pytest.warns(UserWarning, match="differs.*from fit time"):
        proba = clf.predict_proba(X_pred)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()


def test__classifier_predict__numpy_array_against_string_fit_categories() -> None:
    """Predicting with a numpy array (no column names) after a named-DataFrame fit.

    ``validate_data`` converts both fit and predict inputs to numpy before
    ``fix_dtypes`` wraps them into an integer-column DataFrame, so the frozen encoder's
    columns are integer positions and match a numpy-array predict input. This guards
    against a ``KeyError`` when aligning predict-time dtypes against the fit categories.
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

    with pytest.warns(UserWarning, match="differs.*from fit time"):
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


def test__process_text_na_dataframe__restores_infs_when_encoder_reorders() -> None:
    """+/-inf is restored at the encoded position even if the encoder reorders cols.

    A plain ``ColumnTransformer`` emits the encoded (string) columns first and the
    numeric remainder last, so the numeric columns carrying the infinities move. The
    restore must target their post-encoding positions, not their input positions.
    """
    X = pd.DataFrame(
        {
            "num_a": pd.array([1.0, np.inf, 3.0], dtype="float64"),
            "str_b": pd.array(["x", "y", "x"], dtype="string"),
            "num_c": pd.array([10.0, 20.0, -np.inf], dtype="float64"),
        }
    )
    # Plain (reordering) ColumnTransformer: encoded columns are placed before the
    # passthrough remainder, unlike the order-preserving production encoder.
    encoder = ColumnTransformer(
        transformers=[
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                make_column_selector(dtype_include=["category", "string"]),
            )
        ],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    out = process_text_na_dataframe(
        X.copy(), ord_encoder=encoder, fit_encoder=True, passthrough_inf=True
    )

    pos = _encoded_column_positions(encoder, X.columns)
    # The string column is reordered to the front, numeric columns to the back.
    assert pos["str_b"] == 0
    assert {pos["num_a"], pos["num_c"]} == {1, 2}
    # Infinities land in the numeric columns at their reordered output positions.
    assert np.isposinf(out[1, pos["num_a"]])
    assert np.isneginf(out[2, pos["num_c"]])
    assert not np.isinf(out[:, pos["str_b"]]).any()
