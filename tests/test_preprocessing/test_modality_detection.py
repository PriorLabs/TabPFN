#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for feature type detection functionality."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing import modality_detection
from tabpfn.preprocessing.datamodel import (
    INPUT_FEATURE_PREFIX,
    Feature,
    FeatureModality,
    FeatureSchema,
)
from tabpfn.preprocessing.modality_detection import (
    _EARLY_EXIT_PREFIX_ROWS,
    _MAX_TEXT_COLUMNS_IN_WARNING,
    _detect_feature_modality,
    _underspecified_date_values,
    _warn_on_dates,
    _warn_on_texts,
    detect_feature_modalities,
)
from tabpfn.preprocessing.type_detection import infer_categorical_features


def test__detect_feature_modalities_basic():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "c", "a", "b"],
            "cat_num": [0, 1, 2, 1, 2],
            "text": ["longer", "texts", "appear", "here", "yay"],
            "const": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    feature_schema = detect_feature_modalities(
        X=df.values,
        feature_names=df.columns.tolist(),
        min_samples_for_inference=1,
        max_unique_for_category=3,
        min_unique_for_numerical=5,
        min_cardinality_for_text=3,
    )
    assert feature_schema.indices_for(FeatureModality.NUMERICAL) == [0]
    assert feature_schema.indices_for(FeatureModality.CATEGORICAL) == [1, 2]
    assert feature_schema.indices_for(FeatureModality.TEXT) == [3]
    assert feature_schema.indices_for(FeatureModality.CONSTANT) == [4]
    # Input column names are namespaced with the "input_" prefix so they cannot
    # collide with names generated for features added by preprocessing steps.
    assert feature_schema.feature_names == [f"input_{c}" for c in df.columns]


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
            np.array([["a", "x"], ["b", "y"], ["a", "x"], ["b", "y"]], dtype=object),
            {FeatureModality.NUMERICAL: [], FeatureModality.CATEGORICAL: [0, 1]},
            id="string_array_categorical",
        ),
        pytest.param(
            np.array([[1.5, "a"], [3.1, "b"], [5.2, "a"], [7.4, "b"]], dtype=object),
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
def test__detect_feature_modalities__input_types(
    input_data: np.ndarray,
    expected_modalities: dict[FeatureModality, list[int]],
) -> None:
    """Test that different input types are correctly tagged and sanitized."""
    feature_schema = detect_feature_modalities(
        X=input_data,
        feature_names=None,
        min_samples_for_inference=1,
        max_unique_for_category=3,
        min_unique_for_numerical=4,  # small so we detect numericals
        min_cardinality_for_text=3,
    )
    assert (
        feature_schema.indices_for(FeatureModality.NUMERICAL)
        == expected_modalities[FeatureModality.NUMERICAL]
    )
    assert (
        feature_schema.indices_for(FeatureModality.CATEGORICAL)
        == expected_modalities[FeatureModality.CATEGORICAL]
    )


def _for_test_detect_with_defaults(
    s: pd.Series,
    max_unique_for_category: int = 10,
    min_unique_for_numerical: int = 5,
    *,
    reported_categorical: bool = False,
    big_enough_n_to_infer_cat: bool = True,
    min_cardinality_for_text: int = 10,
) -> FeatureModality:
    return _detect_feature_modality(
        s,
        reported_categorical=reported_categorical,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        min_cardinality_for_text=min_cardinality_for_text,
        big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
    )


def _for_test_detect_modality(
    series_data: list[Any], test_name: str, expected: FeatureModality
) -> None:
    s = pd.Series(series_data)
    result = _for_test_detect_with_defaults(s)
    if result != expected:
        error = f"Expected {expected} but got {result} for {test_name}: {series_data}"
        raise AssertionError(error)


@pytest.mark.parametrize(
    ("series_data", "test_name"),
    [
        ([1.0, 1.0, 1.0, 1.0], "multiple floats"),
        ([1.0], "single float"),
        ([np.nan], "single NaN"),
        ([None], "single None"),
        (["a"], "single string"),
        ([True], "single boolean"),
        (["a", "a", "a", "a"], "multiple strings"),
        ([True, True, True, True], "multiple booleans"),
        ([], "empty"),
        ([np.nan, np.nan, np.nan, np.nan], "multiple NaN values"),
        ([np.nan, None, np.nan, None], "mixed NaN and None values"),
    ],
)
def test__detect_for_constant(series_data: list[Any], test_name: str) -> None:
    return _for_test_detect_modality(series_data, test_name, FeatureModality.CONSTANT)


@pytest.mark.parametrize(
    ("series_data", "test_name"),
    [
        (["a", "b", "c", "a", "b", "c", "c"], "multiple strings"),
        ([True, False, False, False], "multiple booleans"),
        (["True", "False", "True", "False"], "multiple boolean-like strings"),
        ([1.0, 0.0, 0.0, 1.0, 0.0], "multiple floats"),
        ([np.nan, 1.0, np.nan, 1.0], "constant value with missing ones"),
        ([0.0, 1.0, np.nan, 2.0], "multiple floats with missing ones"),
    ],
)
def test__detect_for_categorical(series_data: list[Any], test_name: str) -> None:
    return _for_test_detect_modality(
        series_data, test_name, FeatureModality.CATEGORICAL
    )


def test__numerical_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.NUMERICAL


def test__numerical_series_from_strings():
    s = pd.Series(
        ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]
    )
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.NUMERICAL


def test__detect_numerical_as_string_with_nulls():
    # Note that in pandas 3.0, None and np.nan both become pd.NA, so n_unique=4.
    # Ideally tests shouldn't depend on pandas version
    s = pd.Series([None, np.nan, "1.0", "2.0", "3.0"])
    result = _for_test_detect_with_defaults(s, min_unique_for_numerical=4)
    assert result == FeatureModality.NUMERICAL


def test__numerical_series_with_nan():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.NUMERICAL


def test__numerical_but_stored_as_string():
    s = pd.Series(
        ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]
    )
    s = s.astype(str)
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.NUMERICAL


def test__categorical_series():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL


def test__categorical_series_with_nan():
    s = pd.Series(["a", "b", "c", "a", "b", "c", np.nan])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL
    s = pd.Series(["a", "b", "c", "a", "b", "c", np.nan, None])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL
    s = pd.Series([None, np.nan, pd.NA, "house", "garden"])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL


def test__numerical_reported_as_categorical():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _for_test_detect_with_defaults(s, reported_categorical=True)
    assert result == FeatureModality.CATEGORICAL


def test__numerical_reported_as_categorical_but_too_many_unique_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _for_test_detect_with_defaults(
        s, reported_categorical=True, max_unique_for_category=9
    )
    assert result == FeatureModality.NUMERICAL


def test__detected_categorical_without_reporting():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = _for_test_detect_with_defaults(
        s, reported_categorical=False, min_unique_for_numerical=5
    )
    assert result == FeatureModality.CATEGORICAL

    # Even with floats, this should be categorical
    s = pd.Series([3.43, 3.54, 3.43, 3.53, 3.43, 3.54, 657.3])
    result = _for_test_detect_with_defaults(
        s, reported_categorical=False, min_unique_for_numerical=5
    )
    assert result == FeatureModality.CATEGORICAL


def test__detect_for_categorical_with_category_dtype():
    s = pd.Series(["a", "b", "c", "a", "b", "c"], dtype="category")
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL


def test__detect_textual_feature():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
    result = _for_test_detect_with_defaults(s, min_cardinality_for_text=2)
    assert result == FeatureModality.TEXT


def test__detect_long_texts():
    s = pd.Series(
        [
            "This is a long text",
            "Another long text here",
            "Yet another different text",
            "More text content",
            "Even more text",
            "Text continues",
            "More strings",
            "Additional text",
            "More content",
            "Final text",
            "Extra text",
            "Last one",
        ]
    )
    result = _for_test_detect_with_defaults(s, min_cardinality_for_text=2)
    assert result == FeatureModality.TEXT
    result = _for_test_detect_with_defaults(s, min_cardinality_for_text=15)
    assert result == FeatureModality.CATEGORICAL


def test__detect_text_as_object():
    s = pd.Series(["a", "b", "c", "e", "f"], dtype=object)
    s = s.astype(object)
    result = _for_test_detect_with_defaults(s, min_cardinality_for_text=2)
    assert result == FeatureModality.TEXT
    result = _for_test_detect_with_defaults(s, min_cardinality_for_text=15)
    assert result == FeatureModality.CATEGORICAL


@pytest.mark.parametrize(
    (
        "X",
        "provided",
        "min_samples_for_inference",
        "max_unique_for_category",
        "min_unique_for_numerical",
        "expected",
    ),
    [
        pytest.param(
            np.array([[np.nan, "NA"]], dtype=object).reshape(-1, 1),
            [0],
            0,
            2,
            5,
            [0],
            id="str_and_nan_provided_included",
        ),
        pytest.param(
            np.array([[np.nan], ["NA"], ["NA"]], dtype=object),
            [0],
            0,
            2,
            5,
            [0],
            id="str_and_nan_multiple_rows_provided_included",
        ),
        pytest.param(
            np.array([[1.0], [1.0], [np.nan]]),
            None,
            3,
            2,
            4,
            [],
            id="auto_inference_blocked_when_not_enough_samples",
        ),
        pytest.param(
            np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0], [np.nan, 9.0]]),
            None,
            3,
            3,
            4,
            [0],
            id="auto_inference_enabled_with_enough_samples",
        ),
        pytest.param(
            np.array([[0], [1], [2], [3], [np.nan]], dtype=float),
            [0],
            0,
            3,
            2,
            [],
            id="provided_column_excluded_if_exceeds_max_unique",
        ),
    ],
)
def test__infer_categorical_features(
    X: np.ndarray,
    provided: list[int] | None,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    expected: list[int],
):
    out_old_api = infer_categorical_features(
        X,
        provided=provided,
        min_samples_for_inference=min_samples_for_inference,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
    )
    feature_schema = detect_feature_modalities(
        X=X,
        feature_names=None,
        min_samples_for_inference=min_samples_for_inference,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        min_cardinality_for_text=30,
        provided_categorical_indices=provided,
    )
    assert (
        out_old_api
        == expected
        == feature_schema.indices_for(FeatureModality.CATEGORICAL)
    )


def test_infer_categorical_with_dict_raises_error():
    X = np.array([[{"a": 1}], [{"b": 2}]], dtype=object)
    with pytest.raises(TypeError):
        infer_categorical_features(
            X,
            provided=None,
            min_samples_for_inference=0,
            max_unique_for_category=2,
            min_unique_for_numerical=2,
        )


@pytest.mark.parametrize(
    ("series_data", "expected"),
    [
        # Prefix (1024 rows) already shows > 10 distinct -> decided early.
        (np.arange(5000, dtype=float), FeatureModality.NUMERICAL),
        # Prefix stays below the threshold -> exact full scan must run.
        (np.tile([1.0, 2.0, 3.0], 2000), FeatureModality.CATEGORICAL),
        (np.zeros(5000), FeatureModality.CONSTANT),
    ],
)
def test__long_columns_early_exit_decisions(
    series_data: np.ndarray, expected: FeatureModality
) -> None:
    """Columns longer than the early-exit prefix get the exact-count modality."""
    assert _for_test_detect_with_defaults(pd.Series(series_data)) == expected


def test__early_exit_not_fooled_by_uninformative_prefix():
    # The prefix is constant; only the tail is distinct. The prefix must not
    # decide anything -- the full scan has to run.
    s = pd.Series(
        np.concatenate([np.zeros(_EARLY_EXIT_PREFIX_ROWS), np.arange(1.0, 4000.0)])
    )
    assert _for_test_detect_with_defaults(s) == FeatureModality.NUMERICAL


def test__early_exit_accounts_for_min_cardinality_for_text() -> None:
    """The early-exit threshold must include `min_cardinality_for_text` too.

    Regression: `decided_at` only considered `max_unique_for_category` and
    `min_unique_for_numerical`. If `min_cardinality_for_text` is configured
    above both, a string column whose prefix already clears the first two, but
    not the text threshold, stopped scanning early and used the undercounted
    prefix value to decide category-vs-text -- silently leaving a genuinely
    high-cardinality column CATEGORICAL instead of TEXT.
    """
    # Prefix cycles through only 8 distinct values, clearing
    # max_unique_for_category (5) and min_unique_for_numerical (3) alone, but
    # the tail introduces new values the prefix never saw, pushing the true
    # count past min_cardinality_for_text (10).
    prefix = [f"v{i % 8}" for i in range(_EARLY_EXIT_PREFIX_ROWS)]
    tail = [f"new{i}" for i in range(10)]
    s = pd.Series(prefix + tail)

    result = _for_test_detect_with_defaults(
        s,
        max_unique_for_category=5,
        min_unique_for_numerical=3,
        min_cardinality_for_text=10,
    )
    assert result == FeatureModality.TEXT


def _text_schema(*names: str) -> FeatureSchema:
    """Schema of TEXT features with the `input_` prefix real input names carry."""
    return FeatureSchema(
        features=[
            Feature(name=f"{INPUT_FEATURE_PREFIX}{name}", modality=FeatureModality.TEXT)
            for name in names
        ]
    )


class TestWarnOnTexts:
    """Schema-level unit tests for `_warn_on_texts`."""

    def test__no_text_features__does_not_warn(self) -> None:
        schema = FeatureSchema(
            features=[
                Feature(name="input_a", modality=FeatureModality.NUMERICAL),
                Feature(name="input_b", modality=FeatureModality.CATEGORICAL),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_texts(schema)

    def test__text_features__warn_with_column_names_and_remedies(self) -> None:
        with pytest.warns(UserWarning, match="look like free text") as record:
            _warn_on_texts(_text_schema("review"))

        message = str(record[0].message)
        # Column names are shown as the user wrote them, without the input_ prefix.
        assert "'review'" in message
        assert INPUT_FEATURE_PREFIX not in message
        # The message must state all remedies.
        assert "numeric dtype" in message
        assert "https://github.com/PriorLabs/tabpfn-client" in message
        assert "categorical_features_indices" in message

    def test__declared_cat_indices__are_not_reported(self) -> None:
        schema = _text_schema("sku", "review")

        with pytest.warns(UserWarning, match="look like free text") as record:
            _warn_on_texts(schema, declared_cat_indices=[0])
        message = str(record[0].message)
        assert "'review'" in message
        assert "'sku'" not in message

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_texts(schema, declared_cat_indices=[0, 1])

    def test__many_text_columns__message_is_truncated(self) -> None:
        n_extra = 5
        n_columns = _MAX_TEXT_COLUMNS_IN_WARNING + n_extra
        schema = _text_schema(*(f"t{i}" for i in range(n_columns)))

        with pytest.warns(UserWarning, match="look like free text") as record:
            _warn_on_texts(schema)

        message = str(record[0].message)
        assert f"(and {n_extra} more)" in message
        assert f"'t{_MAX_TEXT_COLUMNS_IN_WARNING - 1}'" in message
        assert f"'t{_MAX_TEXT_COLUMNS_IN_WARNING}'" not in message


class TestWarnOnDates:
    """Schema-level unit tests for `_warn_on_dates`."""

    def test__demoted_dates__warn_with_column_names_and_remedies(self) -> None:
        with pytest.warns(UserWarning, match="hold dates") as record:
            _warn_on_dates(["signed_on"])
        message = str(record[0].message)
        assert "'signed_on'" in message
        assert "USE_DATES" in message
        assert "categorical_features_indices" in message

    def test__no_demoted_dates__does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_dates([])


class TestDetectFeatureModalitiesWarnsOnText:
    """`detect_feature_modalities` emits the text warning over real columns.

    The warning is now produced inside `detect_feature_modalities`, so these
    exercise the whole path: which columns actually get labelled TEXT and thus
    reach the warning, which the schema-level tests above cannot (they build
    schemas by hand).
    """

    n_rows = 200

    def _numeric_column(self) -> np.ndarray:
        return np.random.default_rng(0).normal(size=self.n_rows)

    def _detect(
        self, X: pd.DataFrame, declared: list[int] | None = None
    ) -> FeatureSchema:
        """Run modality detection over a frame, as `fit()` does."""
        return detect_feature_modalities(
            X=X.to_numpy(dtype=object),
            feature_names=list(X.columns),
            provided_categorical_indices=declared,
            min_samples_for_inference=100,
            max_unique_for_category=30,
            min_unique_for_numerical=4,
            min_cardinality_for_text=30,
        )

    def test__free_text_column__warns(self) -> None:
        X = pd.DataFrame(
            {
                "num": self._numeric_column(),
                "review": [f"review {i}, a fairly long sentence" for i in range(200)],
            }
        )

        with pytest.warns(UserWarning, match="look like free text") as record:
            self._detect(X)

        assert "'review'" in str(record[0].message)

    def test__ordinary_columns__do_not_warn(self) -> None:
        """Neither low-cardinality strings nor fully numeric strings are TEXT.

        The former are ordinary categoricals and the latter are
        detected NUMERICAL.
        """
        values = np.random.default_rng(1).normal(size=200)
        X = pd.DataFrame(
            {
                "num": self._numeric_column(),
                "color": ["red", "green", "blue"] * 66 + ["red", "red"],
                "as_str": [str(round(float(v), 4)) for v in values],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._detect(X)

    def test__numeric_column_with_one_stray_token__warns(self) -> None:
        """A single non-numeric token flips a whole numeric column to TEXT.

        `_is_numeric_pandas_series` requires *every* value to be coercible, so one
        stray "N/A" makes the column ordinal-encoded as a near-unique categorical.
        Warning here is the point of the feature: the fix is a numeric dtype.
        """
        values = np.random.default_rng(2).normal(size=200)
        mostly_numeric = [str(round(float(v), 4)) for v in values]
        mostly_numeric[7] = "N/A"
        X = pd.DataFrame(
            {"num": self._numeric_column(), "mostly_numeric": mostly_numeric}
        )

        with pytest.warns(UserWarning, match="look like free text") as record:
            self._detect(X)

        assert "'mostly_numeric'" in str(record[0].message)

    def test__column_with_a_crash_prone_token__does_not_crash(self) -> None:
        """A value that used to segfault `pandas.to_numeric` must not crash the fit.

        `"8e2569614270f3d8b9e7038efac9f116"` is a hash-like string whose leading
        digits read as scientific notation with an exponent in `[2**31, 2**32)`,
        which crashes `pandas.to_numeric` outright on some pandas/numpy versions.
        A crash cannot be caught with `pytest.raises`, so surviving this call at all
        is the assertion; the column is otherwise unremarkable free text.
        """
        values = [f"id_{i}" for i in range(200)]
        values[7] = "8e2569614270f3d8b9e7038efac9f116"
        X = pd.DataFrame({"num": self._numeric_column(), "ids": values})

        with pytest.warns(UserWarning, match="look like free text") as record:
            self._detect(X)

        assert "'ids'" in str(record[0].message)

    def test__declared_categorical_columns__do_not_warn(self) -> None:
        """Declaring a column categorical states intent, so it must stay quiet.

        Covers both a plain string column and an explicit pandas `category`
        dtype, each above the cardinality threshold.
        """
        X = pd.DataFrame(
            {
                "num": self._numeric_column(),
                "sku": [f"sku_{i % 60}" for i in range(200)],
                "sku_cat": pd.Series(
                    [f"sku_{i % 60}" for i in range(200)], dtype="category"
                ),
            }
        )
        declared = [1, 2]

        # Without the declaration the columns really are detected as TEXT and warn.
        with pytest.warns(UserWarning, match="look like free text"):
            self._detect(X)

        # Declaring them silences the warning; the columns are still labelled TEXT.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            schema = self._detect(X, declared)
        assert schema.indices_for(FeatureModality.TEXT) == declared


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_text_column__warns_at_call_site(estimator_cls: type) -> None:
    """`fit` runs `detect_feature_modalities`, so a free-text column warns.

    Both estimators share the detection path, so one parametrized test pins the
    estimator-level behaviour: `fit` emits the warning naming the column and
    blaming this file's `fit` call (the stacklevel), declaring the column in
    `categorical_features_indices` silences it, and `predict` stays quiet.
    """
    n = 120
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "review": [f"review {i}, a fairly long sentence" for i in range(n)],
        }
    )
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="look like free text") as record:
        model.fit(X, y)
    assert "'review'" in str(record[0].message)
    # Pins the stacklevel: the warning must blame this file's `fit` call, not a
    # frame inside tabpfn or the contextlib wrapper around `fit`.
    assert record[0].filename == __file__

    # Only `fit` runs modality detection, so `predict` must not warn again.
    # catch_warnings collects any warning instead of failing on unrelated ones.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(X)
    assert not [w for w in caught if "look like free text" in str(w.message)]

    model = estimator_cls(
        n_estimators=1, device="cpu", categorical_features_indices=[1]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    assert not [w for w in caught if "look like free text" in str(w.message)]


def test__category_and_text_thresholds__move_independently() -> None:
    """Numerical-vs-categorical and categorical-vs-text are separate decisions.

    A numeric column with 35 distinct values and a string column with 35
    distinct values, under the same `max_unique_for_category`, land on
    opposite sides of it for reasons that have nothing to do with each other:
    the number is simply above the categorical cutoff, the string is below
    the (independent) text cutoff. Coupling the two would make
    `min_cardinality_for_text` silently move whenever `max_unique_for_category`
    did.
    """
    rng = np.random.default_rng(0)
    numeric_column = rng.choice(35, size=200).astype(float)
    string_column = np.array([f"g{i % 35}" for i in range(200)], dtype=object)
    X = np.column_stack([numeric_column, string_column])

    schema = detect_feature_modalities(
        X=X,
        feature_names=["num", "str"],
        min_samples_for_inference=100,
        max_unique_for_category=30,  # below 35: the number is NUMERICAL
        min_unique_for_numerical=4,
        min_cardinality_for_text=40,  # above 35: the string is CATEGORICAL
    )

    assert schema.features[0].modality is FeatureModality.NUMERICAL
    assert schema.features[1].modality is FeatureModality.CATEGORICAL


class TestDateLikeColumnDetection:
    """`detect_feature_modalities` recognizing date-like string columns.

    Nothing expands a date into calendar features yet, so a recognized date is
    always demoted to whichever of CATEGORICAL/TEXT its cardinality implies --
    exactly the modality a non-date string of the same shape would get. The
    only observable difference recognizing it makes right now is the warning:
    a demoted date is named as a date, not reported as generic free text.
    """

    n_rows = 200

    def _numeric_column(self) -> np.ndarray:
        return np.random.default_rng(0).normal(size=self.n_rows)

    def _detect(
        self,
        X: pd.DataFrame,
        *,
        use_dates: bool = False,
        declared: list[int] | None = None,
    ) -> FeatureSchema:
        return detect_feature_modalities(
            X=X.to_numpy(dtype=object),
            feature_names=list(X.columns),
            provided_categorical_indices=declared,
            min_samples_for_inference=100,
            max_unique_for_category=30,
            min_unique_for_numerical=4,
            min_cardinality_for_text=30,
            use_dates=use_dates,
        )

    def _dates(self, n_unique: int) -> list[str]:
        pool = pd.date_range("2020-01-01", periods=n_unique).strftime("%Y-%m-%d")
        return [pool[i % n_unique] for i in range(self.n_rows)]

    def test__high_cardinality_date__is_text_like_a_same_shaped_string(self) -> None:
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(60)})
        with pytest.warns(UserWarning, match="hold dates"):
            schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.TEXT

    def test__low_cardinality_date__is_categorical(self) -> None:
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(4)})
        with pytest.warns(UserWarning, match="hold dates"):
            schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.CATEGORICAL

    def test__clock_time_column__is_not_a_date(self) -> None:
        """A bare time needs a year/month/day filled in to parse at all, and
        `pd.to_datetime` fills it in from today's date -- not a real calendar
        signal, and not reproducible across days, so it must not count.
        """
        times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
        clock = [times[i % len(times)] for i in range(self.n_rows)]
        X = pd.DataFrame({"num": self._numeric_column(), "clock": clock})
        # High cardinality, so it now reads as free text instead -- that warning
        # is expected here, unlike the "no warnings at all" cases below.
        with pytest.warns(UserWarning, match="look like free text"):
            schema = self._detect(X, use_dates=True)
        assert schema.features[1].modality is not FeatureModality.DATE

    def test__month_and_day_without_a_year__is_not_a_date(self) -> None:
        """Missing just the year is the same gap as a missing time: `pd.to_datetime`
        fills it in (from today's year, or a fixed one, depending on pandas
        version) rather than raising.
        """
        values = [f"May {(i % 28) + 1}" for i in range(self.n_rows)]
        X = pd.DataFrame({"num": self._numeric_column(), "date": values})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            schema = self._detect(X, use_dates=True)
        assert schema.features[1].modality is not FeatureModality.DATE

    def test__one_underspecified_value_among_full_dates__is_not_a_date(self) -> None:
        """All-or-nothing, like the rest of date detection: one bare time
        mixed into an otherwise fully-specified date column disqualifies it.
        """
        values = self._dates(60)
        values[0] = "12:00"
        X = pd.DataFrame({"num": self._numeric_column(), "date": values})
        schema = self._detect(X, use_dates=True)
        assert schema.features[1].modality is not FeatureModality.DATE

    def test__strict_iso_dates__skip_the_expensive_per_value_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Performance guard: the exact per-value check is expensive (two full
        `dateutil` parses each), so a column of plain ISO dates -- unambiguous
        by construction -- must never reach it.
        """

        def _fail(_value: object) -> bool:
            raise AssertionError("must not be called for strict ISO dates")

        monkeypatch.setattr(modality_detection, "_is_fully_specified_date_value", _fail)
        values = pd.Series(self._dates(60))
        assert _underspecified_date_values(values) == set()

    def test__use_dates__high_cardinality_date__stays_date_and_does_not_warn(
        self,
    ) -> None:
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(60)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            schema = self._detect(X, use_dates=True)
        assert schema.features[1].modality is FeatureModality.DATE

    def test__use_dates__low_cardinality_date__is_still_expanded(self) -> None:
        """`_demote_dates` never runs once `use_dates` is on, low-cardinality
        or not: every detected date is left for the caller to expand.
        """
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(3)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            schema = self._detect(X, use_dates=True)
        assert schema.features[1].modality is FeatureModality.DATE

    def test__demoted_date__is_not_also_reported_as_free_text(self) -> None:
        """The date warning fires; the free-text warning must not repeat it."""
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(60)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._detect(X)
        date_warnings = [w for w in caught if "hold dates" in str(w.message)]
        text_warnings = [w for w in caught if "look like free text" in str(w.message)]
        assert len(date_warnings) == 1
        assert not text_warnings

    def test__genuine_free_text__is_not_a_date(self) -> None:
        X = pd.DataFrame(
            {
                "num": self._numeric_column(),
                "review": [f"review {i}, a fairly long sentence" for i in range(200)],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.TEXT
        assert not [w for w in caught if "hold dates" in str(w.message)]

    def test__mostly_dates_with_one_bad_value__is_not_a_date(self) -> None:
        """All-or-nothing, like the numeric check: one bad value disqualifies it."""
        values = self._dates(60)
        values[0] = "not a date at all"
        X = pd.DataFrame({"num": self._numeric_column(), "date": values})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._detect(X)
        assert not [w for w in caught if "hold dates" in str(w.message)]

    def test__numeric_string_column__is_not_a_date(self) -> None:
        """The numeric check runs first and wins, even for an 8-digit date shape."""
        values = [f"2024010{i % 9 + 1}" for i in range(self.n_rows)]
        X = pd.DataFrame({"num": self._numeric_column(), "code": values})
        schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.NUMERICAL

    def test__mixed_date_and_datetime_string_formats__is_still_a_date(self) -> None:
        """Regression: naive `pd.to_datetime` infers a format from an early
        value and silently coerces a later, differently-shaped but valid value
        to NaT, which would otherwise fail the all-or-nothing date check.
        """
        values = self._dates(60)
        values[0] = "2020-06-15 13:45:30"  # date-only pool plus one full datetime
        X = pd.DataFrame({"num": self._numeric_column(), "date": values})
        with pytest.warns(UserWarning, match="hold dates"):
            schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.TEXT

    def test__declared_categorical_date_column__declaration_has_no_effect(
        self,
    ) -> None:
        """A declared-categorical date gets no special treatment once demoted,
        exactly like a declared-categorical non-date string already doesn't
        (see `test__declared_categorical_columns__do_not_warn`): only
        `min_cardinality_for_text` decides, `max_unique_for_category` is not
        even consulted, even though it would have saved this column.
        """
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(40)})
        schema = detect_feature_modalities(
            X=X.to_numpy(dtype=object),
            feature_names=list(X.columns),
            provided_categorical_indices=[1],
            min_samples_for_inference=100,
            max_unique_for_category=50,
            min_unique_for_numerical=4,
            min_cardinality_for_text=30,
        )
        assert schema.features[1].modality is FeatureModality.TEXT

    def test__declared_categorical_date_column__is_not_reported_in_warning(
        self,
    ) -> None:
        """The declaration doesn't change the demotion target (see
        `test__declared_categorical_date_column__declaration_has_no_effect`),
        but it does silence the warning, like a declared-categorical text
        column already does.
        """
        X = pd.DataFrame(
            {
                "num": self._numeric_column(),
                "declared": self._dates(60),
                "undeclared": self._dates(60),
            }
        )
        with pytest.warns(UserWarning, match="hold dates") as record:
            schema = self._detect(X, declared=[1])
        message = str(record[0].message)
        assert "'undeclared'" in message
        assert "'declared'" not in message
        assert schema.features[1].modality is FeatureModality.TEXT

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._detect(X, declared=[1, 2])
