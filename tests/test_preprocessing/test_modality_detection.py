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
from tabpfn.preprocessing.clean import PANDAS_BELOW_3
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
    _is_numeric_or_missing_for_old_pandas,
    _is_numeric_pandas_series,
    _warn_on_text,
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


def _reference_is_numeric(s: pd.Series) -> bool:
    """`_is_numeric_pandas_series` without the prefix rejection."""
    if pd.api.types.is_numeric_dtype(s.dtype):
        return True
    if PANDAS_BELOW_3:
        return all(_is_numeric_or_missing_for_old_pandas(value) for value in s)
    coerced = pd.to_numeric(s, errors="coerce")
    return bool((coerced.notna() | s.isna()).all())


#: Prefix these tests run the rejection at. The logic is identical at any size,
#: so a smaller one than production keeps the columns below cheap to build and
#: easy to reason about. Only `test__prefix_rejection__fires_within_the_real_prefix`
#: uses the production value, since that is the one thing a shrunk prefix cannot
#: check.
_TEST_PREFIX_ROWS = 50


@pytest.fixture
def small_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the rejection prefix so test columns can stay a handful of rows."""
    monkeypatch.setattr(
        modality_detection, "_EARLY_EXIT_PREFIX_ROWS", _TEST_PREFIX_ROWS
    )


#: Columns whose prefix and tail disagree, so the rejection either has to fire
#: correctly or fall through. Named by where the disagreement sits. Every column
#: is longer than `_TEST_PREFIX_ROWS`, or the guard would not run at all.
_TAIL = 10
_NUMBERS = [str(i) for i in range(_TEST_PREFIX_ROWS + _TAIL)]
_PROBE_COLUMNS: dict[str, list[Any]] = {
    # Fails on its very first value.
    "text": [f"a sentence, number {i}" for i in range(_TEST_PREFIX_ROWS + _TAIL)],
    "missing_prefix_then_text": [None] * _TEST_PREFIX_ROWS + ["not a number"] * _TAIL,
    "all_missing": [None] * (_TEST_PREFIX_ROWS + _TAIL),
    # Numeric strings, with the offending value on either side of the prefix.
    "numeric_strings": _NUMBERS,
    "non_numeric_inside_prefix": [*_NUMBERS[: _TEST_PREFIX_ROWS - 1], "abc", *_NUMBERS],
    "non_numeric_beyond_prefix": [*_NUMBERS, "abc"],
}


@pytest.mark.parametrize("name", list(_PROBE_COLUMNS))
@pytest.mark.usefixtures("small_prefix")
def test__prefix_rejection__agrees_with_parsing_the_whole_column(name: str) -> None:
    """The prefix rejection is exact, not an approximation.

    One non-numeric value settles an all-or-nothing check, so rejecting on a
    prefix can only skip work. A clean prefix proves nothing about the tail and
    must fall through, including when the prefix is entirely missing.
    """
    s = pd.Series(_PROBE_COLUMNS[name], dtype=object)
    assert _is_numeric_pandas_series(s) == _reference_is_numeric(s)


@pytest.mark.skipif(
    PANDAS_BELOW_3,
    reason=(
        "Below pandas 3 the numeric check walks values through an `all(...)` "
        "generator that already stops at the first non-numeric one, so it has "
        "no prefix guard to skip and never calls `_all_numeric_or_missing`."
    ),
)
@pytest.mark.usefixtures("small_prefix")
def test__prefix_rejection__skips_the_guard_on_a_short_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A column no longer than the prefix is parsed once, not twice.

    Counting the parses rather than the answer, since parsing twice gives the
    same answer. A longer column that clears its prefix is parsed twice, which
    is the cost this trade accepts.
    """
    real = modality_detection._all_numeric_or_missing
    parses = []

    def counting(s: pd.Series) -> bool:
        parses.append(len(s))
        return real(s)

    monkeypatch.setattr(modality_detection, "_all_numeric_or_missing", counting)

    def passing_values(n: int) -> list[str]:
        return [str(i) for i in range(n)]

    s = pd.Series(passing_values(_TEST_PREFIX_ROWS), dtype=object)
    assert _is_numeric_pandas_series(s) is True
    assert len(parses) == 1

    parses.clear()
    assert (
        _is_numeric_pandas_series(
            pd.Series(passing_values(_TEST_PREFIX_ROWS + 2), dtype=object)
        )
        is True
    )
    assert len(parses) == 2


def _text_schema(*names: str) -> FeatureSchema:
    """Schema of TEXT features with the `input_` prefix real input names carry."""
    return FeatureSchema(
        features=[
            Feature(name=f"{INPUT_FEATURE_PREFIX}{name}", modality=FeatureModality.TEXT)
            for name in names
        ]
    )


class TestWarnOnText:
    """Schema-level unit tests for `_warn_on_text`."""

    def test__no_text_features__does_not_warn(self) -> None:
        schema = FeatureSchema(
            features=[
                Feature(name="input_a", modality=FeatureModality.NUMERICAL),
                Feature(name="input_b", modality=FeatureModality.CATEGORICAL),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_text(schema)

    def test__text_features__warn_with_column_names_and_remedies(self) -> None:
        with pytest.warns(UserWarning, match="look like free text") as record:
            _warn_on_text(_text_schema("review"))

        message = str(record[0].message)
        # Column names are shown as the user wrote them, without the input_ prefix.
        assert "'review'" in message
        assert INPUT_FEATURE_PREFIX not in message
        # The message must state all remedies.
        assert "numeric dtype" in message
        assert "TRANSFORM_TEXT" in message
        assert "https://github.com/PriorLabs/tabpfn-client" in message
        assert "categorical_features_indices" in message

    def test__declared_cat_indices__are_not_reported(self) -> None:
        schema = _text_schema("sku", "review")

        with pytest.warns(UserWarning, match="look like free text") as record:
            _warn_on_text(schema, declared_cat_indices=[0])
        message = str(record[0].message)
        assert "'review'" in message
        assert "'sku'" not in message

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_text(schema, declared_cat_indices=[0, 1])

    def test__many_text_columns__message_is_truncated(self) -> None:
        n_extra = 5
        n_columns = _MAX_TEXT_COLUMNS_IN_WARNING + n_extra
        schema = _text_schema(*(f"t{i}" for i in range(n_columns)))

        with pytest.warns(UserWarning, match="look like free text") as record:
            _warn_on_text(schema)

        message = str(record[0].message)
        assert f"(and {n_extra} more)" in message
        assert f"'t{_MAX_TEXT_COLUMNS_IN_WARNING - 1}'" in message
        assert f"'t{_MAX_TEXT_COLUMNS_IN_WARNING}'" not in message


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
