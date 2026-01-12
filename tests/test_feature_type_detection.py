"""Tests for feature type detection functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabpfn.preprocessing.type_detection import (
    FeatureType,
    _detect_feature_type,
    detect_feature_types,
)


def _for_test_detect_with_defaults(
    s: pd.Series,
    max_unique_for_category: int = 10,
    min_unique_for_numerical: int = 5,
    *,
    reported_categorical: bool = False,
    big_enough_n_to_infer_cat: bool = True,
) -> FeatureType:
    return _detect_feature_type(
        s,
        reported_categorical=reported_categorical,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
    )


def test__dataset_view_end_to_end():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "c", "d", "e"],
            "cat_num": [0, 1, 2, 3, 4],
            "text": ["longer", "texts", "appear", "here", "yay"],
            "const": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    view = detect_feature_types(
        df,
        min_samples_for_inference=3,
        max_unique_for_category=2,
        min_unique_for_numerical=3,
    )
    assert view.feature_type_to_columns[FeatureType.NUMERICAL] == ["num"]
    assert view.feature_type_to_columns[FeatureType.CATEGORICAL] == ["cat", "cat_num"]
    assert view.feature_type_to_columns[FeatureType.TEXT] == ["text"]
    assert view.feature_type_to_columns[FeatureType.CONSTANT] == ["const"]


def test__numerical_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.NUMERICAL


def test__numerical_series_from_strings():
    s = pd.Series(
        ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]
    )
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.NUMERICAL


def test__numerical_series_with_nan():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.NUMERICAL


def test__numerical_but_stored_as_string():
    s = pd.Series(
        ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]
    )
    s = s.astype(str)
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.NUMERICAL


def test__categorical_series():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CATEGORICAL


def test__categorical_series_with_nan():
    s = pd.Series(["a", "b", "c", "a", "b", "c", np.nan])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CATEGORICAL
    s = pd.Series(["a", "b", "c", "a", "b", "c", np.nan, None])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CATEGORICAL


def test__numerical_reported_as_categorical():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _for_test_detect_with_defaults(s, reported_categorical=True)
    assert result == FeatureType.CATEGORICAL


def test__numerical_reported_as_categorical_but_too_many_unique_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _for_test_detect_with_defaults(
        s, reported_categorical=True, max_unique_for_category=9
    )
    assert result == FeatureType.NUMERICAL


def test__detected_categorical_without_reporting():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = _for_test_detect_with_defaults(
        s, reported_categorical=False, min_unique_for_numerical=5
    )
    assert result == FeatureType.CATEGORICAL

    # Even with floats, this should be categorical
    s = pd.Series([3.43, 3.54, 3.43, 3.53, 3.43, 3.54, 657.3])
    result = _for_test_detect_with_defaults(
        s, reported_categorical=False, min_unique_for_numerical=5
    )
    assert result == FeatureType.CATEGORICAL


def test__detect_textual_feature():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
    result = _for_test_detect_with_defaults(s, max_unique_for_category=2)
    assert result == FeatureType.TEXT


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
    result = _for_test_detect_with_defaults(s, max_unique_for_category=2)
    assert result == FeatureType.TEXT
    result = _for_test_detect_with_defaults(s, max_unique_for_category=15)
    assert result == FeatureType.CATEGORICAL


def test__detect_text_as_object():
    s = pd.Series(["a", "b", "c", "e", "f"], dtype=object)
    s = s.astype(object)
    result = _for_test_detect_with_defaults(s, max_unique_for_category=2)
    assert result == FeatureType.CATEGORICAL


def test__detect_for_boolean():
    s = pd.Series([True, False, True, False])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CATEGORICAL


def test__detect_for_boolean_with_floats():
    s = pd.Series([1.0, 0.0, 1.0, 0.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CATEGORICAL


def test__detect_for_boolean_with_strings():
    s = pd.Series(["True", "False", "True", "False"])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CATEGORICAL


def test__detect_for_constant():
    s = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT


def test__detect_for_constant_from_single_value():
    s = pd.Series([1.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT
    s = pd.Series([np.nan])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT
    s = pd.Series([None])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT
    s = pd.Series(["a"])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT
    s = pd.Series([True])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT


def test__detect_for_constant_with_strings():
    s = pd.Series(["a", "a", "a", "a"])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT


def test__detect_for_constant_with_booleans():
    s = pd.Series([True, True, True, True])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT


def test__detect_for_empty_series():
    s = pd.Series([])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT


def test__detect_for_series_with_nan():
    s = pd.Series([np.nan, np.nan, np.nan, np.nan])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT


def test__detect_for_series_with_nan_and_floats():
    s = pd.Series([np.nan, 1.0, np.nan, 1.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CATEGORICAL


def test__detect_for_series_with_few_null_types():
    s = pd.Series([np.nan, None, np.nan, None])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.CONSTANT
