"""Tests for feature type detection functionality."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tabpfn.preprocessing.modality_detection import (
    FeatureModality,
    _detect_feature_modality,
    detect_feature_modalities,
)


def _for_test_detect_with_defaults(
    s: pd.Series,
    max_unique_for_category: int = 10,
    min_unique_for_numerical: int = 5,
    *,
    reported_categorical: bool = False,
    big_enough_n_to_infer_cat: bool = True,
) -> FeatureModality:
    return _detect_feature_modality(
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
            "cat": ["a", "b", "c", "a", "b"],
            "cat_num": [0, 1, 2, 1, 2],
            "text": ["longer", "texts", "appear", "here", "yay"],
            "const": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    view = detect_feature_modalities(
        df,
        min_samples_for_inference=1,
        max_unique_for_category=3,
        min_unique_for_numerical=5,
    )
    assert view.columns_by_modality[FeatureModality.NUMERICAL] == ["num"]
    assert view.columns_by_modality[FeatureModality.CATEGORICAL] == ["cat", "cat_num"]
    assert view.columns_by_modality[FeatureModality.TEXT] == ["text"]
    assert view.columns_by_modality[FeatureModality.CONSTANT] == ["const"]


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


def test__detect_textual_feature():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
    result = _for_test_detect_with_defaults(s, max_unique_for_category=2)
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
    result = _for_test_detect_with_defaults(s, max_unique_for_category=2)
    assert result == FeatureModality.TEXT
    result = _for_test_detect_with_defaults(s, max_unique_for_category=15)
    assert result == FeatureModality.CATEGORICAL


def test__detect_text_as_object():
    s = pd.Series(["a", "b", "c", "e", "f"], dtype=object)
    s = s.astype(object)
    result = _for_test_detect_with_defaults(s, max_unique_for_category=2)
    assert result == FeatureModality.TEXT
    result = _for_test_detect_with_defaults(s, max_unique_for_category=15)
    assert result == FeatureModality.CATEGORICAL


def test__detect_for_boolean():
    s = pd.Series([True, False, True, False])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL


def test__detect_for_boolean_with_floats():
    s = pd.Series([1.0, 0.0, 1.0, 0.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL


def test__detect_for_boolean_with_strings():
    s = pd.Series(["True", "False", "True", "False"])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL


def test__detect_for_series_with_nan_and_floats_not_constant():
    s = pd.Series([np.nan, 1.0, np.nan, 1.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureModality.CATEGORICAL


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
    s = pd.Series(series_data)
    result = _for_test_detect_with_defaults(s)
    if result != FeatureModality.CONSTANT:
        error = f"Expected CONSTANT but got {result} for {test_name}: {series_data}"
        raise AssertionError(error)
