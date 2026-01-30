"""Tests for feature type detection functionality."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.modality_detection import (
    _detect_feature_modality,
    detect_feature_modalities,
)
from tabpfn.preprocessing.type_detection import infer_categorical_features


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
    feature_schema = detect_feature_modalities(
        X=df.values,
        feature_names=df.columns.tolist(),
        min_samples_for_inference=1,
        max_unique_for_category=3,
        min_unique_for_numerical=5,
    )
    assert feature_schema.indices_for(FeatureModality.NUMERICAL) == [0]
    assert feature_schema.indices_for(FeatureModality.CATEGORICAL) == [1, 2]
    assert feature_schema.indices_for(FeatureModality.TEXT) == [3]
    assert feature_schema.indices_for(FeatureModality.CONSTANT) == [4]


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


def test__detect_for_categorical_with_category_dtype():
    s = pd.Series(["a", "b", "c", "a", "b", "c"], dtype="category")
    result = _for_test_detect_with_defaults(s)
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
        reported_categorical_indices=provided,
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
