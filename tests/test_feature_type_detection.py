"""Tests for feature type detection functionality."""

from __future__ import annotations

import pandas as pd

from tabpfn.preprocessing.type_detection import FeatureType, _detect_feature_type


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


def test__numerical_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = _for_test_detect_with_defaults(s)
    assert result == FeatureType.NUMERICAL


def test__categorical_series():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
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


def test__detect_textual_feature():
    s = pd.Series(["a", "b", "c", "a", "b", "c"])
    result = _for_test_detect_with_defaults(s, max_unique_for_category=2)
    assert result == FeatureType.TEXT
