#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for converting columns to the type their contents imply."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier
from tabpfn.preprocessing.input_conversion import (
    DEFAULT_TEXT_N_COMPONENTS,
    InputTypeConverter,
)

N_ROWS = 120


def _date_strings(n: int = N_ROWS) -> list[str]:
    return [f"2021-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]


def test__fit_transform__datetime_column__expands_into_numeric_features() -> None:
    frame = pd.DataFrame({"signed_on": pd.to_datetime(_date_strings())})
    out = InputTypeConverter().fit_transform(frame)

    assert out.shape[1] > 1
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in out.dtypes)
    assert any(name.startswith("signed_on_") for name in out.columns)


def test__fit_transform__date_strings__are_parsed_then_expanded() -> None:
    frame = pd.DataFrame({"signed_on": _date_strings()})
    out = InputTypeConverter().fit_transform(frame)

    assert out.shape[1] > 1
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in out.dtypes)


def test__fit_transform__numeric_strings__become_numeric() -> None:
    frame = pd.DataFrame({"amount": [f"{i}.5" for i in range(N_ROWS)]})
    out = InputTypeConverter().fit_transform(frame)

    assert pd.api.types.is_numeric_dtype(out["amount"].dtype)
    assert out["amount"].iloc[3] == pytest.approx(3.5)


def test__fit_transform__categorical_column__is_passed_through() -> None:
    """Categoricals stay as they are, so the estimator's own detection still runs."""
    frame = pd.DataFrame({"kind": ["a", "b", "c"] * (N_ROWS // 3)})
    out = InputTypeConverter().fit_transform(frame)

    assert out.shape == frame.shape
    assert out["kind"].tolist() == frame["kind"].tolist()


def test__fit_transform__non_dataframe__passes_through() -> None:
    converter = InputTypeConverter()
    array = np.arange(20.0).reshape(10, 2)

    assert converter.fit_transform(array) is array
    assert converter.transform(array) is array


def test__transform__decisions__are_frozen_at_fit() -> None:
    """A column parsed as a number at fit stays numeric even if predict data differs."""
    converter = InputTypeConverter()
    converter.fit_transform(pd.DataFrame({"amount": [f"{i}.5" for i in range(N_ROWS)]}))

    out = converter.transform(pd.DataFrame({"amount": ["not a number", "7.5"]}))
    assert pd.api.types.is_numeric_dtype(out["amount"].dtype)
    assert pd.isna(out["amount"].iloc[0])
    assert out["amount"].iloc[1] == pytest.approx(7.5)


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"n": np.arange(float(N_ROWS))}, id="datetime_and_numeric"),
        pytest.param({}, id="datetime_only"),
        pytest.param({"s": list("abcd") * (N_ROWS // 4)}, id="datetime_and_string"),
    ],
)
def test__classifier_fit_predict__datetime_column__does_not_raise(
    extra: dict[str, object],
) -> None:
    """Every shape of datetime input used to fail somewhere in preprocessing."""
    frame = pd.DataFrame({"when": pd.to_datetime(_date_strings()), **extra})
    y = np.arange(N_ROWS) % 2

    classifier = TabPFNClassifier(device="cpu", n_estimators=1, random_state=0)
    classifier.fit(frame, y)

    assert classifier.predict_proba(frame.iloc[:5]).shape == (5, 2)


def test__classifier_fit_predict__expanded_dates__feature_counts_agree() -> None:
    """`n_features_in_` describes the converted frame at fit and at predict alike."""
    frame = pd.DataFrame(
        {"when": _date_strings(), "value": np.arange(float(N_ROWS))},
    )
    y = np.arange(N_ROWS) % 2

    classifier = TabPFNClassifier(device="cpu", n_estimators=1, random_state=0)
    classifier.fit(frame, y)

    assert classifier.n_features_in_ > frame.shape[1]
    assert classifier.n_features_in_ == len(classifier.feature_names_in_)
    assert classifier.predict_proba(frame.iloc[:5]).shape == (5, 2)


def _text_frame(n_unique: int, n: int = N_ROWS) -> pd.DataFrame:
    return pd.DataFrame(
        {"notes": [f"free text number {i % n_unique}" for i in range(n)]}
    )


def test__fit_transform__text_column__is_encoded_into_numeric_features() -> None:
    frame = _text_frame(n_unique=100)
    out = InputTypeConverter().fit_transform(frame)

    assert out.shape[1] == DEFAULT_TEXT_N_COMPONENTS
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in out.dtypes)


def test__fit_transform__use_text_false__leaves_text_as_strings() -> None:
    frame = _text_frame(n_unique=100)
    out = InputTypeConverter(use_text=False).fit_transform(frame)

    assert out.shape == frame.shape
    assert out["notes"].tolist() == frame["notes"].tolist()


def test__fit_transform__text_cardinality_threshold__decides_what_is_text() -> None:
    """Below the threshold a string column is a category, above it it is text."""
    frame = _text_frame(n_unique=35)

    below = InputTypeConverter(text_cardinality_threshold=40).fit_transform(frame)
    above = InputTypeConverter(text_cardinality_threshold=20).fit_transform(frame)

    assert below.shape == frame.shape
    assert above.shape[1] == DEFAULT_TEXT_N_COMPONENTS


def test__fit_transform__text_column__warns_naming_the_column() -> None:
    with pytest.warns(UserWarning, match="encoded as text") as record:
        InputTypeConverter().fit_transform(_text_frame(n_unique=100))
    assert "'notes'" in str(record[0].message)


def test__fit_transform__use_dates_false__leaves_date_strings_alone() -> None:
    """Turning dates off must not leave a dtype the pipeline cannot represent."""
    frame = pd.DataFrame({"signed_on": _date_strings()})
    out = InputTypeConverter(use_dates=False).fit_transform(frame)

    assert out.shape == frame.shape
    assert out["signed_on"].tolist() == frame["signed_on"].tolist()


def test__fit_transform__use_dates_false__leaves_datetime_columns_alone() -> None:
    frame = pd.DataFrame({"signed_on": pd.to_datetime(_date_strings())})
    out = InputTypeConverter(use_dates=False).fit_transform(frame)

    assert out.shape == frame.shape
    assert pd.api.types.is_datetime64_any_dtype(out["signed_on"].dtype)


def test__transform__array_after_frame_fit__is_still_converted() -> None:
    """Fitting on a named frame and predicting with a bare array is supported."""
    frame = pd.DataFrame(
        {"notes": [f"free text {i}" for i in range(N_ROWS)], "n": range(N_ROWS)}
    )
    converter = InputTypeConverter()
    fitted = converter.fit_transform(frame)

    out = converter.transform(frame.to_numpy())
    assert list(out.columns) == list(fitted.columns)


@pytest.mark.parametrize("use_text", [True, False])
def test__classifier_fit_predict__text_column__round_trips(use_text: bool) -> None:
    frame = pd.DataFrame(
        {"notes": [f"note {i}" for i in range(N_ROWS)], "n": np.arange(float(N_ROWS))}
    )
    y = np.arange(N_ROWS) % 2

    classifier = TabPFNClassifier(
        device="cpu", n_estimators=1, random_state=0, use_text=use_text
    )
    classifier.fit(frame, y)

    expected = DEFAULT_TEXT_N_COMPONENTS + 1 if use_text else 2
    assert classifier.n_features_in_ == expected
    assert classifier.predict_proba(frame.iloc[:5]).shape == (5, 2)


def test__fit_transform__text_n_components__sets_the_width() -> None:
    frame = _text_frame(n_unique=100)
    out = InputTypeConverter(text_n_components=5).fit_transform(frame)

    assert out.shape[1] == 5


def test__classifier_fit_predict__text_n_components__sets_the_feature_count() -> None:
    frame = _text_frame(n_unique=100)
    y = np.arange(N_ROWS) % 2

    classifier = TabPFNClassifier(
        device="cpu", n_estimators=1, random_state=0, text_n_components=4
    )
    classifier.fit(frame, y)

    assert classifier.n_features_in_ == 4
    assert classifier.predict_proba(frame.iloc[:5]).shape == (5, 2)


def test__fit_transform__use_dates_false__warns_naming_the_date_columns() -> None:
    """Turning dates off must say so, or the column is only reported as free text."""
    frame = pd.DataFrame({"signed_on": _date_strings()})

    with pytest.warns(UserWarning, match="`use_dates` is off") as record:
        InputTypeConverter(use_dates=False).fit_transform(frame)
    assert "'signed_on'" in str(record[0].message)


def test__fit_transform__use_dates_true__does_not_warn_about_dates() -> None:
    frame = pd.DataFrame({"signed_on": _date_strings()})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InputTypeConverter().fit_transform(frame)
    assert not [w for w in caught if "use_dates" in str(w.message)]


def test__fit_transform__use_text_false__does_not_warn_about_encoding() -> None:
    """The off switch must not claim the column was encoded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InputTypeConverter(use_text=False).fit_transform(_text_frame(n_unique=100))
    assert not [w for w in caught if "encoded as text" in str(w.message)]
