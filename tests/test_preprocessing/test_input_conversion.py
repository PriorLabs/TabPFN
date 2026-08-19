#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for converting columns to the type their contents imply."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier
from tabpfn.preprocessing.input_conversion import InputTypeConverter

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
