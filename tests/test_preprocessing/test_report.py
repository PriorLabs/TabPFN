#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the input report: `inspect_input` and the fitted `input_report_`."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from tabpfn.preprocessing import ColumnReport, InputReport
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.modality_detection import _EARLY_EXIT_PREFIX_ROWS

N_ROWS = 200


def _mixed_frame(n: int = N_ROWS) -> pd.DataFrame:
    """Every decision the report can make about a kept column, one column each."""
    return pd.DataFrame(
        {
            "age": (np.arange(n) * 7) % 60 + 18,
            "n_children": np.arange(n) % 3,
            "store_id": np.arange(n) % 50,
            "city": [f"city{i % 12}" for i in range(n)],
            "comment": [f"comment number {i} with words" for i in range(n)],
            "price": [f"{i / 4:.2f}" for i in range(n)],
        }
    )


def _by_name(report: InputReport) -> dict[str, ColumnReport]:
    return {column.name: column for column in report.columns}


def _target(estimator_cls: type, n: int = N_ROWS) -> np.ndarray:
    if estimator_cls is TabPFNClassifier:
        return np.arange(n) % 2
    return np.arange(n, dtype=float)


def test__inspect_input__mixed_frame__reports_each_input_column_once() -> None:
    X = _mixed_frame()
    report = TabPFNClassifier(categorical_features_indices=[2]).inspect_input(X)

    assert isinstance(report, InputReport)
    assert [column.name for column in report.columns] == list(X.columns)
    assert [column.index for column in report.columns] == list(range(X.shape[1]))
    assert report.n_samples == N_ROWS
    assert report.n_features_out == X.shape[1]
    assert [column.modality for column in report.columns] == [
        FeatureModality.NUMERICAL,
        FeatureModality.CATEGORICAL,
        FeatureModality.NUMERICAL,
        FeatureModality.CATEGORICAL,
        FeatureModality.TEXT,
        FeatureModality.NUMERICAL,
    ]
    assert [column.ordinal_encoded for column in report.columns] == [
        False,
        True,
        False,
        True,
        True,
        True,
    ]
    assert [column.declared_categorical for column in report.columns] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert [column.n_unique for column in report.columns] == [60, 3, 50, 12, 200, 200]
    assert all(column.n_unique_is_exact for column in report.columns)
    assert all(column.n_missing == 0 for column in report.columns)
    assert all(column.expanded_by is None for column in report.columns)
    assert report.settings["MAX_UNIQUE_FOR_CATEGORICAL_FEATURES"] == 30
    assert report.settings["TRANSFORM_DATES"] is False
    assert report.columns_for(FeatureModality.TEXT) == [report.columns[4]]


def test__inspect_input__does_not_load_the_model() -> None:
    clf = TabPFNClassifier()
    clf.inspect_input(_mixed_frame())

    assert not hasattr(clf, "models_")
    assert not hasattr(clf, "inference_config_")
    assert not hasattr(clf, "input_report_")


def test__inspect_input__numeric_string_column__is_numerical_and_ordinal_encoded() -> (
    None
):
    price = _by_name(TabPFNClassifier().inspect_input(_mixed_frame()))["price"]

    assert price.modality is FeatureModality.NUMERICAL
    assert price.ordinal_encoded
    assert "stored as a string" in price.reason
    assert "numeric dtype" in price.how_to_change
    # Strings keep their quotes, so the column reads as text at a glance.
    assert price.examples == ("'0.00'", "'0.25'", "'0.50'")


def test__inspect_input__declared_high_cardinality_int__is_numerical() -> None:
    X = _mixed_frame()
    store = _by_name(
        TabPFNClassifier(categorical_features_indices=[2]).inspect_input(X)
    )["store_id"]
    assert store.declared_categorical
    assert store.modality is FeatureModality.NUMERICAL
    assert not store.ordinal_encoded
    assert "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES=30" in store.reason
    assert (
        "Raise MAX_UNIQUE_FOR_CATEGORICAL_FEATURES to 50 or above"
        in store.how_to_change
    )

    raised = TabPFNClassifier(
        categorical_features_indices=[2],
        inference_config={"MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": 50},
    ).inspect_input(X)
    store = _by_name(raised)["store_id"]
    assert store.modality is FeatureModality.CATEGORICAL
    assert store.ordinal_encoded
    assert raised.settings["MAX_UNIQUE_FOR_CATEGORICAL_FEATURES"] == 50


def test__inspect_input__category_dtype_column__counts_as_declared() -> None:
    X = pd.DataFrame(
        {
            "num": np.arange(N_ROWS, dtype=float),
            "code": pd.Series(
                [f"code{i % 40}" for i in range(N_ROWS)], dtype="category"
            ),
        }
    )
    code = _by_name(TabPFNClassifier().inspect_input(X))["code"]

    assert code.declared_categorical
    assert code.modality is FeatureModality.CATEGORICAL
    assert code.n_unique == 40
    assert "`category` dtype" in code.reason


def test__inspect_input__datetime_with_transform_dates__is_expanded() -> None:
    X = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=N_ROWS, freq="D"),
            "num": np.arange(N_ROWS, dtype=float),
        }
    )
    report = TabPFNClassifier(inference_config={"TRANSFORM_DATES": True}).inspect_input(
        X
    )
    when, num = report.columns

    assert when.name == "when"
    assert when.expanded_by == "dates"
    assert when.modality is None
    assert when.read_as == "expanded"
    assert when.n_unique is None
    assert when.generated_features
    assert all(name.startswith("when") for name in when.generated_features)
    assert "TRANSFORM_DATES=True" in when.reason
    assert num.modality is FeatureModality.NUMERICAL
    assert report.n_features_out == 1 + len(when.generated_features)


def test__inspect_input__datetime_without_transform_dates__raises_like_fit() -> None:
    X = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=N_ROWS, freq="D")})
    with pytest.raises(TabPFNValidationError):
        TabPFNClassifier().inspect_input(X)


def test__inspect_input__transform_text__expands_and_shifts_positions() -> None:
    X = pd.DataFrame(
        {
            "num": np.arange(N_ROWS, dtype=float),
            "review": pd.Series(
                [f"review {i}, a fairly long sentence" for i in range(N_ROWS)],
                dtype="string",
            ),
            "cat": [f"c{i % 12}" for i in range(N_ROWS)],
        }
    )
    report = TabPFNClassifier(inference_config={"TRANSFORM_TEXT": True}).inspect_input(
        X
    )
    num, review, cat = report.columns

    assert review.expanded_by == "text"
    assert review.modality is None
    assert review.generated_features
    assert all(name.startswith("review") for name in review.generated_features)
    assert "TRANSFORM_TEXT=True" in review.reason
    # `cat` moved down one position in the expanded frame, and is still found.
    assert cat.name == "cat"
    assert cat.modality is FeatureModality.CATEGORICAL
    assert cat.ordinal_encoded
    assert num.modality is FeatureModality.NUMERICAL
    assert not num.ordinal_encoded
    assert report.n_features_out == 2 + len(review.generated_features)


def test__inspect_input__object_text_with_transform_text__stays_text() -> None:
    X = pd.DataFrame(
        {
            "num": np.arange(N_ROWS, dtype=float),
            "review": pd.Series(
                [f"review {i}, a fairly long sentence" for i in range(N_ROWS)],
                dtype=object,
            ),
        }
    )
    review = _by_name(
        TabPFNClassifier(inference_config={"TRANSFORM_TEXT": True}).inspect_input(X)
    )["review"]

    assert review.expanded_by is None
    assert review.modality is FeatureModality.TEXT
    assert review.ordinal_encoded
    assert "Not expanded" in review.reason
    assert "object" in review.reason


def test__inspect_input__timedelta__is_converted_then_detected() -> None:
    X = pd.DataFrame(
        {
            "dur": pd.to_timedelta(np.arange(N_ROWS), unit="h"),
            "num": np.arange(N_ROWS, dtype=float),
        }
    )
    dur = _by_name(TabPFNClassifier().inspect_input(X))["dur"]

    assert dur.dtype.startswith("timedelta64")
    assert dur.modality is FeatureModality.NUMERICAL
    assert dur.reason.startswith("timedelta converted to seconds first;")


def test__inspect_input__array_input__uses_positional_names_and_array_dtype() -> None:
    X = np.arange(N_ROWS * 3, dtype=float).reshape(N_ROWS, 3)
    report = TabPFNClassifier().inspect_input(X)

    assert [column.name for column in report.columns] == ["f0", "f1", "f2"]
    assert [column.dtype for column in report.columns] == ["float64"] * 3
    assert all(
        column.modality is FeatureModality.NUMERICAL for column in report.columns
    )
    assert not any(column.ordinal_encoded for column in report.columns)
    assert report.columns[0].examples == ("0.0", "3.0", "6.0")


def test__inspect_input__integer_labels__are_named_by_position() -> None:
    X = pd.DataFrame(np.arange(N_ROWS * 2, dtype=float).reshape(N_ROWS, 2))
    report = TabPFNClassifier().inspect_input(X)

    assert [column.name for column in report.columns] == ["0", "1"]


def test__inspect_input__all_missing_column__is_constant() -> None:
    X = pd.DataFrame(
        {"num": np.arange(N_ROWS, dtype=float), "gone": np.full(N_ROWS, np.nan)}
    )
    gone = _by_name(TabPFNClassifier().inspect_input(X))["gone"]

    assert gone.modality is FeatureModality.CONSTANT
    assert gone.n_missing == N_ROWS
    assert gone.n_unique == 1
    assert gone.examples == ()
    assert gone.reason.startswith("Every value is missing")


def test__inspect_input__past_prefix_rows__n_unique_is_lower_bound() -> None:
    n = _EARLY_EXIT_PREFIX_ROWS + 500
    X = pd.DataFrame(
        {"low": np.arange(n) % 3, "text": [f"free text {i}" for i in range(n)]}
    )
    report = TabPFNClassifier().inspect_input(X)
    low, text = report.columns

    assert low.n_unique == 3
    assert low.n_unique_is_exact
    assert text.modality is FeatureModality.TEXT
    assert text.n_unique == _EARLY_EXIT_PREFIX_ROWS
    assert not text.n_unique_is_exact
    assert f">={_EARLY_EXIT_PREFIX_ROWS}" in str(report)
    assert f"at least {_EARLY_EXIT_PREFIX_ROWS} distinct values" in text.reason


def test__inspect_input__emits_no_text_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        report = TabPFNClassifier().inspect_input(_mixed_frame())
    assert report.columns_for(FeatureModality.TEXT)


def test__inspect_input__unknown_config_key__raises() -> None:
    with pytest.raises(ValueError, match="NOT_A_FIELD"):
        TabPFNClassifier(inference_config={"NOT_A_FIELD": 1}).inspect_input(
            _mixed_frame()
        )


def test__input_report__to_frame__has_one_row_per_input_column() -> None:
    report = TabPFNClassifier().inspect_input(_mixed_frame())

    frame = report.to_frame()
    assert frame.shape == (6, 14)
    assert frame["name"].tolist() == list(_mixed_frame().columns)
    assert frame["modality"].tolist() == [
        "numerical",
        "categorical",
        "numerical",
        "categorical",
        "text",
        "numerical",
    ]

    as_dict = report.to_dict()
    assert json.loads(json.dumps(as_dict))["columns"][4]["modality"] == "text"
    assert as_dict["settings"] == report.settings

    markdown = report.to_markdown().splitlines()
    assert markdown[0].startswith("| # | column | dtype |")
    assert markdown[1].startswith("| --- |")
    assert len(markdown) == 2 + 6


def test__input_report__str__names_every_column() -> None:
    report = TabPFNClassifier().inspect_input(_mixed_frame())
    text = str(report)

    assert text.startswith(
        "InputReport: 6 input columns, 200 rows, 6 columns after expansion"
    )
    assert all(name in text for name in _mixed_frame().columns)
    assert "MIN_CARDINALITY_FOR_TEXT=30" in text
    assert repr(report) == text


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__sets_input_report_equal_to_inspect_input(estimator_cls: type) -> None:
    X = _mixed_frame()
    model = estimator_cls(
        n_estimators=1, device="cpu", categorical_features_indices=[2]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X, _target(estimator_cls))

    assert model.input_report_ == model.inspect_input(X)
    assert model.input_report_.settings["MAX_UNIQUE_FOR_CATEGORICAL_FEATURES"] == (
        model.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES
    )
    assert (
        _by_name(model.input_report_)["store_id"].modality is FeatureModality.NUMERICAL
    )


def test__fit__text_warning_points_at_input_report() -> None:
    model = TabPFNClassifier(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="look like free text") as record:
        model.fit(_mixed_frame(), _target(TabPFNClassifier))
    assert "input_report_" in str(record[0].message)


def test__save_and_load_fitted_model__keeps_input_report(tmp_path: Path) -> None:
    X = _mixed_frame()
    model = TabPFNClassifier(n_estimators=1, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X, _target(TabPFNClassifier))
    path = tmp_path / "model.tabpfn_fit"
    save_fitted_tabpfn_model(model, path)

    loaded = load_fitted_tabpfn_model(path, device="cpu")
    assert loaded.input_report_ == model.input_report_
