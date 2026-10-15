#  Copyright (c) Prior Labs GmbH 2026.
"""Tests for the inspection fit keeps of its input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from tabpfn.preprocessing.clean import PANDAS_BELOW_3, clean_data
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.datetimes import DateTransformer
from tabpfn.preprocessing.input_inspection import (
    MAX_EXAMPLE_LENGTH,
    MAX_EXAMPLES,
    InputInspection,
    build_input_inspection,
)
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.preprocessing.text import TextTransformer

NUMERICAL = FeatureModality.NUMERICAL
CATEGORICAL = FeatureModality.CATEGORICAL
TEXT = FeatureModality.TEXT
CONSTANT = FeatureModality.CONSTANT

#: The default string dtype's name: `str` from pandas 3.0, `object` before.
STR = "object" if PANDAS_BELOW_3 else "str"

#: Positions of `code`, `grade` and `tag` below, declared categorical by index.
DECLARED = [2, 3, 12]


def _every_kind_of_column(n: int = 200) -> pd.DataFrame:
    """One column of every kind fit reads, in a frame of `n` rows."""
    i = np.arange(n)
    return pd.DataFrame(
        {
            # Numbers with many distinct values.
            "amount": np.random.default_rng(seed=0).normal(size=n),
            # Numbers with few distinct values.
            "count": i % 3,
            # Declared categorical, with more distinct values than the cap allows.
            "code": i % 50,
            # Declared categorical, within the cap.
            "grade": i % 5,
            # Declared categorical through the dtype.
            "level": pd.Series([("low", "mid", "high")[k % 3] for k in i]).astype(
                "category"
            ),
            "flag": i % 2 == 0,
            # A nullable integer dtype, with missing values.
            "score": pd.array(
                [None if k % 9 == 0 else k % 40 for k in i], dtype="Int64"
            ),
            # Strings with few distinct values, some missing.
            "city": np.where(i % 7 == 0, None, [f"c{k % 8}" for k in i]),
            # Numbers stored as strings, few distinct.
            "size": [str(k % 3 + 1) for k in i],
            # Numbers stored as strings, many distinct.
            "zip": [f"{10000 + k % 60}" for k in i],
            # Free text in the `string` dtype, which text expansion reads.
            "review": pd.Series(
                [f"quite a long free text value number {k} in this cell" for k in i],
                dtype="string",
            ),
            # Free text in the `object` dtype, which it does not.
            "note": pd.Series(
                [f"another long free text value number {k} here" for k in i],
                dtype=object,
            ),
            # Strings with many distinct values, declared categorical.
            "tag": [f"tag{k}" for k in i],
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "elapsed": pd.to_timedelta(i, unit="h"),
            "fixed": np.full(n, 7),
            "empty": np.full(n, np.nan),
        }
    )


#: How each column above is read, with the detection thresholds at their defaults
#: (categorical up to 30 distinct values when declared, below 4 when inferred; text
#: above 30 distinct strings) and both expansions on. A kept column is
#: `(modality, n_unique, numeric_like, ordinal_encoded)`, an expanded one
#: `(kind, n_features)`.
EXPECTED_READING = {
    "amount": (NUMERICAL, 200, True, False),
    "count": (CATEGORICAL, 3, True, True),
    # Declared, but 50 distinct values exceed the cap: read as numerical after all.
    "code": (NUMERICAL, 50, True, False),
    "grade": (CATEGORICAL, 5, True, True),
    "level": (CATEGORICAL, 3, False, True),
    "flag": (CATEGORICAL, 2, True, True),
    # 40 values and the missing one.
    "score": (NUMERICAL, 41, True, False),
    # 8 cities and the missing one.
    "city": (CATEGORICAL, 9, False, True),
    "size": (CATEGORICAL, 3, True, True),
    # Counted as numbers, yet ordinal-encoded by dtype: the numeric order is lost.
    "zip": (NUMERICAL, 60, True, True),
    "review": ("text", 30),
    # Not expanded, so read as text and ordinal-encoded like any string column.
    "note": (TEXT, 200, False, True),
    # Declared, so categorical whatever its cardinality.
    "tag": (CATEGORICAL, 200, False, True),
    "when": ("dates", 9),
    # A duration becomes its length in seconds first.
    "elapsed": (NUMERICAL, 200, True, False),
    "fixed": (CONSTANT, 1, None, False),
    "empty": (CONSTANT, 1, None, False),
}

EXPECTED_DTYPES = (
    "float64",
    "int64",
    "int64",
    "int64",
    "category",
    "bool",
    "Int64",
    STR,
    STR,
    STR,
    "string",
    "object",
    STR,
    "datetime64",
    "timedelta64",
    "int64",
    "float64",
)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__every_kind_of_column__inspection_tells_how_each_was_read(
    estimator_cls: type,
) -> None:
    X = _every_kind_of_column()
    y = np.arange(len(X)) % 2 if estimator_cls is TabPFNClassifier else X["amount"]
    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=DECLARED,
        inference_config={"TRANSFORM_DATES": True, "TRANSFORM_TEXT": True},
    )
    with pytest.warns(UserWarning, match="look like free text"):
        model.fit(X, y)

    inspection = model.input_inspection_
    assert isinstance(inspection, InputInspection)
    assert inspection.labels == tuple(EXPECTED_READING)
    assert all(
        dtype.startswith(expected)
        for dtype, expected in zip(inspection.dtypes, EXPECTED_DTYPES, strict=True)
    )
    assert inspection.declared_positions == tuple(DECLARED)
    assert inspection.category_dtype_positions == (X.columns.get_loc("level"),)

    # The decisions cover the expanded input, column for column.
    schema = model.inferred_feature_schema_
    assert len(inspection.decisions) == schema.num_columns
    assert [d.modality for d in inspection.decisions] == [
        f.modality for f in schema.features
    ]
    n_kept = X.shape[1] - len(inspection.expansions)
    assert all(p < n_kept for p in inspection.ordinal_encoded_positions)

    # Every input column is accounted for: kept and decided on, or expanded.
    for position, (label, expected) in enumerate(EXPECTED_READING.items()):
        after = inspection.position_after_expansion(position)
        if len(expected) == 2:
            kind, n_features = expected
            expansion = next(e for e in inspection.expansions if e.position == position)
            assert after is None, label
            assert expansion.kind == kind, label
            assert len(expansion.features) == n_features, label
            assert all(name.startswith(f"{label}_") for name in expansion.features)
            continue
        modality, n_unique, numeric_like, encoded = expected
        decision = inspection.decisions[after]
        assert decision.modality is modality, label
        assert decision.n_unique == n_unique, label
        assert decision.n_unique_is_exact, label
        assert decision.numeric_like is numeric_like, label
        assert (after in inspection.ordinal_encoded_positions) is encoded, label

    # The generated features follow the kept columns, dates first, and are numeric
    # except where a calendar feature cannot vary: every date here is in 2024.
    assert [e.kind for e in inspection.expansions] == ["dates", "text"]
    dates, text = inspection.expansions
    date_decisions = inspection.decisions[n_kept : n_kept + len(dates.features)]
    text_decisions = inspection.decisions[n_kept + len(dates.features) :]
    assert "when_year" in dates.features
    assert [d.modality for d in date_decisions] == [
        CONSTANT if name == "when_year" else NUMERICAL for name in dates.features
    ]
    assert len(text_decisions) == len(text.features)
    assert all(d.modality is NUMERICAL for d in text_decisions)

    # The examples show each column as it arrived.
    examples = dict(zip(inspection.labels, inspection.examples, strict=True))
    assert len(examples["amount"]) == MAX_EXAMPLES
    assert examples["level"] == ("'low'", "'mid'", "'high'")
    assert examples["flag"] == ("True", "False")
    assert examples["city"][:2] == ("'c1'", "'c2'")
    assert all(len(value) == MAX_EXAMPLE_LENGTH for value in examples["review"])
    assert examples["when"][0] == "2024-01-01 00:00:00"
    assert examples["fixed"] == ("7",)
    assert examples["empty"] == ()


def _inspect(X: pd.DataFrame | np.ndarray | list) -> InputInspection:
    """Inspect `X` the way fit does, with nothing declared and nothing to expand."""
    values = X if isinstance(X, np.ndarray) else np.asarray(X, dtype=object)
    schema, decisions = detect_feature_modalities(
        values,
        feature_names=None,
        min_samples_for_inference=100,
        max_unique_for_category=30,
        min_unique_for_numerical=4,
        min_cardinality_for_text=30,
    )
    _, ordinal_encoder, _ = clean_data(values, schema)
    return build_input_inspection(
        X,
        declared_positions=None,
        date_transformer=DateTransformer().fit(X),
        text_transformer=TextTransformer().fit(X),
        decisions=decisions,
        ordinal_encoder=ordinal_encoder,
    )


def test__build_input_inspection__array_input__has_positional_facts():
    X = np.column_stack([np.arange(10, dtype=float), np.arange(10) % 2])

    inspection = _inspect(X)

    assert inspection.labels is None
    assert inspection.dtypes == ("float64", "float64")
    assert inspection.declared_positions == ()
    assert inspection.category_dtype_positions == ()
    assert inspection.expansions == ()
    assert inspection.ordinal_encoded_positions == ()
    assert inspection.examples[0] == ("0.0", "1.0", "2.0", "3.0", "4.0")
    assert inspection.examples[1] == ("0.0", "1.0")
    assert inspection.position_after_expansion(1) == 1


def test__build_input_inspection__list_input__has_no_dtypes():
    X = [[1.0, "a"], [2.0, "b"], [3.0, "a"]]

    inspection = _inspect(X)

    assert inspection.labels is None
    assert inspection.dtypes is None
    assert inspection.examples == (("1.0", "2.0", "3.0"), ("'a'", "'b'"))
    assert inspection.ordinal_encoded_positions == (1,)


def test__build_input_inspection__integer_labels__are_kept_as_text():
    X = pd.DataFrame(np.arange(20, dtype=float).reshape(10, 2))

    inspection = _inspect(X)

    assert inspection.labels == ("0", "1")


def test__save_and_load_fitted_model__keeps_input_inspection(tmp_path: Path) -> None:
    X = _every_kind_of_column()
    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=DECLARED,
        inference_config={"TRANSFORM_DATES": True, "TRANSFORM_TEXT": True},
    )
    with pytest.warns(UserWarning, match="look like free text"):
        model.fit(X, np.arange(len(X)) % 2)

    save_fitted_tabpfn_model(model, tmp_path / "model.tabpfn_fit")
    loaded = load_fitted_tabpfn_model(tmp_path / "model.tabpfn_fit", device="cpu")

    assert loaded.input_inspection_ == model.input_inspection_
