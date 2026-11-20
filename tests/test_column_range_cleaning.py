#  Copyright (c) Prior Labs GmbH 2026.

"""A fit input cleaned a range of columns at a time matches it cleaned whole."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import tabpfn.base
import tabpfn.classifier
import tabpfn.regressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.clean import ColumnRangeEncoders

N_ROWS = 240
N_TEST = 60


def _mixed_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """One column of each kind the range cleaning has to handle."""
    rng = np.random.default_rng(seed)
    missing = rng.random(n) < 0.1
    words = np.array(["red", "green", "blue", "cyan"], dtype=object)
    strings = words[rng.integers(0, len(words), n)]
    strings[missing] = None
    return pd.DataFrame(
        {
            "f32": rng.standard_normal(n).astype(np.float32),
            "f32_b": rng.standard_normal(n).astype(np.float32),
            "i32": rng.integers(-1000, 1000, n, dtype=np.int32),
            "few_ints": rng.integers(0, 4, n, dtype=np.int32),
            "flag": rng.random(n) < 0.3,
            "nullable": pd.arrays.IntegerArray(
                rng.integers(0, 50, n, dtype=np.int32), missing
            ),
            "category": pd.Categorical(words[rng.integers(0, len(words), n)]),
            "strings": strings,
            "constant": np.full(n, 7.0),
            "f64": rng.standard_normal(n),
        }
    )


def _fit_predict(
    estimator_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    *,
    whole_table: bool,
) -> tuple[Any, ...]:
    """The cleaned fit table, schema, encoders, cleaned test table and predictions."""
    X = _mixed_frame(N_ROWS + N_TEST)
    rng = np.random.default_rng(1)
    y = (
        rng.integers(0, 2, N_ROWS)
        if estimator_cls is TabPFNClassifier
        else rng.standard_normal(N_ROWS)
    )
    module = (
        tabpfn.classifier if estimator_cls is TabPFNClassifier else tabpfn.regressor
    )
    captured: dict[str, Any] = {}

    def fit_spy(*args: Any, **kwargs: Any) -> Any:
        captured["fit"] = tabpfn.base.validate_and_clean_fit_input(*args, **kwargs)
        return captured["fit"]

    def predict_spy(*args: Any, **kwargs: Any) -> Any:
        captured["predict"] = tabpfn.base.clean_predict_input(*args, **kwargs)
        return captured["predict"]

    monkeypatch.setattr(module, "validate_and_clean_fit_input", fit_spy)
    monkeypatch.setattr(module, "clean_predict_input", predict_spy)
    with monkeypatch.context() as patch:
        if whole_table:
            patch.setattr(tabpfn.base, "_cleans_by_column_ranges", lambda *_: False)
            patch.setattr(
                tabpfn.base, "cleaned_table_dtype", lambda _: np.dtype(np.float64)
            )
        else:
            # Ranges of a column or two, so range boundaries fall inside each run of
            # numeric or non-numeric columns.
            patch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)
        model = estimator_cls(
            n_estimators=2,
            device="cpu",
            random_state=0,
            inference_config={"ENABLE_GPU_PREPROCESSING": True},
        )
        model.fit(X.iloc[:N_ROWS], y)
        predictions = model.predict(X.iloc[N_ROWS:])
    table, _, _, schema, encoders = captured["fit"]
    return table, schema, encoders, captured["predict"], predictions


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__mixed_frame__cleaned_by_ranges__matches_the_whole_table(
    estimator_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = _fit_predict(estimator_cls, monkeypatch, whole_table=True)
    by_ranges = _fit_predict(estimator_cls, monkeypatch, whole_table=False)

    table, schema, encoders, test_table, predictions = by_ranges
    assert isinstance(encoders, ColumnRangeEncoders)
    assert len(encoders.ranges) > 2
    assert table.dtype == np.float32
    assert test_table.dtype == np.float32
    assert schema == whole[1]
    np.testing.assert_array_equal(table, whole[0].astype(np.float32))
    np.testing.assert_array_equal(test_table, whole[3].astype(np.float32))
    np.testing.assert_array_equal(predictions, whole[4])


def test__gpu_preprocessing_off__keeps_a_float64_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fit_spy(*args: Any, **kwargs: Any) -> Any:
        captured["fit"] = tabpfn.base.validate_and_clean_fit_input(*args, **kwargs)
        return captured["fit"]

    monkeypatch.setattr(tabpfn.classifier, "validate_and_clean_fit_input", fit_spy)
    X = _mixed_frame(N_ROWS)
    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        inference_config={"ENABLE_GPU_PREPROCESSING": False},
    )
    model.fit(X, np.arange(N_ROWS) % 2)

    assert captured["fit"][0].dtype == np.float64


def test__column_ranges__split_by_kind_and_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame(
        {
            "a": np.zeros(10, np.float32),
            "b": np.zeros(10, np.int32),
            "c": np.zeros(10, np.float64),
            "s": ["x"] * 10,
            "t": pd.Categorical(["y"] * 10),
            "d": np.zeros(10),
        }
    )
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * 10 * 32)

    assert list(tabpfn.base._column_ranges(X)) == [
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]
