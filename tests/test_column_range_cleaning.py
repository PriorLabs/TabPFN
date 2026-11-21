#  Copyright (c) Prior Labs GmbH 2026.

"""A fit input cleaned a range of columns at a time matches it cleaned whole, and
inside `overwrite_fit_input()` lets go of the caller's memory.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tabpfn.base
import tabpfn.classifier
import tabpfn.regressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.clean import (
    DELETING_COLUMNS_COPIES,
    ColumnRangeEncoders,
    overwrite_fit_input,
)
from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable

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


def _gpu_preprocessing_model(estimator_cls: type) -> TabPFNClassifier | TabPFNRegressor:
    return estimator_cls(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={"ENABLE_GPU_PREPROCESSING": True},
    )


def _targets(estimator_cls: type, n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    if estimator_cls is TabPFNClassifier:
        return rng.integers(0, 2, n)
    return rng.standard_normal(n)


def _predict(model: TabPFNClassifier | TabPFNRegressor, X: Any) -> np.ndarray:
    if isinstance(model, TabPFNClassifier):
        return model.predict_proba(X)
    return model.predict(X)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__overwrite_fit_input__frame__is_emptied_and_predicts_as_a_copy(
    estimator_cls: type,
) -> None:
    X = _mixed_frame(N_ROWS + N_TEST)
    y = _targets(estimator_cls, N_ROWS)
    expected = _gpu_preprocessing_model(estimator_cls).fit(X.iloc[:N_ROWS].copy(), y)
    given = X.iloc[:N_ROWS].copy()

    with overwrite_fit_input():
        model = _gpu_preprocessing_model(estimator_cls).fit(given, y)

    assert given.shape == (N_ROWS, 0)
    np.testing.assert_array_equal(
        _predict(model, X.iloc[N_ROWS:]), _predict(expected, X.iloc[N_ROWS:])
    )


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.int32])
def test__overwrite_fit_input__wide_array__members_share_it_narrowed(
    estimator_cls: type, dtype: type
) -> None:
    """The members read one float32 table in the caller's own memory, and predict as
    members of a fit on a copy.
    """
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((N_ROWS + N_TEST, 6)) * 1000).astype(dtype)
    y = _targets(estimator_cls, N_ROWS)
    expected = _gpu_preprocessing_model(estimator_cls).fit(X[:N_ROWS].copy(), y)
    given = X[:N_ROWS].copy()

    with overwrite_fit_input():
        model = _gpu_preprocessing_model(estimator_cls).fit(given, y)

    for member in model.executor_.ensemble_members:
        table = member.X_train
        base = table.base if isinstance(table, SharedColumnsTable) else table
        assert base.dtype == np.float32
        assert np.shares_memory(base, given)
    np.testing.assert_array_equal(
        _predict(model, X[N_ROWS:]), _predict(expected, X[N_ROWS:])
    )
    with overwrite_fit_input(), pytest.raises(ValueError, match="overwritten"):
        _gpu_preprocessing_model(estimator_cls).fit(given, y)


def test__overwrite_fit_input__wide_array_with_a_categorical__is_copied() -> None:
    """A column that is encoded keeps the table from being the array narrowed."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_ROWS + N_TEST, 4))
    X[:, 1] = rng.integers(0, 3, len(X))
    y = _targets(TabPFNClassifier, N_ROWS)
    expected = _gpu_preprocessing_model(TabPFNClassifier).fit(X[:N_ROWS].copy(), y)
    given = X[:N_ROWS].copy()

    with overwrite_fit_input():
        model = _gpu_preprocessing_model(TabPFNClassifier).fit(given, y)

    np.testing.assert_array_equal(given, X[:N_ROWS])
    np.testing.assert_array_equal(
        model.predict_proba(X[N_ROWS:]), expected.predict_proba(X[N_ROWS:])
    )


@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES, reason="The frame is emptied at the end on pandas < 2"
)
def test__overwrite_fit_input__frame__frees_the_smaller_blocks_before_the_largest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each block is freed with its last cleaned column, the largest block last, so
    it is never beside the table together with the others.
    """
    rng = np.random.default_rng(0)
    floats = rng.standard_normal((N_ROWS, 6)).astype(np.float32)
    ints = rng.integers(0, 1000, (N_ROWS, 2)).astype(np.int32)
    given = pd.concat(
        [pd.DataFrame(floats, copy=False), pd.DataFrame(ints, columns=[6, 7])],
        axis=1,
        copy=False,
    )
    # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
    blocks = {b.values.dtype: weakref.ref(b.values) for b in given._mgr.blocks}  # noqa: PD011
    del floats, ints
    alive_when_cleaned: list[tuple[str, bool]] = []
    clean_data = tabpfn.base.clean_data

    def spy(*, X: np.ndarray, **kwargs: Any) -> Any:
        alive_when_cleaned.append(
            (str(X.dtype), blocks[np.dtype(np.int32)]() is not None)
        )
        return clean_data(X=X, **kwargs)

    monkeypatch.setattr(tabpfn.base, "clean_data", spy)
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)

    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(
            given, _targets(TabPFNClassifier, N_ROWS)
        )

    first_float = next(
        i for i, (dtype, _) in enumerate(alive_when_cleaned) if dtype == "float32"
    )
    assert all(dtype == "int32" for dtype, _ in alive_when_cleaned[:first_float])
    assert not alive_when_cleaned[first_float][1]
    gc.collect()
    assert all(ref() is None for ref in blocks.values())
