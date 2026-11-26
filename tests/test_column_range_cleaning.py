#  Copyright (c) Prior Labs GmbH 2026.

"""Cleaning a fit input one range of columns at a time matches cleaning it whole, and
frees the caller's memory inside `overwrite_fit_input()`.
"""

from __future__ import annotations

import ctypes
import gc
import mmap
import weakref
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import tabpfn.base
import tabpfn.classifier
import tabpfn.preprocessing.clean
import tabpfn.regressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.clean import (
    DELETING_COLUMNS_COPIES,
    ColumnRangeEncoders,
    _validates_alike_in_runs,
    clean_data_transform,
    narrow_fit_input_in_place,
    overwrite_fit_input,
)
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable
from tabpfn.validation import ensure_compatible_predict_input_sklearn

N_ROWS = 240
N_TEST = 60


def _mixed_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """A frame with one column of each kind that range cleaning must handle."""
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
            # Ranges of one or two columns, so that range boundaries fall inside each
            # run of numeric or non-numeric columns.
            patch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)
        model = estimator_cls(
            n_estimators=2,
            device="cpu",
            random_state=0,
            inference_config={"ENABLE_GPU_PREPROCESSING": True},
        )
        model.fit(X.iloc[:N_ROWS], y)
        predictions = model.predict(X.iloc[N_ROWS:])
    table, _, _, schema, encoders, _ = captured["fit"]
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


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__clean_data_transform__range_encoders__match_clean_predict_input(
    estimator_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validating predict input whole and cleaning it with `ordinal_encoder_` gives
    the same table as `predict` cleans.
    """
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)
    X = _mixed_frame(N_ROWS + N_TEST)
    y = np.arange(N_ROWS) % 2
    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        inference_config={"ENABLE_GPU_PREPROCESSING": True},
    ).fit(X.iloc[:N_ROWS], y)
    X_test, _ = tabpfn.base.transform_columns(model, X.iloc[N_ROWS:])

    cleaned = clean_data_transform(
        ensure_compatible_predict_input_sklearn(X_test, model),
        cat_indices=model.inferred_feature_schema_.indices_for(
            FeatureModality.CATEGORICAL
        ),
        ord_encoder=model.ordinal_encoder_,
        passthrough_inf=model.get_inference_config().PASSTHROUGH_INF,
    )

    expected = tabpfn.base.clean_predict_input(model, X_test)
    assert cleaned.dtype == expected.dtype
    np.testing.assert_array_equal(cleaned, expected)


@pytest.mark.parametrize("container", ["frame", "array"])
def test__clean_predict_input__ranges_validated_in_runs__match_ranges_validated_alone(
    container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = N_ROWS + N_TEST
    rng = np.random.default_rng(0)
    if container == "frame":
        # Adjacent categoricals of one dtype: one converts to its categories' dtype
        # alone, but two convert to objects together.
        X = _mixed_frame(n).assign(
            codes_a=pd.Categorical(rng.integers(0, 3, n), categories=[0, 1, 2]),
            codes_b=pd.Categorical(rng.integers(0, 3, n), categories=[0, 1, 2]),
        )
        X_fit, X_test = X.iloc[:N_ROWS], X.iloc[N_ROWS:]
    else:
        X = rng.standard_normal((n, 6))
        X_fit, X_test = X[:N_ROWS], X[N_ROWS:]
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", N_ROWS * 32)
    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        inference_config={"ENABLE_GPU_PREPROCESSING": True},
    ).fit(X_fit, np.arange(N_ROWS) % 2)
    X_test, _ = tabpfn.base.transform_columns(model, X_test)
    encoders = model.ordinal_encoder_
    assert len(encoders._runs(X_test)) < len(encoders.ranges)

    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 1 << 30)
    in_runs = tabpfn.base.clean_predict_input(model, X_test)
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 1)
    alone = tabpfn.base.clean_predict_input(model, X_test)

    assert in_runs.dtype == alone.dtype
    np.testing.assert_array_equal(in_runs, alone)


@pytest.mark.parametrize(
    "column",
    [
        np.array([0.5, -1.0, np.nan, 2.0], dtype=np.float32),
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([True, False, True, True]),
        np.array(["a", None, "b", "a"], dtype=object),
        pd.array([1, None, 3, 4], dtype="Int32"),
        pd.array([True, None, False, True], dtype="boolean"),
        pd.array(["a", None, "b", "a"], dtype="string"),
        pd.Categorical([0, 1, 2, 1], categories=[0, 1, 2]),
    ],
    ids=[
        "float32",
        "int32",
        "bool",
        "object",
        "Int32",
        "boolean",
        "string",
        "category",
    ],
)
def test__validates_alike_in_runs__iff_one_column_validates_as_in_two(
    column: Any,
) -> None:
    X = pd.DataFrame({"a": column, "b": column})

    def validate(part: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            ensure_compatible_predict_input_sklearn(
                part, TabPFNClassifier(), check_finite=False
            )
        )

    alike = validate(X).iloc[:, :1].equals(validate(X.iloc[:, :1]))

    assert _validates_alike_in_runs(X.dtypes.iloc[0]) == alike


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


@pytest.mark.parametrize("gpu_preprocessing", [False, True])
def test__float32_frame_in_several_blocks__cleaned_whole_into_a_float32_table(
    monkeypatch: pytest.MonkeyPatch, gpu_preprocessing: bool
) -> None:
    captured: dict[str, Any] = {}

    def fit_spy(*args: Any, **kwargs: Any) -> Any:
        captured["fit"] = tabpfn.base.validate_and_clean_fit_input(*args, **kwargs)
        return captured["fit"]

    monkeypatch.setattr(tabpfn.classifier, "validate_and_clean_fit_input", fit_spy)
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", N_ROWS * 32)
    rng = np.random.default_rng(0)
    X = pd.concat(
        [
            pd.DataFrame(
                rng.standard_normal((N_ROWS, 2), np.float32), columns=["a", "b"]
            ),
            pd.DataFrame(
                rng.standard_normal((N_ROWS, 2), np.float32), columns=["c", "d"]
            ),
        ],
        axis=1,
    )
    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        inference_config={"ENABLE_GPU_PREPROCESSING": gpu_preprocessing},
    )
    model.fit(X, np.arange(N_ROWS) % 2)

    assert not isinstance(captured["fit"][4], ColumnRangeEncoders)
    assert captured["fit"][0].dtype == np.float32


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


@pytest.mark.parametrize(
    ("other", "refused"),
    [
        (np.arange(N_ROWS, dtype=np.int32), True),
        (np.where(np.arange(N_ROWS) % 2, "a", "b").astype(object), False),
    ],
    ids=["numeric", "strings"],
)
@pytest.mark.parametrize("by_range", [False, True])
def test__fit__infinity_after_the_first_range__refused_as_the_whole_table_is(
    monkeypatch: pytest.MonkeyPatch,
    other: np.ndarray,
    refused: bool,
    by_range: bool,
) -> None:
    """The whole-table validation reads a frame with a string column as one object
    array and allows its infinities. It refuses an infinity in a numeric frame, with
    the hint.
    """
    if by_range:
        monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", N_ROWS * 32)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"o": other, "f": rng.standard_normal(N_ROWS)})
    X.loc[3, "f"] = np.inf
    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        inference_config={"ENABLE_GPU_PREPROCESSING": True},
    )
    y = rng.integers(0, 2, N_ROWS)

    if refused:
        with pytest.raises(ValueError, match="PASSTHROUGH_INF"):
            model.fit(X, y)
        X.loc[3, "f"] = 0.0
        model.fit(X, y)
        X.loc[3, "f"] = np.inf
        with pytest.raises(ValueError, match="infinity"):
            model.predict_proba(X)
    else:
        model.fit(X, y)
        model.predict_proba(X)


def test__cleans_by_column_ranges__input_within_one_range__is_read_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = _mixed_frame(N_ROWS)

    assert not tabpfn.base._cleans_by_column_ranges(X, np.dtype(np.float32))
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)
    assert tabpfn.base._cleans_by_column_ranges(X, np.dtype(np.float32))


@pytest.fixture
def ranges_narrower_than_the_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ranges of two numeric columns. Without this, the small inputs of these tests
    are read whole, and the fit never frees or overwrites them.
    """
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)


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


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
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


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.int32])
def test__overwrite_fit_input__wide_array__members_share_it_narrowed(
    estimator_cls: type, dtype: type
) -> None:
    """The members read one float32 table in the caller's memory, and predict the
    same as after a fit on a copy.
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


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
def test__overwrite_fit_input__memory_mapped_array__leaves_the_file_as_it_is(
    tmp_path: Any,
) -> None:
    """A writeable memory map is not narrowed in place: that would write the file."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal((N_ROWS + N_TEST, 6)) * 1000
    path = tmp_path / "X.f64"
    values[:N_ROWS].tofile(path)
    given = np.memmap(path, dtype=np.float64, mode="r+", shape=(N_ROWS, 6))
    y = _targets(TabPFNClassifier, N_ROWS)
    expected = _gpu_preprocessing_model(TabPFNClassifier).fit(values[:N_ROWS], y)

    with overwrite_fit_input():
        model = _gpu_preprocessing_model(TabPFNClassifier).fit(given, y)
    given.flush()

    np.testing.assert_array_equal(np.fromfile(path), values[:N_ROWS].reshape(-1))
    np.testing.assert_array_equal(
        _predict(model, values[N_ROWS:]), _predict(expected, values[N_ROWS:])
    )


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.int32])
def test__narrow_fit_input_in_place__many_blocks__holds_the_cast_values(
    monkeypatch: pytest.MonkeyPatch, dtype: type
) -> None:
    """With one row per block and each wave's blocks in reverse order, the table holds
    `X` cast to float32.
    """

    class Backwards:
        """Runs a wave's blocks one by one, last first."""

        def __init__(self, max_workers: int) -> None: ...

        def __enter__(self) -> Backwards:
            return self

        def __exit__(self, *exc: object) -> None: ...

        def map(self, fn: Any, *spans: Any) -> list[None]:
            return [fn(*span) for span in reversed(list(zip(*spans, strict=True)))]

    monkeypatch.setattr(tabpfn.preprocessing.clean, "_NARROW_BLOCK_BYTES", 1)
    monkeypatch.setattr(tabpfn.preprocessing.clean, "ThreadPoolExecutor", Backwards)
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((1000, 3)) * 1000).astype(dtype)
    expected = X.astype(np.float32)

    table = narrow_fit_input_in_place(X)

    assert np.shares_memory(table, X)
    np.testing.assert_array_equal(table, expected)


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
def test__overwrite_fit_input__wide_array_with_a_categorical__is_copied() -> None:
    """An encoded column makes the fit copy the array instead of casting it in place."""
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


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES,
    reason="Deleting a column copies its block, so blocks are freed only at the end",
)
def test__overwrite_fit_input__frame__streams_the_largest_block_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fit cleans the largest block, copies it into the table slice by slice and
    frees it before any other range writes to the table. The smaller blocks follow.
    The model predicts the same as after a fit on a copy.
    """
    rng = np.random.default_rng(0)
    floats = rng.standard_normal((N_ROWS + N_TEST, 6)).astype(np.float32)
    ints = rng.integers(0, 1000, (N_ROWS + N_TEST, 2)).astype(np.int32)
    X = pd.concat(
        [pd.DataFrame(floats, copy=False), pd.DataFrame(ints, columns=[6, 7])],
        axis=1,
        copy=False,
    )
    del floats, ints
    y = _targets(TabPFNClassifier, N_ROWS)
    expected = _gpu_preprocessing_model(TabPFNClassifier).fit(X.iloc[:N_ROWS].copy(), y)
    X_test = X.iloc[N_ROWS:].copy()
    given = X.iloc[:N_ROWS]
    del X
    # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
    blocks = {b.values.dtype: weakref.ref(b.values) for b in given._mgr.blocks}  # noqa: PD011
    events: list[tuple[str, bool | int]] = []
    clean_data = tabpfn.base.clean_data

    def clean_spy(*, X: np.ndarray, **kwargs: Any) -> Any:
        events.append((str(X.dtype), blocks[np.dtype(np.float32)]() is not None))
        return clean_data(X=X, **kwargs)

    copy_into = tabpfn.base._StreamedBlock.copy_into

    def copy_spy(self: Any, table: np.ndarray) -> None:
        events.append(("copy", len(self.positions)))
        copy_into(self, table)

    monkeypatch.setattr(tabpfn.base, "clean_data", clean_spy)
    monkeypatch.setattr(tabpfn.base._StreamedBlock, "copy_into", copy_spy)
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)
    # Several slices of rows.
    monkeypatch.setattr(tabpfn.base, "_STREAM_BLOCK_BYTES", 7 * 6 * 4)

    with overwrite_fit_input():
        model = _gpu_preprocessing_model(TabPFNClassifier).fit(given, y)

    copy_at = events.index(("copy", 6))
    assert {kind for kind, _ in events[:copy_at]} == {"float32"}
    assert all(kind == "int32" for kind, _ in events[copy_at + 1 :])
    assert not any(alive for _, alive in events[copy_at + 1 :])
    del given
    gc.collect()
    assert all(ref() is None for ref in blocks.values())
    np.testing.assert_array_equal(
        model.predict_proba(X_test), expected.predict_proba(X_test)
    )


@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES,
    reason="Deleting a column copies its block, so blocks are freed only at the end",
)
def test__overwrite_fit_input__frame__frees_each_block_after_its_last_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A block that is not streamed is freed once its last range is cleaned, before
    the ranges of the next block.
    """
    rng = np.random.default_rng(0)
    floats = rng.standard_normal((N_ROWS, 6)).astype(np.float32)
    ints = rng.integers(0, 1000, (N_ROWS, 2)).astype(np.int32)
    wide = rng.standard_normal((N_ROWS, 3))
    given = pd.concat(
        [
            pd.DataFrame(floats, copy=False),
            pd.DataFrame(ints, columns=[6, 7], copy=False),
            pd.DataFrame(wide, columns=[8, 9, 10], copy=False),
        ],
        axis=1,
        copy=False,
    )
    del floats, ints, wide
    y = _targets(TabPFNClassifier, N_ROWS)
    # A block's `.values` is a view; the array that owns its memory is its root.
    # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
    ints_memory = next(
        weakref.ref(_root(b.values))
        for b in given._mgr.blocks
        if b.values.dtype == np.int32  # noqa: PD011
    )
    alive_while_cleaning: list[tuple[str, bool]] = []
    clean_data = tabpfn.base.clean_data

    def clean_spy(*, X: np.ndarray, **kwargs: Any) -> Any:
        alive_while_cleaning.append((str(X.dtype), ints_memory() is not None))
        return clean_data(X=X, **kwargs)

    monkeypatch.setattr(tabpfn.base, "clean_data", clean_spy)
    # Two columns per range: the int32 block is one range, cleaned before the
    # float64 block's ranges because it is smaller.
    monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * N_ROWS * 32)

    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(given, y)

    assert ("int32", True) in alive_while_cleaning
    assert [alive for kind, alive in alive_while_cleaning if kind == "float64"] == [
        False,
        False,
    ]


def _root(array: np.ndarray) -> np.ndarray:
    """The array that owns the memory `array` views."""
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array


@pytest.fixture
def pages_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Released pages read as zeros on every system, as after `madvise` on Linux."""

    def zero_pages(X: np.ndarray, start: int, stop: int | None = None) -> None:
        page = mmap.PAGESIZE
        stop = X.nbytes if stop is None else stop
        first = -(-(X.ctypes.data + start) // page) * page
        last = (X.ctypes.data + stop) // page * page
        if last > first:
            ctypes.memset(first, 0, last - first)

    monkeypatch.setattr(tabpfn.preprocessing.clean, "_release_pages", zero_pages)


PAGED_ROWS = 4096
"""Enough rows for six float32 columns to span several whole pages of 16 KiB."""


def _frame_over(values: np.ndarray, n_ints: int = 2) -> pd.DataFrame:
    """A frame whose float block is `values` itself, beside a block of ints."""
    rng = np.random.default_rng(2)
    ints = pd.DataFrame(
        rng.integers(0, 1000, (len(values), n_ints)).astype(np.int32),
        columns=range(values.shape[1], values.shape[1] + n_ints),
    )
    return pd.concat([pd.DataFrame(values, copy=False), ints], axis=1, copy=False)


@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES,
    reason="Deleting a column copies its block, so blocks are freed only at the end",
)
@pytest.mark.usefixtures("ranges_narrower_than_the_input", "pages_dropped")
def test__overwrite_fit_input__frame_over_a_read_only_array__leaves_it_as_it_is() -> (
    None
):
    rng = np.random.default_rng(0)
    values = rng.standard_normal((PAGED_ROWS, 6)).astype(np.float32)
    kept = values.copy()
    values.setflags(write=False)
    X = _frame_over(values)

    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(
            X, _targets(TabPFNClassifier, PAGED_ROWS)
        )

    np.testing.assert_array_equal(values, kept)


@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES,
    reason="Deleting a column copies its block, so blocks are freed only at the end",
)
@pytest.mark.usefixtures("ranges_narrower_than_the_input", "pages_dropped")
def test__overwrite_fit_input__frame_over_an_array_a_model_reads__leaves_it() -> None:
    """A first fit keeps the caller's float32 array. A second fit, given a frame over
    that array, does not change it.
    """
    rng = np.random.default_rng(0)
    train = rng.standard_normal((PAGED_ROWS, 6)).astype(np.float32)
    kept = train.copy()
    y = _targets(TabPFNClassifier, PAGED_ROWS)
    with overwrite_fit_input():
        first = _gpu_preprocessing_model(TabPFNClassifier).fit(train, y)
    table = first.executor_.ensemble_members[0].X_train
    base = table.base if isinstance(table, SharedColumnsTable) else table
    assert np.shares_memory(base, train)

    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(_frame_over(train), y)

    np.testing.assert_array_equal(train, kept)


@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES,
    reason="Deleting a column copies its block, so blocks are freed only at the end",
)
@pytest.mark.usefixtures("ranges_narrower_than_the_input", "pages_dropped")
def test__overwrite_fit_input__memory_given_back__a_later_fit_on_it_raises() -> None:
    rng = np.random.default_rng(0)
    values = rng.standard_normal((PAGED_ROWS, 6)).astype(np.float32)
    y = _targets(TabPFNClassifier, PAGED_ROWS)
    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(_frame_over(values), y)

    with pytest.raises(ValueError, match="overwritten"):
        _gpu_preprocessing_model(TabPFNClassifier).fit(values, y)


@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES,
    reason="Deleting a column copies its block, so blocks are freed only at the end",
)
@pytest.mark.usefixtures("ranges_narrower_than_the_input", "pages_dropped")
def test__overwrite_fit_input__released_tensor_memory__a_later_fit_on_it_raises() -> (
    None
):
    """The memory of `Tensor.numpy()` is owned by the tensor, not by an array."""
    torch.manual_seed(0)
    values = torch.randn(PAGED_ROWS, 6)
    y = _targets(TabPFNClassifier, PAGED_ROWS)
    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(_frame_over(values.numpy()), y)

    with pytest.raises(ValueError, match="overwritten"):
        _gpu_preprocessing_model(TabPFNClassifier).fit(_frame_over(values.numpy()), y)
    with pytest.raises(ValueError, match="overwritten"):
        _gpu_preprocessing_model(TabPFNClassifier).fit(values, y)


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
def test__overwrite_fit_input__narrowed_array__a_later_fit_on_a_view_raises() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_ROWS, 6))
    y = _targets(TabPFNClassifier, N_ROWS)
    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(X, y)

    with pytest.raises(ValueError, match="overwritten"):
        _gpu_preprocessing_model(TabPFNClassifier).fit(X.view(), y)


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
def test__overwrite_fit_input__narrowed_rows__a_fit_on_other_rows_is_accepted() -> None:
    """Only the rows a fit wrote into are refused, not the rest of their array."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2 * N_ROWS, 6))
    rest = X[N_ROWS:].copy()
    y = _targets(TabPFNClassifier, N_ROWS)
    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(X[:N_ROWS], y)

    with overwrite_fit_input():
        model = _gpu_preprocessing_model(TabPFNClassifier).fit(X[N_ROWS:], y)

    expected = _gpu_preprocessing_model(TabPFNClassifier).fit(rest, y)
    np.testing.assert_array_equal(
        _predict(model, rest[:N_TEST]), _predict(expected, rest[:N_TEST])
    )


@pytest.mark.skipif(
    DELETING_COLUMNS_COPIES,
    reason="Deleting a column copies its block, so blocks are freed only at the end",
)
@pytest.mark.usefixtures("ranges_narrower_than_the_input", "pages_dropped")
def test__overwrite_fit_input__frame_over_an_array_a_model_fitted_outside_reads() -> (
    None
):
    """A model fitted outside the block that keeps the caller's array is protected as
    well: a later fit inside the block does not release that memory.
    """
    rng = np.random.default_rng(0)
    train = rng.standard_normal((PAGED_ROWS, 6)).astype(np.float32)
    kept = train.copy()
    y = _targets(TabPFNClassifier, PAGED_ROWS)
    TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        fit_mode="low_memory",
        inference_config={"ENABLE_GPU_PREPROCESSING": True},
    ).fit(train, y)

    with overwrite_fit_input():
        _gpu_preprocessing_model(TabPFNClassifier).fit(_frame_over(train), y)

    np.testing.assert_array_equal(train, kept)


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
def test__overwrite_fit_input__frame__invalid_targets__leave_it_whole() -> None:
    X = _mixed_frame(N_ROWS)
    kept = X.copy()
    y = _targets(TabPFNClassifier, N_ROWS).astype(float)
    y[5] = np.nan

    with overwrite_fit_input(), pytest.raises(ValueError, match="NaN"):
        _gpu_preprocessing_model(TabPFNClassifier).fit(X, y)

    pd.testing.assert_frame_equal(X, kept)


@pytest.mark.usefixtures("ranges_narrower_than_the_input")
def test__overwrite_fit_input__frame__failing_midway__is_empty_and_refused() -> None:
    """A fit that fails after it empties the frame leaves the frame empty, not partly
    cleaned. A later fit refuses the frame.
    """
    rng = np.random.default_rng(0)
    X = _frame_over(rng.standard_normal((N_ROWS, 6)).astype(np.float32))
    # The smallest block, cleaned last.
    X[8] = rng.standard_normal(N_ROWS)
    X.loc[3, 8] = np.inf
    y = _targets(TabPFNClassifier, N_ROWS)

    with overwrite_fit_input(), pytest.raises(ValueError, match="infinity"):
        _gpu_preprocessing_model(TabPFNClassifier).fit(X, y)

    assert X.shape[1] == 0
    with pytest.raises(ValueError, match="overwritten"):
        _gpu_preprocessing_model(TabPFNClassifier).fit(X, y)
