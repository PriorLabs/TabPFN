#  Copyright (c) Prior Labs GmbH 2026.

"""Cleaning a fit input one range of columns at a time matches cleaning it whole."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import tabpfn.base
import tabpfn.classifier
import tabpfn.regressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.clean import (
    ColumnRangeEncoders,
    _validates_alike_in_runs,
    clean_data_transform,
)
from tabpfn.preprocessing.datamodel import FeatureModality
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
