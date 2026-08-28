#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `DateFeatureExpander`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import tabpfn.base
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.base import get_embeddings
from tabpfn.preprocessing.clean import clean_data
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.date_encoding import DateFeatureExpander

N = 20


def _numeric_and_date_schema() -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(name="input_num", modality=FeatureModality.NUMERICAL),
            Feature(name="input_signed_on", modality=FeatureModality.DATE),
        ]
    )


def _numeric_and_date_frame(dates: list[str]) -> np.ndarray:
    return np.column_stack(
        [np.arange(len(dates), dtype=object), np.array(dates, dtype=object)]
    )


def _dates(n: int = N) -> list[str]:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return dates.strftime("%Y-%m-%d").tolist()


def test__no_date_columns__is_a_noop() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    schema = FeatureSchema(
        features=[
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.NUMERICAL),
        ]
    )
    expander = DateFeatureExpander()
    X_out, schema_out = expander.fit_transform(X, schema)
    assert X_out is X
    assert schema_out is schema
    assert expander.expanded_indices == []


def test__fit__removes_raw_column_and_appends_numeric_features() -> None:
    X = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()

    expander = DateFeatureExpander()
    X_out, schema_out = expander.fit_transform(X, schema)

    assert schema_out is not None
    assert schema_out.indices_for(FeatureModality.DATE) == []
    assert all(f.modality is FeatureModality.NUMERICAL for f in schema_out.features)
    assert schema_out.num_columns == X_out.shape[1]
    assert X_out.shape[0] == N
    # More than just the original numeric column survives: the date expanded.
    assert X_out.shape[1] > 2

    assert expander.expanded_indices == [1]
    output_names = schema_out.feature_names[1:]
    assert len(output_names) == X_out.shape[1] - 1
    assert all(name.startswith("input_signed_on_") for name in output_names)
    # Every expanded feature is real-valued for a fully populated date column.
    assert np.isfinite(X_out[:, 1:].astype(float)).all()


def test__fit__output_names_are_skrubs_own_descriptive_names() -> None:
    """Skrub's own per-feature names (e.g. "_year", "_month_circular_0") are
    kept as-is, not replaced with a generic "_0", "_1", ... -- readable, and
    independent of skrub ever changing its output order.
    """
    X = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()

    _, schema_out = DateFeatureExpander().fit_transform(X, schema)

    assert schema_out.feature_names[1:] == [
        "input_signed_on_year",
        "input_signed_on_total_seconds",
        "input_signed_on_day_of_year",
        "input_signed_on_month_circular_0",
        "input_signed_on_month_circular_1",
        "input_signed_on_day_circular_0",
        "input_signed_on_day_circular_1",
        "input_signed_on_weekday_circular_0",
        "input_signed_on_weekday_circular_1",
    ]


def test__fit__every_date_column__is_expanded_unconditionally() -> None:
    """The expander weighs nothing: a `DATE` column is one to expand, full stop.

    Which columns count as dates -- including the caller's
    `categorical_features_indices`, which stops a column being called a date at
    all -- is settled in `detect_feature_modalities` before the schema gets
    here. See `TestDateLikeColumnDetection` for that half.
    """
    X = np.column_stack(
        [np.array(_dates(), dtype=object), np.array(_dates(), dtype=object)]
    )
    schema = FeatureSchema(
        features=[
            Feature(name="input_a", modality=FeatureModality.DATE),
            Feature(name="input_b", modality=FeatureModality.DATE),
        ]
    )

    expander = DateFeatureExpander()
    _, schema_out = expander.fit_transform(X, schema)

    assert expander.expanded_indices == [0, 1]
    assert schema_out.indices_for(FeatureModality.DATE) == []


def test__fit__output_names_avoid_collision_with_existing_columns() -> None:
    """A pre-existing column can happen to look like a generated output name."""
    X = np.column_stack([np.array(_dates(), dtype=object), np.zeros(N, dtype=object)])
    schema = FeatureSchema(
        features=[
            Feature(name="input_signed_on", modality=FeatureModality.DATE),
            Feature(name="input_signed_on_year", modality=FeatureModality.NUMERICAL),
        ]
    )

    _, schema_out = DateFeatureExpander().fit_transform(X, schema)

    names = schema_out.feature_names
    assert len(names) == len(set(names))
    assert "input_signed_on_year" in names
    # The pre-existing column keeps the name; the newly generated one deduped.
    assert names[0] == "input_signed_on_year"
    assert names[1] != "input_signed_on_year"


def test__expand_before_clean__vs__clean_before_expand() -> None:
    """Ordering guard: `clean_data` has no notion of `DATE` at all.

    Expand-then-clean (what fit/predict actually do) keeps the real date
    value: the year feature reads 2020. Clean-before-expand -- swapping the
    two calls -- doesn't error; `clean_data` just doesn't recognize the
    still-`DATE`-tagged column as anything special and silently ordinal-codes
    it like any other string column, replacing the dates with 0, 1, 2, ...
    """
    X = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()

    # Correct order: expand, then clean.
    X_expanded, schema_expanded = DateFeatureExpander().fit_transform(X, schema)
    X_right_order, _, schema_right_order = clean_data(X_expanded, schema_expanded)
    year_index = schema_right_order.feature_names.index("input_signed_on_year")
    np.testing.assert_array_equal(X_right_order[:, year_index], 2020.0)

    # Swapped order: clean first, on the still-DATE-tagged column.
    X_wrong_order, _, _ = clean_data(X, schema)
    date_column_index = 1  # unchanged: clean_data never expands/removes columns
    assert not np.allclose(X_wrong_order[:, date_column_index], 2020.0)
    np.testing.assert_array_equal(
        X_wrong_order[:, date_column_index], np.arange(N, dtype=float)
    )


def test__predict__reapplies_fitted_encoder_positionally() -> None:
    X_fit = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()
    expander = DateFeatureExpander()
    X_fit_out, _ = expander.fit_transform(X_fit, schema)

    X_test = _numeric_and_date_frame(_dates())
    X_test_out = expander.transform(X_test)

    assert X_test_out.shape[1] == X_fit_out.shape[1]
    # Same dates in, same encoded values out.
    np.testing.assert_array_equal(
        X_test_out[:, 1:].astype(float), X_fit_out[:, 1:].astype(float)
    )


def test__predict__value_that_no_longer_parses_as_a_date__becomes_nan() -> None:
    """A predict-time column can drift dtype like any other fitted column; a
    value that no longer parses coerces to NaN rather than crashing.
    """
    X_fit = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()
    expander = DateFeatureExpander()
    expander.fit_transform(X_fit, schema)

    dates = _dates()
    dates[3] = "not a date at all"
    X_test = _numeric_and_date_frame(dates)

    X_test_out = expander.transform(X_test)

    assert np.isnan(X_test_out[3, 1:].astype(float)).all()
    # Every other row is unaffected.
    other_rows = [i for i in range(N) if i != 3]
    assert np.isfinite(X_test_out[other_rows][:, 1:].astype(float)).all()


def test__predict__underspecified_value__becomes_nan_not_todays_date() -> None:
    """A predict-time value missing a year/month/day (e.g. a bare time) must
    not silently take on today's date -- that would make the same input map
    to different features depending on which day predict runs.
    """
    X_fit = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()
    expander = DateFeatureExpander()
    expander.fit_transform(X_fit, schema)

    dates = _dates()
    dates[3] = "12:00"
    X_test = _numeric_and_date_frame(dates)

    X_test_out = expander.transform(X_test)

    assert np.isnan(X_test_out[3, 1:].astype(float)).all()
    other_rows = [i for i in range(N) if i != 3]
    assert np.isfinite(X_test_out[other_rows][:, 1:].astype(float)).all()


@pytest.mark.parametrize(
    ("label", "column"),
    [
        # `to_datetime` refuses this column outright rather than per value, so
        # `errors="coerce"` does not cover it and it used to raise at predict.
        (
            "mixed utc offsets",
            [
                f"2020-01-0{i % 9 + 1}T00:00:00" + ("+02:00" if i % 2 else "-05:00")
                for i in range(N)
            ],
        ),
        # Not strings at all, so the ISO fast path had no `.str` to reach for.
        ("numbers", list(range(N))),
        ("bare times", ["12:00:00"] * N),
        ("garbage", [f"junk{i}" for i in range(N)]),
    ],
)
def test__predict__column_detection_would_reject__is_nan_not_an_error(
    label: str, column: list
) -> None:
    """Fit and predict must agree on what a date is.

    Detection turns a column down at fit time by returning False; expansion
    meets the same column at predict time, on data detection never saw, and has
    no such option. Both go through one parse so that a column detection would
    have rejected comes back all-NaT here -- degraded calendar features -- and
    never as an exception from a column that fit accepted.
    """
    expander = DateFeatureExpander()
    expander.fit_transform(
        _numeric_and_date_frame(_dates()), _numeric_and_date_schema()
    )

    X_test_out = expander.transform(_numeric_and_date_frame(column))

    assert np.isnan(X_test_out[:, 1:].astype(float)).all(), label


def test__mixed_date_and_datetime_string_formats__all_parse() -> None:
    """Regression: naive `pd.to_datetime` infers a format from an early value
    and silently coerces a later, differently-shaped but valid value to NaT.
    Verified directly: mixing "2020-01-01" and "2020-06-15 13:45:30" drops the
    second to NaT under the default (non-"mixed") format inference.
    """
    dates = ["2020-01-01", "2020-06-15 13:45:30", "2021-12-31 23:59:59"]
    X = _numeric_and_date_frame(dates)
    schema = _numeric_and_date_schema()

    X_out, _ = DateFeatureExpander().fit_transform(X, schema)

    assert np.isfinite(X_out[:, 1:].astype(float)).all()


def _classification_or_regression_target(
    estimator_cls: type, rng: np.random.Generator, n: int
) -> np.ndarray:
    if estimator_cls is TabPFNClassifier:
        return rng.integers(0, 2, size=n)
    return rng.normal(size=n)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_predict__transform_dates__expands_date_and_predicts(
    estimator_cls: type,
) -> None:
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)

    assert model.inferred_feature_schema_.indices_for(FeatureModality.DATE) == []
    assert 1 in model.date_expander_.expanded_indices  # "signed_on" is 2nd column

    if estimator_cls is TabPFNClassifier:
        out = model.predict_proba(X)
    else:
        out = model.predict(X)
    assert np.isfinite(out).all()


def test__fit__declared_categorical_date__transform_dates_has_no_effect() -> None:
    """A date column declared categorical must stay excluded from
    `DatetimeEncoder` -- `TRANSFORM_DATES` must not override that intent.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = rng.integers(0, 2, size=n)

    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[1],
        inference_config={"TRANSFORM_DATES": True},
    )
    model.fit(X, y)

    assert model.date_expander_.expanded_indices == []
    assert model.inferred_feature_schema_.indices_for(FeatureModality.DATE) == []
    out = model.predict_proba(X)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__predict__date_expander_attribute_missing__does_not_crash(
    estimator_cls: type,
) -> None:
    """A path that skips `_initialize_dataset_preprocessing` (e.g.
    `fit_from_preprocessed`) never sets `date_expander_` at all -- predict
    must not crash on that, the same way it already tolerates a missing
    `ordinal_encoder_`.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"num": rng.normal(size=n)})
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(n_estimators=1, device="cpu")
    model.fit(X, y)
    del model.date_expander_

    if estimator_cls is TabPFNClassifier:
        out = model.predict_proba(X)
    else:
        out = model.predict(X)
    assert np.isfinite(out).all()


def test__predict_proba_batched__transform_dates__reapplies_encoder_on_worker() -> None:
    """The ensemble-worker predict path also reapplies the fitted date encoder,
    not just the direct-`self` path.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = rng.integers(0, 2, size=n)

    clf = TabPFNClassifier(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    proba = clf.predict_proba_batched([X], [y], [X[:5]])
    assert proba.shape == (1, 5, 2)
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__get_embeddings__transform_dates__expands_before_the_ordinal_encoder(
    estimator_cls: type,
) -> None:
    """`get_embeddings` has its own predict-input path, separate from
    `predict`/`predict_proba` -- it must also expand dates first, or the
    ordinal encoder sees a column count it was never fitted with.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    model.fit(X, y)

    embeddings = get_embeddings(model, X, data_source="test")
    assert np.isfinite(embeddings).all()


def test__get_embeddings__transform_dates__categorical_indices_shift_with_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A date column expanding *before* a declared-categorical one shifts that
    column's position. `get_embeddings` must pass the post-expansion index to
    `fix_dtypes`, not the raw, pre-expansion `categorical_features_indices` --
    otherwise it silently marks the wrong (date-derived, numeric) column as
    categorical instead, with no error to reveal it.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
            "num": rng.normal(size=n),
            "cat": rng.integers(0, 3, size=n),
        }
    )
    y = rng.integers(0, 2, size=n)

    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[2],  # "cat", before expansion
        inference_config={"TRANSFORM_DATES": True},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    # The date column expanded and was moved to the end, so "cat" shifted from
    # raw index 2 down to 1 -- distinct from the stale raw index.
    assert model.categorical_features_indices == [2]
    assert model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL) == [
        1
    ]

    seen_cat_indices = []
    original_fix_dtypes = tabpfn.base.fix_dtypes

    def _spy_fix_dtypes(X, cat_indices, **kwargs) -> pd.DataFrame:
        seen_cat_indices.append(cat_indices)
        return original_fix_dtypes(X, cat_indices, **kwargs)

    monkeypatch.setattr(tabpfn.base, "fix_dtypes", _spy_fix_dtypes)

    get_embeddings(model, X, data_source="test")

    assert seen_cat_indices == [[1]]
