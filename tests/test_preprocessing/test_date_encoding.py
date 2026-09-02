#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `DateTimeExpander` and the date handling built on it: a datetime
column is expanded into calendar features with `TRANSFORM_DATES=True`, and
rejected with an informative error otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

import tabpfn.base
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.base import get_embeddings
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.date_encoding import DateTimeExpander

N = 20


def _dates(n: int = N) -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))


def _numeric_and_date_frame(date_column: object) -> pd.DataFrame:
    return pd.DataFrame({"num": np.arange(N, dtype=float), "signed_on": date_column})


def _classification_or_regression_target(
    estimator_cls: type, rng: np.random.Generator, n: int
) -> np.ndarray:
    if estimator_cls is TabPFNClassifier:
        return rng.integers(0, 2, size=n)
    return rng.normal(size=n)


# --------------------------------------------------------------------------
# DateTimeExpander: basic shape / no-op behavior
# --------------------------------------------------------------------------


def test__no_temporal_columns__is_a_noop() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    expander = DateTimeExpander().fit(X)
    assert expander.transform(X) is X
    assert expander.encoders_ == {}
    assert expander.output_indices_for([0, 1]) == [0, 1]


def test__not_a_dataframe__is_a_noop() -> None:
    """Regression: `output_indices_for` must still answer the identity here --
    a declared `categorical_features_indices` on a plain-array fit needs to
    look itself up.
    """
    X = np.array([[1.0, 2.0]])
    expander = DateTimeExpander().fit(X)
    assert expander.transform(X) is X
    assert expander.encoders_ == {}
    assert expander.output_indices_for([0, 1]) == [0, 1]


def test__transform__before_fit__raises() -> None:
    """It is a transformer: nothing is resolved before the decision is made."""
    with pytest.raises(NotFittedError):
        DateTimeExpander().transform(_numeric_and_date_frame(_dates()))


def test__input_frame_is_not_mutated() -> None:
    X = _numeric_and_date_frame(_dates())
    before = X.dtypes.tolist()
    DateTimeExpander(transform_dates=True).fit(X).transform(X)
    assert X.dtypes.tolist() == before


def test__values_of_the_input_frame_are_not_written_through() -> None:
    """The copy is shallow, so this pins that no value is written through it."""
    X = _numeric_and_date_frame(_dates())
    before = X.copy(deep=True)
    DateTimeExpander(transform_dates=True).fit(X).transform(X)
    pd.testing.assert_frame_equal(X, before)


def test__duplicate_column_labels__are_replaced_by_position() -> None:
    """The labels are the caller's, so they can repeat -- the same case
    `build_input_feature_names` exists for. Replacing by label would be
    ambiguous, so replacement is positional.
    """
    X = pd.concat(
        [pd.Series([1.0, 2.0, 3.0]), pd.Series(pd.date_range("2020-01-01", periods=3))],
        axis=1,
    )
    X.columns = ["same", "same"]

    expander = DateTimeExpander(transform_dates=True).fit(X)
    out = expander.transform(X)

    assert out.iloc[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert list(out.columns) == ["same", *expander.output_names_[1]]
    assert out["same_year"].tolist() == [2020, 2020, 2020]


def test__non_unique_index__rows_stay_aligned() -> None:
    """Expansion must not depend on the index being unique or ordered."""
    X = pd.DataFrame(
        {"n": [1.0, 2.0, 3.0], "d": pd.date_range("2020-01-01", periods=3)},
        index=[7, 7, 2],
    )
    out = DateTimeExpander(transform_dates=True).fit(X).transform(X)
    assert out["n"].tolist() == [1.0, 2.0, 3.0]
    assert out["d_day_of_year"].tolist() == [1, 2, 3]


def test__timedelta__becomes_seconds_regardless_of_transform_dates() -> None:
    """A duration is a quantity, not a point on a calendar, so it needs no
    opt-in and is never expanded.
    """
    X = pd.DataFrame({"d": pd.to_timedelta([1, 2, 3], unit="D")})
    expander = DateTimeExpander().fit(X)
    out = expander.transform(X)
    assert expander.encoders_ == {}
    assert out["d"].tolist() == [86400.0, 172800.0, 259200.0]


# --------------------------------------------------------------------------
# Datetime dtypes: rejected by default, expanded with transform_dates=True
# --------------------------------------------------------------------------

_DATETIME_COLUMNS = [
    pytest.param(pd.date_range("2020-01-01", periods=3), id="datetime64"),
    pytest.param(pd.date_range("2020-01-01", periods=3, tz="UTC"), id="tz aware"),
    pytest.param(
        pd.date_range("2020-01-01 13:45", periods=3, freq="D"), id="with time"
    ),
    pytest.param(pd.date_range("2020-01-01", periods=3).to_period("M"), id="period"),
]


@pytest.mark.parametrize("column", _DATETIME_COLUMNS)
def test__fit__datetime_column__is_rejected_by_default(column: pd.Index) -> None:
    X = pd.DataFrame({"n": [1.0, 2.0, 3.0], "d": column})
    with pytest.raises(TabPFNValidationError, match="TRANSFORM_DATES"):
        DateTimeExpander().fit(X)


@pytest.mark.parametrize("column", _DATETIME_COLUMNS)
def test__fit__datetime_column__expands_with_transform_dates(column: pd.Index) -> None:
    X = pd.DataFrame({"n": [1.0, 2.0, 3.0], "d": column})
    expander = DateTimeExpander(transform_dates=True).fit(X)
    out = expander.transform(X)
    assert expander.expanded_input_indices == [1]
    assert out["d_year"].tolist() == [2020, 2020, 2020]
    assert not any(pd.api.types.is_datetime64_any_dtype(d) for d in out.dtypes)


def test__fit__rejection_names_the_columns_and_the_way_out() -> None:
    X = pd.DataFrame({f"d{i}": _dates() for i in range(12)})
    with pytest.raises(TabPFNValidationError) as excinfo:
        DateTimeExpander().fit(X)
    message = str(excinfo.value)
    assert "'d0'" in message
    assert "(and 2 more)" in message
    assert "TRANSFORM_DATES" in message
    assert ".astype(str)" in message


def test__fit__declared_categorical_date_column__is_rejected() -> None:
    """A datetime column expands into many numerical columns, so there is no
    single column a categorical declaration could apply to.
    """
    X = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True, categorical_features_indices=[1])
    with pytest.raises(TabPFNValidationError, match="categorical_features_indices"):
        expander.fit(X)


# --------------------------------------------------------------------------
# transform_dates=True: expand into calendar features
# --------------------------------------------------------------------------


def test__fit__expands_and_removes_the_raw_column() -> None:
    X = _numeric_and_date_frame(_dates())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expander = DateTimeExpander(transform_dates=True).fit(X)
        out = expander.transform(X)

    assert expander.expanded_input_indices == [1]
    assert expander.output_indices_for([0]) == [0]
    output_names = expander.output_names_[1]
    assert "signed_on" not in out.columns
    assert all(name.startswith("signed_on_") for name in output_names)
    assert out.shape == (N, 1 + len(output_names))
    # Every expanded feature is real-valued for a fully populated date column.
    assert np.isfinite(out[output_names].to_numpy()).all()
    assert expander.expanded_output_indices == list(range(1, 1 + len(output_names)))


def test__fit__output_names_are_skrubs_own_descriptive_names() -> None:
    """Skrub's own per-feature names (e.g. "_year", "_month_circular_0") are
    kept as-is, unprefixed -- prefixing and deduplication happen exactly
    once, later, in `build_input_feature_names`.
    """
    X = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True).fit(X)
    assert expander.output_names_[1] == [
        "signed_on_year",
        "signed_on_total_seconds",
        "signed_on_day_of_year",
        "signed_on_month_circular_0",
        "signed_on_month_circular_1",
        "signed_on_day_circular_0",
        "signed_on_day_circular_1",
        "signed_on_weekday_circular_0",
        "signed_on_weekday_circular_1",
    ]


def test__fit__multiple_date_columns__each_expand() -> None:
    X = pd.DataFrame({"a": _dates(), "b": _dates()})
    expander = DateTimeExpander(transform_dates=True).fit(X)
    out = expander.transform(X)
    assert expander.expanded_input_indices == [0, 1]
    total_width = len(expander.output_names_[0]) + len(expander.output_names_[1])
    assert out.shape[1] == total_width
    # Nothing is kept, so the expanded block starts at position 0.
    assert expander.expanded_output_indices == list(range(total_width))


def test__output_feature_names__splices_kept_and_expanded_names() -> None:
    X = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True).fit(X)
    merged = expander.output_feature_names(["num", "signed_on"])
    assert merged is not None
    assert merged[0] == "num"
    assert merged[1:] == expander.output_names_[1]
    assert len(merged) == expander.transform(X).shape[1]


def test__output_feature_names__none_in_none_out() -> None:
    X = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True).fit(X)
    assert expander.output_feature_names(None) is None


# --------------------------------------------------------------------------
# Predict-time reapplication
# --------------------------------------------------------------------------


def test__predict__reapplies_fitted_encoder_positionally() -> None:
    X_fit = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True).fit(X_fit)
    fit_out = expander.transform(X_fit)

    test_out = expander.transform(_numeric_and_date_frame(_dates()))

    output_names = expander.output_names_[1]
    np.testing.assert_array_equal(
        test_out[output_names].to_numpy(), fit_out[output_names].to_numpy()
    )


def test__predict__differently_labelled_date_column__keeps_the_fit_time_names() -> None:
    """Skrub's `DatetimeEncoder` names its outputs after the label of the
    column it is handed, and rewrites its own `all_outputs_` on every
    transform. The expander snapshots those names at fit, so a predict frame
    that labels the column differently still produces the fitted layout
    rather than silently renaming the fitted features.
    """
    X_fit = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True).fit(X_fit)
    fitted_names = list(expander.output_names_[1])

    X_test = _numeric_and_date_frame(_dates()).rename(columns={"signed_on": "renamed"})
    out = expander.transform(X_test)

    assert list(out.columns) == ["num", *fitted_names]
    assert expander.output_names_[1] == fitted_names


def test__predict__no_native_value_for_a_fitted_date_column__becomes_nan() -> None:
    """A predict-time column can drift dtype like any other fitted column,
    and detection has already moved on by then -- there is no re-detection
    to fall back to. A fitted date column that is no longer a genuine
    datetime dtype degrades to NaN, the same as any other missing value,
    rather than a best-effort parse of whatever is actually sitting there.
    """
    X_fit = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True).fit(X_fit)

    out = expander.transform(_numeric_and_date_frame(["whatever, never inspected"] * N))

    assert np.isnan(out[expander.output_names_[1]].to_numpy()).all()


def test__predict__datetime_column_that_was_not_a_date_at_fit__is_rejected() -> None:
    """Which positions expand is settled at fit; a datetime dtype turning up
    elsewhere at predict time cannot be expanded and is not silently read as
    something else either.
    """
    X_fit = _numeric_and_date_frame(["a string, not a date"] * N)
    expander = DateTimeExpander(transform_dates=True).fit(X_fit)
    assert expander.encoders_ == {}

    with pytest.raises(TabPFNValidationError, match="did not when `fit` ran"):
        expander.transform(_numeric_and_date_frame(_dates()))


def test__predict__native_value_has_a_missing_row__only_that_row_becomes_nan() -> None:
    """A still-genuine datetime column can carry a per-row `NaT`; that row
    degrades to NaN and the rest of the column is unaffected.
    """
    X_fit = _numeric_and_date_frame(_dates())
    expander = DateTimeExpander(transform_dates=True).fit(X_fit)

    dates_with_a_gap = _dates()
    dates_with_a_gap.iloc[3] = pd.NaT
    out = expander.transform(_numeric_and_date_frame(dates_with_a_gap))

    values = out[expander.output_names_[1]].to_numpy()
    assert np.isnan(values[3]).all()
    other_rows = [i for i in range(N) if i != 3]
    assert np.isfinite(values[other_rows]).all()


# --------------------------------------------------------------------------
# End-to-end, through the public estimator API
# --------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_predict__native_datetime_column_with_missing_value__does_not_crash(
    estimator_cls: type,
) -> None:
    """Regression: a genuine `datetime64` column with a `NaT` must not crash
    `fit`/`predict` under `TRANSFORM_DATES=True` -- the missing row's calendar
    features degrade to `NaN`, like any other missing value.
    """
    n = 60
    rng = np.random.default_rng(0)
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))
    dates.iloc[3] = pd.NaT
    X = pd.DataFrame({"num": rng.normal(size=n), "signed_on": dates})
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)
        out = (
            model.predict_proba(X)
            if estimator_cls is TabPFNClassifier
            else model.predict(X)
        )

    assert np.isfinite(out).all()


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

    assert model.date_expander_.expanded_input_indices == [1]  # "signed_on"
    # Every calendar feature is tagged numerical outright, however few distinct
    # values it holds (e.g. `year` on a dataset spanning one year).
    schema = model.inferred_feature_schema_
    assert schema.indices_for(FeatureModality.NUMERICAL) == list(
        range(len(schema.feature_names))
    )

    if estimator_cls is TabPFNClassifier:
        out = model.predict_proba(X)
    else:
        out = model.predict(X)
    assert np.isfinite(out).all()


def test__fit__records_the_raw_input_shape_and_predict_checks_against_it() -> None:
    """`feature_names_in_`/`n_features_in_` describe the frame the caller
    passed, not the wider expanded one -- and a predict frame is checked
    against that raw shape, before expansion could hide a mismatch.
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
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    model.fit(X, y)

    assert list(model.feature_names_in_) == ["num", "signed_on"]
    assert model.n_features_in_ == 2
    assert len(model.inferred_feature_schema_.feature_names) > 2

    with pytest.raises(TabPFNValidationError, match="feature names"):
        model.predict(X.rename(columns={"signed_on": "renamed"}))
    with pytest.raises(TabPFNValidationError, match="2 features"):
        model.predict(X.to_numpy(dtype=object)[:, :1])


def test__fit__datetime64_numpy_array__points_at_a_dataframe() -> None:
    """Only a DataFrame column can be a date. A bare `datetime64` array has no
    per-column dtype to recast, so it is rejected with the way out, rather
    than with numpy's own promotion error.
    """
    n = 60
    X = pd.date_range("2020-01-01", periods=n, freq="D").to_numpy().reshape(-1, 1)
    y = np.arange(n) % 2

    with pytest.raises(ValueError, match=r"pandas DataFrame.*TRANSFORM_DATES"):
        TabPFNClassifier(n_estimators=1, device="cpu").fit(X, y)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__datetime_column_without_transform_dates__raises_naming_the_flag(
    estimator_cls: type,
) -> None:
    """With the flag off, a datetime column is a clear error at `fit`, not the
    opaque numpy promotion error sklearn's validation would otherwise hit.
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

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.raises(TabPFNValidationError, match="'signed_on'") as excinfo:
        model.fit(X, y)
    assert "TRANSFORM_DATES" in str(excinfo.value)


def test__fit_with_differentiable_input__leaves_a_usable_expander() -> None:
    """The differentiable fit path sets `date_expander_` too: `predict` is
    reachable from it, and transforms through the expander unconditionally --
    a tensor fit simply has nothing to expand.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = torch.as_tensor(rng.normal(size=(n, 3)), dtype=torch.float32)
    y = torch.as_tensor(rng.normal(size=n), dtype=torch.float32)

    model = TabPFNRegressor(n_estimators=1, device="cpu", differentiable_input=True)
    model.fit_with_differentiable_input(X, y)

    assert model.date_expander_.encoders_ == {}
    assert model.date_expander_.transform(X) is X


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
