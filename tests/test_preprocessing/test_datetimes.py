#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `DateTransformer`: refusing a date, expanding it, converting a duration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.inference_tuning import ClassifierTuningConfig, RegressorTuningConfig
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.datetimes import DateTransformer


def _frame(dates: pd.Series | pd.DatetimeIndex | pd.PeriodIndex | list) -> pd.DataFrame:
    return pd.DataFrame({"num": [1.0, 2.0, 3.0], "date": dates})


def _estimator_data(
    estimator_cls: type, column: pd.DatetimeIndex | pd.TimedeltaIndex
) -> tuple[pd.DataFrame, np.ndarray]:
    """A numeric column next to `column`, plus a `y` matching the estimator."""
    n = len(column)
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame({"num": rng.normal(size=n), "date": column})
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )
    return X, y


class TestRefusal:
    """`TRANSFORM_DATES` off: a point in time is refused, naming it.

    Runs on the raw DataFrame, before validation flattens it into one numpy
    array, so a genuine temporal dtype is still visible. Dtype is all it reads:
    a string that merely looks like a date is not a date here.
    """

    def test__datetime_columns__are_refused_by_index_and_label(self) -> None:
        X = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "start": pd.date_range("2020-01-01", periods=3),
                "end": pd.date_range("2021-01-01", periods=3),
            }
        )

        with pytest.raises(TabPFNValidationError, match="does not support") as excinfo:
            DateTransformer().fit_transform(X)

        message = str(excinfo.value)
        assert "1 ('start'), 2 ('end')" in message
        assert '"TRANSFORM_DATES": True' in message

    @pytest.mark.parametrize(
        "dates",
        [
            pd.date_range("2020-01-01", periods=3, tz="Europe/Berlin"),
            pd.period_range("2020-01-01", periods=3, freq="D"),
        ],
        ids=["tz_aware", "period"],
    )
    def test__other_points_in_time__are_refused_too(
        self, dates: pd.DatetimeIndex | pd.PeriodIndex
    ) -> None:
        with pytest.raises(TabPFNValidationError, match=r"1 \('date'\)"):
            DateTransformer().fit_transform(_frame(dates))

    def test__date_like_string_column__is_not_refused(self) -> None:
        X = _frame(["2020-01-01", "2020-01-02", "2020-01-03"])
        assert DateTransformer().fit_transform(X) is X

    def test__declared_categorical_date__is_refused_like_any_other(self) -> None:
        """Declaring the column categorical does not make it supported."""
        X = _frame(pd.date_range("2020-01-01", periods=3))

        with pytest.raises(TabPFNValidationError, match=r"1 \('date'\)"):
            DateTransformer(categorical_indices=[1]).fit_transform(X)

    def test__non_dataframe_input__is_a_noop(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        transformer = DateTransformer()
        assert transformer.fit_transform(X) is X
        assert transformer.transform(X) is X

    def test__transform__refuses_a_date_too(self) -> None:
        transformer = DateTransformer()
        transformer.fit_transform(_frame([1.0, 2.0, 3.0]))

        with pytest.raises(TabPFNValidationError, match=r"1 \('date'\)"):
            transformer.transform(_frame(pd.date_range("2020-01-01", periods=3)))


class TestDurations:
    """A duration has no calendar, so its length in seconds is all of its meaning:
    converted flag or no flag, and nothing to refuse.
    """

    @pytest.mark.parametrize("transform_dates", [False, True])
    def test__duration_column__becomes_seconds(self, transform_dates: bool) -> None:
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

        out = DateTransformer(transform_dates=transform_dates).fit_transform(X)

        np.testing.assert_array_equal(out["date"], [86400.0, 172800.0, 259200.0])
        # The caller's own frame is left untouched.
        assert pd.api.types.is_timedelta64_dtype(X["date"])

    def test__declared_categorical_duration__is_still_converted(self) -> None:
        """Unlike a point in time: leaving it alone only crashes validation, and
        a whole number of seconds ordinal-encodes as a category just as well.
        """
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

        out = DateTransformer(categorical_indices=[1]).fit_transform(X)

        assert pd.api.types.is_numeric_dtype(out["date"])

    def test__transform__converts_the_same_way(self) -> None:
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))
        transformer = DateTransformer()
        fitted = transformer.fit_transform(X)

        np.testing.assert_array_equal(transformer.transform(X)["date"], fitted["date"])

    def test__duplicate_column_labels__are_converted_positionally(self) -> None:
        """Pandas allows repeated labels, so only position identifies a column."""
        X = pd.DataFrame({0: pd.to_timedelta([1, 2, 3], unit="D"), 1: [1.0, 2.0, 3.0]})
        X.columns = ["same", "same"]

        out = DateTransformer().fit_transform(X)

        assert [str(dtype) for dtype in out.dtypes] == ["float64", "float64"]
        np.testing.assert_array_equal(out.iloc[:, 1], X.iloc[:, 1])


class TestExpansion:
    """`TRANSFORM_DATES` on: a date becomes calendar features instead of an error."""

    def _expander(self, **kwargs: object) -> DateTransformer:
        return DateTransformer(transform_dates=True, **kwargs)  # type: ignore[arg-type]

    def test__date_column__becomes_calendar_features(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3, freq="37h"))

        transformer = self._expander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = transformer.fit_transform(X)

        names = transformer.feature_names_out_
        assert out.shape[0] == 3
        assert out.shape[1] > X.shape[1]
        assert list(out.columns) == names
        # The kept columns come first, in order, then the expansion.
        assert names[0] == "num"
        assert all(name.startswith("date_") for name in names[1:])
        assert "date_year" in names
        assert out.notna().all().all()

    def test__period_column__is_expanded_from_the_instant_it_starts_at(self) -> None:
        """A period is a span, not an instant; its start is the instant that
        orders identically, which is all the encoder needs.
        """
        X = _frame(pd.period_range("2020-01-01", periods=3, freq="D"))

        transformer = self._expander()
        out = transformer.fit_transform(X)

        assert "date_year" in transformer.feature_names_out_
        assert out.notna().all().all()

    def test__declared_categorical_indices__are_remapped(self) -> None:
        """The expanded column is gone from where it was, so what came after it
        has moved down.
        """
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "cat": ["a", "b", "c"],
            }
        )

        transformer = self._expander(categorical_indices=[1])
        transformer.fit_transform(X)

        assert transformer.output_indices([1]) == [0]
        assert transformer.output_indices(None) is None
        assert transformer.feature_names_out_[0] == "cat"

    def test__declared_categorical_date__is_refused(self) -> None:
        """Expansion makes many numeric columns of it, so there is no single
        column the declaration could apply to.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))

        with pytest.raises(TabPFNValidationError, match="listed in") as excinfo:
            self._expander(categorical_indices=[1]).fit_transform(X)
        assert "1 ('date')" in str(excinfo.value)

    def test__refit_on_a_frame_with_nothing_to_expand__forgets_the_last_fit(
        self,
    ) -> None:
        """Otherwise `transform` reapplies encoders for the previous input's
        columns, and comes out a width the new fit never produced.
        """
        transformer = self._expander()
        transformer.fit_transform(_frame(pd.date_range("2020-01-01", periods=3)))
        transformer.fit_transform(np.zeros((3, 2)))

        assert transformer.expanded_indices == []
        assert transformer.transform(_frame([1.0, 2.0, 3.0])).shape[1] == 2

    def test__expanded_indices__reports_what_was_expanded(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        transformer = self._expander()
        transformer.fit_transform(X)
        assert transformer.expanded_indices == [1]

    def test__generated_name_colliding_with_an_existing_column__is_deduped(
        self,
    ) -> None:
        X = pd.DataFrame(
            {
                "date_year": [1.0, 2.0, 3.0],
                "date": pd.date_range("2020-01-01", periods=3),
            }
        )

        transformer = self._expander()
        transformer.fit_transform(X)

        names = transformer.feature_names_out_
        assert len(set(names)) == len(names)
        assert "date_year" in names
        assert "date_year_1" in names


class TestExpansionAtPredictTime:
    """`transform` reapplies the encoder fit on the training column, not a new one."""

    def test__same_data__reproduces_the_fitted_columns(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3, freq="37h"))
        transformer = DateTransformer(transform_dates=True)
        fitted = transformer.fit_transform(X)

        out = transformer.transform(X)

        assert list(out.columns) == list(fitted.columns)
        np.testing.assert_array_equal(out.to_numpy(), fitted.to_numpy())

    def test__predict_data_with_a_time_of_day__keeps_the_fitted_width(self) -> None:
        """The encoder decides how many features it makes when it is fit: a date-only
        training column has no hour to encode, and predict cannot add one without
        making the frame the wrong width for the fitted model.
        """
        transformer = DateTransformer(transform_dates=True)
        fitted = transformer.fit_transform(
            _frame(pd.date_range("2020-01-01", periods=3))
        )

        out = transformer.transform(
            _frame(pd.date_range("2020-01-01 13:45", periods=3, freq="37h"))
        )

        assert list(out.columns) == list(fitted.columns)

    def test__position_that_is_no_longer_a_date__is_refused(self) -> None:
        """The encoder fitted there needs dates; strings are not parsed for it."""
        transformer = DateTransformer(transform_dates=True)
        transformer.fit_transform(_frame(pd.date_range("2020-01-01", periods=3)))

        with pytest.raises(TabPFNValidationError, match="but do not now") as excinfo:
            transformer.transform(_frame(["2020-01-01", "2020-01-02", "2020-01-03"]))
        assert "1 ('date')" in str(excinfo.value)

    def test__date_at_a_position_that_was_not_one_at_fit__is_refused(self) -> None:
        """No encoder was fit for it, so there is nothing to expand it with, and
        the flag is already on, so the error must not point at it.
        """
        transformer = DateTransformer(transform_dates=True)
        transformer.fit_transform(_frame(pd.date_range("2020-01-01", periods=3)))

        with pytest.raises(TabPFNValidationError, match="did not when") as excinfo:
            transformer.transform(
                pd.DataFrame(
                    {
                        "num": pd.date_range("2020-01-01", periods=3),
                        "date": pd.date_range("2020-01-01", periods=3),
                    }
                )
            )
        assert "0 ('num')" in str(excinfo.value)
        assert "TRANSFORM_DATES" not in str(excinfo.value)

    @pytest.mark.parametrize(
        ("fit_tz", "predict_tz"),
        [("UTC", "America/Los_Angeles"), ("UTC", None), (None, "UTC")],
        ids=["other_zone", "aware_to_naive", "naive_to_aware"],
    )
    def test__column_in_another_timezone_than_at_fit__is_refused(
        self, fit_tz: str | None, predict_tz: str | None
    ) -> None:
        """The encoder reads calendar features in the column's own timezone, so
        the same instants in another zone come out as other days and hours.
        """

        def frame(tz: str | None) -> pd.DataFrame:
            dates = pd.date_range("2020-08-28 03:00", periods=3, freq="D", tz="UTC")
            return _frame(
                dates.tz_localize(None) if tz is None else dates.tz_convert(tz)
            )

        transformer = DateTransformer(transform_dates=True)
        transformer.fit_transform(frame(fit_tz))

        with pytest.raises(TabPFNValidationError, match="another timezone") as excinfo:
            transformer.transform(frame(predict_tz))
        assert f"1 ('date': {predict_tz or 'naive'}, was {fit_tz or 'naive'})" in str(
            excinfo.value
        )

    def test__same_timezone_as_at_fit__passes(self) -> None:
        X = _frame(pd.date_range("2020-08-28 03:00", periods=3, freq="D", tz="UTC"))
        transformer = DateTransformer(transform_dates=True)
        fitted = transformer.fit_transform(X)

        pd.testing.assert_frame_equal(transformer.transform(X), fitted)

    def test__array_after_an_expanding_fit__is_refused(self) -> None:
        """Its raw width matches the fit input, so only the transformer can tell
        that it cannot be widened to the expanded layout.
        """
        transformer = DateTransformer(transform_dates=True)
        transformer.fit_transform(_frame(pd.date_range("2020-01-01", periods=3)))

        with pytest.raises(TabPFNValidationError, match="has to be a DataFrame"):
            transformer.transform(np.zeros((3, 2)))

    def test__array_after_a_fit_that_expanded_nothing__passes(self) -> None:
        transformer = DateTransformer(transform_dates=True)
        transformer.fit_transform(_frame([1.0, 2.0, 3.0]))

        X = np.zeros((3, 2))
        assert transformer.transform(X) is X


class TestInterface:
    """`fit`, `fit_transform` and `transform` relate the way sklearn's do."""

    def test__fit__returns_itself_and_transform_matches_fit_transform(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3, freq="37h"))

        fitted = DateTransformer(transform_dates=True).fit(X)
        via_transform = fitted.transform(X)
        via_fit_transform = DateTransformer(transform_dates=True).fit_transform(X)

        assert isinstance(fitted, DateTransformer)
        pd.testing.assert_frame_equal(via_transform, via_fit_transform)

    def test__transform__before_fit__raises(self) -> None:
        with pytest.raises(NotFittedError):
            DateTransformer().transform(_frame([1.0, 2.0, 3.0]))

    def test__expanded_indices__before_fit__raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = DateTransformer().expanded_indices

    def test__fit_on_an_array__has_no_output_names(self) -> None:
        """An array carries no labels, so there are none to report, and no
        column was expanded, so every index stays where it was.
        """
        transformer = DateTransformer(transform_dates=True).fit(np.zeros((3, 2)))

        assert transformer.feature_names_out_ is None
        assert transformer.output_indices([0, 1]) == [0, 1]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_real_datetime_column__raises_naming_it(estimator_cls: type) -> None:
    """`fit` refuses a genuine `datetime64` column, naming it.

    Both estimators share the transformer, so one parametrized test pins the
    estimator-level behaviour, including that declaring the column in
    `categorical_features_indices` changes nothing about the refusal.
    """
    X, y = _estimator_data(estimator_cls, pd.date_range("2020-01-01", periods=120))

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.raises(TabPFNValidationError, match=r"1 \('date'\)"):
        model.fit(X, y)

    model = estimator_cls(
        n_estimators=1, device="cpu", categorical_features_indices=[1]
    )
    with pytest.raises(TabPFNValidationError, match=r"1 \('date'\)"):
        model.fit(X, y)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__predict_with_real_datetime_column__raises_naming_it(
    estimator_cls: type,
) -> None:
    """A column cast to a number at fit but left as `datetime64` at predict is
    refused there too, before validation compares it against the fitted frame.
    """
    X, y = _estimator_data(estimator_cls, pd.date_range("2020-01-01", periods=120))
    X_numeric = X.assign(date=X["date"].astype("int64"))

    model = estimator_cls(n_estimators=1, device="cpu").fit(X_numeric, y)
    with pytest.raises(TabPFNValidationError, match=r"1 \('date'\)"):
        model.predict(X)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_duration_column__no_longer_crashes(estimator_cls: type) -> None:
    """A `timedelta64` column used to abort a `fit` outright: it has no common
    numpy dtype with the numeric column beside it.
    """
    X, y = _estimator_data(estimator_cls, pd.to_timedelta(np.arange(120), unit="D"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = estimator_cls(n_estimators=1, device="cpu").fit(X, y)
        predictions = model.predict(X)

    assert len(predictions) == len(X)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_dates__expands_the_date_column(
    estimator_cls: type,
) -> None:
    """`TRANSFORM_DATES` reaches the transformer, and the wider frame survives
    the whole fit/predict path: the schema describes the calendar features.
    """
    X, y = _estimator_data(
        estimator_cls, pd.date_range("2020-01-01", periods=150, freq="37h")
    )
    X = X.rename(columns={"date": "signed_on"})

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X, y)
    predictions = model.predict(X)

    names = [feature.name for feature in model.inferred_feature_schema_.features]
    assert len(names) > X.shape[1]
    assert any(name.endswith("signed_on_year") for name in names)
    assert any(name.endswith("signed_on_weekday_circular_0") for name in names)
    assert len(predictions) == len(X)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_dates__reports_the_caller_s_own_columns(
    estimator_cls: type,
) -> None:
    """Expansion widens the frame internally, which must not leak into the
    sklearn attributes: they describe what the caller passed, so the error a
    caller gets for the wrong columns names their columns, not ours.
    """
    X, y = _estimator_data(
        estimator_cls, pd.date_range("2020-01-01", periods=150, freq="37h")
    )
    X = X.rename(columns={"date": "signed_on"})

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    ).fit(X, y)

    assert model.n_features_in_ == 2
    assert list(model.feature_names_in_) == ["num", "signed_on"]
    # The schema is the internal, expanded view, and stays that way.
    assert len(model.inferred_feature_schema_.features) > 2

    with pytest.raises(TabPFNValidationError, match="feature names should match"):
        model.predict(X.assign(extra=1.0))
    with pytest.raises(TabPFNValidationError, match="feature names should match"):
        model.predict(X.rename(columns={"num": "nums"}))
    # Positional input has no names to check, only a width.
    with pytest.raises(TabPFNValidationError, match="expecting 2 features"):
        model.predict(np.zeros((10, 3)))
    # The right width, but no columns to expand.
    with pytest.raises(TabPFNValidationError, match="has to be a DataFrame"):
        model.predict(np.zeros((10, 2)))


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_differentiable_input__sets_a_date_transformer(
    estimator_cls: type,
) -> None:
    """The differentiable path fits on a tensor, which holds no dates, and still
    sets the attribute: every predict path converts through it unconditionally.
    """
    X = torch.randn(20, 3)
    y = (
        torch.tensor([0, 1] * 10)
        if estimator_cls is TabPFNClassifier
        else torch.randn(20)
    )
    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        differentiable_input=True,
        ignore_pretraining_limits=True,
    ).fit_with_differentiable_input(X, y)

    assert model.date_transformer_.transform(np.zeros((2, 3))).shape == (2, 3)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_dates_and_tuning__tuning_estimator_gets_shifted_indices(
    estimator_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tuning estimators are fit on the expanded array, where a declared
    categorical column has moved down past the expanded date, so they must be
    handed the shifted index rather than the caller's.
    """
    n = 80
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "num": rng.normal(size=n),
        }
    )
    is_classifier = estimator_cls is TabPFNClassifier
    y = rng.integers(0, 2, size=n) if is_classifier else rng.normal(size=n)
    config_cls = ClassifierTuningConfig if is_classifier else RegressorTuningConfig
    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[1],
        inference_config={"TRANSFORM_DATES": True},
        tuning_config=config_cls(
            calibrate_temperature=True, tuning_holdout_frac=0.25, tuning_n_folds=1
        ),
    )
    getter = "_get_tuning_classifier" if is_classifier else "_get_tuning_regressor"
    tuning_estimators: list[TabPFNClassifier | TabPFNRegressor] = []
    original = getattr(model, getter)

    def capture(**kwargs: object) -> TabPFNClassifier | TabPFNRegressor:
        tuning_estimators.append(original(**kwargs))
        return tuning_estimators[-1]

    monkeypatch.setattr(model, getter, capture)
    with warnings.catch_warnings():
        # Tuning on this few rows warns; the point here is the index bookkeeping.
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X, y)

    # Expansion drops "date" from position 0 and appends its features, so "cat"
    # sits at 0 in the array the tuning estimator sees.
    assert model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL) == [
        0
    ]
    assert len(tuning_estimators) == 1
    tuning = tuning_estimators[0]
    assert tuning.categorical_features_indices == [0]
    assert tuning.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL) == [
        0
    ]
