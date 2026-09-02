#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `DateTransformer`: refusing a date, expanding it, converting a duration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNUserError, TabPFNValidationError
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

        with pytest.raises(TabPFNUserError, match="does not support") as excinfo:
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
        with pytest.raises(TabPFNUserError, match=r"1 \('date'\)"):
            DateTransformer().fit_transform(_frame(dates))

    def test__date_like_string_column__is_not_refused(self) -> None:
        X = _frame(["2020-01-01", "2020-01-02", "2020-01-03"])
        assert DateTransformer().fit_transform(X).X is X

    def test__declared_categorical_date__is_left_alone(self) -> None:
        """The user's declared intent for the column wins over reading it as a
        date, so it is neither refused nor converted.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))
        assert DateTransformer(categorical_indices=[1]).fit_transform(X).X is X

    def test__non_dataframe_input__is_a_noop(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert DateTransformer().fit_transform(X).X is X
        assert DateTransformer().transform(X) is X

    def test__transform__refuses_a_date_too(self) -> None:
        transformer = DateTransformer()
        transformer.fit_transform(_frame([1.0, 2.0, 3.0]))

        with pytest.raises(TabPFNUserError, match=r"1 \('date'\)"):
            transformer.transform(_frame(pd.date_range("2020-01-01", periods=3)))


class TestDurations:
    """A duration has no calendar, so its length in seconds is all of its meaning:
    converted flag or no flag, and nothing to refuse.
    """

    @pytest.mark.parametrize("transform_dates", [False, True])
    def test__duration_column__becomes_seconds(self, transform_dates: bool) -> None:
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

        out = DateTransformer(transform_dates=transform_dates).fit_transform(X).X

        np.testing.assert_array_equal(out["date"], [86400.0, 172800.0, 259200.0])
        # The caller's own frame is left untouched.
        assert pd.api.types.is_timedelta64_dtype(X["date"])

    def test__declared_categorical_duration__is_still_converted(self) -> None:
        """Unlike a point in time: leaving it alone only crashes validation, and
        a whole number of seconds ordinal-encodes as a category just as well.
        """
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

        out = DateTransformer(categorical_indices=[1]).fit_transform(X).X

        assert pd.api.types.is_numeric_dtype(out["date"])

    def test__transform__converts_the_same_way(self) -> None:
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))
        transformer = DateTransformer()
        fitted = transformer.fit_transform(X).X

        np.testing.assert_array_equal(transformer.transform(X)["date"], fitted["date"])

    def test__duplicate_column_labels__are_converted_positionally(self) -> None:
        """Pandas allows repeated labels, so only position identifies a column."""
        X = pd.DataFrame({0: pd.to_timedelta([1, 2, 3], unit="D"), 1: [1.0, 2.0, 3.0]})
        X.columns = ["same", "same"]

        out = DateTransformer().fit_transform(X).X

        assert [str(dtype) for dtype in out.dtypes] == ["float64", "float64"]
        np.testing.assert_array_equal(out.iloc[:, 1], X.iloc[:, 1])


class TestExpansion:
    """`TRANSFORM_DATES` on: a date becomes calendar features instead of an error."""

    def _expander(self, **kwargs: object) -> DateTransformer:
        return DateTransformer(transform_dates=True, **kwargs)  # type: ignore[arg-type]

    def test__date_column__becomes_calendar_features(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3, freq="37h"))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            conversion = self._expander().fit_transform(X)

        assert conversion.X.shape[0] == 3
        assert conversion.X.shape[1] > X.shape[1]
        # The kept columns come first, in order, then the expansion.
        assert conversion.feature_names[0] == "num"
        assert all(name.startswith("date_") for name in conversion.feature_names[1:])
        assert "date_year" in conversion.feature_names
        assert conversion.X.notna().all().all()

    def test__period_column__is_expanded_from_the_instant_it_starts_at(self) -> None:
        """A period is a span, not an instant; its start is the instant that
        orders identically, which is all the encoder needs.
        """
        X = _frame(pd.period_range("2020-01-01", periods=3, freq="D"))

        conversion = self._expander().fit_transform(X)

        assert "date_year" in conversion.feature_names
        assert conversion.X.notna().all().all()

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

        conversion = self._expander(categorical_indices=[1]).fit_transform(X)

        assert conversion.categorical_indices == [0]
        assert conversion.feature_names[0] == "cat"

    def test__declared_categorical_date__is_not_expanded(self) -> None:
        """The user's declared intent still wins, flag or no flag."""
        X = _frame(pd.date_range("2020-01-01", periods=3))

        conversion = self._expander(categorical_indices=[1]).fit_transform(X)

        assert conversion.X is X

    def test__refit_on_a_frame_with_nothing_to_expand__forgets_the_last_fit(
        self,
    ) -> None:
        """Otherwise `transform` reapplies encoders for the previous input's
        columns, and comes out a width the new fit never produced.
        """
        transformer = self._expander()
        transformer.fit_transform(_frame(pd.date_range("2020-01-01", periods=3)))
        transformer.fit_transform(np.zeros((3, 2)))

        assert transformer._expanded_indices == []
        assert transformer.transform(_frame([1.0, 2.0, 3.0])).shape[1] == 2

    def test__expanded_indices__reports_what_was_expanded(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        transformer = self._expander()
        assert transformer._expanded_indices == []
        transformer.fit_transform(X)
        assert transformer._expanded_indices == [1]

    def test__generated_name_colliding_with_an_existing_column__is_deduped(
        self,
    ) -> None:
        X = pd.DataFrame(
            {
                "date_year": [1.0, 2.0, 3.0],
                "date": pd.date_range("2020-01-01", periods=3),
            }
        )

        conversion = self._expander().fit_transform(X)

        assert len(set(conversion.feature_names)) == len(conversion.feature_names)
        assert "date_year" in conversion.feature_names
        assert "date_year_1" in conversion.feature_names


class TestExpansionAtPredictTime:
    """`transform` reapplies the encoder fit on the training column, not a new one."""

    def test__same_data__reproduces_the_fitted_columns(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3, freq="37h"))
        transformer = DateTransformer(transform_dates=True)
        fitted = transformer.fit_transform(X)

        out = transformer.transform(X)

        assert list(out.columns) == list(fitted.X.columns)
        np.testing.assert_array_equal(out.to_numpy(), fitted.X.to_numpy())

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

        assert list(out.columns) == list(fitted.X.columns)

    def test__position_that_is_no_longer_a_date__becomes_nan_features(self) -> None:
        transformer = DateTransformer(transform_dates=True)
        fitted = transformer.fit_transform(
            _frame(pd.date_range("2020-01-01", periods=3))
        )

        out = transformer.transform(_frame(["not", "a", "date"]))

        assert list(out.columns) == list(fitted.X.columns)
        assert out[fitted.feature_names[1:]].isna().all().all()
        # The column that was never a date is untouched.
        assert out["num"].tolist() == [1.0, 2.0, 3.0]

    def test__date_at_a_position_that_was_not_one_at_fit__is_refused(self) -> None:
        """No encoder was fit for it, so there is nothing to expand it with."""
        transformer = DateTransformer(transform_dates=True)
        transformer.fit_transform(_frame(pd.date_range("2020-01-01", periods=3)))

        with pytest.raises(TabPFNUserError, match=r"0 \('num'\)"):
            transformer.transform(
                pd.DataFrame(
                    {
                        "num": pd.date_range("2020-01-01", periods=3),
                        "date": pd.date_range("2020-01-01", periods=3),
                    }
                )
            )


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_real_datetime_column__raises_naming_it(estimator_cls: type) -> None:
    """`fit` refuses a genuine `datetime64` column, naming it.

    Both estimators share the transformer, so one parametrized test pins the
    estimator-level behaviour. Declaring the column in
    `categorical_features_indices` excludes it from the transformer (see
    `TestRefusal`), so it is left as a real `datetime64` dtype all the way to
    validation and hits the pre-existing, unrelated `np.result_type` failure
    instead: an accepted limitation of declaring a genuine date column
    categorical, not something this silences.
    """
    X, y = _estimator_data(estimator_cls, pd.date_range("2020-01-01", periods=120))

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.raises(TabPFNUserError, match=r"1 \('date'\)"):
        model.fit(X, y)

    model = estimator_cls(
        n_estimators=1, device="cpu", categorical_features_indices=[1]
    )
    with pytest.raises(TabPFNValidationError, match="could not be promoted"):
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
    with pytest.raises(TabPFNUserError, match=r"1 \('date'\)"):
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
