#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for expanding a point in time into calendar features."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datetimes import SkrubDateTransformer


def _frame(dates: pd.Series | pd.DatetimeIndex | list) -> pd.DataFrame:
    return pd.DataFrame({"num": [1.0, 2.0, 3.0], "date": dates})


class TestExpansion:
    """`TRANSFORM_DATES` on: a date becomes calendar features instead of a number."""

    def _expander(self, **kwargs: object) -> SkrubDateTransformer:
        return SkrubDateTransformer(**kwargs)  # type: ignore[arg-type]

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

    def test__duration_column__still_becomes_seconds(self) -> None:
        """A duration has no calendar to expand into, flag or no flag."""
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

        conversion = self._expander().fit_transform(X)

        np.testing.assert_array_equal(
            conversion.X["date"], [86400.0, 172800.0, 259200.0]
        )

    def test__expanded_column__is_not_warned_about(self) -> None:
        """The warning is about a date read as one plain number; an expanded one
        is not, so there is nothing to report.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._expander().fit_transform(X)

        assert not [w for w in caught if "hold dates" in str(w.message)]

    def test__refit_on_a_frame_with_nothing_to_expand__forgets_the_last_fit(
        self,
    ) -> None:
        """Otherwise `transform` reapplies encoders for the previous input's
        columns, and comes out a width the new fit never produced.
        """
        transformer = SkrubDateTransformer()
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
        transformer = SkrubDateTransformer()
        fitted = transformer.fit_transform(X)

        out = transformer.transform(X)

        assert list(out.columns) == list(fitted.X.columns)
        np.testing.assert_array_equal(out.to_numpy(), fitted.X.to_numpy())

    def test__predict_data_with_a_time_of_day__keeps_the_fitted_width(self) -> None:
        """The encoder decides how many features it makes when it is fit: a date-only
        training column has no hour to encode, and predict cannot add one without
        making the frame the wrong width for the fitted model.
        """
        transformer = SkrubDateTransformer()
        fitted = transformer.fit_transform(
            _frame(pd.date_range("2020-01-01", periods=3))
        )

        out = transformer.transform(
            _frame(pd.date_range("2020-01-01 13:45", periods=3, freq="37h"))
        )

        assert list(out.columns) == list(fitted.X.columns)

    def test__position_that_is_no_longer_a_date__becomes_nan_features(self) -> None:
        transformer = SkrubDateTransformer()
        fitted = transformer.fit_transform(
            _frame(pd.date_range("2020-01-01", periods=3))
        )

        out = transformer.transform(_frame(["not", "a", "date"]))

        assert list(out.columns) == list(fitted.X.columns)
        assert out[fitted.feature_names[1:]].isna().all().all()
        # The column that was never a date is untouched.
        assert out["num"].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_dates__expands_the_date_column(
    estimator_cls: type,
) -> None:
    """`TRANSFORM_DATES` reaches the transformer, and the wider frame survives
    the whole fit/predict path: the schema describes the calendar features, and
    the date warning is gone since the column is no longer one plain number.
    """
    n = 150
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="37h"),
        }
    )
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )

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
    assert len(predictions) == n


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_dates__reports_the_caller_s_own_columns(
    estimator_cls: type,
) -> None:
    """Expansion widens the frame internally, which must not leak into the
    sklearn attributes: they describe what the caller passed, so the error a
    caller gets for the wrong columns names their columns, not ours.
    """
    n = 150
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="37h"),
        }
    )
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )

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
