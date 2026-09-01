#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for converting temporal columns before validation runs."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.date_encoding import DateTransformer, convert_dates
from tabpfn.preprocessing.modality_detection import detect_feature_modalities


def _frame(dates: pd.Series | pd.DatetimeIndex | list) -> pd.DataFrame:
    return pd.DataFrame({"num": [1.0, 2.0, 3.0], "date": dates})


class TestFitTransform:
    """`DateTransformer.fit_transform`: which columns convert, and to what.

    Runs on the raw DataFrame, before validation flattens it into one numpy
    array, so a genuine `datetime64` dtype is still visible. Dtype is all it
    reads: a string that merely looks like a date is not a date here.
    """

    def test__real_datetime_column__is_cast_to_numeric(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X).X

        assert pd.api.types.is_numeric_dtype(out["date"])
        # Exact: converting must not collide distinct dates together.
        assert out["date"].nunique() == 3
        # The caller's own frame is left untouched.
        np.testing.assert_array_equal(out["num"], X["num"])
        assert pd.api.types.is_datetime64_any_dtype(X["date"])

    def test__date_like_string_column__is_not_converted(self) -> None:
        X = _frame(["2020-01-01", "2020-01-02", "2020-01-03"])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = DateTransformer().fit_transform(X).X

        assert out is X

    def test__declared_categorical_date__is_left_alone(self) -> None:
        """The user's declared intent for the column wins over reading it as a
        date, so it is not converted and not warned about.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = DateTransformer(categorical_indices=[1]).fit_transform(X).X

        assert out is X

    def test__non_dataframe_input__is_a_noop(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert DateTransformer().fit_transform(X).X is X

    def test__missing_date__becomes_nan_not_a_sentinel_int(self) -> None:
        """`NaT.astype('int64')` is a huge sentinel, not `NaN`: must be masked."""
        X = _frame(pd.to_datetime(["2020-01-01", None, "2020-01-03"]))

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X).X

        assert out["date"].isna().tolist() == [False, True, False]

    def test__timezone_aware_column__is_converted(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3, tz="Europe/Berlin"))

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X).X

        assert pd.api.types.is_numeric_dtype(out["date"])
        assert out["date"].nunique() == 3

    def test__period_column__is_converted(self) -> None:
        """A period is a span; the instant it starts at orders identically."""
        X = _frame(pd.period_range("2020-01-01", periods=3, freq="D"))

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X).X

        assert pd.api.types.is_numeric_dtype(out["date"])
        assert out["date"].nunique() == 3

    def test__duration_column__becomes_seconds(self) -> None:
        """A duration carries no calendar, so its length is all of its meaning:
        converted whatever else is going on, and nothing to warn about.
        """
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = DateTransformer().fit_transform(X).X

        np.testing.assert_array_equal(out["date"], [86400.0, 172800.0, 259200.0])

    def test__declared_categorical_duration__is_still_converted(self) -> None:
        """Unlike a point in time: leaving it alone only crashes validation, and
        a whole number of seconds ordinal-encodes as a category just as well.
        """
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

        out = DateTransformer(categorical_indices=[1]).fit_transform(X).X

        assert pd.api.types.is_numeric_dtype(out["date"])

    def test__duplicate_column_labels__are_converted_positionally(self) -> None:
        """Pandas allows repeated labels, so only position identifies a column."""
        X = pd.DataFrame(
            {0: pd.date_range("2020-01-01", periods=3), 1: [1.0, 2.0, 3.0]}
        )
        X.columns = ["same", "same"]

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X).X

        assert [str(dtype) for dtype in out.dtypes] == ["float64", "float64"]
        np.testing.assert_array_equal(out.iloc[:, 1], X.iloc[:, 1])

    def test__warning__names_every_converted_column(self) -> None:
        X = pd.DataFrame(
            {
                "signed_on": pd.date_range("2020-01-01", periods=3),
                "num": [1.0, 2.0, 3.0],
                "shipped_on": pd.date_range("2021-01-01", periods=3),
            }
        )

        with pytest.warns(UserWarning, match="hold dates") as record:
            DateTransformer().fit_transform(X)

        message = str(record[0].message)
        assert "'signed_on'" in message
        assert "'shipped_on'" in message


class TestResolutionScaling:
    """A datetime column converts to the same numbers whatever its resolution.

    `astype("int64")` counts ticks of the column's own resolution, so the same
    timestamps arriving as `datetime64[us]` (what pandas 3 reads every ordinary
    date source as) came out 1000x smaller than as `[ns]` (what pandas 2 read
    them as).
    """

    stamps = ("2020-01-01", "2020-06-01", "2021-01-01")
    expected = (1.5778368e18, 1.5909696e18, 1.6094592e18)

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test__every_resolution__converts_to_the_same_numbers(self, unit: str) -> None:
        dates = pd.to_datetime(list(self.stamps)).astype(f"datetime64[{unit}]")

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(_frame(dates)).X

        np.testing.assert_array_equal(out["date"], self.expected)

    def test__fit_and_predict_at_different_resolutions__agree(self) -> None:
        """The bug that matters: a column normalised against fit-time statistics
        does not renormalise, so a predict column arriving 1000x off lands far
        outside the fitted distribution.
        """
        transformer = DateTransformer()
        with pytest.warns(UserWarning, match="hold dates"):
            fitted = transformer.fit_transform(
                _frame(pd.to_datetime(list(self.stamps)).astype("datetime64[us]"))
            ).X

        out = transformer.transform(
            _frame(pd.to_datetime(list(self.stamps)).astype("datetime64[ns]"))
        )

        np.testing.assert_array_equal(out["date"], fitted["date"])

    @pytest.mark.skipif(
        Version(pd.__version__) < Version("2.0.0"),
        reason="pandas 1 holds every datetime as [ns], so such a date cannot be built",
    )
    def test__date_outside_the_nanosecond_range__still_converts(self) -> None:
        """Scaled in `float64` rather than by casting the column to `[ns]` first,
        which raises `OutOfBoundsDatetime` outside 1678-2262.

        Skipped below pandas 2: `to_datetime` there raises on the date itself, so
        the scenario cannot be constructed, let alone converted.
        """
        dates = pd.to_datetime(["1500-01-01", "2600-01-01"]).astype("datetime64[s]")

        with pytest.warns(UserWarning, match="hold dates"):
            out = (
                DateTransformer()
                .fit_transform(pd.DataFrame({"num": [1.0, 2.0], "date": dates}))
                .X
            )

        assert out["date"].tolist() == [-1.48317696e19, 1.98808992e19]

    def test__timezone_aware_column__is_scaled_by_its_own_unit(self) -> None:
        """A tz-aware dtype reports its unit itself, where a plain `datetime64`
        only does so through numpy.
        """
        dates = pd.to_datetime(list(self.stamps)).tz_localize("UTC")

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(_frame(dates)).X

        np.testing.assert_array_equal(out["date"], self.expected)


class TestExpansion:
    """`TRANSFORM_DATES` on: a date becomes calendar features instead of a number."""

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

    def test__expanded_output__is_reported_numerical(self) -> None:
        """Positions the caller passes to `detect_feature_modalities`.

        A narrow training window gives a cyclical pair few enough distinct values
        to look categorical, and the ensemble members that ordinal-encode a
        category turn a value unseen at fit into `NaN`. A third month at predict
        time is ordinary for such a window, so the feature would go missing on
        exactly the rows carrying the new information.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))

        conversion = self._expander().fit_transform(X)

        assert conversion.numerical_indices == list(
            range(1, len(conversion.feature_names))
        )

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
        assert conversion.numerical_indices == []

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

    def test__expanded_indices__reports_what_was_expanded(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        transformer = self._expander()
        assert transformer.expanded_indices == []
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


class TestTransform:
    """`DateTransformer.transform`: the predict-time side of the same conversion."""

    def test__converts_the_same_way_without_warning(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        transformer = DateTransformer()
        with pytest.warns(UserWarning, match="hold dates"):
            fitted = transformer.fit_transform(X).X

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = transformer.transform(X)

        np.testing.assert_array_equal(out["date"], fitted["date"])

    def test__column_that_is_a_date_only_at_predict__is_still_converted(self) -> None:
        """Which columns are temporal is read from the dtypes again, not frozen at
        fit: an unconverted `datetime64` column would crash validation.
        """
        transformer = DateTransformer()
        transformer.fit_transform(_frame([1.0, 2.0, 3.0]))

        out = transformer.transform(_frame(pd.date_range("2020-01-01", periods=3)))

        assert pd.api.types.is_numeric_dtype(out["date"])

    def test__declared_categorical_date__is_left_alone_at_predict_too(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        assert DateTransformer(categorical_indices=[1]).transform(X) is X

    def test__non_dataframe_input__is_a_noop(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert DateTransformer().transform(X) is X


class TestConvertDates:
    """`convert_dates`: the predict paths' guard for an unset attribute."""

    class _Source:
        def __init__(self, **attributes: object) -> None:
            self.__dict__.update(attributes)

    def test__source_without_a_transformer__still_converts(self) -> None:
        """`fit_from_preprocessed` never sets `date_transformer_`, exactly like the
        pre-existing `ordinal_encoder_` guard.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))
        out = convert_dates(X, self._Source(categorical_features_indices=None))
        assert pd.api.types.is_numeric_dtype(out["date"])

    def test__source_without_a_transformer__honours_declared_categoricals(
        self,
    ) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        out = convert_dates(X, self._Source(categorical_features_indices=[1]))
        assert out is X

    def test__source_with_a_fitted_transformer__uses_it(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        source = self._Source(
            date_transformer_=DateTransformer(categorical_indices=[1]),
            categorical_features_indices=None,
        )
        assert convert_dates(X, source) is X


class TestConvertedDateClassification:
    """How a converted date column is classified by `detect_feature_modalities`.

    Nothing expands a date into calendar features yet, so it arrives at detection
    as the plain number `DateTransformer` made of it, and is classified like any
    other number.
    """

    n_rows = 200

    def _numeric_column(self) -> np.ndarray:
        return np.random.default_rng(0).normal(size=self.n_rows)

    def _detect(self, X: pd.DataFrame) -> FeatureSchema:
        """Convert, then detect, in the order `fit` does it."""
        converted = DateTransformer().fit_transform(X).X
        return detect_feature_modalities(
            X=converted.to_numpy(dtype=object),
            feature_names=list(X.columns),
            min_samples_for_inference=100,
            max_unique_for_category=30,
            min_unique_for_numerical=4,
            min_cardinality_for_text=30,
        )

    def _dates(self, n_unique: int) -> pd.DatetimeIndex:
        pool = pd.date_range("2020-01-01", periods=n_unique)
        return pd.DatetimeIndex([pool[i % n_unique] for i in range(self.n_rows)])

    def test__high_cardinality_date__is_numerical(self) -> None:
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(60)})
        with pytest.warns(UserWarning, match="hold dates"):
            schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.NUMERICAL

    def test__low_cardinality_date__is_categorical(self) -> None:
        """Below `min_unique_for_numerical`, so categorical like any other number."""
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(3)})
        with pytest.warns(UserWarning, match="hold dates"):
            schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.CATEGORICAL

    def test__converted_date__is_not_also_reported_as_free_text(self) -> None:
        """The date warning fires once; the free-text warning must not repeat it."""
        X = pd.DataFrame({"num": self._numeric_column(), "date": self._dates(60)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._detect(X)
        date_warnings = [w for w in caught if "hold dates" in str(w.message)]
        text_warnings = [w for w in caught if "look like free text" in str(w.message)]
        assert len(date_warnings) == 1
        assert not text_warnings

    def test__date_like_string__is_not_a_date(self) -> None:
        """A string column that merely looks like a date is left as text, and no
        date warning names it.
        """
        X = pd.DataFrame(
            {
                "num": self._numeric_column(),
                "date": [d.strftime("%Y-%m-%d") for d in self._dates(60)],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            schema = self._detect(X)
        assert schema.features[1].modality is FeatureModality.TEXT
        assert not [w for w in caught if "hold dates" in str(w.message)]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_real_datetime_column__warns_at_call_site(
    estimator_cls: type,
) -> None:
    """`fit` converts a genuine `datetime64` column and warns naming it.

    Both estimators share the path, so one parametrized test pins the
    estimator-level behaviour: `fit` emits the warning naming the column and
    blaming this file's `fit` call (the stacklevel). Declaring the column in
    `categorical_features_indices` excludes it from conversion (see
    `TestFitTransform`), so it is left as a real `datetime64` dtype all the way
    to validation and hits the pre-existing, unrelated `np.result_type` crash
    instead: an accepted limitation of declaring a genuine date column
    categorical, not something this silences.
    """
    n = 120
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {"num": rng.normal(size=n), "date": pd.date_range("2020-01-01", periods=n)}
    )
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="hold dates") as record:
        model.fit(X, y)
    assert "'date'" in str(record[0].message)
    # Pins the stacklevel: the warning must blame this file's `fit` call, not a
    # frame inside tabpfn or the contextlib wrapper around `fit`.
    assert record[0].filename == __file__
    model.predict(X)

    model = estimator_cls(
        n_estimators=1, device="cpu", categorical_features_indices=[1]
    )
    with pytest.raises(TabPFNValidationError, match="could not be promoted"):
        model.fit(X, y)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("dtype", ["period", "duration"])
def test__fit_with_period_or_duration_column__no_longer_crashes(
    estimator_cls: type,
    dtype: str,
) -> None:
    """Both used to abort a `fit` outright.

    A `period` column reached modality detection intact and was rejected there
    ("Unknown dtype: period[D]"); a `timedelta64` column never got that far,
    since it has no common numpy dtype with the numeric column beside it.
    """
    n = 120
    rng = np.random.default_rng(seed=42)
    column = (
        pd.period_range("2020-01-01", periods=n, freq="D")
        if dtype == "period"
        else pd.to_timedelta(np.arange(n), unit="D")
    )
    X = pd.DataFrame({"num": rng.normal(size=n), "col": column})
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = estimator_cls(n_estimators=1, device="cpu").fit(X, y)
        predictions = model.predict(X)

    assert len(predictions) == n


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
