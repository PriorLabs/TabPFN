#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for converting temporal columns before validation runs."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.date_encoding import DateTransformer, apply_date_conversion
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
            out = DateTransformer().fit_transform(X)

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
            out = DateTransformer().fit_transform(X)

        assert out is X

    def test__declared_categorical_date__is_left_alone(self) -> None:
        """The user's declared intent for the column wins over reading it as a
        date, so it is not converted and not warned about.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = DateTransformer(categorical_indices=[1]).fit_transform(X)

        assert out is X

    def test__non_dataframe_input__is_a_noop(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert DateTransformer().fit_transform(X) is X

    def test__missing_date__becomes_nan_not_a_sentinel_int(self) -> None:
        """`NaT.astype('int64')` is a huge sentinel, not `NaN`: must be masked."""
        X = _frame(pd.to_datetime(["2020-01-01", None, "2020-01-03"]))

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X)

        assert out["date"].isna().tolist() == [False, True, False]

    def test__timezone_aware_column__is_converted(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3, tz="Europe/Berlin"))

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X)

        assert pd.api.types.is_numeric_dtype(out["date"])
        assert out["date"].nunique() == 3

    def test__duplicate_column_labels__are_converted_positionally(self) -> None:
        """Pandas allows repeated labels, so only position identifies a column."""
        X = pd.DataFrame(
            {0: pd.date_range("2020-01-01", periods=3), 1: [1.0, 2.0, 3.0]}
        )
        X.columns = ["same", "same"]

        with pytest.warns(UserWarning, match="hold dates"):
            out = DateTransformer().fit_transform(X)

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


class TestTransform:
    """`DateTransformer.transform`: the predict-time side of the same conversion."""

    def test__converts_the_same_way_without_warning(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        transformer = DateTransformer()
        with pytest.warns(UserWarning, match="hold dates"):
            fitted = transformer.fit_transform(X)

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


class TestApplyDateConversion:
    """`apply_date_conversion`: the predict paths' guard for an unset attribute."""

    class _Source:
        def __init__(self, **attributes: object) -> None:
            self.__dict__.update(attributes)

    def test__source_without_a_transformer__still_converts(self) -> None:
        """`fit_from_preprocessed` never sets `date_transformer_`, exactly like the
        pre-existing `ordinal_encoder_` guard.
        """
        X = _frame(pd.date_range("2020-01-01", periods=3))
        out = apply_date_conversion(X, self._Source(categorical_features_indices=None))
        assert pd.api.types.is_numeric_dtype(out["date"])

    def test__source_without_a_transformer__honours_declared_categoricals(
        self,
    ) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        out = apply_date_conversion(X, self._Source(categorical_features_indices=[1]))
        assert out is X

    def test__source_with_a_fitted_transformer__uses_it(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))
        source = self._Source(
            date_transformer_=DateTransformer(categorical_indices=[1]),
            categorical_features_indices=None,
        )
        assert apply_date_conversion(X, source) is X


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
        converted = DateTransformer().fit_transform(X)
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
