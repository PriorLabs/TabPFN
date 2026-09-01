#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for picking a date transformer and for the predict-time guard."""

from __future__ import annotations

import pandas as pd

from tabpfn.preprocessing.datetimes import (
    NumericalDateTransformer,
    SkrubDateTransformer,
    convert_dates,
    make_date_transformer,
)


def _frame(dates: pd.Series | pd.DatetimeIndex | list) -> pd.DataFrame:
    return pd.DataFrame({"num": [1.0, 2.0, 3.0], "date": dates})


class TestMakeDateTransformer:
    """`make_date_transformer`: `TRANSFORM_DATES` picks the implementation."""

    def test__flag_off__reads_dates_as_plain_numbers(self) -> None:
        transformer = make_date_transformer(transform_dates=False)

        assert isinstance(transformer, NumericalDateTransformer)

    def test__flag_on__expands_dates_into_calendar_features(self) -> None:
        transformer = make_date_transformer(transform_dates=True)

        assert isinstance(transformer, SkrubDateTransformer)

    def test__declared_categoricals__reach_the_transformer(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))

        out = make_date_transformer(
            categorical_indices=[1], transform_dates=True
        ).fit_transform(X)

        assert out.X is X


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
            date_transformer_=NumericalDateTransformer(categorical_indices=[1]),
            categorical_features_indices=None,
        )
        assert convert_dates(X, source) is X
