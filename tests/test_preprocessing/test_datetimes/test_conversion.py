#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the predict paths' guard around `date_transformer_`."""

from __future__ import annotations

import pandas as pd
import pytest

from tabpfn.errors import TabPFNUserError
from tabpfn.preprocessing.datetimes import DateTransformer, convert_dates


def _frame(dates: pd.Series | pd.DatetimeIndex | list) -> pd.DataFrame:
    return pd.DataFrame({"num": [1.0, 2.0, 3.0], "date": dates})


class TestConvertDates:
    """`convert_dates`: the predict paths' guard for an unset attribute.

    `fit_from_preprocessed` never sets `date_transformer_`, exactly like the
    pre-existing `ordinal_encoder_` guard, so the fallback has nothing fitted
    to expand a date with.
    """

    class _Source:
        def __init__(self, **attributes: object) -> None:
            self.__dict__.update(attributes)

    def test__source_without_a_transformer__refuses_a_date(self) -> None:
        X = _frame(pd.date_range("2020-01-01", periods=3))

        with pytest.raises(TabPFNUserError, match=r"1 \('date'\)"):
            convert_dates(X, self._Source(categorical_features_indices=None))

    def test__source_without_a_transformer__still_converts_a_duration(self) -> None:
        X = _frame(pd.to_timedelta([1, 2, 3], unit="D"))

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
