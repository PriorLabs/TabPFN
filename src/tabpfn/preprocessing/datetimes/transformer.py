#  Copyright (c) Prior Labs GmbH 2026.

"""The one transformer temporal columns go through, before validation runs."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from skrub import DatetimeEncoder

from tabpfn.errors import TabPFNUserError
from tabpfn.preprocessing.datamodel import make_names_unique
from tabpfn.preprocessing.datetimes.columns import (
    as_timestamp,
    drop_and_append,
    is_instant_dtype,
    replace_columns_positionally,
    to_seconds,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType


@dataclasses.dataclass(frozen=True)
class DateConversion:
    """What `DateTransformer.fit_transform` did, as the fit path needs it.

    Attributes:
        X: The converted data.
        feature_names: `X`'s column labels as strings, in order, or `None` when
            the input was not a `DataFrame` and so has no labels.
        categorical_indices: The caller's declared categorical indices, moved to
            where those columns ended up.
    """

    X: XType
    feature_names: list[str] | None
    categorical_indices: list[int] | None


@dataclasses.dataclass
class FittedDateColumn:
    """One input column's fitted encoder, and the features it makes.

    Attributes:
        encoder: The encoder fitted on that column.
        output_names: The names of its output features, in order.
    """

    encoder: DatetimeEncoder
    output_names: list[str]


class DateTransformer:
    """Expands each point in time into calendar features, or refuses it.

    Not a `PreprocessingStep` (`pipeline_interface.py`): that tier runs per
    ensemble member on already-numeric arrays, well past where this has to run.
    Not `BaseEstimator`/`TransformerMixin` either: `fit_transform` returns more
    than the transformed data, which does not fit sklearn's shape.

    Usage mirrors `ordinal_encoder_`: call `fit_transform` once at fit time and
    keep the instance around (as `self.date_transformer_`), then call `transform`
    on it at predict time.

    Args:
        categorical_indices: Indices the caller declared categorical. A point in
            time among them is left alone entirely, at fit and at predict alike:
            the user's declared intent for it wins over reading it as a date.
        transform_dates: Whether a point in time is expanded into a year, a day
            of year, the seconds since the epoch, and cyclical month, day and
            weekday pairs, so a December and the December before it are near
            each other rather than a year apart. Off, such a column is refused
            with an error naming it.
    """

    def __init__(
        self,
        *,
        categorical_indices: Sequence[int] | None = None,
        transform_dates: bool = False,
    ) -> None:
        self._categorical_indices = (
            None if categorical_indices is None else list(categorical_indices)
        )
        self._declared_categorical = set(self._categorical_indices or ())
        self._transform_dates = transform_dates
        self._fitted: dict[int, FittedDateColumn] = {}

    @property
    def _expanded_indices(self) -> list[int]:
        """Input positions that were expanded into calendar features, ascending.

        Empty before `fit_transform` runs, and whenever it expanded nothing.
        """
        return sorted(self._fitted)

    def fit_transform(self, X: XType) -> DateConversion:
        """Convert every temporal column in `X`, refusing a date it cannot.

        Expansion changes the column count, so the conversion carries the
        resolved feature names and the declared categorical indices moved to
        their new positions: everything downstream indexes the wider frame.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            The conversion, including what the caller has to pass on to
            `detect_feature_modalities`.

        Raises:
            TabPFNUserError: On a point in time, with `transform_dates` off.
        """
        # Cleared before anything else, so that refitting on an input with no
        # columns to expand still forgets the last fit.
        self._fitted = {}
        if not isinstance(X, pd.DataFrame):
            return DateConversion(
                X=X, feature_names=None, categorical_indices=self._categorical_indices
            )

        instants, durations = self._temporal_positions(X)
        if not self._transform_dates:
            _refuse(X, instants)
        converted = self._durations_to_seconds(X, durations)
        if not instants:
            return DateConversion(
                X=converted,
                feature_names=[str(column) for column in converted.columns],
                categorical_indices=self._categorical_indices,
            )

        # `drop_and_append` concatenates the kept columns against skrub's own
        # (freshly default-indexed) output, so the row index has to be the
        # default range already or the two align by label instead of position.
        converted = converted.reset_index(drop=True)
        kept_names = [
            str(column)
            for i, column in enumerate(converted.columns)
            if i not in set(instants)
        ]
        expanded_names: list[str] = []
        blocks: list[pd.DataFrame] = []
        for position in instants:
            column = as_timestamp(X.iloc[:, position]).rename(str(X.columns[position]))
            block, fitted = self._fit_one(column, kept_names + expanded_names)
            self._fitted[position] = fitted
            expanded_names += fitted.output_names
            blocks.append(block)

        return DateConversion(
            X=drop_and_append(converted, instants, blocks),
            feature_names=kept_names + expanded_names,
            categorical_indices=self._remap(self._categorical_indices, instants),
        )

    def transform(self, X: XType) -> XType:
        """Reapply the conversion `fit_transform` decided on, so the width holds.

        A position expanded at fit that no longer holds a point in time degrades
        to `NaN` calendar features, like any other missing value: there is no
        attempt to parse whatever is sitting there instead. A point in time at a
        position that held something else at fit has no encoder to go through,
        so it is refused whatever the flag: the model was not fit on a date there.

        Args:
            X: The data, before any dtype fixing.

        Raises:
            TabPFNUserError: On a point in time no fitted encoder covers.
        """
        if not isinstance(X, pd.DataFrame):
            return X
        instants, durations = self._temporal_positions(X)
        _refuse(X, [i for i in instants if i not in self._fitted])
        converted = self._durations_to_seconds(X, durations)
        expanded = self._expanded_indices
        if not expanded:
            return converted

        # `drop_and_append` aligns the kept columns against skrub's own
        # default-indexed output by position, which needs the default index.
        converted = converted.reset_index(drop=True)
        blocks = [self._apply_one(X.iloc[:, i], self._fitted[i]) for i in expanded]
        return drop_and_append(converted, expanded, blocks)

    def _temporal_positions(self, X: pd.DataFrame) -> tuple[list[int], list[int]]:
        """The positions of `X`'s points in time and of its durations.

        A declared-categorical instant is not among them at all. A declared
        categorical duration is: leaving it alone only crashes validation, and a
        whole number of seconds ordinal-encodes as a category just as well.
        """
        dtypes = list(X.dtypes)
        instants = [
            i
            for i, dtype in enumerate(dtypes)
            if is_instant_dtype(dtype) and i not in self._declared_categorical
        ]
        durations = [
            i
            for i, dtype in enumerate(dtypes)
            if pd.api.types.is_timedelta64_dtype(dtype)
        ]
        return instants, durations

    @staticmethod
    def _durations_to_seconds(
        X: pd.DataFrame, durations: Sequence[int]
    ) -> pd.DataFrame:
        """Replace each duration column with its length in seconds."""
        return replace_columns_positionally(
            X, {i: to_seconds(X.iloc[:, i]) for i in durations}
        )

    @staticmethod
    def _remap(indices: list[int] | None, expanded: Sequence[int]) -> list[int] | None:
        """Move `indices` to where those columns end up once `expanded` is gone.

        A kept column keeps its relative order, so it just shifts down by however
        many expanded columns sat ahead of it. An expanded index is not remapped
        here: a declared-categorical column is never expanded to begin with.
        """
        if indices is None:
            return None
        return [i - sum(1 for j in expanded if j < i) for i in indices]

    @staticmethod
    def _make_encoder() -> DatetimeEncoder:
        """Build the encoder one date column is expanded through.

        Returns:
            An encoder producing the year, the day of year, the seconds since the
            epoch, and the cyclical month, day and weekday pairs, plus the time
            of day when the column carries one.
        """
        return DatetimeEncoder(
            resolution="second",
            add_weekday=True,
            add_day_of_year=True,
            periodic_encoding="circular",
        )

    @staticmethod
    def _fit_one(
        column: pd.Series,
        existing_names: Sequence[str],
    ) -> tuple[pd.DataFrame, FittedDateColumn]:
        """Fit an encoder on one column, naming its output after that column.

        skrub names each feature after the column it came from (e.g.
        "signed_on_year"), which is kept as-is, deduplicated only against names
        already in the frame. How many features there are is settled here too:
        skrub drops the ones a column cannot vary in, e.g. the time of day of a
        date-only column, which is why `transform` reuses this encoder rather
        than fitting a fresh one.
        """
        encoder = DateTransformer._make_encoder()
        encoded = pd.DataFrame(encoder.fit_transform(column))
        output_names = make_names_unique(
            [str(name) for name in encoded.columns], existing=existing_names
        )
        return (
            encoded.set_axis(output_names, axis=1).reset_index(drop=True),
            FittedDateColumn(encoder=encoder, output_names=output_names),
        )

    @staticmethod
    def _apply_one(
        column: pd.Series,
        fitted: FittedDateColumn,
    ) -> pd.DataFrame:
        """Reapply one fitted encoder, or produce its features as all-`NaN`."""
        if not is_instant_dtype(column.dtype):
            return pd.DataFrame(
                {name: np.full(len(column), np.nan) for name in fitted.output_names}
            )
        encoded = pd.DataFrame(fitted.encoder.transform(as_timestamp(column)))
        return encoded.set_axis(fitted.output_names, axis=1).reset_index(drop=True)


def _refuse(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on the points in time at `positions`, naming each; a no-op when empty.

    sklearn's array machinery has no dtype holding a `datetime64` column beside a
    numeric one, so validation would fail on such a column with numpy's opaque
    dtype-promotion error otherwise. Nothing downstream reads a date as a date,
    so the column is refused outright rather than silently read as a number.
    """
    if not positions:
        return
    columns = ", ".join(f"{i} ({X.columns[i]!r})" for i in positions)
    raise TabPFNUserError(
        f"These columns hold datetimes, which TabPFN does not support: {columns}. "
        'Set `inference_config={"TRANSFORM_DATES": True}` to expand them into '
        "calendar features, or preprocess them yourself first."
    )
