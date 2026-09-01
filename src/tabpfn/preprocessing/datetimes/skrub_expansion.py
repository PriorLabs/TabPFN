#  Copyright (c) Prior Labs GmbH 2026.

"""Expanding a point in time into calendar features, via `skrub.DatetimeEncoder`."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from skrub import DatetimeEncoder

from tabpfn.preprocessing.datamodel import make_names_unique
from tabpfn.preprocessing.datetimes.base import DateConversion, DateTransformer
from tabpfn.preprocessing.datetimes.dtypes import as_timestamp, is_instant_dtype
from tabpfn.preprocessing.datetimes.frames import drop_and_append

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType


def _make_datetime_encoder() -> DatetimeEncoder:
    """Build the encoder that turns one date column into calendar features.

    Returns:
        An encoder producing the year, the day of year, the seconds since the
        epoch, and the cyclical month, day and weekday pairs, plus the time of
        day when the column carries one.
    """
    return DatetimeEncoder(
        resolution="second",
        add_weekday=True,
        add_day_of_year=True,
        periodic_encoding="circular",
    )


class SkrubDateTransformer(DateTransformer):
    """Expands each point in time into the calendar features it stands for.

    A year, a day of year, the seconds since the epoch, and cyclical month, day
    and weekday pairs, so a December and the December before it are near each
    other rather than a year apart. This changes the column count, which is why
    the conversion reports its resolved feature names and remapped categorical
    indices: everything downstream indexes the wider frame.

    What each column expands into is decided at fit time and frozen: skrub drops
    the features a column cannot vary in, so a date-only column and one carrying
    a time of day do not produce the same width, and `transform` reuses the
    encoder rather than re-deciding. A duration is still converted to seconds,
    and so is a point in time that appears at a position which held something
    else at fit time -- there is no fitted encoder for it to go through.
    """

    @dataclasses.dataclass
    class _FittedColumn:
        """One column's fitted encoder, and the names of the features it makes."""

        encoder: DatetimeEncoder
        output_names: list[str]

    def __init__(self, *, categorical_indices: Sequence[int] | None = None) -> None:
        super().__init__(categorical_indices=categorical_indices)
        self._fitted: dict[int, SkrubDateTransformer._FittedColumn] = {}

    @property
    def expanded_indices(self) -> list[int]:
        """Input positions that were expanded into calendar features, ascending.

        Empty before `fit_transform` runs, and whenever it expanded nothing.
        """
        return sorted(self._fitted)

    def _fit_transform_frame(self, X: pd.DataFrame) -> DateConversion:
        self._fitted = {}
        to_expand, durations = self._temporal_positions(X)
        converted = self._convert_in_place(X, instants=[], durations=durations)
        if not to_expand:
            return self._conversion(converted)

        # `drop_and_append` concatenates the kept columns against skrub's own
        # (freshly default-indexed) output, so the row index has to be the
        # default range already or the two align by label instead of position.
        converted = converted.reset_index(drop=True)
        kept_names = [
            str(column)
            for i, column in enumerate(converted.columns)
            if i not in set(to_expand)
        ]
        expanded_names: list[str] = []
        blocks: list[pd.DataFrame] = []
        for position in to_expand:
            column = as_timestamp(X.iloc[:, position]).rename(str(X.columns[position]))
            block, fitted = self._fit_one(column, kept_names + expanded_names)
            self._fitted[position] = fitted
            expanded_names += fitted.output_names
            blocks.append(block)

        return DateConversion(
            X=drop_and_append(converted, to_expand, blocks),
            feature_names=kept_names + expanded_names,
            categorical_indices=self._remap(self._categorical_indices, to_expand),
        )

    def _transform_frame(self, X: pd.DataFrame) -> XType:
        """Reapply the fitted encoders, so the width cannot change.

        A position expanded at fit time that no longer holds a point in time
        degrades to `NaN` calendar features, like any other missing value: there
        is no attempt to parse whatever is sitting there instead.
        """
        to_expand = self.expanded_indices
        instants, durations = self._temporal_positions(X)
        converted = self._convert_in_place(
            X,
            instants=[i for i in instants if i not in set(to_expand)],
            durations=durations,
        )
        if not to_expand:
            return converted

        # See the identical comment in `_fit_transform_frame`.
        converted = converted.reset_index(drop=True)
        blocks = [
            self._apply_one(X.iloc[:, position], self._fitted[position])
            for position in to_expand
        ]
        return drop_and_append(converted, to_expand, blocks)

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
    def _fit_one(
        column: pd.Series,
        existing_names: Sequence[str],
    ) -> tuple[pd.DataFrame, SkrubDateTransformer._FittedColumn]:
        """Fit an encoder on one column, naming its output after that column.

        skrub names each feature after the column it came from (e.g.
        "signed_on_year"), which is kept as-is, deduplicated only against names
        already in the frame.
        """
        encoder = _make_datetime_encoder()
        encoded = pd.DataFrame(encoder.fit_transform(column))
        output_names = make_names_unique(
            [str(name) for name in encoded.columns], existing=existing_names
        )
        return (
            encoded.set_axis(output_names, axis=1).reset_index(drop=True),
            SkrubDateTransformer._FittedColumn(
                encoder=encoder, output_names=output_names
            ),
        )

    @staticmethod
    def _apply_one(
        column: pd.Series,
        fitted: SkrubDateTransformer._FittedColumn,
    ) -> pd.DataFrame:
        """Reapply one fitted encoder, or produce its features as all-`NaN`."""
        if not is_instant_dtype(column.dtype):
            return pd.DataFrame(
                {name: np.full(len(column), np.nan) for name in fitted.output_names}
            )
        encoded = pd.DataFrame(fitted.encoder.transform(as_timestamp(column)))
        return encoded.set_axis(fitted.output_names, axis=1).reset_index(drop=True)
