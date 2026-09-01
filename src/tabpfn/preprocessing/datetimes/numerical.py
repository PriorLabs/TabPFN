#  Copyright (c) Prior Labs GmbH 2026.

"""Reading a point in time as one plain number: nanoseconds since the epoch."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tabpfn.preprocessing.datetimes.base import DateConversion, DateTransformer
from tabpfn.preprocessing.modality_detection import format_names_for_warning

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from tabpfn.constants import XType


def _warn_on_dates(column_names: Sequence[str]) -> None:
    """Warn about date columns read as a plain number rather than a calendar.

    Points in time held as one number only: an expanded column has its calendar
    features, and a duration's length in seconds loses nothing, so neither has
    anything to report.
    """
    if not column_names:
        return
    warnings.warn(
        f"These columns hold dates, which are read as plain numbers "
        f"(nanoseconds since the epoch): "
        f"{format_names_for_warning(list(column_names))}.\n"
        'Raise `inference_config={"TRANSFORM_DATES": True}` to expand them into '
        "calendar features (year, day of year, cyclical month/day/weekday) "
        "instead.",
        UserWarning,
        # stacklevel=7 reaches the `estimator.fit(X, y)` call site, counting the
        # base class's `fit_transform` as well as this subclass's; pinned by the
        # `warning.filename` asserts in the tests.
        stacklevel=7,
    )


class NumericalDateTransformer(DateTransformer):
    """Converts each point in time to a single ordered number.

    Nanoseconds since the epoch keep every distinct instant distinct, and keep
    them in order, but say nothing about weekdays or seasons: nothing here can
    tell the model that two Decembers a year apart have anything in common. That
    is what `SkrubDateTransformer` is for; this is the default because it adds no
    columns and cannot mislead about a calendar it never read.

    Nothing is fitted, so `transform` re-reads the dtypes and converts whatever
    is temporal at the time. A column that only becomes a date at predict time is
    therefore still converted, which is the point: an unconverted one crashes
    validation whether or not it was a date when we fit.
    """

    def _fit_transform_frame(self, X: pd.DataFrame) -> DateConversion:
        instants, durations = self._temporal_positions(X)
        converted = self._convert_in_place(X, instants=instants, durations=durations)
        _warn_on_dates([str(X.columns[i]) for i in instants])
        return self._conversion(converted)

    def _transform_frame(self, X: pd.DataFrame) -> XType:
        instants, durations = self._temporal_positions(X)
        return self._convert_in_place(X, instants=instants, durations=durations)
