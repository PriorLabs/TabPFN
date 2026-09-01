#  Copyright (c) Prior Labs GmbH 2026.

"""Reading and converting single temporal columns, one column at a time."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def is_instant_dtype(dtype: Any) -> bool:
    """Whether `dtype` holds points in time: `datetime64`, tz-aware, or `period`."""
    return pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(
        dtype, pd.PeriodDtype
    )


def as_timestamp(column: pd.Series) -> pd.Series:
    """The instant a `period` column starts at, or the column unchanged.

    A period is a span, not an instant; its start is the instant that orders
    identically, which is all any conversion here needs.
    """
    if isinstance(column.dtype, pd.PeriodDtype):
        return column.dt.to_timestamp()
    return column


def to_nanoseconds(column: pd.Series) -> pd.Series:
    """Cast one point-in-time column to nanoseconds since the epoch.

    Scaled by the column's own resolution, since `astype("int64")` counts ticks
    of that resolution rather than nanoseconds: pandas 3 reads a date as
    `datetime64[us]` where pandas 2 read it as `[ns]`, and the same timestamps
    have to come out as the same number either way, or a fit and a predict of
    one column that arrived at different resolutions are scaled apart. Scaled in
    `float64` rather than by converting the column to `[ns]` first, which raises
    `OutOfBoundsDatetime` outside 1678-2262.

    As `float64`, so a missing date (`NaT`) survives as `NaN`: `NaT.astype`
    `("int64")` maps it to that dtype's huge sentinel value instead, which
    `float64` has no equivalent of. Exact enough for any realistic datetime
    column: nanosecond-since-epoch magnitudes are still ~256ns short of
    float64's precision limit for a present-day date, far below anything a
    tabular feature would need to distinguish.
    """
    is_missing = column.isna().to_numpy()
    # A tz-aware dtype carries `.unit`; a plain one reports it only through numpy.
    unit = getattr(column.dtype, "unit", None) or np.datetime_data(column.dtype)[0]
    as_ns = column.astype("int64").astype("float64") * (
        np.timedelta64(1, unit) / np.timedelta64(1, "ns")
    )
    as_ns[is_missing] = np.nan
    return as_ns


def to_seconds(column: pd.Series) -> pd.Series:
    """Cast one `timedelta64` column to its length in seconds.

    A duration carries no calendar, so its length is the whole of its meaning:
    unlike a point in time, there is nothing an expansion could add later.
    """
    return column.dt.total_seconds()
