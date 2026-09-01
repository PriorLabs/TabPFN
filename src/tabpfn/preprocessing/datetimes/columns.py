#  Copyright (c) Prior Labs GmbH 2026.

"""Reading one temporal column, and moving columns around a frame.

The frame surgery is positional throughout, because the labels are the caller's
and so can repeat (the same duplicate-name case `build_input_feature_names`
exists for), which makes anything driven by a label ambiguous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def replace_columns_positionally(
    X: pd.DataFrame,
    replacements: dict[int, pd.Series],
) -> pd.DataFrame:
    """Return `X` with the given column positions replaced, leaving `X` untouched.

    Positional via a temporary integer column axis: numbering the axis makes
    every label unique and equal to its own position, so a plain assignment is
    unambiguous, and the caller's labels go back afterwards.

    The copy is shallow and the frame handed in is never written through: each
    assignment replaces a whole column rather than any value inside one.
    """
    if not replacements:
        return X
    out = X.copy(deep=False)
    original_columns = out.columns
    out.columns = pd.RangeIndex(out.shape[1])
    for position, values in replacements.items():
        out[position] = values.to_numpy()
    out.columns = original_columns
    return out


def drop_and_append(
    frame: pd.DataFrame,
    expanded: Sequence[int],
    blocks: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Drop every expanded column and append its calendar features instead.

    Positional, not `frame.drop(columns=...)`, which would take the wrong column
    when labels repeat. Every expanded column's output lands after all the kept
    ones, in original-position order, which is what makes the appended block's
    positions predictable.
    """
    keep = [i for i in range(frame.shape[1]) if i not in set(expanded)]
    return pd.concat([frame.iloc[:, keep], *blocks], axis=1)
