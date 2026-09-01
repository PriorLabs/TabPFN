#  Copyright (c) Prior Labs GmbH 2026.

"""Resolve every temporal column before validation ever sees it.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric
one in one array (`DTypePromotionError`: no common dtype exists), so a
temporal column has to stop looking like one before `check_array`/`check_X_y`
run. `DateTimeExpander` is where that happens: a point in time (`datetime64`,
tz-aware, or `period`) is either expanded into calendar features via
`skrub.DatetimeEncoder` (when `TRANSFORM_DATES` is on and the column isn't
declared categorical) or rendered to ISO 8601 text (otherwise, so it reads as
an ordinary high-cardinality category downstream). A duration (`timedelta64`)
always becomes its length in seconds -- a quantity with no calendar in it, so
the number is the whole of its meaning, independent of `TRANSFORM_DATES`.

The expander is used like a `ColumnTransformer`: `fit` on the raw fit input,
`transform` that same input, and `transform` every later predict input --
`fit` alone decides what happens to which column, so one `transform` body
serves both.

Because this runs before validation, there is no `FeatureSchema` yet:
`detect_feature_modalities` only ever sees the *result* -- a fully validated,
already-expanded array -- and never learns a column was ever a date at all.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _num_features
from skrub import DatetimeEncoder

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np

    from tabpfn.constants import XType

#: Cap on how many column names the "holds dates" warning lists, so a wide
#: frame of date columns does not produce an unreadable multi-kilobyte message.
_MAX_DATE_COLUMNS_IN_WARNING = 10


def _is_datetime_like_dtype(dtype: Any) -> bool:
    """Whether `dtype` holds points in time: `datetime64`, tz-aware, or `period`."""
    return pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(
        dtype, pd.PeriodDtype
    )


def _make_datetime_encoder() -> DatetimeEncoder:
    """Build the encoder that turns a datetime column into calendar features.

    Returns:
        An encoder producing the year, the day of year, the seconds since epoch,
        and the cyclical month, day and weekday pairs, plus the time of day when
        the column carries one.
    """
    return DatetimeEncoder(
        resolution="second",
        add_weekday=True,
        add_day_of_year=True,
        periodic_encoding="circular",
    )


class DateTimeExpander:
    """Expands or text-renders every temporal column of an input.

    Used like a `ColumnTransformer`: construct it with the two fit-time
    decisions it needs, `fit` it on the raw fit input, then `transform` that
    same input and every later predict input. `fit` settles which columns are
    expanded, once; `transform` reapplies that decision positionally, exactly
    like `ordinal_encoder_`.

    A predict-time position that was expanded at fit time but is no longer a
    genuine datetime dtype degrades to a `NaN` calendar feature, the same as
    any other missing value -- there is no attempt to parse it from whatever
    is sitting there instead, since a column is a date because of its dtype,
    at fit and at predict alike.

    Attributes:
        encoders_: Raw column position -> the encoder fit on that column.
        output_names_: Raw column position -> the calendar-feature names that
            column expands into.
        n_input_columns_: The raw input's column count, or `None` when it can't
            be determined (e.g. a 1D array).
    """

    encoders_: dict[int, DatetimeEncoder]
    output_names_: dict[int, list[str]]
    n_input_columns_: int | None

    def __init__(
        self,
        *,
        transform_dates: bool = False,
        categorical_features_indices: Sequence[int] = (),
    ) -> None:
        """Initialize the expander.

        Args:
            transform_dates: Whether an eligible date column is expanded into
                calendar features, rather than rendered to text.
            categorical_features_indices: Raw indices the caller declared
                categorical; a date column among them is never expanded,
                regardless of `transform_dates`.
        """
        self.transform_dates = transform_dates
        self.categorical_features_indices = categorical_features_indices

    def fit(self, X: XType) -> DateTimeExpander:
        """Pick the date columns to expand and fit one encoder for each.

        Warns about every date column left to be read as a plain category or
        text. Nothing else about `X` is inspected -- only its dtypes -- so this
        is safe to call before any validation.

        Args:
            X: The input data, before any dtype fixing. Only a `DataFrame` can
                carry a temporal column; anything else fits an expander with
                nothing to do.

        Returns:
            Itself, fitted.
        """
        self.encoders_ = {}
        self.output_names_ = {}
        self.n_input_columns_ = _num_columns_or_none(X)
        if not isinstance(X, pd.DataFrame):
            return self

        date_indices = [
            i for i, dtype in enumerate(X.dtypes) if _is_datetime_like_dtype(dtype)
        ]
        declared_categorical = set(self.categorical_features_indices)
        to_expand = [
            i
            for i in date_indices
            if self.transform_dates and i not in declared_categorical
        ]
        # Warned from `fit` itself rather than from a helper of it: the
        # `stacklevel` inside counts the frames from here up to the caller's
        # own `estimator.fit(X, y)`.
        _warn_on_dates_held_as_text(
            [
                X.columns[i]
                for i in date_indices
                if i not in to_expand and i not in declared_categorical
            ]
        )

        for position in to_expand:
            encoder = _make_datetime_encoder()
            encoded = pd.DataFrame(
                encoder.fit_transform(_as_timestamp(X.iloc[:, position]))
            )
            self.encoders_[position] = encoder
            # Snapshotted, rather than read back off the encoder when needed:
            # skrub's `DatetimeEncoder.transform` reassigns its own
            # `all_outputs_` (what `get_feature_names_out` returns) from the
            # label of the column it was just handed, so those names drift as
            # soon as a predict-time frame labels the column differently.
            self.output_names_[position] = list(encoded.columns)
        return self

    def transform(self, X: XType) -> XType:
        """Resolve every temporal column, by expanding it or rendering it to text.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            The resolved data: every fitted position replaced by its calendar
            features (appended after the kept columns, in original-index
            order), every other date column rendered to ISO 8601 text, and
            every duration column turned into seconds.

        Raises:
            NotFittedError: If `fit` has not run yet -- which columns are
                expanded is its decision to make, not this one's.
        """
        # Checked by hand rather than with sklearn's `check_is_fitted`, which
        # requires its argument to be a `BaseEstimator`; this is a plain
        # object, since nothing clones it or reads parameters off it.
        if not hasattr(self, "encoders_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call "
                "`fit` before using `transform`."
            )
        if not isinstance(X, pd.DataFrame):
            return X

        dtypes = list(X.dtypes)
        date_indices = [
            i for i, dtype in enumerate(dtypes) if _is_datetime_like_dtype(dtype)
        ]
        duration_indices = [
            i
            for i, dtype in enumerate(dtypes)
            if pd.api.types.is_timedelta64_dtype(dtype)
        ]
        if not date_indices and not duration_indices and not self.encoders_:
            # Nothing to expand, no fitted columns needing a degraded (NaN)
            # reapplication either -- a genuine no-op, unlike the case where
            # `encoders_` still has positions to (re)produce even though none
            # of them holds a date dtype right now.
            return X

        expanded_blocks = {
            position: self._encode_block(X, position, is_date=position in date_indices)
            for position in sorted(self.encoders_)
        }
        single_column_replacements: dict[int, np.ndarray] = {}
        for position in date_indices:
            if position in self.encoders_:
                continue
            column = _as_timestamp(X.iloc[:, position])
            single_column_replacements[position] = (
                column.astype(str).where(column.notna(), None).to_numpy()
            )
        for position in duration_indices:
            if position in self.encoders_:
                continue
            single_column_replacements[position] = (
                X.iloc[:, position].dt.total_seconds().to_numpy()
            )

        recast_frame = _replace_columns_positionally(X, single_column_replacements)
        return _drop_expanded_and_append(recast_frame, expanded_blocks)

    @property
    def expanded_input_indices(self) -> list[int]:
        """Raw column indices that are expanded, ascending."""
        return sorted(self.encoders_)

    @property
    def expanded_output_indices(self) -> list[int]:
        """Post-transform positions holding every expanded column's output.

        Always contiguous, right after every kept column: `transform` drops
        every expanded column, keeping the rest in relative order, then
        appends the expanded blocks after them, in original-index order.
        """
        n_kept = (self.n_input_columns_ or 0) - len(self.encoders_)
        total_width = sum(len(names) for names in self.output_names_.values())
        return list(range(n_kept, n_kept + total_width))

    def output_indices_for(self, raw_indices: Iterable[int]) -> list[int]:
        """Where each of `raw_indices` sits after `transform`.

        A kept column keeps its relative order; it just shifts down by however
        many expanded columns were removed ahead of it. An expanded position
        has no single answer -- it became many columns -- and is never asked
        for: the only caller passes declared-categorical indices, and a
        declared-categorical column is never expanded.
        """
        expanded = self.expanded_input_indices
        return [index - sum(1 for e in expanded if e < index) for index in raw_indices]

    def output_feature_names(self, raw_names: Sequence[str] | None) -> list[str] | None:
        """`raw_names`, with each expanded column's generated names appended.

        The name list `detect_feature_modalities` needs, matching a
        transformed frame column-for-column. Deliberately not named
        `get_feature_names_out`: unlike sklearn's, it answers `None` for
        `None`, since an unnamed array input has no names to build on (and
        never had a date column either).
        """
        if raw_names is None:
            return None
        kept = [name for i, name in enumerate(raw_names) if i not in self.encoders_]
        expanded = [
            name
            for position in sorted(self.encoders_)
            for name in self.output_names_[position]
        ]
        return [*kept, *expanded]

    def _encode_block(
        self, X: pd.DataFrame, position: int, *, is_date: bool
    ) -> pd.DataFrame:
        """One expanded column's calendar features, named as they were at fit."""
        names = self.output_names_[position]
        if is_date:
            encoded = pd.DataFrame(
                self.encoders_[position].transform(_as_timestamp(X.iloc[:, position]))
            )
        else:
            # Fit expanded this position, but it is no longer a genuine
            # datetime dtype right now -- degrade to NaN rather than guess.
            encoded = pd.DataFrame({name: [float("nan")] * len(X) for name in names})
        return encoded.set_axis(names, axis=1).reset_index(drop=True)


def _warn_on_dates_held_as_text(column_names: list[Any]) -> None:
    """Warn about date columns read as a plain category or text.

    Empty whenever every date column was declared categorical or expanded --
    both routes reach this call with nothing to report.
    """
    if not column_names:
        return
    shown = column_names[:_MAX_DATE_COLUMNS_IN_WARNING]
    printed = ", ".join(repr(str(name)) for name in shown)
    if len(column_names) > len(shown):
        printed += f" (and {len(column_names) - len(shown)} more)"
    warnings.warn(
        f"These columns hold dates, which are read as plain categories or "
        f"text: {printed}.\n"
        'Raise `inference_config={"TRANSFORM_DATES": True}` to expand them into '
        "calendar features instead. To silence this for a column that should "
        "stay a plain category or text, pass its index in "
        "`categorical_features_indices`.",
        UserWarning,
        # stacklevel=6 reaches the `estimator.fit(X, y)` call site; pinned by
        # the `warning.filename` assert in the tests.
        stacklevel=6,
    )


def _as_timestamp(column: pd.Series) -> pd.Series:
    """A point-in-time column, as plain (non-period) timestamps.

    A period is a span, not an instant; its start is the instant that orders
    identically, which is all the calendar features -- or the ISO text
    rendering -- need.
    """
    if isinstance(column.dtype, pd.PeriodDtype):
        return column.dt.to_timestamp()
    return column


def _num_columns_or_none(X: XType) -> int | None:
    """`X`'s column count, or `None` if it can't be determined (e.g. a 1D array).

    Nothing in a shapeless `X` can be a date column either way, and downstream
    value validation (`check_array`/`check_X_y`) rejects such an `X` with its
    own, clearer message before anything would ever consult this count.
    """
    try:
        return _num_features(X)
    except TypeError:
        return None


def _replace_columns_positionally(
    X: pd.DataFrame,
    replacements: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Return `X` with the given column positions replaced, leaving `X` untouched.

    Positional, and via a temporary integer column axis rather than
    ``isetitem``: the labels are the caller's, so they can repeat (the same
    duplicate-name case ``build_input_feature_names`` exists for), which makes
    assignment by label ambiguous -- and ``isetitem`` only arrived in pandas
    1.5, below this package's floor. Numbering the axis makes every label unique
    and equal to its own position, so a plain assignment is unambiguous, and the
    caller's labels go back afterwards.

    The copy is shallow and the frame handed in is never written through: each
    assignment replaces a whole column rather than any value inside one.
    """
    if not replacements:
        return X
    out = X.copy(deep=False)
    original_columns = out.columns
    out.columns = pd.RangeIndex(out.shape[1])
    for position, values in replacements.items():
        out[position] = values
    out.columns = original_columns
    return out


def _drop_expanded_and_append(
    frame: pd.DataFrame,
    expanded_blocks: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Drop the expanded columns and append their replacements.

    Positional, not `frame.drop(columns=...)`: dropping by label instead of
    position would silently misbehave for duplicate labels (the same case
    `build_input_feature_names` exists to handle elsewhere).
    """
    if not expanded_blocks:
        return frame
    expanded = set(expanded_blocks)
    keep = [i for i in range(frame.shape[1]) if i not in expanded]
    remaining = frame.iloc[:, keep].reset_index(drop=True)
    ordered_blocks = [expanded_blocks[i] for i in sorted(expanded_blocks)]
    return pd.concat([remaining, *ordered_blocks], axis=1)
