#  Copyright (c) Prior Labs GmbH 2026.

"""Expand datetime columns into calendar features before validation runs.

A datetime column (`datetime64`, tz-aware, or `period`) is only accepted with
`inference_config={"TRANSFORM_DATES": True}`: `DateTimeExpander` then expands
it into calendar features via `skrub.DatetimeEncoder`. With the flag off (the
default), `fit` raises a `TabPFNValidationError` naming the columns and the
flag. A `timedelta64` column always becomes its length in seconds. A string
column that merely looks like a date is never treated as one.

This runs before sklearn's `check_array`/`check_X_y`, which cannot hold a
`datetime64` column beside a numeric one in a single array. The expander is
used like a `ColumnTransformer`: `fit_transform` on the raw fit input decides
which columns expand and expands them, and `transform` reapplies that decision
to every predict input.

Only `TabPFNClassifier` and `TabPFNRegressor` run this. The fine-tuning
estimators do not: they accept no `inference_config`, so a datetime column
must be converted before fine-tuning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _num_features
from skrub import DatetimeEncoder

from tabpfn.errors import TabPFNValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tabpfn.constants import XType

#: Cap on how many column names an error lists, so a wide frame of date
#: columns does not produce an unreadable multi-kilobyte message.
_MAX_COLUMNS_IN_MESSAGE = 10


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
    """Expands every datetime column of an input into calendar features.

    Used like a `ColumnTransformer`: `fit_transform` it on the raw fit input,
    then `transform` every later predict input. Fitting settles which columns
    are expanded, once, and rejects a datetime column it is not allowed to
    expand; `transform` reapplies that decision positionally, exactly like
    `ordinal_encoder_`.

    A column is a date because of its dtype, at fit and at predict alike. A
    position that was expanded at fit time must therefore still carry a
    datetime dtype at predict time; anything else there is rejected rather
    than parsed or silently read as missing.

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
            transform_dates: Whether a datetime column is expanded into calendar
                features. With this off, `fit` rejects any datetime column.
            categorical_features_indices: Raw indices the caller declared
                categorical. A datetime column among them is rejected by
                `fit`: it is expanded into many numerical columns, so there is
                no single column the declaration could apply to.
        """
        self.transform_dates = transform_dates
        self.categorical_features_indices = categorical_features_indices

    def fit(self, X: XType) -> DateTimeExpander:
        """Fit one encoder for every datetime column, or reject the input.

        Nothing about `X` is inspected but its dtypes, so this is safe to call
        before any validation. Prefer `fit_transform` when the transformed fit
        input is needed too: fitting an encoder already encodes its column, and
        `fit_transform` keeps that result instead of encoding a second time.

        Args:
            X: The input data, before any dtype fixing. Only a `DataFrame` can
                carry a datetime column; anything else fits an expander with
                nothing to do.

        Returns:
            Itself, fitted.

        Raises:
            TabPFNValidationError: If `X` holds a datetime column while
                `transform_dates` is off, or one that is declared categorical;
                or if `X` is a `datetime64` array, which no flag can expand.
        """
        self._fit(X)
        return self

    def fit_transform(self, X: XType) -> XType:
        """`fit(X).transform(X)`, encoding each datetime column only once.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            `X`, transformed as `transform` would.

        Raises:
            TabPFNValidationError: As `fit`.
        """
        expanded_blocks = self._fit(X)
        if not isinstance(X, pd.DataFrame):
            return self.transform(X)
        return self._assemble(X, expanded_blocks)

    def transform(self, X: XType) -> XType:
        """Expand every fitted datetime column and turn durations into seconds.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            `X` with every fitted position replaced by its calendar features
            (appended after the kept columns, in original-index order) and
            every duration column turned into seconds.

        Raises:
            NotFittedError: If `fit` has not run yet -- which columns are
                expanded is its decision to make, not this one's.
            TabPFNValidationError: If `X` holds a datetime column at a position
                `fit` did not expand, i.e. one that was not a date at fit time;
                if a position `fit` expanded no longer holds a datetime dtype;
                or if `fit` expanded columns and `X` is not a `DataFrame`, which
                is the only input that can carry them.
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
            if self.encoders_:
                # Same raw width as at fit, so the shape check upstream passed,
                # but nothing here can widen an array to the expanded layout.
                raise TabPFNValidationError(
                    f"`fit` expanded datetime columns at positions "
                    f"{self.expanded_input_indices} into calendar features, so "
                    f"predict input must be a pandas DataFrame carrying those "
                    f"columns with a datetime dtype; got {type(X).__name__}."
                )
            if isinstance(X, np.ndarray) and X.dtype.kind == "m":
                return X / np.timedelta64(1, "s")
            return X

        date_indices = _datetime_column_indices(X)
        if unexpected := [i for i in date_indices if i not in self.encoders_]:
            raise TabPFNValidationError(
                f"These columns hold dates (datetime dtype) now but did not when "
                f"`fit` ran: {_format_names([X.columns[i] for i in unexpected])}.\n"
                "Only a column that was a datetime column at fit time is expanded "
                "into calendar features; pass every column with the dtype it had "
                "at fit time."
            )
        if missing := [i for i in self.expanded_input_indices if i not in date_indices]:
            raise TabPFNValidationError(
                f"These columns held dates (datetime dtype) when `fit` ran but do "
                f"not now: {_format_names([X.columns[i] for i in missing])}.\n"
                "A column expanded into calendar features at fit time must carry "
                "a datetime dtype at predict time too; convert it first (e.g. "
                "`pd.to_datetime`) rather than passing it as strings or numbers."
            )
        expanded_blocks = {
            position: self._encode_block(X, position)
            for position in self.expanded_input_indices
        }
        return self._assemble(X, expanded_blocks)

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
        for: the only caller passes declared-categorical indices, and `fit`
        rejects a declared-categorical datetime column.
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
            for position in self.expanded_input_indices
            for name in self.output_names_[position]
        ]
        return [*kept, *expanded]

    def _fit(self, X: XType) -> dict[int, pd.DataFrame]:
        """Fit the encoders; return each expanded column's calendar features."""
        self.encoders_ = {}
        self.output_names_ = {}
        self.n_input_columns_ = _num_columns_or_none(X)
        if not isinstance(X, pd.DataFrame):
            _reject_datetime_array(X)
            return {}

        date_indices = _datetime_column_indices(X)
        if date_indices and not self.transform_dates:
            raise TabPFNValidationError(
                f"These columns hold dates (datetime dtype), which are only read "
                f'with `inference_config={{"TRANSFORM_DATES": True}}`: '
                f"{_format_names([X.columns[i] for i in date_indices])}.\n"
                "Set that flag to expand each of them into calendar features "
                "(year, month, day, weekday, ...), or convert them yourself first "
                "(e.g. `.astype(str)` to read them as plain categories)."
            )
        declared_categorical = set(self.categorical_features_indices)
        if declared := [i for i in date_indices if i in declared_categorical]:
            raise TabPFNValidationError(
                f"These columns hold dates (datetime dtype) but are listed in "
                f"`categorical_features_indices`: "
                f"{_format_names([X.columns[i] for i in declared])}.\n"
                "A datetime column is expanded into calendar features, so it "
                "cannot be declared categorical. Drop it from "
                "`categorical_features_indices`, or convert it to strings yourself "
                "first (e.g. `.astype(str)`)."
            )

        expanded_blocks: dict[int, pd.DataFrame] = {}
        for position in date_indices:
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
            expanded_blocks[position] = encoded.reset_index(drop=True)
        return expanded_blocks

    def _encode_block(self, X: pd.DataFrame, position: int) -> pd.DataFrame:
        """One expanded column's calendar features, named as they were at fit."""
        encoded = pd.DataFrame(
            self.encoders_[position].transform(_as_timestamp(X.iloc[:, position]))
        )
        return encoded.set_axis(self.output_names_[position], axis=1).reset_index(
            drop=True
        )

    def _assemble(
        self, X: pd.DataFrame, expanded_blocks: dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """Turn durations into seconds, then swap expanded columns for their blocks."""
        duration_indices = [
            i
            for i, dtype in enumerate(X.dtypes)
            if pd.api.types.is_timedelta64_dtype(dtype) and i not in self.encoders_
        ]
        if not duration_indices and not expanded_blocks:
            return X
        durations_in_seconds = {
            position: X.iloc[:, position].dt.total_seconds().to_numpy()
            for position in duration_indices
        }
        recast_frame = _replace_columns_positionally(X, durations_in_seconds)
        return _drop_expanded_and_append(recast_frame, expanded_blocks)


def _datetime_column_indices(X: pd.DataFrame) -> list[int]:
    """Positions of the columns holding points in time."""
    return [i for i, dtype in enumerate(X.dtypes) if _is_datetime_like_dtype(dtype)]


def _reject_datetime_array(X: XType) -> None:
    """Reject a `datetime64` array: only a `DataFrame` column can be expanded.

    Without this, the array reaches modality detection, which cannot read a
    datetime dtype and says so in terms of a column that is already a date.
    """
    if isinstance(X, np.ndarray) and X.dtype.kind == "M":
        raise TabPFNValidationError(
            f"Got a numpy array of dtype {X.dtype}, which holds dates. Only a "
            "pandas DataFrame column can be expanded into calendar features: pass "
            'the data as a DataFrame with `inference_config={"TRANSFORM_DATES": '
            "True}` instead."
        )


def _format_names(column_names: list[Any]) -> str:
    """Quote the names for an error message, listing at most a handful."""
    shown = column_names[:_MAX_COLUMNS_IN_MESSAGE]
    printed = ", ".join(repr(str(name)) for name in shown)
    if len(column_names) > len(shown):
        printed += f" (and {len(column_names) - len(shown)} more)"
    return printed


def _as_timestamp(column: pd.Series) -> pd.Series:
    """A point-in-time column, as plain (non-period) timestamps under a string name.

    A period is a span, not an instant; its start is the instant that orders
    identically, which is all the calendar features need. The name is made a
    string because skrub builds its output names by concatenating onto it,
    which fails on an integer label (e.g. a frame with default columns).
    """
    if isinstance(column.dtype, pd.PeriodDtype):
        column = column.dt.to_timestamp()
    return column.rename(str(column.name))


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
