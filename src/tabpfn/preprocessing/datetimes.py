#  Copyright (c) Prior Labs GmbH 2026.

"""Converting temporal columns, before validation ever sees them.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric one
in one array (no common dtype exists), so a temporal column has to stop looking
like one before `check_array`/`check_X_y` run, which is why this tier exists at
all. A point in time (`datetime64`, tz-aware, or `period`) is expanded into
calendar features when `TRANSFORM_DATES` is on, and refused with an error naming
it otherwise. A duration (`timedelta64`) always becomes its length in seconds.

The frame surgery is positional throughout, because the labels are the caller's
and so can repeat (the same duplicate-name case `build_input_feature_names`
exists for), which makes anything driven by a label ambiguous.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skrub import DatetimeEncoder

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import make_names_unique

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType

__all__ = ["DateTransformer"]


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
    Not `BaseEstimator`/`TransformerMixin` either: nothing clones it or reads
    parameters off it.

    Usage mirrors `ordinal_encoder_`: call `fit_transform` once at fit time and
    keep the instance around (as `self.date_transformer_`), then call `transform`
    on it at predict time.

    Args:
        categorical_indices: Indices the caller declared categorical. A point in
            time among them is refused at fit: it is expanded into many numeric
            columns, or refused outright, so there is no single column the
            declaration could apply to.
        transform_dates: Whether a point in time is expanded into a year, a day
            of year, the seconds since the epoch, and cyclical month, day and
            weekday pairs, so a December and the December before it are near
            each other rather than a year apart. Off, such a column is refused
            with an error naming it.

    Attributes:
        fitted_columns_: Input position -> the encoder fitted on that column and
            the names of the features it makes. Empty when nothing was expanded.
        feature_names_out_: The transformed frame's column labels as strings, in
            order, or `None` when the input was not a `DataFrame` and so has no
            labels.
    """

    fitted_columns_: dict[int, FittedDateColumn]
    feature_names_out_: list[str] | None

    def __init__(
        self,
        *,
        categorical_indices: Sequence[int] | None = None,
        transform_dates: bool = False,
    ) -> None:
        self._declared_categorical = set(categorical_indices or ())
        self._transform_dates = transform_dates

    @property
    def expanded_indices(self) -> list[int]:
        """Input positions that were expanded into calendar features, ascending."""
        self._check_is_fitted()
        return sorted(self.fitted_columns_)

    def fit(self, X: XType) -> DateTransformer:
        """Fit one encoder per point in time in `X`, refusing a date it cannot.

        Prefer `fit_transform` when the converted fit input is needed too:
        fitting an encoder already encodes its column, and `fit_transform` keeps
        that result rather than encoding a second time.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            Itself, fitted.

        Raises:
            TabPFNValidationError: On a point in time with `transform_dates` off,
                on one declared categorical, or on a `datetime64` array, which
                no flag can expand.
        """
        self._fit(X)
        return self

    def fit_transform(self, X: XType) -> XType:
        """`fit(X).transform(X)`, encoding each date column only once.

        Expansion changes the column count, so the fitted transformer also
        reports the transformed frame's labels (`feature_names_out_`) and moves
        input indices to where those columns ended up (`output_indices`):
        everything downstream indexes the wider frame.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            `X`, converted as `transform` would.

        Raises:
            TabPFNValidationError: As `fit`.
        """
        blocks = self._fit(X)
        if not isinstance(X, pd.DataFrame):
            return _duration_array_to_seconds(X)
        return self._assemble(X, blocks)

    def transform(self, X: XType) -> XType:
        """Reapply the conversion `fit` decided on, so the width holds.

        Which positions hold a point in time has to match what `fit` saw: a date
        where there was none has no encoder to go through, and a position that
        was expanded needs a date to feed the encoder it has. Either mismatch is
        refused rather than guessed at, since the model was fit on a frame of
        one particular shape.

        Args:
            X: The data, before any dtype fixing.

        Raises:
            NotFittedError: If `fit` has not run yet; which columns are expanded
                is its decision.
            TabPFNValidationError: If `X`'s points in time sit at other positions
                than at fit, or if `fit` expanded columns and `X` is not a
                `DataFrame`, the only input that can carry them.
        """
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            _refuse_datetime_array(X)
            _refuse_array_after_expansion(X, self.expanded_indices)
            return _duration_array_to_seconds(X)
        instants = self._instant_positions(X)
        if not self._transform_dates:
            _refuse(X, instants)
        _refuse_unexpected_dates(
            X, [i for i in instants if i not in self.fitted_columns_]
        )
        _refuse_missing_dates(
            X, [i for i in self.expanded_indices if i not in instants]
        )
        blocks = [
            self._apply_one(X.iloc[:, i], self.fitted_columns_[i])
            for i in self.expanded_indices
        ]
        return self._assemble(X, blocks)

    def output_indices(self, indices: Sequence[int] | None) -> list[int] | None:
        """Where each of `indices`, input positions, sits in the transformed frame.

        A kept column keeps its relative order, so it just shifts down by however
        many expanded columns sat ahead of it. An expanded position has no single
        answer, it became many columns, and is never asked for: the one caller
        passes declared-categorical indices, which are never expanded.

        Args:
            indices: Input positions, or `None` for none declared.

        Returns:
            The same positions in the transformed frame, or `None` for `None`.
        """
        self._check_is_fitted()
        if indices is None:
            return None
        expanded = self.expanded_indices
        return [i - sum(1 for j in expanded if j < i) for i in indices]

    def _check_is_fitted(self) -> None:
        # By hand rather than sklearn's `check_is_fitted`, which requires a
        # `BaseEstimator`.
        if not hasattr(self, "fitted_columns_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call "
                "`fit` before using `transform`."
            )

    def _fit(self, X: XType) -> list[pd.DataFrame]:
        """Fit the encoders; return each expanded column's features, in order."""
        # Cleared before anything else, so that refitting on an input with no
        # columns to expand still forgets the last fit.
        self.fitted_columns_ = {}
        self.feature_names_out_ = None
        if not isinstance(X, pd.DataFrame):
            _refuse_datetime_array(X)
            return []

        instants = self._instant_positions(X)
        if not self._transform_dates:
            _refuse(X, instants)
        _refuse_declared_categorical(
            X, [i for i in instants if i in self._declared_categorical]
        )
        kept_names = [
            str(column) for i, column in enumerate(X.columns) if i not in set(instants)
        ]
        expanded_names: list[str] = []
        blocks: list[pd.DataFrame] = []
        for position in instants:
            column = as_timestamp(X.iloc[:, position]).rename(str(X.columns[position]))
            block, fitted = self._fit_one(column, kept_names + expanded_names)
            self.fitted_columns_[position] = fitted
            expanded_names += fitted.output_names
            blocks.append(block)
        self.feature_names_out_ = kept_names + expanded_names
        return blocks

    def _assemble(
        self, X: pd.DataFrame, blocks: Sequence[pd.DataFrame]
    ) -> pd.DataFrame:
        """Turn durations into seconds, then swap expanded columns for `blocks`."""
        converted = replace_columns_positionally(
            X, {i: to_seconds(X.iloc[:, i]) for i in self._duration_positions(X)}
        )
        if not blocks:
            return converted
        # `drop_and_append` concatenates the kept columns against skrub's own
        # (freshly default-indexed) output, so the row index has to be the
        # default range already or the two align by label instead of position.
        return drop_and_append(
            converted.reset_index(drop=True), self.expanded_indices, blocks
        )

    @staticmethod
    def _instant_positions(X: pd.DataFrame) -> list[int]:
        """The positions of `X`'s points in time."""
        return [i for i, dtype in enumerate(X.dtypes) if is_instant_dtype(dtype)]

    @staticmethod
    def _duration_positions(X: pd.DataFrame) -> list[int]:
        """The positions of `X`'s durations, declared categorical or not.

        Leaving a declared-categorical duration alone only crashes validation,
        and a whole number of seconds ordinal-encodes as a category just as well.
        """
        return [
            i
            for i, dtype in enumerate(X.dtypes)
            if pd.api.types.is_timedelta64_dtype(dtype)
        ]

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
        """Reapply one fitted encoder, naming its features as at fit."""
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
    raise TabPFNValidationError(
        f"These columns hold datetimes, which TabPFN does not support: "
        f"{_name_columns(X, positions)}. "
        'Set `inference_config={"TRANSFORM_DATES": True}` to expand them into '
        "calendar features, or preprocess them yourself first."
    )


def _refuse_declared_categorical(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on the points in time at `positions`, all declared categorical.

    Expansion turns one such column into many numeric ones, so there is no
    single column the declaration could apply to; refusing is clearer than
    quietly picking one reading over the other.
    """
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns hold datetimes but are listed in "
        f"`categorical_features_indices`: {_name_columns(X, positions)}. A "
        "datetime column is "
        "expanded into calendar features, so it cannot be a category as well: "
        "drop it from `categorical_features_indices`, or cast it to strings "
        "yourself first."
    )


def _refuse_unexpected_dates(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on points in time at `positions`, where `fit` saw none.

    No encoder was fitted for them, so there is nothing to expand them with; the
    frame the model was fit on had something else there.
    """
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns hold datetimes now but did not when `fit` ran: "
        f"{_name_columns(X, positions)}. Only a column that held datetimes at fit "
        "is expanded into calendar features: pass every column with the dtype it "
        "had at fit."
    )


def _refuse_missing_dates(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on `positions` that `fit` expanded but that hold no points in time now.

    The encoder fitted there needs dates to run on; whatever is sitting there
    instead is not parsed, and not read as missing either.
    """
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns held datetimes when `fit` ran but do not now: "
        f"{_name_columns(X, positions)}. A column expanded into calendar features "
        "at fit needs datetimes at predict too: convert it first (e.g. with "
        "`pd.to_datetime`) rather than passing strings or numbers."
    )


def _refuse_array_after_expansion(X: XType, expanded: Sequence[int]) -> None:
    """Raise on a non-`DataFrame` predict input once `fit` expanded columns.

    Its raw width may match `n_features_in_`, so the shape check upstream let it
    through, but nothing here can widen an array to the expanded layout.
    """
    if not expanded:
        return
    raise TabPFNValidationError(
        f"`fit` expanded the datetime columns at positions {list(expanded)} into "
        "calendar features, so predict input has to be a DataFrame carrying those "
        f"columns as datetimes; got {type(X).__name__}."
    )


def _name_columns(X: pd.DataFrame, positions: Sequence[int]) -> str:
    """Name each of `positions` by index and label, e.g. `1 ('signed_on')`."""
    return ", ".join(f"{i} ({X.columns[i]!r})" for i in positions)


def _refuse_datetime_array(X: XType) -> None:
    """Raise on a `datetime64` array: only a `DataFrame` column can be expanded.

    Left alone, it would reach `fix_dtypes`, which rejects the dtype without
    saying what to do about it.
    """
    if isinstance(X, np.ndarray) and X.dtype.kind == "M":
        raise TabPFNValidationError(
            f"Got a numpy array of dtype {X.dtype}, which holds datetimes. Only a "
            "DataFrame column can be expanded into calendar features: pass a "
            'DataFrame with `inference_config={"TRANSFORM_DATES": True}`, or '
            "preprocess the dates yourself first."
        )


def _duration_array_to_seconds(X: XType) -> XType:
    """A `timedelta64` array as its lengths in seconds, or `X` unchanged.

    The same conversion a duration column gets, so an array and a frame holding
    the same durations reach the model as the same numbers.
    """
    if isinstance(X, np.ndarray) and X.dtype.kind == "m":
        return X / np.timedelta64(1, "s")
    return X


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


def to_seconds(column: pd.Series) -> pd.Series:
    """Cast one `timedelta64` column to its length in seconds.

    A duration carries no calendar, so its length is the whole of its meaning:
    unlike a point in time, there is nothing an expansion could add.
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
