#  Copyright (c) Prior Labs GmbH 2026.

"""Convert temporal columns before validation sees them.

sklearn's `check_array`/`check_X_y` cannot hold a `datetime64` column beside a
numeric one in a single array, so a temporal column has to stop looking like one
first. A point in time (`datetime64`, tz-aware, or `period`) is expanded into
calendar features when `TRANSFORM_DATES` is on and refused with an error naming
it otherwise; a duration (`timedelta64`) always becomes its length in seconds.
Only a genuine temporal dtype counts: a string that looks like a date is a string.

Only `TabPFNClassifier` and `TabPFNRegressor` run this. The fine-tuning
estimators validate their input directly, so a datetime column has to be
converted before fine-tuning.

Column handling is positional throughout: labels are the caller's and can repeat.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

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
    """One input column's fitted encoder, its timezone, and its features' names."""

    encoder: DatetimeEncoder
    output_names: list[str]
    timezone: str | None
    """The column's timezone by name, or `None` for a naive column. Calendar
    features are read in the column's own timezone, so `transform` refuses a
    column that arrives in another."""


class DateTransformer:
    """Expands each point in time into calendar features, or refuses it.

    Used like `ordinal_encoder_`: `fit_transform` once at fit time, keep the
    instance as `date_transformer_`, `transform` at predict time. Not a
    `PreprocessingStep`, which runs per ensemble member on already-numeric
    arrays, well past where this has to run.

    Args:
        categorical_indices: Indices the caller declared categorical. A point in
            time among them is refused: it becomes many numeric columns, so there
            is no single column the declaration could apply to.
        transform_dates: Whether a point in time is expanded into a year, a day
            of year, the seconds since the epoch, and cyclical month, day and
            weekday pairs. Off, such a column is refused with an error naming it.

    Attributes:
        fitted_columns_: Input position -> the encoder fitted on that column, its
            timezone, and the names of the features it makes. Empty when nothing
            was expanded.
        feature_names_out_: The transformed frame's column labels as strings, or
            `None` when the input was not a `DataFrame` and so has no labels.
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

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            Itself, fitted.

        Raises:
            TabPFNValidationError: On a point in time with `transform_dates` off,
                or on one declared categorical.
        """
        self._fit(X)
        return self

    def fit_transform(self, X: XType) -> XType:
        """`fit(X).transform(X)`, encoding each date column only once.

        Expansion changes the column count: `feature_names_out_` reports the
        resulting labels and `output_indices` moves input indices to match.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            `X`, converted as `transform` would.

        Raises:
            TabPFNValidationError: As `fit`.
        """
        blocks = self._fit(X)
        if not isinstance(X, pd.DataFrame):
            return X
        return self._assemble(X, blocks)

    def transform(self, X: XType) -> XType:
        """Reapply the conversion `fit` decided on, so the width holds.

        Which positions hold a point in time, and in which timezone, has to match
        what `fit` saw; any mismatch is refused rather than guessed at.

        Args:
            X: The data, before any dtype fixing.

        Raises:
            NotFittedError: If `fit` has not run yet.
            TabPFNValidationError: If `X`'s points in time sit at other positions
                than at fit or come in another timezone, or if `fit` expanded
                columns and `X` is not a `DataFrame`, the only input that can
                carry them.
        """
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            _refuse_array_after_expansion(X, self.expanded_indices)
            return X
        instants = _instant_positions(X)
        if not self._transform_dates:
            _refuse(X, instants)
        _refuse_unexpected_dates(
            X, [i for i in instants if i not in self.fitted_columns_]
        )
        _refuse_missing_dates(
            X, [i for i in self.expanded_indices if i not in instants]
        )
        mismatched_timezones = {}
        for i in self.expanded_indices:
            timezone = _timezone_of(_as_timestamp(X.iloc[:, i]))
            if timezone != self.fitted_columns_[i].timezone:
                mismatched_timezones[i] = (self.fitted_columns_[i].timezone, timezone)
        _refuse_other_timezone(X, mismatched_timezones)
        blocks = [
            self._apply_one(X.iloc[:, i], self.fitted_columns_[i])
            for i in self.expanded_indices
        ]
        return self._assemble(X, blocks)

    def output_indices(self, indices: Sequence[int] | None) -> list[int] | None:
        """Where each of `indices`, input positions, sits in the transformed frame.

        A kept column shifts down by however many expanded columns sat ahead of
        it. An expanded position is never asked for: the one caller passes
        declared-categorical indices, which are never expanded.

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
        # By hand: sklearn's `check_is_fitted` requires a `BaseEstimator`.
        if not hasattr(self, "fitted_columns_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call "
                "`fit` before using `transform`."
            )

    def _fit(self, X: XType) -> list[pd.DataFrame]:
        """Fit the encoders; return each expanded column's features, in order."""
        # Cleared first, so refitting on an input with nothing to expand still
        # forgets the last fit.
        self.fitted_columns_ = {}
        self.feature_names_out_ = None
        if not isinstance(X, pd.DataFrame):
            return []

        instants = _instant_positions(X)
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
            column = _as_timestamp(X.iloc[:, position]).rename(str(X.columns[position]))
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
        converted = _replace_columns_positionally(
            X,
            {
                i: X.iloc[:, i].dt.total_seconds()
                for i, dtype in enumerate(X.dtypes)
                if pd.api.types.is_timedelta64_dtype(dtype)
            },
        )
        if not blocks:
            return converted
        # skrub's output is default-indexed, so the kept columns must be too, or
        # `concat` aligns the two by label instead of position.
        return _drop_and_append(
            converted.reset_index(drop=True), self.expanded_indices, blocks
        )

    @staticmethod
    def _fit_one(
        column: pd.Series,
        existing_names: Sequence[str],
    ) -> tuple[pd.DataFrame, FittedDateColumn]:
        """Fit an encoder on one column, naming its output after that column.

        skrub drops the features a column cannot vary in (e.g. the time of day of
        a date-only column), so how many there are is settled here, which is why
        `transform` reuses this encoder rather than fitting a fresh one.
        """
        encoder = DatetimeEncoder(
            resolution="second",
            add_weekday=True,
            add_day_of_year=True,
            periodic_encoding="circular",
        )
        encoded = pd.DataFrame(encoder.fit_transform(column))
        output_names = make_names_unique(
            [str(name) for name in encoded.columns], existing=existing_names
        )
        return (
            encoded.set_axis(output_names, axis=1).reset_index(drop=True),
            FittedDateColumn(
                encoder=encoder,
                output_names=output_names,
                timezone=_timezone_of(column),
            ),
        )

    @staticmethod
    def _apply_one(column: pd.Series, fitted: FittedDateColumn) -> pd.DataFrame:
        """Reapply one fitted encoder, naming its features as at fit."""
        encoded = pd.DataFrame(fitted.encoder.transform(_as_timestamp(column)))
        return encoded.set_axis(fitted.output_names, axis=1).reset_index(drop=True)


def _instant_positions(X: pd.DataFrame) -> list[int]:
    """Positions of `X`'s points in time: `datetime64`, tz-aware, or `period`."""
    return [
        i
        for i, dtype in enumerate(X.dtypes)
        if pd.api.types.is_datetime64_any_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    ]


def _as_timestamp(column: pd.Series) -> pd.Series:
    """A `period` column as the instants it starts at; any other column unchanged."""
    if isinstance(column.dtype, pd.PeriodDtype):
        return column.dt.to_timestamp()
    return column


def _timezone_of(column: pd.Series) -> str | None:
    """The datetime column's timezone by name, or `None` for a naive column."""
    timezone = column.dt.tz
    return None if timezone is None else str(timezone)


def _replace_columns_positionally(
    X: pd.DataFrame, replacements: dict[int, pd.Series]
) -> pd.DataFrame:
    """Return `X` with the given column positions replaced; `X` is left untouched."""
    if not replacements:
        return X
    # A temporary integer axis makes every label unique and equal to its
    # position, so assignment is unambiguous even when the caller's labels repeat.
    out = X.copy(deep=False)
    original_columns = out.columns
    out.columns = pd.RangeIndex(out.shape[1])
    for position, values in replacements.items():
        out[position] = values.to_numpy()
    out.columns = original_columns
    return out


def _drop_and_append(
    frame: pd.DataFrame, expanded: Sequence[int], blocks: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    """Drop the `expanded` positions and append `blocks` after the kept columns."""
    keep = [i for i in range(frame.shape[1]) if i not in set(expanded)]
    return pd.concat([frame.iloc[:, keep], *blocks], axis=1)


def _name_columns(X: pd.DataFrame, positions: Sequence[int]) -> str:
    """Name each of `positions` by index and label, e.g. `1 ('signed_on')`."""
    return ", ".join(f"{i} ({X.columns[i]!r})" for i in positions)


def _refuse(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on the points in time at `positions`; a no-op when there are none."""
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns hold datetimes, which TabPFN does not support: "
        f"{_name_columns(X, positions)}. "
        'Set `inference_config={"TRANSFORM_DATES": True}` to expand them into '
        "calendar features, or preprocess them yourself first."
    )


def _refuse_declared_categorical(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on the points in time at `positions`, all declared categorical."""
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns hold datetimes but are listed in "
        f"`categorical_features_indices`: {_name_columns(X, positions)}. A "
        "datetime column is expanded into calendar features, so it cannot be a "
        "category as well: drop it from `categorical_features_indices`, or cast "
        "it to strings yourself first."
    )


def _refuse_unexpected_dates(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on points in time at `positions`, where `fit` saw none."""
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns hold datetimes now but did not when `fit` ran: "
        f"{_name_columns(X, positions)}. Only a column that held datetimes at fit "
        "is expanded into calendar features: pass every column with the dtype it "
        "had at fit."
    )


def _refuse_missing_dates(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on `positions` that `fit` expanded but that hold no dates now."""
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns held datetimes when `fit` ran but do not now: "
        f"{_name_columns(X, positions)}. A column expanded into calendar features "
        "at fit needs datetimes at predict too: convert it first (e.g. with "
        "`pd.to_datetime`) rather than passing strings or numbers."
    )


def _refuse_other_timezone(
    X: pd.DataFrame, mismatches: dict[int, tuple[str | None, str | None]]
) -> None:
    """Raise on positions whose timezone differs from fit: `{position: (fit, now)}`."""
    if not mismatches:
        return
    described = ", ".join(
        f"{i} ({X.columns[i]!r}: {now or 'naive'}, was {fit or 'naive'})"
        for i, (fit, now) in mismatches.items()
    )
    raise TabPFNValidationError(
        f"These columns hold datetimes in another timezone than when `fit` ran: "
        f"{described}. Calendar features are read in the column's own timezone, "
        "so bring it back to the timezone it had at fit first."
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
