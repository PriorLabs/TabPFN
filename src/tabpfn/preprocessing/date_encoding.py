#  Copyright (c) Prior Labs GmbH 2026.

"""Convert every temporal column to numbers, before validation ever sees it.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric one
in one array (no common dtype exists), so a temporal column has to stop looking
like one before `check_array`/`check_X_y` run. `DateTransformer` is where that
happens. A point in time (`datetime64`, tz-aware, or `period`) is expanded into
calendar features via `skrub.DatetimeEncoder` when `TRANSFORM_DATES` is on, and
otherwise becomes nanoseconds since the epoch: a single ordered number, which
keeps every distinct instant distinct but says nothing about weekdays or
seasons. A duration (`timedelta64`) always becomes its length in seconds, a
quantity with no calendar in it either way.

Because this runs before detection, `detect_feature_modalities` only ever sees
the result, and never learns a column was a date at all.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from skrub import DatetimeEncoder

from tabpfn.preprocessing.datamodel import make_names_unique
from tabpfn.preprocessing.modality_detection import format_names_for_warning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType


def _is_instant_dtype(dtype: Any) -> bool:
    """Whether `dtype` holds points in time: `datetime64`, tz-aware, or `period`."""
    return pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(
        dtype, pd.PeriodDtype
    )


def _as_timestamp(column: pd.Series) -> pd.Series:
    """The instant a `period` column starts at, or the column unchanged.

    A period is a span, not an instant; its start is the instant that orders
    identically, which is all any conversion here needs.
    """
    if isinstance(column.dtype, pd.PeriodDtype):
        return column.dt.to_timestamp()
    return column


def _to_nanoseconds(column: pd.Series) -> pd.Series:
    """Cast one point-in-time column to nanoseconds since the epoch.

    As `float64`, so a missing date (`NaT`) survives as `NaN`: `NaT.astype`
    `("int64")` maps it to that dtype's huge sentinel value instead, which
    `float64` has no equivalent of. Exact enough for any realistic datetime
    column: nanosecond-since-epoch magnitudes are still ~256ns short of
    float64's precision limit for a present-day date, far below anything a
    tabular feature would need to distinguish.
    """
    is_missing = column.isna().to_numpy()
    as_ns = column.astype("int64").astype("float64")
    as_ns[is_missing] = np.nan
    return as_ns


def _to_seconds(column: pd.Series) -> pd.Series:
    """Cast one `timedelta64` column to its length in seconds.

    A duration carries no calendar, so its length is the whole of its meaning:
    unlike a point in time, there is nothing an expansion could add later.
    """
    return column.dt.total_seconds()


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


def _replace_columns_positionally(
    X: pd.DataFrame,
    replacements: dict[int, pd.Series],
) -> pd.DataFrame:
    """Return `X` with the given column positions replaced, leaving `X` untouched.

    Positional, via a temporary integer column axis: the labels are the caller's,
    so they can repeat (the same duplicate-name case `build_input_feature_names`
    exists for), which makes assignment by label ambiguous. Numbering the axis
    makes every label unique and equal to its own position, so a plain assignment
    is unambiguous, and the caller's labels go back afterwards.

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


def _drop_and_append(
    frame: pd.DataFrame,
    expanded: Sequence[int],
    blocks: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Drop every expanded column and append its calendar features instead.

    Positional, not `frame.drop(columns=...)`: labels can repeat, so dropping by
    label would take the wrong column in that case. Every expanded column's
    output lands after all the kept ones, in original-position order, which is
    what makes the appended block's positions predictable.
    """
    keep = [i for i in range(frame.shape[1]) if i not in set(expanded)]
    return pd.concat([frame.iloc[:, keep], *blocks], axis=1)


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
        # stacklevel=6 reaches the `estimator.fit(X, y)` call site; pinned by the
        # `warning.filename` asserts in the tests.
        stacklevel=6,
    )


@dataclasses.dataclass(frozen=True)
class DateConversion:
    """What `DateTransformer.fit_transform` did, as the fit path needs it.

    Attributes:
        X: The converted data.
        feature_names: `X`'s column labels as strings, in order, or `None` when
            the input was not a `DataFrame` and so has no labels.
        categorical_indices: The caller's declared categorical indices, moved to
            where those columns ended up.
        numerical_indices: Positions holding calendar-expansion output. Numbers
            by construction, so they bypass the cardinality heuristics that
            would read a cyclical month pair spanning two months as a category.
    """

    X: XType
    feature_names: list[str] | None
    categorical_indices: list[int] | None
    numerical_indices: list[int]


class DateTransformer:
    """Converts every temporal column in an input to numbers.

    Not a `PreprocessingStep` (`pipeline_interface.py`): that tier runs per
    ensemble member on already-numeric arrays, well past where this has to run.
    Not `BaseEstimator`/`TransformerMixin` either: `fit_transform` returns more
    than the transformed data, which does not fit sklearn's shape.

    Usage mirrors `ordinal_encoder_`: construct one, call `fit_transform` once at
    fit time and keep the instance around (as `self.date_transformer_`), then
    call `transform` on it at predict time.

    Args:
        categorical_indices: Indices the caller declared categorical. A point in
            time among them is left alone entirely, at fit and at predict alike:
            the user's declared intent for it wins over reading it as a date.
        transform_dates: Whether a point in time is expanded into calendar
            features rather than read as one plain number.
    """

    @dataclasses.dataclass
    class _FittedColumn:
        """One column's fitted encoder, and the names of the features it makes."""

        encoder: DatetimeEncoder
        output_names: list[str]

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
        self._fitted: dict[int, DateTransformer._FittedColumn] = {}

    @property
    def expanded_indices(self) -> list[int]:
        """Input positions that were expanded into calendar features, ascending.

        Empty before `fit_transform` runs, and whenever it expanded nothing.
        """
        return sorted(self._fitted)

    def fit_transform(self, X: XType) -> DateConversion:
        """Convert every temporal column in `X`, and warn naming the unexpanded.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            The conversion, including what the caller has to pass on to
            `detect_feature_modalities`.
        """
        self._fitted = {}
        if not isinstance(X, pd.DataFrame):
            return DateConversion(
                X=X,
                feature_names=None,
                categorical_indices=self._categorical_indices,
                numerical_indices=[],
            )

        instants, durations = self._temporal_positions(X)
        to_expand = instants if self._transform_dates else []
        converted = self._convert_in_place(
            X,
            instants=[i for i in instants if i not in set(to_expand)],
            durations=durations,
        )
        _warn_on_dates([str(X.columns[i]) for i in instants if i not in set(to_expand)])
        if not to_expand:
            return DateConversion(
                X=converted,
                feature_names=[str(column) for column in converted.columns],
                categorical_indices=self._categorical_indices,
                numerical_indices=[],
            )

        # `_drop_and_append` concatenates the kept columns against skrub's own
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
            column = _as_timestamp(X.iloc[:, position]).rename(str(X.columns[position]))
            block, fitted = self._fit_one(column, kept_names + expanded_names)
            self._fitted[position] = fitted
            expanded_names += fitted.output_names
            blocks.append(block)

        return DateConversion(
            X=_drop_and_append(converted, to_expand, blocks),
            feature_names=kept_names + expanded_names,
            categorical_indices=self._remap(self._categorical_indices, to_expand),
            numerical_indices=list(
                range(len(kept_names), len(kept_names) + len(expanded_names))
            ),
        )

    def transform(self, X: XType) -> XType:
        """Reapply the conversion `fit_transform` decided on, silently.

        Only `expanded_indices`, frozen at fit time, is ever expanded, and always
        with the encoder fitted then, so the columns come out the same width and
        in the same order. A position expanded at fit time that no longer holds a
        point in time degrades to `NaN` calendar features, like any other missing
        value: there is no attempt to parse whatever is sitting there instead.

        Every other temporal column is found from the dtypes again rather than
        frozen, since an unconverted one crashes validation whether or not it was
        a date when we fit.

        Args:
            X: The data, before any dtype fixing.
        """
        if not isinstance(X, pd.DataFrame):
            return X
        to_expand = self.expanded_indices
        instants, durations = self._temporal_positions(X)
        converted = self._convert_in_place(
            X,
            instants=[i for i in instants if i not in set(to_expand)],
            durations=durations,
        )
        if not to_expand:
            return converted

        # See the identical comment in `fit_transform`.
        converted = converted.reset_index(drop=True)
        blocks = [
            self._apply_one(X.iloc[:, position], self._fitted[position])
            for position in to_expand
        ]
        return _drop_and_append(converted, to_expand, blocks)

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
            if _is_instant_dtype(dtype) and i not in self._declared_categorical
        ]
        durations = [
            i
            for i, dtype in enumerate(dtypes)
            if pd.api.types.is_timedelta64_dtype(dtype)
        ]
        return instants, durations

    @staticmethod
    def _convert_in_place(
        X: pd.DataFrame,
        *,
        instants: Sequence[int],
        durations: Sequence[int],
    ) -> pd.DataFrame:
        """Replace each given column with its single-number conversion."""
        replacements = {
            i: _to_nanoseconds(_as_timestamp(X.iloc[:, i])) for i in instants
        }
        replacements.update({i: _to_seconds(X.iloc[:, i]) for i in durations})
        return _replace_columns_positionally(X, replacements)

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
    ) -> tuple[pd.DataFrame, DateTransformer._FittedColumn]:
        """Fit an encoder on one column, naming its output after that column.

        skrub names each feature after the column it came from (e.g.
        "signed_on_year"), which is kept as-is, deduplicated only against names
        already in the frame. How many features there are is decided here too:
        skrub drops the ones a column cannot vary in, e.g. the time of day of a
        date-only column, which is why `transform` has to reuse this encoder
        rather than fit a new one.
        """
        encoder = _make_datetime_encoder()
        encoded = pd.DataFrame(encoder.fit_transform(column))
        output_names = make_names_unique(
            [str(name) for name in encoded.columns], existing=existing_names
        )
        return (
            encoded.set_axis(output_names, axis=1).reset_index(drop=True),
            DateTransformer._FittedColumn(encoder=encoder, output_names=output_names),
        )

    @staticmethod
    def _apply_one(
        column: pd.Series,
        fitted: DateTransformer._FittedColumn,
    ) -> pd.DataFrame:
        """Reapply one fitted encoder, or produce its features as all-`NaN`."""
        if not _is_instant_dtype(column.dtype):
            return pd.DataFrame(
                {name: np.full(len(column), np.nan) for name in fitted.output_names}
            )
        encoded = pd.DataFrame(fitted.encoder.transform(_as_timestamp(column)))
        return encoded.set_axis(fitted.output_names, axis=1).reset_index(drop=True)


def apply_date_conversion(X: XType, source: object) -> XType:
    """Convert `X`'s temporal columns via `source`'s fitted `date_transformer_`.

    `source` (a fitted estimator or ensemble worker) may never have set
    `date_transformer_` at all, e.g. `fit_from_preprocessed` skips the step that
    would, exactly like the pre-existing `ordinal_encoder_` guard. One built from
    `source`'s own declared categoricals converts the same way, minus an
    expansion there was no fit to decide on.
    """
    transformer = getattr(source, "date_transformer_", None) or DateTransformer(
        categorical_indices=getattr(source, "categorical_features_indices", None)
    )
    return transformer.transform(X)
