#  Copyright (c) Prior Labs GmbH 2026.

"""Resolve every temporal column before validation ever sees it.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric
one in one array (`DTypePromotionError`: no common dtype exists), so a
temporal column has to stop looking like one before `check_array`/`check_X_y`
run. `resolve_date_columns` is where that happens: a point in time
(`datetime64`, tz-aware, or `period`) is either expanded into calendar
features via `skrub.DatetimeEncoder` (when `TRANSFORM_DATES` is on and the
column isn't declared categorical) or rendered to ISO 8601 text (otherwise,
so it reads as an ordinary high-cardinality category downstream). A duration
(`timedelta64`) always becomes its length in seconds -- a quantity with no
calendar in it, so the number is the whole of its meaning, independent of
`TRANSFORM_DATES`.

Because this runs before validation, there is no `FeatureSchema` yet:
`detect_feature_modalities` only ever sees the *result* -- a fully validated,
already-expanded array -- and never learns a column was ever a date at all.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.utils.validation import _num_features
from skrub import DatetimeEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

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


@dataclasses.dataclass
class FittedDateColumn:
    """A fitted `DatetimeEncoder` for one column, and its (raw) output names."""

    encoder: DatetimeEncoder
    output_names: list[str]


@dataclasses.dataclass
class FittedDateColumns:
    """Which columns `resolve_date_columns` expanded at fit time.

    Usage mirrors `ordinal_encoder_`: keep the instance returned by fitting
    around (e.g. as `self.date_expander_`), and pass its `by_index` back into
    `resolve_date_columns` at predict time.
    """

    by_index: dict[int, FittedDateColumn] = dataclasses.field(default_factory=dict)

    @property
    def expanded_indices(self) -> list[int]:
        """Raw column indices that were expanded, ascending."""
        return sorted(self.by_index)


def fitted_date_columns_of(source: object) -> dict[int, FittedDateColumn]:
    """The `by_index` of `source.date_expander_`, or `{}` if never set.

    `source` (a fitted estimator or ensemble worker) may never have set
    `date_expander_` at all -- e.g. `fit_from_preprocessed` skips the step
    that would, exactly like the pre-existing `ordinal_encoder_` guard. `{}`
    (not `None`) so `resolve_date_columns` takes this as "nothing was fit to
    expand" rather than "fit new encoders now" -- the same predict-time
    behavior as when the attribute is present but genuinely empty.
    """
    date_expander = getattr(source, "date_expander_", None)
    return date_expander.by_index if date_expander is not None else {}


@dataclasses.dataclass
class DateResolution:
    """The result of resolving every temporal column in one input."""

    X: XType
    fitted: dict[int, FittedDateColumn]
    old_to_new_index: dict[int, int]
    """Original index -> new index, for every column that was *not*
    numerically expanded (untouched, text-rendered, or duration-to-seconds).
    A numerically expanded column has no single new index -- it became many.
    """

    @property
    def expanded_new_indices(self) -> list[int]:
        """New-layout positions holding every numerically-expanded column's
        output, combined.

        Always contiguous, right after every kept column: `_assemble` drops
        every expanded column, keeping the rest in relative order (exactly
        `old_to_new_index`'s positions), then appends the expanded blocks
        after them, in original-index order. Meaningful only right after a
        *fit*-time call (`fitted` is only ever populated then, by
        construction) -- `detect_feature_modalities`, the sole consumer,
        never runs at predict time either.
        """
        n_kept = len(self.old_to_new_index)
        total_width = sum(len(f.output_names) for f in self.fitted.values())
        return list(range(n_kept, n_kept + total_width))

    def merged_feature_names(self, raw_names: Sequence[str] | None) -> list[str] | None:
        """`raw_names` (the caller's own, pre-resolution feature names),
        spliced with each expanded column's generated names -- the name list
        `detect_feature_modalities` needs, matching this resolution's `X`
        column-for-column. `None` if `raw_names` is `None` (an unnamed array
        input never has a date column either, so nothing to splice).
        """
        if raw_names is None:
            return None
        merged: list[str | None] = [None] * (
            len(self.old_to_new_index) + len(self.expanded_new_indices)
        )
        for original_index, new_index in self.old_to_new_index.items():
            merged[new_index] = raw_names[original_index]
        expanded_names = [
            name
            for original_index in sorted(self.fitted)
            for name in self.fitted[original_index].output_names
        ]
        for new_index, name in zip(
            self.expanded_new_indices, expanded_names, strict=True
        ):
            merged[new_index] = name
        return merged  # type: ignore[return-value]


def resolve_date_columns(
    X: XType,
    *,
    transform_dates: bool = False,
    categorical_features_indices: Sequence[int] = (),
    fitted: dict[int, FittedDateColumn] | None = None,
) -> DateResolution:
    """Resolve every temporal column, by expanding it or rendering it to text.

    Called once per fit/predict, before any validation. At fit time (`fitted`
    left as `None`), a new encoder is fit for every date column that is
    eligible to expand. At predict time, `fitted` is the dict a prior fit
    call returned: only those exact positions are ever (re-)expanded --
    `transform_dates`/`categorical_features_indices` are only consulted at
    fit time, since which columns to expand is decided once, then reapplied
    positionally, exactly like `ordinal_encoder_`.

    A predict-time position that was expanded at fit time but is no longer a
    genuine datetime dtype right now degrades to a `NaN` calendar feature,
    the same as any other missing value -- there is no attempt to parse it
    from whatever is actually sitting there instead, since a column is a date
    because of its dtype, at fit and at predict alike.

    Args:
        X: The input data, before any dtype fixing.
        transform_dates: Whether an eligible date column is expanded into
            calendar features, rather than rendered to text. Ignored at
            predict time (see above).
        categorical_features_indices: Raw indices the caller declared
            categorical; a date column among them is never expanded,
            regardless of `transform_dates`. Ignored at predict time.
        fitted: `None` at fit time. The `by_index` of a prior fit call's
            `FittedDateColumns`, at predict time.

    Returns:
        The resolved data, this call's newly fitted columns (empty at
        predict time), and the old-to-new index map for every column that
        was not numerically expanded.
    """
    if not isinstance(X, pd.DataFrame):
        return DateResolution(X=X, fitted={}, old_to_new_index=_identity_mapping(X))

    dtypes = list(X.dtypes)
    date_indices = [
        i for i, dtype in enumerate(dtypes) if _is_datetime_like_dtype(dtype)
    ]
    duration_indices = [
        i for i, dtype in enumerate(dtypes) if pd.api.types.is_timedelta64_dtype(dtype)
    ]
    if not date_indices and not duration_indices and not fitted:
        # Nothing to expand, no fitted columns needing a degraded (NaN)
        # reapplication either -- a genuine no-op, unlike the case just below
        # where `fitted` still has positions to (re)produce even though none
        # of them are a date dtype right now.
        return DateResolution(
            X=X, fitted={}, old_to_new_index=_old_to_new_index(X.shape[1], [])
        )

    if fitted is None:
        categorical = set(categorical_features_indices)
        to_expand = [
            i for i in date_indices if transform_dates and i not in categorical
        ]
        _warn_on_dates_held_as_text(
            [
                X.columns[i]
                for i in date_indices
                if i not in to_expand and i not in categorical
            ]
        )
    else:
        to_expand = sorted(fitted)

    new_fitted: dict[int, FittedDateColumn] = {}
    single_column_replacements: dict[int, np.ndarray] = {}
    expanded_blocks: dict[int, pd.DataFrame] = {}
    for position in to_expand:
        if position in date_indices:
            column = _as_timestamp(X.iloc[:, position])
            if fitted is None:
                raw_name = X.columns[position]
                encoder = _make_datetime_encoder()
                raw_encoded = pd.DataFrame(
                    encoder.fit_transform(column.rename(raw_name))
                )
                fitted_column = FittedDateColumn(
                    encoder=encoder, output_names=list(raw_encoded.columns)
                )
                new_fitted[position] = fitted_column
            else:
                fitted_column = fitted[position]
                raw_encoded = pd.DataFrame(fitted_column.encoder.transform(column))
        else:
            # Fit expanded this position, but it is no longer a genuine
            # datetime dtype right now -- degrade to NaN rather than guess.
            fitted_column = fitted[position]  # type: ignore[index]
            raw_encoded = pd.DataFrame(
                {name: [float("nan")] * len(X) for name in fitted_column.output_names}
            )
        expanded_blocks[position] = raw_encoded.set_axis(
            fitted_column.output_names, axis=1
        ).reset_index(drop=True)

    for position in date_indices:
        if position in to_expand:
            continue
        column = _as_timestamp(X.iloc[:, position])
        single_column_replacements[position] = (
            column.astype(str).where(column.notna(), None).to_numpy()
        )
    for position in duration_indices:
        single_column_replacements[position] = (
            X.iloc[:, position].dt.total_seconds().to_numpy()
        )

    recast_frame = _replace_columns_positionally(X, single_column_replacements)
    resolved = _assemble(recast_frame, to_expand, expanded_blocks)
    old_to_new_index = _old_to_new_index(X.shape[1], to_expand)
    return DateResolution(
        X=resolved, fitted=new_fitted, old_to_new_index=old_to_new_index
    )


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


def _assemble(
    frame: pd.DataFrame,
    to_expand: list[int],
    expanded_blocks: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Drop the numerically-expanded columns and append their replacements.

    Positional, not `frame.drop(columns=...)`: dropping by label instead of
    position would silently misbehave for duplicate labels (the same case
    `build_input_feature_names` exists to handle elsewhere).
    """
    if not to_expand:
        return frame
    keep = [i for i in range(frame.shape[1]) if i not in set(to_expand)]
    remaining = frame.iloc[:, keep].reset_index(drop=True)
    ordered_blocks = [expanded_blocks[i] for i in sorted(expanded_blocks)]
    return pd.concat([remaining, *ordered_blocks], axis=1)


def _identity_mapping(X: XType) -> dict[int, int]:
    """Every column's old-to-new index, unchanged, for a non-`DataFrame` `X`.

    Empty if `X`'s column count can't even be determined (e.g. a 1D array):
    nothing here can be a date column either way, and downstream value
    validation (`check_array`/`check_X_y`) rejects such an `X` with its own,
    clearer message before anything would ever consult this mapping.
    """
    try:
        num_columns = _num_features(X)
    except TypeError:
        return {}
    return _old_to_new_index(num_columns, [])


def _old_to_new_index(num_columns: int, to_expand: list[int]) -> dict[int, int]:
    """Map every not-numerically-expanded position to its new position.

    A column keeps its relative order; it just shifts down by however many
    expanded columns were removed ahead of it -- the same reindexing
    `_assemble`'s positional `keep` list produces.
    """
    expanded = set(to_expand)
    mapping: dict[int, int] = {}
    removed_so_far = 0
    for original_index in range(num_columns):
        if original_index in expanded:
            removed_so_far += 1
            continue
        mapping[original_index] = original_index - removed_so_far
    return mapping
