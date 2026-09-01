#  Copyright (c) Prior Labs GmbH 2026.

"""Convert every temporal column to numbers, before validation ever sees it.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric one
in one array (no common dtype exists), so a temporal column has to stop looking
like one before `check_array`/`check_X_y` run. `DateTransformer` is where that
happens: a point in time becomes nanoseconds since the epoch.

Because this runs before detection, `detect_feature_modalities` only ever sees
the result, and never learns a column was a date at all.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tabpfn.preprocessing.modality_detection import format_names_for_warning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType


def _is_datetime_like_dtype(dtype: Any) -> bool:
    """Whether `dtype` holds points in time."""
    return pd.api.types.is_datetime64_any_dtype(dtype)


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
    out = X.copy(deep=False)
    original_columns = out.columns
    out.columns = pd.RangeIndex(out.shape[1])
    for position, values in replacements.items():
        out[position] = values.to_numpy()
    out.columns = original_columns
    return out


def _warn_on_dates(column_names: Sequence[str]) -> None:
    """Warn about date columns read as a plain number rather than a calendar."""
    if not column_names:
        return
    warnings.warn(
        f"These columns hold dates, which are not yet expanded into calendar "
        f"features, so they are read as plain numbers (nanoseconds since the "
        f"epoch): {format_names_for_warning(list(column_names))}.",
        UserWarning,
        # stacklevel=6 reaches the `estimator.fit(X, y)` call site; pinned by the
        # `warning.filename` asserts in the tests.
        stacklevel=6,
    )


class DateTransformer:
    """Converts every temporal column in an input to numbers.

    Not a `PreprocessingStep` (`pipeline_interface.py`): that tier runs per
    ensemble member on already-numeric arrays, well past where this has to run.
    Not `BaseEstimator`/`TransformerMixin` either: `fit_transform` takes fitting
    parameters that do not fit sklearn's `fit(X, y=None)` shape.

    Usage mirrors `ordinal_encoder_`: construct one, call `fit_transform` once at
    fit time and keep the instance around (as `self.date_transformer_`), then
    call `transform` on it at predict time.

    Args:
        categorical_indices: Indices the caller declared categorical. A date
            column among them is left alone, at fit and at predict alike: the
            user's declared intent for it wins over reading it as a date.
    """

    def __init__(self, *, categorical_indices: Sequence[int] | None = None) -> None:
        self._declared_categorical = set(categorical_indices or ())

    def fit_transform(self, X: XType) -> XType:
        """Convert every temporal column in `X`, and warn naming them.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            `X` with every converted column in place, or `X` itself when there
            was nothing to convert.
        """
        converted, names = self._convert(X)
        _warn_on_dates(names)
        return converted

    def transform(self, X: XType) -> XType:
        """Convert `X`'s temporal columns the way `fit_transform` did, silently.

        Which columns are temporal is read from the dtypes again rather than
        frozen at fit time: the conversion is the same for every point-in-time
        column, so a column that only turns up as a date here still has to stop
        looking like one before validation runs.

        Args:
            X: The data, before any dtype fixing.
        """
        converted, _ = self._convert(X)
        return converted

    def _convert(self, X: XType) -> tuple[XType, list[str]]:
        """The conversion itself, and the names of the columns it converted."""
        if not isinstance(X, pd.DataFrame):
            return X, []
        positions = [
            i
            for i, dtype in enumerate(X.dtypes)
            if _is_datetime_like_dtype(dtype) and i not in self._declared_categorical
        ]
        if not positions:
            return X, []
        replacements = {i: _to_nanoseconds(X.iloc[:, i]) for i in positions}
        converted = _replace_columns_positionally(X, replacements)
        return converted, [str(X.columns[i]) for i in positions]


def apply_date_conversion(X: XType, source: object) -> XType:
    """Convert `X`'s temporal columns via `source`'s fitted `date_transformer_`.

    `source` (a fitted estimator or ensemble worker) may never have set
    `date_transformer_` at all, e.g. `fit_from_preprocessed` skips the step that
    would, exactly like the pre-existing `ordinal_encoder_` guard. One built from
    `source`'s own declared categoricals converts the same way, so that case
    needs no special handling.
    """
    transformer = getattr(source, "date_transformer_", None) or DateTransformer(
        categorical_indices=getattr(source, "categorical_features_indices", None)
    )
    return transformer.transform(X)
