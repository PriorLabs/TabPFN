#  Copyright (c) Prior Labs GmbH 2026.

"""What both date transformers share: when they run, and what they hand back.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric one
in one array (no common dtype exists), so a temporal column has to stop looking
like one before `check_array`/`check_X_y` run. A `DateTransformer` is where that
happens, and because it runs before detection, `detect_feature_modalities` only
ever sees the result and never learns a column was a date at all.

Which transformer runs is the whole of the difference between reading a date as
one number and expanding it into a calendar (`numerical.py`, `skrub_expansion.py`).
A duration (`timedelta64`) is not part of that choice: it always becomes its
length in seconds, a quantity with no calendar in it either way, so it is
converted here.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING

import pandas as pd

from tabpfn.preprocessing.datetimes.dtypes import (
    as_timestamp,
    is_instant_dtype,
    to_nanoseconds,
    to_seconds,
)
from tabpfn.preprocessing.datetimes.frames import replace_columns_positionally

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType


@dataclasses.dataclass(frozen=True)
class DateConversion:
    """What `DateTransformer.fit_transform` did, as the fit path needs it.

    Attributes:
        X: The converted data.
        feature_names: `X`'s column labels as strings, in order, or `None` when
            the input was not a `DataFrame` and so has no labels.
        categorical_indices: The caller's declared categorical indices, moved to
            where those columns ended up.
    """

    X: XType
    feature_names: list[str] | None
    categorical_indices: list[int] | None


class DateTransformer(abc.ABC):
    """Converts every temporal column in an input to numbers.

    Not a `PreprocessingStep` (`pipeline_interface.py`): that tier runs per
    ensemble member on already-numeric arrays, well past where this has to run.
    Not `BaseEstimator`/`TransformerMixin` either: `fit_transform` returns more
    than the transformed data, which does not fit sklearn's shape.

    Usage mirrors `ordinal_encoder_`: build one with `make_date_transformer`,
    call `fit_transform` once at fit time and keep the instance around (as
    `self.date_transformer_`), then call `transform` on it at predict time.

    Args:
        categorical_indices: Indices the caller declared categorical. A point in
            time among them is left alone entirely, at fit and at predict alike:
            the user's declared intent for it wins over reading it as a date.
    """

    def __init__(self, *, categorical_indices: Sequence[int] | None = None) -> None:
        self._categorical_indices = (
            None if categorical_indices is None else list(categorical_indices)
        )
        self._declared_categorical = set(self._categorical_indices or ())

    def fit_transform(self, X: XType) -> DateConversion:
        """Convert every temporal column in `X`, warning about what it cost.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            The conversion, including what the caller has to pass on to
            `detect_feature_modalities`.
        """
        if not isinstance(X, pd.DataFrame):
            return DateConversion(
                X=X,
                feature_names=None,
                categorical_indices=self._categorical_indices,
            )
        return self._fit_transform_frame(X)

    def transform(self, X: XType) -> XType:
        """Reapply the conversion `fit_transform` decided on, silently.

        Args:
            X: The data, before any dtype fixing.
        """
        if not isinstance(X, pd.DataFrame):
            return X
        return self._transform_frame(X)

    @abc.abstractmethod
    def _fit_transform_frame(self, X: pd.DataFrame) -> DateConversion:
        """Convert a real `DataFrame`, recording whatever predict has to reuse."""

    @abc.abstractmethod
    def _transform_frame(self, X: pd.DataFrame) -> XType:
        """Convert a real `DataFrame` the way `_fit_transform_frame` decided."""

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
            if is_instant_dtype(dtype) and i not in self._declared_categorical
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
        replacements = {i: to_nanoseconds(as_timestamp(X.iloc[:, i])) for i in instants}
        replacements.update({i: to_seconds(X.iloc[:, i]) for i in durations})
        return replace_columns_positionally(X, replacements)

    def _conversion(self, converted: pd.DataFrame) -> DateConversion:
        """The result for a frame whose columns are still the caller's own."""
        return DateConversion(
            X=converted,
            feature_names=[str(column) for column in converted.columns],
            categorical_indices=self._categorical_indices,
        )
