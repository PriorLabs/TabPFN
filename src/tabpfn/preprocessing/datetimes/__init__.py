#  Copyright (c) Prior Labs GmbH 2026.

"""Converting temporal columns to numbers, before validation ever sees them.

A point in time (`datetime64`, tz-aware, or `period`) either becomes one plain
number (`NumericalDateTransformer`) or is expanded into calendar features
(`SkrubDateTransformer`), decided by `TRANSFORM_DATES` through
`make_date_transformer`. A duration (`timedelta64`) always becomes its length in
seconds. See `base.py` for why any of it has to happen before validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabpfn.preprocessing.datetimes.base import DateConversion, DateTransformer
from tabpfn.preprocessing.datetimes.numerical import NumericalDateTransformer
from tabpfn.preprocessing.datetimes.skrub_expansion import SkrubDateTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType

__all__ = [
    "DateConversion",
    "DateTransformer",
    "NumericalDateTransformer",
    "SkrubDateTransformer",
    "convert_dates",
    "make_date_transformer",
]


def make_date_transformer(
    *,
    categorical_indices: Sequence[int] | None = None,
    transform_dates: bool = False,
) -> DateTransformer:
    """Build the transformer `TRANSFORM_DATES` asks for.

    Args:
        categorical_indices: Indices the caller declared categorical.
        transform_dates: Whether a point in time is expanded into calendar
            features rather than read as one plain number.

    Returns:
        The transformer to fit, and to keep for predict.
    """
    transformer_cls = (
        SkrubDateTransformer if transform_dates else NumericalDateTransformer
    )
    return transformer_cls(categorical_indices=categorical_indices)


def convert_dates(X: XType, source: object) -> XType:
    """Convert `X`'s temporal columns via `source`'s fitted `date_transformer_`.

    `source` (a fitted estimator or ensemble worker) may never have set
    `date_transformer_` at all, e.g. `fit_from_preprocessed` skips the step that
    would, exactly like the pre-existing `ordinal_encoder_` guard. The fallback
    converts the same way, minus an expansion there was no fit to decide on.
    """
    transformer = getattr(source, "date_transformer_", None) or (
        NumericalDateTransformer(
            categorical_indices=getattr(source, "categorical_features_indices", None)
        )
    )
    return transformer.transform(X)
