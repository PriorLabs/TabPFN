#  Copyright (c) Prior Labs GmbH 2026.

"""Converting temporal columns, before validation ever sees them.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric one
in one array (no common dtype exists), so a temporal column has to stop looking
like one before `check_array`/`check_X_y` run, which is why this tier exists at
all. A point in time (`datetime64`, tz-aware, or `period`) is expanded into
calendar features when `TRANSFORM_DATES` is on, and refused with an error naming
it otherwise. A duration (`timedelta64`) always becomes its length in seconds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabpfn.preprocessing.datetimes.transformer import (
    DateConversion,
    DateTransformer,
)

if TYPE_CHECKING:
    from tabpfn.constants import XType

__all__ = ["DateConversion", "DateTransformer", "convert_dates"]


def convert_dates(X: XType, source: object) -> XType:
    """Convert `X`'s temporal columns via `source`'s fitted `date_transformer_`.

    `source` (a fitted estimator or ensemble worker) may never have set
    `date_transformer_` at all, e.g. `fit_from_preprocessed` skips the step that
    would, exactly like the pre-existing `ordinal_encoder_` guard. The fallback
    has nothing fitted to expand a date with, so it refuses one, and converts a
    duration as any transformer does.
    """
    transformer = getattr(source, "date_transformer_", None) or DateTransformer(
        categorical_indices=getattr(source, "categorical_features_indices", None)
    )
    return transformer.transform(X)
