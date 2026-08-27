#  Copyright (c) Prior Labs GmbH 2026.

"""Expand a detected `DATE` column into calendar features via `skrub.DatetimeEncoder`.

Only reached when `USE_DATES` is on.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd
from skrub import DatetimeEncoder

from tabpfn.preprocessing.datamodel import (
    FeatureModality,
    FeatureSchema,
    make_names_unique,
)
from tabpfn.preprocessing.modality_detection import _underspecified_date_values

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


@dataclasses.dataclass
class FittedDatetimeEncoder:
    """A fitted `DatetimeEncoder` for one column, and its output column names."""

    encoder: DatetimeEncoder
    output_names: list[str]


def make_datetime_encoder() -> DatetimeEncoder:
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


def _parse_dates(column: pd.Series) -> pd.Series:
    """Parse to datetime, tolerant of per-row format differences.

    format="mixed": a format inferred from one value would otherwise silently
    coerce a later, differently-shaped but valid date to NaT. A value needing
    a defaulted year/month/day (e.g. a bare time) is masked to NaT the same
    way, instead of silently taking on today's date.

    Inference-only in practice: at fit time this never actually masks
    anything, since `_is_date_like_pandas_series` already rejected the whole
    column unless every value here was fully specified. It only fires for a
    value that drifts into being underspecified after fitting.
    """
    parsed = pd.to_datetime(column, errors="coerce", format="mixed")
    underspecified = _underspecified_date_values(column.dropna())
    if underspecified:
        parsed[column.isin(underspecified)] = pd.NaT
    return parsed


def _fit_one_column(
    column: pd.Series,
    existing_names: list[str],
) -> tuple[pd.DataFrame, FittedDatetimeEncoder]:
    """Fit a new encoder on one column.

    `column` must already carry the real feature name: skrub names each
    output after it (e.g. "signed_on_year", "signed_on_month_circular_0"),
    and those are kept as-is here rather than replaced with a generic
    "{name}_{i}", deduped only against name collisions with existing columns.
    """
    encoder = make_datetime_encoder()
    raw_encoded = pd.DataFrame(encoder.fit_transform(_parse_dates(column)))
    output_names = make_names_unique(list(raw_encoded.columns), existing=existing_names)
    encoded = raw_encoded.set_axis(output_names, axis=1)
    fitted_encoder = FittedDatetimeEncoder(encoder=encoder, output_names=output_names)
    return encoded, fitted_encoder


def _apply_one_column(column: pd.Series, fitted: FittedDatetimeEncoder) -> pd.DataFrame:
    """Reapply an already-fitted encoder to one column."""
    encoded = fitted.encoder.transform(_parse_dates(column))
    return pd.DataFrame(encoded).set_axis(fitted.output_names, axis=1)


def expand_date_features(
    X: np.ndarray,
    feature_schema: FeatureSchema | None,
    *,
    provided_categorical_indices: Sequence[int] | None = None,
    fitted: dict[int, FittedDatetimeEncoder] | None = None,
) -> tuple[np.ndarray, FeatureSchema | None, dict[int, FittedDatetimeEncoder]]:
    """Expand every `DATE`-modality column into numbers, via `DatetimeEncoder`.

    Args:
        X: The data, before any dtype fixing.
        feature_schema: The schema to fit against; `None` at predict time.
        provided_categorical_indices: Indices declared categorical by the
            caller. Detection tags a date-like column `DATE` regardless of
            this declaration, so it must be excluded here instead, or the
            declaration would have no effect once `USE_DATES` is on.
        fitted: Previously fitted encoders, keyed by column index, to reuse at
            predict time instead of fitting new ones.

    Returns:
        The (possibly wider) data, the updated schema, and the fitted encoders.
    """
    if fitted is not None:
        to_expand = sorted(fitted)
    else:
        assert feature_schema is not None, "feature_schema is required to fit"
        declared = set(provided_categorical_indices or ())
        date_indices = feature_schema.indices_for(FeatureModality.DATE)
        declared_dates = [i for i in date_indices if i in declared]
        if declared_dates:
            # A declared date can't just be skipped: unlike a genuine TEXT
            # column, a column left tagged DATE has no safe fallback -- clean_data
            # doesn't recognize it and would silently ordinal-code the raw
            # strings. Demote it exactly like `_demote_dates` already does when
            # USE_DATES is off, since the declaration means the same thing here.
            features = list(feature_schema.features)
            for index in declared_dates:
                features[index] = dataclasses.replace(
                    features[index], modality=FeatureModality.CATEGORICAL
                )
            feature_schema = FeatureSchema(features=features)
        to_expand = sorted(i for i in date_indices if i not in declared)
    if not to_expand:
        return X, feature_schema, fitted or {}

    frame = pd.DataFrame(X, copy=False).reset_index(drop=True)
    new_fitted: dict[int, FittedDatetimeEncoder] = {}
    existing_names = list(feature_schema.feature_names) if feature_schema else []
    encoded_blocks: list[pd.DataFrame] = []
    for index in to_expand:
        if fitted is not None:
            # The exact name doesn't matter here: `_apply_one_column` overrides
            # the output labels with the already-fitted ones regardless. Still
            # needs to be a string, not the bare `pd.DataFrame`'s int label --
            # some of skrub's own naming code concatenates onto it directly.
            column = frame.iloc[:, index].rename(str(index))
            encoded = _apply_one_column(column, fitted[index])
        else:
            assert feature_schema is not None
            name = feature_schema.features[index].name
            column = frame.iloc[:, index].rename(name)
            encoded, fitted_encoder = _fit_one_column(column, existing_names)
            existing_names += fitted_encoder.output_names
            new_fitted[index] = fitted_encoder
        encoded_blocks.append(encoded.reset_index(drop=True))

    # Positional, not `frame.drop(columns=...)`: `X` is always an ndarray here,
    # so `frame`'s labels are its default positions today, but dropping by
    # label instead of position would silently misbehave the day that stops
    # being true (e.g. duplicate labels, which `build_input_feature_names`
    # exists to handle elsewhere).
    keep = [i for i in range(frame.shape[1]) if i not in set(to_expand)]
    remaining = frame.iloc[:, keep]
    out = pd.concat([remaining, *encoded_blocks], axis=1)

    schema = feature_schema
    if fitted is None:
        assert schema is not None
        schema = schema.remove_columns(to_expand)
        for fitted_encoder in new_fitted.values():
            schema = schema.append_columns(
                FeatureModality.NUMERICAL,
                len(fitted_encoder.output_names),
                names=fitted_encoder.output_names,
            )

    return out.to_numpy(), schema, (fitted if fitted is not None else new_fitted)


def encode_multimodal_data(
    X: np.ndarray,
    feature_schema: FeatureSchema | None,
    *,
    provided_categorical_indices: Sequence[int] | None = None,
    fitted: dict[int, FittedDatetimeEncoder] | None = None,
) -> tuple[np.ndarray, FeatureSchema | None, dict[int, FittedDatetimeEncoder]]:
    """Encode every modality with an encoder available (today: dates)."""
    return expand_date_features(
        X,
        feature_schema,
        provided_categorical_indices=provided_categorical_indices,
        fitted=fitted,
    )
