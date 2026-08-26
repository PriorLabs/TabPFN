#  Copyright (c) Prior Labs GmbH 2026.

"""Expand a detected `DATE` column into calendar features via `skrub.DatetimeEncoder`.

Only reached when `USE_DATES` is on.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd

from tabpfn.preprocessing.datamodel import FeatureModality, make_names_unique

if TYPE_CHECKING:
    import numpy as np
    from skrub import DatetimeEncoder

    from tabpfn.preprocessing.datamodel import FeatureSchema


@dataclasses.dataclass
class FittedDatetimeEncoder:
    """A fitted `DatetimeEncoder` for one column, and what's needed to reapply it."""

    column_index: int
    encoder: DatetimeEncoder
    output_names: list[str]


def make_datetime_encoder() -> DatetimeEncoder:
    """Build the encoder that turns a datetime column into calendar features.

    Returns:
        An encoder producing the year, the day of year, the seconds since epoch,
        and the cyclical month, day and weekday pairs, plus the time of day when
        the column carries one.
    """
    # Local import: skrub depends on matplotlib.
    from skrub import DatetimeEncoder  # noqa: PLC0415

    return DatetimeEncoder(
        resolution="second",
        add_weekday=True,
        add_day_of_year=True,
        periodic_encoding="circular",
    )


def _parse_dates(column: pd.Series) -> pd.Series:
    """Parse to datetime, tolerant of per-row format differences.

    format="mixed": a format inferred from one value would otherwise silently
    coerce a later, differently-shaped but valid date to NaT.
    """
    return pd.to_datetime(column, errors="coerce", format="mixed")


def _fit_one_column(
    column: pd.Series,
    index: int,
    name: str,
    existing_names: list[str],
) -> tuple[pd.DataFrame, FittedDatetimeEncoder]:
    """Fit a new encoder on one column."""
    encoder = make_datetime_encoder()
    raw_encoded = pd.DataFrame(encoder.fit_transform(_parse_dates(column)))
    output_names = make_names_unique(
        [f"{name}_{i}" for i in range(raw_encoded.shape[1])], existing=existing_names
    )
    encoded = raw_encoded.set_axis(output_names, axis=1)
    fitted_encoder = FittedDatetimeEncoder(
        column_index=index, encoder=encoder, output_names=output_names
    )
    return encoded, fitted_encoder


def _apply_one_column(column: pd.Series, fitted: FittedDatetimeEncoder) -> pd.DataFrame:
    """Reapply an already-fitted encoder to one column."""
    encoded = fitted.encoder.transform(_parse_dates(column))
    return pd.DataFrame(encoded).set_axis(fitted.output_names, axis=1)


def expand_date_features(
    X: np.ndarray,
    feature_schema: FeatureSchema | None,
    *,
    fitted: dict[str, FittedDatetimeEncoder] | None = None,
) -> tuple[np.ndarray, FeatureSchema | None, dict[str, FittedDatetimeEncoder]]:
    """Expand every `DATE`-modality column into numbers, via `DatetimeEncoder`.

    Args:
        X: The data, before any dtype fixing.
        feature_schema: The schema to fit against; `None` at predict time.
        fitted: Previously fitted encoders, keyed by column name, to reuse at
            predict time instead of fitting new ones.

    Returns:
        The (possibly wider) data, the updated schema, and the fitted encoders.
    """
    if fitted is not None:
        by_index = {fe.column_index: fe for fe in fitted.values()}
        to_expand = sorted(by_index)
    else:
        assert feature_schema is not None, "feature_schema is required to fit"
        to_expand = sorted(feature_schema.indices_for(FeatureModality.DATE))
    if not to_expand:
        return X, feature_schema, fitted or {}

    frame = pd.DataFrame(X, copy=False).reset_index(drop=True)
    new_fitted: dict[str, FittedDatetimeEncoder] = {}
    existing_names = list(feature_schema.feature_names) if feature_schema else []
    encoded_blocks: list[pd.DataFrame] = []
    for index in to_expand:
        # skrub names outputs from the series' `.name`; a bare `pd.DataFrame`
        # column label is an int, which can't take a suffix.
        column = frame.iloc[:, index].rename(str(index))
        if fitted is not None:
            encoded = _apply_one_column(column, by_index[index])
        else:
            assert feature_schema is not None
            name = feature_schema.features[index].name
            encoded, fitted_encoder = _fit_one_column(
                column, index, name, existing_names
            )
            existing_names += fitted_encoder.output_names
            new_fitted[name] = fitted_encoder
        encoded_blocks.append(encoded.reset_index(drop=True))

    remaining = frame.drop(columns=frame.columns[to_expand])
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
    fitted: dict[str, FittedDatetimeEncoder] | None = None,
) -> tuple[np.ndarray, FeatureSchema | None, dict[str, FittedDatetimeEncoder]]:
    """Encode every modality with an encoder available (today: dates)."""
    return expand_date_features(X, feature_schema, fitted=fitted)
