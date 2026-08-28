#  Copyright (c) Prior Labs GmbH 2026.

"""Expand a detected `DATE` column into calendar features via `skrub.DatetimeEncoder`.

Only reached when `TRANSFORM_DATES` is on.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from skrub import DatetimeEncoder

from tabpfn.preprocessing.clean import PANDAS_SUPPORTS_ISO8601_FORMAT
from tabpfn.preprocessing.datamodel import (
    FeatureModality,
    FeatureSchema,
    make_names_unique,
)

if TYPE_CHECKING:
    import numpy as np


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


def _parse_dates(column: pd.Series) -> pd.Series:
    """Parse to datetime, never raising.

    A `DATE` column only ever gets here after `normalize_temporal_columns`
    rendered it from a genuine datetime dtype into ISO 8601 text -- uniformly,
    since one dtype (and so one timezone) covers the whole column. `ISO8601`
    format handles that rendering exactly, including a column mixing
    precisions (a date-only "2020-01-01" beside a full "2020-06-15
    13:45:30"), which the plain, single-inferred-format parse this replaced
    cannot: it infers one shape from the first value and silently drops any
    other shape to `NaT`.

    Still never raises, for a predict-time column that no longer matches what
    fit saw (e.g. it arrived as a genuine datetime dtype at fit but drifted to
    plain, malformed strings at predict): a crash is a worse outcome than a
    degraded (NaN) calendar feature for a column fit already accepted.
    `pd.to_datetime` can also silently succeed with an inconsistent per-value
    result (e.g. mixed UTC offsets, without a uniform dtype behind them)
    instead of raising -- caught here by checking the result is actually
    `datetime64`, not just checking for an exception.

    `ISO8601` needs pandas >= 2.0 (see `PANDAS_SUPPORTS_ISO8601_FORMAT`);
    below it, this falls back to a plain, single-inferred-format parse, which
    cannot lose the main case that matters there -- a column rendered from a
    genuine datetime dtype with everything at the exact same precision.
    """
    try:
        with warnings.catch_warnings():
            # Emitted when pandas cannot infer one format and falls back to
            # parsing value by value -- expected for a drifted column, so noise.
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(
                column,
                errors="coerce",
                **({"format": "ISO8601"} if PANDAS_SUPPORTS_ISO8601_FORMAT else {}),
            )
    except (TypeError, ValueError):
        parsed = None
    if parsed is None or not pd.api.types.is_datetime64_any_dtype(parsed):
        return pd.Series(pd.NaT, index=column.index, dtype="datetime64[ns]")
    return parsed


class DateFeatureExpander:
    """Expands every `DATE`-modality column into numbers via `skrub.DatetimeEncoder`.

    Not a `PreprocessingStep` (`pipeline_interface.py`): that tier runs per
    ensemble member on already-numeric arrays, after `clean_data`/`fix_dtypes`
    -- by which point the raw date strings this needs are already gone. Not
    `BaseEstimator`/`TransformerMixin` either: fitting needs a `FeatureSchema`
    alongside `X`, which doesn't fit sklearn's `fit(X, y=None)` signature.

    Usage mirrors `ordinal_encoder_`: construct one, call `fit_transform` once
    at fit time and keep the instance around (e.g. as `self.date_expander_`),
    then call `transform` on it at predict time.
    """

    @dataclasses.dataclass
    class _FittedColumn:
        """A fitted `DatetimeEncoder` for one column, and its output names."""

        encoder: DatetimeEncoder
        output_names: list[str]

    def __init__(self) -> None:
        self._fitted: dict[int, DateFeatureExpander._FittedColumn] = {}

    @property
    def expanded_indices(self) -> list[int]:
        """Raw column indices that were expanded, ascending.

        Empty both before `fit_transform` is called and after it finds no
        `DATE` columns to expand.
        """
        return sorted(self._fitted)

    def fit_transform(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> tuple[np.ndarray, FeatureSchema]:
        """Fit a new encoder per `DATE` column and expand it into numbers.

        Every `DATE` column is expanded, with no exceptions to weigh: whether a
        column is one is settled by the time the schema arrives, including the
        caller's `categorical_features_indices`, which stops a column being
        called a date in the first place rather than being re-litigated here.

        Args:
            X: The data, before any dtype fixing.
            feature_schema: The schema to fit against.

        Returns:
            The (possibly wider) data and the updated schema.
        """
        to_expand = feature_schema.indices_for(FeatureModality.DATE)
        self._fitted = {}
        if not to_expand:
            return X, feature_schema

        frame = pd.DataFrame(X, copy=False).reset_index(drop=True)
        existing_names = list(feature_schema.feature_names)
        encoded_blocks: list[pd.DataFrame] = []
        for index in to_expand:
            name = feature_schema.features[index].name
            column = frame.iloc[:, index].rename(name)
            encoded, fitted_column = self._fit_one_column(column, existing_names)
            existing_names += fitted_column.output_names
            self._fitted[index] = fitted_column
            encoded_blocks.append(encoded.reset_index(drop=True))

        out = self._assemble(frame, to_expand, encoded_blocks)

        schema = feature_schema.remove_columns(to_expand)
        for index in to_expand:
            fitted_column = self._fitted[index]
            schema = schema.append_columns(
                FeatureModality.NUMERICAL,
                len(fitted_column.output_names),
                names=fitted_column.output_names,
            )
        return out.to_numpy(), schema

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reapply the encoders fit by `fit_transform`, positionally.

        A no-op (returns `X` unchanged) if nothing was fit -- either
        `fit_transform` found no `DATE` columns, or it was never called.
        """
        to_expand = self.expanded_indices
        if not to_expand:
            return X

        frame = pd.DataFrame(X, copy=False).reset_index(drop=True)
        encoded_blocks = []
        for index in to_expand:
            # The exact name doesn't matter: `_apply_one_column` overrides the
            # output labels with the already-fitted ones regardless. Still
            # needs to be a string, not the bare `pd.DataFrame`'s int label --
            # some of skrub's own naming code concatenates onto it directly.
            column = frame.iloc[:, index].rename(str(index))
            encoded = self._apply_one_column(column, self._fitted[index])
            encoded_blocks.append(encoded.reset_index(drop=True))

        return self._assemble(frame, to_expand, encoded_blocks).to_numpy()

    @staticmethod
    def _assemble(
        frame: pd.DataFrame,
        to_expand: list[int],
        encoded_blocks: list[pd.DataFrame],
    ) -> pd.DataFrame:
        # Positional, not `frame.drop(columns=...)`: `X` is always an ndarray
        # here, so `frame`'s labels are its default positions today, but
        # dropping by label instead of position would silently misbehave the
        # day that stops being true (e.g. duplicate labels, which
        # `build_input_feature_names` exists to handle elsewhere).
        keep = [i for i in range(frame.shape[1]) if i not in set(to_expand)]
        remaining = frame.iloc[:, keep]
        return pd.concat([remaining, *encoded_blocks], axis=1)

    @staticmethod
    def _fit_one_column(
        column: pd.Series,
        existing_names: list[str],
    ) -> tuple[pd.DataFrame, DateFeatureExpander._FittedColumn]:
        """Fit a new encoder on one column.

        `column` must already carry the real feature name: skrub names each
        output after it (e.g. "signed_on_year", "signed_on_month_circular_0"),
        and those are kept as-is here rather than replaced with a generic
        "{name}_{i}", deduped only against name collisions with existing
        columns.
        """
        encoder = _make_datetime_encoder()
        raw_encoded = pd.DataFrame(encoder.fit_transform(_parse_dates(column)))
        output_names = make_names_unique(
            list(raw_encoded.columns), existing=existing_names
        )
        encoded = raw_encoded.set_axis(output_names, axis=1)
        fitted_column = DateFeatureExpander._FittedColumn(
            encoder=encoder, output_names=output_names
        )
        return encoded, fitted_column

    @staticmethod
    def _apply_one_column(
        column: pd.Series,
        fitted: DateFeatureExpander._FittedColumn,
    ) -> pd.DataFrame:
        """Reapply an already-fitted encoder to one column."""
        encoded = fitted.encoder.transform(_parse_dates(column))
        return pd.DataFrame(encoded).set_axis(fitted.output_names, axis=1)


def apply_date_expansion(X: np.ndarray, source: object) -> np.ndarray:
    """Reapply `source`'s fitted `date_expander_` at predict time, if any.

    `source` (a fitted estimator or ensemble worker) may never have set
    `date_expander_` at all -- e.g. `fit_from_preprocessed` skips the step
    that would, exactly like the pre-existing `ordinal_encoder_` guard.
    """
    date_expander = getattr(source, "date_expander_", None)
    return X if date_expander is None else date_expander.transform(X)
