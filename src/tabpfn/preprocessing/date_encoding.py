#  Copyright (c) Prior Labs GmbH 2026.

"""Expand a detected `DATE` column into calendar features via `skrub.DatetimeEncoder`.

Only reached when `InferenceConfig.USE_DATES` is on: `detect_feature_modalities`
already demotes a date-like column to `CATEGORICAL`/`TEXT` when it is off, since
nothing would consume `FeatureModality.DATE` otherwise.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd

from tabpfn.preprocessing.datamodel import FeatureModality

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
    # Imported here rather than at module scope: skrub depends on matplotlib, and
    # importing it eagerly would pull a plotting stack into every `import tabpfn`.
    from skrub import DatetimeEncoder  # noqa: PLC0415

    return DatetimeEncoder(
        resolution="second",
        add_weekday=True,
        add_day_of_year=True,
        periodic_encoding="circular",
    )


def expand_date_features(
    X: np.ndarray,
    feature_schema: FeatureSchema | None,
    *,
    fitted: dict[str, FittedDatetimeEncoder] | None = None,
) -> tuple[np.ndarray, FeatureSchema | None, dict[str, FittedDatetimeEncoder]]:
    """Expand every `DATE`-modality column into numbers, via `DatetimeEncoder`.

    A genuine single-column transformer (`.fit(series)`), not
    `skrub.TableVectorizer`: the type decision was already made by
    `detect_feature_modalities`, so no automatic detection is wanted here, only
    the feature engineering. Unconditionally safe to call: a no-op whenever
    there is no `DATE`-tagged column, which is always true when `USE_DATES` is
    off, since detection already demoted everything by then.

    Args:
        X: The data, before any dtype fixing.
        feature_schema: The schema from `detect_feature_modalities`. Only
            consulted when `fitted` is `None` (the fit-time path); at predict
            time the encoders already fitted carry everything they need, so
            this may be left `None`.
        fitted: Previously fitted encoders keyed by original column name, to
            reuse at predict time. Reapplied by each entry's `column_index`,
            not by dict order: sklearn guarantees predict-time column position
            matches fit time, so that index is a reliable, available key,
            unlike the name, which isn't recoverable from a raw array.
            `None` fits new ones instead (the fit-time path).

    Returns:
        The (possibly wider) data, the updated schema (unchanged, and possibly
        `None`, at predict time), and the fitted encoders to store and pass
        back in as `fitted` at predict time.
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
    encoded_blocks: list[pd.DataFrame] = []
    for index in to_expand:
        # Renamed to a plain string regardless of what the frame's own column
        # labels are: skrub's encoder builds its output feature names from the
        # input series' `.name`, and a bare `pd.DataFrame(ndarray)` has integer
        # column labels, which `DatetimeEncoder` cannot concatenate a suffix onto.
        column = frame.iloc[:, index].rename(str(index))
        # `format="mixed"` matters here, not just for speed: without it,
        # `to_datetime` infers one format from an early value and coerces every
        # later value that doesn't match it to NaT, even genuinely valid dates
        # (verified: a column mixing "2020-01-01" and "2020-06-15 13:45:30"
        # silently drops the second to NaT under the default format inference).
        column = pd.to_datetime(column, errors="coerce", format="mixed")
        if fitted is not None:
            fitted_encoder = by_index[index]
            encoded = pd.DataFrame(fitted_encoder.encoder.transform(column)).set_axis(
                fitted_encoder.output_names, axis=1
            )
        else:
            assert feature_schema is not None
            name = feature_schema.features[index].name
            encoder = make_datetime_encoder()
            raw_encoded = pd.DataFrame(encoder.fit_transform(column))
            output_names = [f"{name}_{i}" for i in range(raw_encoded.shape[1])]
            encoded = raw_encoded.set_axis(output_names, axis=1)
            new_fitted[name] = FittedDatetimeEncoder(
                column_index=index, encoder=encoder, output_names=output_names
            )
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
    """Encode every non-numeric modality that has a real encoder to turn to numbers.

    The seam `clean_data` (and `classifier.py`/`regressor.py`'s predict paths)
    call, kept separate from encoding mechanics: today this is only
    `expand_date_features`, but a future `USE_TEXT` is expected to add a
    `StringEncoder` pass here too, mirroring `_warn_on_multimodal`.
    """
    return expand_date_features(X, feature_schema, fitted=fitted)
