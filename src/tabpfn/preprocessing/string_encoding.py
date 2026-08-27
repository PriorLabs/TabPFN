#  Copyright (c) Prior Labs GmbH 2026.

"""Expand a detected `TEXT` column into numbers via `skrub.StringEncoder`.

Only reached when `USE_TEXT` is on.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd

from tabpfn.preprocessing.datamodel import FeatureModality, make_names_unique

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from skrub import StringEncoder

    from tabpfn.preprocessing.datamodel import FeatureSchema

#: Number of features a text column is encoded into. skrub's own default.
DEFAULT_TEXT_N_COMPONENTS = 30


@dataclasses.dataclass
class FittedStringEncoder:
    """A fitted `StringEncoder` for one column, and its output column names."""

    encoder: StringEncoder
    output_names: list[str]


def make_string_encoder(n_components: int = DEFAULT_TEXT_N_COMPONENTS) -> StringEncoder:
    """Build the encoder that turns a text column into numeric features.

    Returns:
        An encoder applying tf-idf over character n-grams, followed by a
        truncated SVD down to `n_components` features.
    """
    # Local import: skrub depends on matplotlib.
    from skrub import StringEncoder  # noqa: PLC0415

    # Seeded independently of the estimator: giving a column its type is a
    # property of the data, so it should not move when the ensemble seed does.
    return StringEncoder(n_components=n_components, random_state=0)


def _stringify(column: pd.Series) -> pd.Series:
    """Cast to pandas' nullable string dtype, which `StringEncoder` requires."""
    return column.astype("string")


def _fit_one_column(
    column: pd.Series,
    name: str,
    existing_names: list[str],
) -> tuple[pd.DataFrame, FittedStringEncoder]:
    """Fit a new encoder on one column."""
    encoder = make_string_encoder()
    raw_encoded = pd.DataFrame(encoder.fit_transform(_stringify(column)))
    output_names = make_names_unique(
        [f"{name}_{i}" for i in range(raw_encoded.shape[1])], existing=existing_names
    )
    encoded = raw_encoded.set_axis(output_names, axis=1)
    fitted_encoder = FittedStringEncoder(encoder=encoder, output_names=output_names)
    return encoded, fitted_encoder


def _apply_one_column(column: pd.Series, fitted: FittedStringEncoder) -> pd.DataFrame:
    """Reapply an already-fitted encoder to one column."""
    encoded = fitted.encoder.transform(_stringify(column))
    return pd.DataFrame(encoded).set_axis(fitted.output_names, axis=1)


def expand_text_features(
    X: np.ndarray,
    feature_schema: FeatureSchema | None,
    *,
    use_text: bool = False,
    provided_categorical_indices: Sequence[int] | None = None,
    fitted: dict[int, FittedStringEncoder] | None = None,
) -> tuple[np.ndarray, FeatureSchema | None, dict[int, FittedStringEncoder]]:
    """Expand every `TEXT`-modality column into numbers, via `StringEncoder`.

    Args:
        X: The data, before any dtype fixing.
        feature_schema: The schema to fit against; `None` at predict time.
        use_text: Whether to fit new encoders. A `TEXT` column stays tagged
            that way in the schema regardless of this flag, unlike `DATE`, so
            this is the only place that decides whether fitting happens.
            Ignored when `fitted` is given.
        provided_categorical_indices: Indices declared categorical by the
            caller. A string column can still classify as `TEXT` despite the
            declaration (detection ignores it, like it already does for a
            declared-categorical date), so it must be excluded here instead,
            or the declaration would have no effect once `USE_TEXT` is on.
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
        to_expand = (
            sorted(
                i
                for i in feature_schema.indices_for(FeatureModality.TEXT)
                if i not in declared
            )
            if use_text
            else []
        )
    if not to_expand:
        return X, feature_schema, fitted or {}

    frame = pd.DataFrame(X, copy=False).reset_index(drop=True)
    new_fitted: dict[int, FittedStringEncoder] = {}
    existing_names = list(feature_schema.feature_names) if feature_schema else []
    encoded_blocks: list[pd.DataFrame] = []
    for index in to_expand:
        # skrub names outputs from the series' `.name`; a bare `pd.DataFrame`
        # column label is an int, which can't take a suffix.
        column = frame.iloc[:, index].rename(str(index))
        if fitted is not None:
            encoded = _apply_one_column(column, fitted[index])
        else:
            assert feature_schema is not None
            name = feature_schema.features[index].name
            encoded, fitted_encoder = _fit_one_column(column, name, existing_names)
            existing_names += fitted_encoder.output_names
            new_fitted[index] = fitted_encoder
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
