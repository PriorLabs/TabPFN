#  Copyright (c) Prior Labs GmbH 2026.

"""Encode every non-numeric modality that has a real encoder into numbers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabpfn.preprocessing.date_encoding import expand_date_features
from tabpfn.preprocessing.string_encoding import expand_text_features

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.date_encoding import FittedDatetimeEncoder
    from tabpfn.preprocessing.string_encoding import FittedStringEncoder


def encode_multimodal_data(
    X: np.ndarray,
    feature_schema: FeatureSchema | None,
    *,
    use_text: bool = False,
    provided_categorical_indices: Sequence[int] | None = None,
    date_fitted: dict[int, FittedDatetimeEncoder] | None = None,
    text_fitted: dict[int, FittedStringEncoder] | None = None,
) -> tuple[
    np.ndarray,
    FeatureSchema | None,
    dict[int, FittedDatetimeEncoder],
    dict[int, FittedStringEncoder],
]:
    """Encode every modality with an encoder available (today: dates and text)."""
    X, feature_schema, date_encoders = expand_date_features(
        X, feature_schema, fitted=date_fitted
    )
    X, feature_schema, text_encoders = expand_text_features(
        X,
        feature_schema,
        fitted=text_fitted,
        use_text=use_text,
        provided_categorical_indices=provided_categorical_indices,
    )
    return X, feature_schema, date_encoders, text_encoders
