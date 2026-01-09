"""Module to infer feature types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy as np


def infer_categorical_features(
    X: np.ndarray,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    provided: Sequence[int] | None = None,
) -> list[int]:
    """Infer the categorical features from the given data.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer the categorical features from.
        provided: Any user provided indices of what is considered categorical.
        min_samples_for_inference:
            The minimum number of samples required
            for automatic inference of features which were not provided
            as categorical.
        max_unique_for_category:
            The maximum number of unique values for a
            feature to be considered categorical.
        min_unique_for_numerical:
            The minimum number of unique values for a
            feature to be considered numerical.

    Returns:
        The indices of inferred categorical features.
    """
    # We presume everything is numerical and go from there
    maybe_categoricals = () if provided is None else provided
    large_enough_x_to_infer_categorical = X.shape[0] > min_samples_for_inference
    indices = []

    for ix, col in enumerate(X.T):
        # Calculate total distinct values once, treating NaN as a category.
        try:
            s = pd.Series(col)
            # counts NaN/None as a category
            num_distinct = s.nunique(dropna=False)
        except TypeError as e:
            # e.g. "unhashable type: 'dict'" when object arrays contain dicts
            raise TypeError(
                "argument must be a string or a number"
                "(columns must only contain strings or numbers)"
            ) from e
        if ix in maybe_categoricals:
            if num_distinct <= max_unique_for_category:
                indices.append(ix)
        elif (
            large_enough_x_to_infer_categorical
            and num_distinct < min_unique_for_numerical
        ):
            indices.append(ix)

    return indices
