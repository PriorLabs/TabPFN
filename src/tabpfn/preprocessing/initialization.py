"""Module for initializing the preprocessing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder
from tabpfn.preprocessing.type_detection import infer_categorical_features

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer


def tag_features_and_sanitize_data(
    X: np.ndarray,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    provided_categorical_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, ColumnTransformer, list[int]]:
    """Tag features and sanitize data.

    This function will:
    - Infer the categorical features
    - Convert dtypes
    - Ensure categories are ordinally encoded
    - Convert the data to a numpy array
    - Return the data, the ordinal encoder, and the inferred categorical indices

    TODO: In the future, this function should be split into multiple steps.
    """
    inferred_categorical_indices = infer_categorical_features(
        X=X,
        provided=provided_categorical_indices,
        min_samples_for_inference=min_samples_for_inference,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
    )

    # Will convert inferred categorical indices to category dtype,
    # to be picked up by the ord_encoder, as well
    # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
    X_pandas: pd.DataFrame = fix_dtypes(X=X, cat_indices=inferred_categorical_indices)
    # Ensure categories are ordinally encoded
    ord_encoder = get_ordinal_encoder()
    X_numpy: np.ndarray = process_text_na_dataframe(
        X=X_pandas, ord_encoder=ord_encoder, fit_encoder=True
    )

    return X_numpy, ord_encoder, inferred_categorical_indices
