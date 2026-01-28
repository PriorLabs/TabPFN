"""Module for initializing the preprocessing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from sklearn.compose import ColumnTransformer


def tag_features_and_sanitize_data(
    X: np.ndarray,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    provided_categorical_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, ColumnTransformer, dict[FeatureModality, list[int]]]:
    """Tag features and sanitize data.

    This function will:
    - Infer feature modalities
    - Convert dtypes
    - Ensure categories are ordinally encoded

    Args:
        X: The data to infer the feature modalities from.
        min_samples_for_inference: The minimum number of samples required for automatic
             inference of features which were not provided as categorical.
        max_unique_for_category: The maximum number of unique values for a feature
            to be considered categorical.
        min_unique_for_numerical: The minimum number of unique values for a feature
            to be considered numerical.
        provided_categorical_indices: Any user provided indices of what is considered
            categorical.

    Returns:
        A tuple containing the data, the ordinal encoder, and the inferred feature
        modalities.
    """
    inferred_column_modalities = detect_feature_modalities(
        X=pd.DataFrame(X),
        min_samples_for_inference=min_samples_for_inference,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        reported_categorical_indices=provided_categorical_indices,
    )

    # Will convert inferred categorical indices to category dtype,
    # to be picked up by the ord_encoder, as well
    # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
    X_pandas: pd.DataFrame = fix_dtypes(
        X=X,
        cat_indices=inferred_column_modalities[FeatureModality.CATEGORICAL],
    )
    # Ensure categories are ordinally encoded
    ord_encoder = get_ordinal_encoder()
    X_numpy: np.ndarray = process_text_na_dataframe(
        X=X_pandas, ord_encoder=ord_encoder, fit_encoder=True
    )

    return X_numpy, ord_encoder, inferred_column_modalities


def convert_to_pandas(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | pd.Index | None,
    y_name: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert the data to a pandas DataFrame and Series.

    Ensures that the feature names are a pandas Index with string dtype
    and the target name is a string.
    """
    num_features = X.shape[1]
    if feature_names is None:
        feature_names = pd.Index([f"c{i}" for i in range(num_features)])
    else:
        feature_names = pd.Index(feature_names).astype(str)
    if y_name is None:
        y_name = "y"

    X_pandas = pd.DataFrame(X, columns=feature_names)
    y_pandas = pd.Series(y, name=y_name)
    return X_pandas, y_pandas
