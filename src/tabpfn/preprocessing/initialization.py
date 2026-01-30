"""Module for initializing the preprocessing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer

    from tabpfn.preprocessing.torch import FeatureMetadata


def tag_features_and_sanitize_data(
    X: np.ndarray,
    feature_names: list[str] | None,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    provided_categorical_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, ColumnTransformer, FeatureMetadata]:
    """Tag features and sanitize data.

    This function will:
    - Infer feature modalities
    - Convert dtypes
    - Ensure categories are ordinally encoded

    Args:
        X: The data to infer the feature modalities from.
        feature_names: The names of the features.
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
    feature_metadata = detect_feature_modalities(
        X=X,
        feature_names=feature_names,
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
        cat_indices=feature_metadata.indices_for(FeatureModality.CATEGORICAL),
    )
    # Ensure categories are ordinally encoded
    ord_encoder = get_ordinal_encoder()
    X_numpy = process_text_na_dataframe(
        X=X_pandas, ord_encoder=ord_encoder, fit_encoder=True
    )

    return X_numpy, ord_encoder, feature_metadata
