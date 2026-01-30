"""Module to infer feature modalities: numerical, categorical, text, etc."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from tabpfn.errors import TabPFNUserError
from tabpfn.preprocessing.datamodel import Feature, FeatureModality
from tabpfn.preprocessing.torch import FeatureMetadata

if TYPE_CHECKING:
    import numpy as np


# This should inheric from FeaturePreprocessingTransformerStep-like object
class FeatureModalityDetector:
    """Detector for feature modalities as defined by FeatureModality."""

    feature_modality_columns: dict[FeatureModality, list[str]]

    def _fit(self, X: pd.DataFrame) -> None:
        raise NotImplementedError("Should be calling `detect_feature_modalities`.")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Should be a no-op.")


def detect_feature_modalities(
    X: np.ndarray,
    feature_names: list[str] | None,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    reported_categorical_indices: Sequence[int] | None = None,
) -> FeatureMetadata:
    """Infer the features modalities from the given data, based on heuristics
    and user-provided indices for categorical features.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer the categorical features from.\
        feature_names: The names of the features.
        reported_categorical_indices: Any user provided indices of what is
            considered categorical.
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
        A dictionary with the feature modalities as keys and the column as
        values.
    """
    features: list[Feature] = []
    big_enough_n_to_infer_cat = len(X) > min_samples_for_inference
    for i, index in enumerate(range(X.shape[1])):
        X_slice: np.ndarray = X[:, index]
        reported_categorical = index in (reported_categorical_indices or ())
        feature_name = feature_names[i] if feature_names is not None else None
        feat_modality = _detect_feature_modality(
            s=pd.Series(X_slice, name=feature_name),
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        )
        features.append(Feature(name=feature_name, modality=feat_modality))
    return FeatureMetadata(features=features)


def _detect_feature_modality(
    s: pd.Series,
    *,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    big_enough_n_to_infer_cat: bool,
) -> FeatureModality:
    # Calculate total distinct values once, treating NaN as a category.
    n_unique = s.nunique(dropna=False)
    if n_unique <= 1:
        # Either all values are missing, or all values are the same.
        # If there's a single value but also missing ones, it's not constant
        return FeatureModality.CONSTANT

    if _is_numeric_pandas_series(s):
        if _detect_numeric_as_categorical(
            n_unique=n_unique,
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        ):
            return FeatureModality.CATEGORICAL
        return FeatureModality.NUMERICAL
    if pd.api.types.is_string_dtype(s.dtype) or isinstance(
        s.dtype, pd.CategoricalDtype
    ):
        if n_unique <= max_unique_for_category:
            return FeatureModality.CATEGORICAL
        return FeatureModality.TEXT
    raise TabPFNUserError(
        f"Unknown dtype: {s.dtype}, with {s.nunique(dropna=False)} unique values"
    )


def _is_numeric_pandas_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return True
    return bool(all(_is_numeric_value(x) for x in s))


def _detect_numeric_as_categorical(
    n_unique: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    *,
    reported_categorical: bool,
    big_enough_n_to_infer_cat: bool,
) -> bool:
    """Detecting if a numerical feature is categorical depending on heuristics:
    - Feature reported as categoricals are treated as such, as long as they
      aren't highly cardinal.
    - For non-reported numerical ones, we infer them as such if they are
      sufficiently low-cardinal.
    """
    if reported_categorical:
        if n_unique <= max_unique_for_category:
            return True
    elif big_enough_n_to_infer_cat and n_unique < min_unique_for_numerical:
        return True
    return False


def _is_numeric_value(x: Any) -> bool:
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, str) and x.isdigit():
        return True
    try:
        x = float(x)
        return True
    except ValueError:
        return False
