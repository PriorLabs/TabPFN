"""Module to infer feature types."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

from tabpfn.errors import TabPFNUserError

if TYPE_CHECKING:
    import numpy as np


# TODO: 'infer_categorical_features' should be deprecated,
# to use 'detect_feature_types'.
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


# TODO: Maybe 'DatasetView' and 'FeatureType' belong to a file with different objects?
@dataclass(frozen=True)
class DatasetView:
    """A view of a dataset split by feature types."""

    X: pd.DataFrame
    feature_type_to_columns: FeatureTypeColumns


class FeatureType(str, Enum):
    """The type of a feature."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    CONSTANT = "constant"


FeatureTypeColumns = dict[FeatureType, list[str]]


# This should inheric from FeaturePreprocessingTransformerStep-like object
class FeatureTypeDetector:
    """Detector for feature types."""

    feature_type_indices_: FeatureTypeColumns

    # TODO: fit, transform etc. the fit should be calling `detect_feature_types`
    # and storing it. Transform should be a no-op.


def detect_feature_types(
    X: pd.DataFrame,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    reported_categorical_indices: Sequence[int] | None = None,
) -> DatasetView:
    """Infer the features types from the given data, based on heuristics
    and user-provided indices for categorical features.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer the categorical features from.
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
        A DatasetView object with the features types.
    """
    feature_type_to_columns = _detect_feature_type_to_columns(
        X,
        min_samples_for_inference=min_samples_for_inference,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        reported_categorical_indices=reported_categorical_indices,
    )
    return DatasetView(X=X, feature_type_to_columns=feature_type_to_columns)


def _detect_feature_type_to_columns(
    X: pd.DataFrame,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    reported_categorical_indices: Sequence[int] | None = None,
) -> FeatureTypeColumns:
    feature_type_to_columns = defaultdict(list)
    big_enough_n_to_infer_cat = len(X) > min_samples_for_inference
    for idx, col in enumerate(X.columns):
        feat = X.iloc[col]
        reported_categorical = idx in (reported_categorical_indices or ())
        feat_type = _detect_feature_type(
            s=feat,
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        )
        feature_type_to_columns[feat_type].append(col)
    return feature_type_to_columns


def _detect_feature_type(
    s: pd.Series,
    *,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    big_enough_n_to_infer_cat: bool,
) -> FeatureType:
    # Calculate total distinct values once, treating NaN as a category.
    nunique = s.nunique(dropna=False)
    if nunique <= 1:
        # Either all values are missing, or all values are the same.
        # If there's a single value but also missing ones, it's not constant
        return FeatureType.CONSTANT

    if _is_numeric_pandas_series(s):
        if _detect_numeric_as_categorical(
            nunique=nunique,
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        ):
            return FeatureType.CATEGORICAL
        return FeatureType.NUMERICAL
    if pd.api.types.is_string_dtype(s.dtype):
        if nunique <= max_unique_for_category:
            return FeatureType.CATEGORICAL
        return FeatureType.TEXT
    raise TabPFNUserError(
        f"Unknown dtype: {s.dtype}, with {s.nunique(dropna=False)} unique values"
    )


def _is_numeric_pandas_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return True
    return bool(all(_is_numeric_value(x) for x in s))


def _detect_numeric_as_categorical(
    nunique: int,
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
        if nunique <= max_unique_for_category:
            return True
    elif big_enough_n_to_infer_cat and nunique < min_unique_for_numerical:
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
