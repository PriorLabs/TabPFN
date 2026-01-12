"""Module to infer feature types."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pandas import Series
from pandas.api.types import is_numeric_dtype
from pandas.io.sql import is_string_dtype

if TYPE_CHECKING:
    import numpy as np

    from tabpfn.constants import XType


# TODO: 'infer_categorical_features' should be deprecated, to use the new 'detect_feature_types'.
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
            s = Series(col)
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


# TODO: Maybe 'DatasetView' and 'FeatureType' belong to a more generic file with the main objects?
@dataclass(frozen=True)
class DatasetView:
    x_num: XType
    x_cat: XType
    x_text: XType


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXTUAL = "textual"
    CONSTANT = "constant"


FeatureTypeIndices = dict[FeatureType, list[int]]


# This should inheric from FeaturePreprocessingTransformerStep-like object
class FeatureTypeDetector:
    feature_type_indices_: FeatureTypeIndices

    # TODO: fit, transform etc. the fit should be calling `detect_feature_types` and storing it. Transform should be a no-op.


# TODO: this functio should basically be the 'fit' function of a FeatureTypeDetector class that wraps "FeaturePreprocessingTransformerStep" or sort.
# This
def detect_feature_types(
    X: np.ndarray,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    reported_categorical_indices: Sequence[int] | None = None,
) -> DatasetView:
    """Infer the features types from the given data, based on heuristics and user-provided indices for categorical features.

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
        A DatasetView object with the features types.
    """
    type2idx = _get_feature_type_indices(
        X,
        min_samples_for_inference=min_samples_for_inference,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        reported_categorical_indices=reported_categorical_indices,
    )
    x_num = X[:, type2idx[FeatureType.NUMERICAL]]
    x_cat = X[:, type2idx[FeatureType.CATEGORICAL]]
    x_text = X[:, type2idx[FeatureType.TEXTUAL]]
    return DatasetView(x_num, x_cat, x_text)


def _get_feature_type_indices(
    X: np.ndarray,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    reported_categorical_indices: Sequence[int] | None = None,
) -> FeatureTypeIndices:
    type2idx = defaultdict(list)
    large_enough_x_to_infer = X.shape[0] > min_samples_for_inference
    for idx, col in enumerate(X.T):
        reported_categorical = idx in (reported_categorical_indices or ())
        feat_type = _detect_feature_type(
            X[:, col],
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            large_enough_x_to_infer_categorical=large_enough_x_to_infer,
        )
        type2idx[feat_type].append(idx)
    return type2idx


def _detect_feature_type(
    col: np.ndarray,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    large_enough_x_to_infer_categorical: bool,
) -> FeatureType:
    # Calculate total distinct values once, treating NaN as a category.
    # TODO: detect whether it's a constant feature
    # TODO: first, check if the feature is numeric or not
    s = _array_to_series(col)
    if _detect_constant(s):
        return FeatureType.CONSTANT
    if _detect_categorical(
        s,
        reported_categorical,
        max_unique_for_category,
        min_unique_for_numerical,
        large_enough_x_to_infer_categorical,
    ):
        return FeatureType.CATEGORICAL
    if _detect_textual(s, max_unique_for_category=max_unique_for_category):
        return FeatureType.TEXTUAL
    if is_numeric_dtype(s.dtype):
        return FeatureType.NUMERICAL
    raise TypeError(f"Unknown feature type: {s.dtype}: {s.unique()}")


def _array_to_series(col: np.ndarray) -> Series:
    # TODO (1): the last part here looks like something that should be part of the validation.
    # TODO (2): what is the point of casting it to a series? wouldn't numpy have built in functions for this?
    try:
        return Series(col)
    except TypeError as e:
        # e.g. "unhashable type: 'dict'" when object arrays contain dicts
        raise TypeError(
            "argument must be a string or a number"
            "(columns must only contain strings or numbers)"
        ) from e


def _detect_categorical(
    s: Series,
    reported_categorical: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    large_enough_x_to_infer_categorical: bool,
) -> bool:
    """Detecting if a numerical feature is categorical depending on heuristics:
    - Feature reported to be categorical are treated as such, as long as they aren't too-highly cardinal.
    - For non-reported numerical ones, we infer them as such if they are sufficiently low-cardinal.
    """
    num_distinct = s.nunique(dropna=False)
    if reported_categorical:
        if num_distinct <= max_unique_for_category:
            return True
    elif (
        large_enough_x_to_infer_categorical and num_distinct < min_unique_for_numerical
    ):
        return True
    return False


def _detect_constant(s: Series) -> bool:
    """A constant feature means that either all values are missing, or all values are the same. If there's a single value but also missing ones, it's not a constant feature."""
    if s.isna().all():
        return True
    return s.nunique(dropna=False) == 1


def _detect_textual(s: Series, max_unique_for_category: int) -> bool:
    """A textual feature means that the feature is a string or a category."""
    if not is_string_dtype(s.dtype):
        return False
    num_distinct = s.nunique(dropna=False)
    if num_distinct <= max_unique_for_category:
        raise ValueError(
            f"A feature with low cardinality should have been detected as categorical. Got {num_distinct} unique values!"
        )
    return True
