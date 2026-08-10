#  Copyright (c) Prior Labs GmbH 2026.

"""Module to infer feature modalities: numerical, categorical, text, etc."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from tabpfn.errors import TabPFNUserError
from tabpfn.preprocessing.datamodel import (
    INPUT_FEATURE_PREFIX,
    Feature,
    FeatureModality,
    FeatureSchema,
    build_input_feature_names,
)

if TYPE_CHECKING:
    import numpy as np

_EARLY_EXIT_PREFIX_ROWS = 1024

#: Cap on how many column names the likely-text warning lists, so a wide frame of
#: text columns does not produce an unreadable multi-kilobyte message.
_MAX_TEXT_COLUMNS_IN_WARNING = 10


def detect_feature_modalities(
    X: np.ndarray,
    feature_names: list[str] | None,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    provided_categorical_indices: Sequence[int] | None = None,
) -> FeatureSchema:
    """Infer the features modalities from the given data, based on heuristics
    and user-provided indices for categorical features.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer the categorical features from.\
        feature_names: The names of the features.
        provided_categorical_indices: Any user provided indices of what is
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
    unique_feature_names = build_input_feature_names(feature_names, X.shape[1])
    for i, index in enumerate(range(X.shape[1])):
        X_slice: np.ndarray = X[:, index]
        reported_categorical = index in (provided_categorical_indices or ())
        feature_name = unique_feature_names[i]
        feat_modality = _detect_feature_modality(
            s=pd.Series(X_slice, name=feature_name),
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        )
        features.append(Feature(name=feature_name, modality=feat_modality))
    feature_schema = FeatureSchema(features=features)
    _warn_if_text_features(
        feature_schema,
        declared_categorical_indices=provided_categorical_indices,
    )
    return feature_schema


def _warn_if_text_features(
    feature_schema: FeatureSchema,
    *,
    declared_categorical_indices: Sequence[int] | None = None,
) -> None:
    """Warn when input columns look like free text rather than categoricals.

    High-cardinality string columns are labelled `FeatureModality.TEXT` by
    `detect_feature_modalities`, but this package has no text handling: they are swept
    into the same `OrdinalEncoder` as real categoricals, which selects columns by dtype
    (see `get_ordinal_encoder`). That turns near-unique text into near-unique integer
    codes, i.e. noise rather than signal, without any error to hint at it.

    Called by `detect_feature_modalities` while the schema still carries the TEXT
    labels, i.e. before the first preprocessing step that rebuilds it, since
    `FeatureSchema.from_only_categorical_indices` collapses TEXT into NUMERICAL.

    Args:
        feature_schema: The schema produced by `detect_feature_modalities`.
        declared_categorical_indices: Positional indices the caller passed as
            `categorical_features_indices`. These are never reported: declaring a
            column categorical states that the user already knows it holds
            non-numeric values and intends them as categories, so warning about it
            would be noise.
    """
    declared = set(declared_categorical_indices or ())
    text_names = [
        feature.name.removeprefix(INPUT_FEATURE_PREFIX)
        for index, feature in enumerate(feature_schema.features)
        if feature.modality is FeatureModality.TEXT and index not in declared
    ]
    if not text_names:
        return

    shown = text_names[:_MAX_TEXT_COLUMNS_IN_WARNING]
    column_names_to_print = ", ".join(repr(name) for name in shown)
    if len(text_names) > len(shown):
        column_names_to_print += f" (and {len(text_names) - len(shown)} more)"

    warnings.warn(
        f"These columns look like free text and are being ordinal-encoded as "
        f"high-cardinality categoricals, which usually adds noise rather than "
        f"signal: {column_names_to_print}.\n"
        "If such a column holds numbers stored as strings, convert it to a numeric "
        "dtype. If it holds genuine text, this package has no text handling -- "
        "consider the tabpfn-client API, which embeds text natively: "
        "https://github.com/PriorLabs/tabpfn-client \n"
        "To silence this for a column that is genuinely a high-cardinality category, "
        "pass its index in `categorical_features_indices`.",
        UserWarning,
        # Points at a direct `estimator.fit(X, y)` call site. Six frames out: this
        # function, `detect_feature_modalities`, `_initialize_dataset_preprocessing`,
        # `fit`, and the contextlib wrapper added by the `@config_context(...)`
        # decorator on `fit`. Pinned by the `warning.filename` asserts in the tests.
        stacklevel=6,
    )


def _detect_feature_modality(
    s: pd.Series,
    *,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    big_enough_n_to_infer_cat: bool,
) -> FeatureModality:
    # Early exit for the distinct count: counts only grow with the number of
    # rows scanned, and every comparison below is against a threshold of at
    # most max(max_unique_for_category, min_unique_for_numerical). So once a
    # small prefix of the column already exceeds all of them, every decision
    # equals the full-column one and scanning the remaining rows is skipped —
    # the common case for continuous columns. Low-cardinality columns fall
    # through to the exact full count, paying one extra prefix pass.
    decided_at = max(max_unique_for_category, min_unique_for_numerical, 1) + 1
    n_unique = 0
    if len(s) > _EARLY_EXIT_PREFIX_ROWS:
        n_unique = _get_unique_with_sklearn_compatible_error(
            s.iloc[:_EARLY_EXIT_PREFIX_ROWS]
        )
    if n_unique < decided_at:
        n_unique = _get_unique_with_sklearn_compatible_error(s)

    if n_unique <= 1 and not reported_categorical:
        # Either all values are missing, or all values are the same.
        # If there's a single value but also missing ones, it's not constant.
        # Columns the user explicitly declared categorical are kept categorical
        # (handled below) even when all-missing, so they route through the ordinal
        # encoder consistently between fit and predict instead of being treated as
        # a constant numeric column that crashes on string values seen at predict.
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
    coerced = pd.to_numeric(s, errors="coerce")
    is_numeric_or_missing = coerced.notna() | s.isna()
    return bool(is_numeric_or_missing.all())


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


def _get_unique_with_sklearn_compatible_error(s: pd.Series) -> int:
    """Calculate total distinct values once, treating NaN as a category."""
    try:
        return s.nunique(dropna=False)
    except TypeError as e:
        # The sklearn test is inserting a dict ({"foo": "bar"}) into the data to verify
        # that the estimator raises a TypeError with a specific message pattern
        # ("argument must be .* string.* number"). However, when pandas tries to
        # compute nunique() on a Series containing a dict, it fails with "unhashable
        # type: 'dict'" which doesn't match sklearn's expected error pattern.
        raise TypeError(
            f"argument must be a string or a number (columns must only contain strings "
            f"or numbers), got `{type(s.iloc[0]).__name__}`"
        ) from e
