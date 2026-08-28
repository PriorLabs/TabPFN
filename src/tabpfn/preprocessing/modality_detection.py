#  Copyright (c) Prior Labs GmbH 2026.

"""Module to infer feature modalities: numerical, categorical, text, etc."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from tabpfn.errors import TabPFNUserError
from tabpfn.preprocessing.clean import PANDAS_BELOW_3
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
    min_cardinality_for_text: int,
    provided_categorical_indices: Sequence[int] | None = None,
    provided_date_indices: Sequence[int] | None = None,
    transform_dates: bool = False,
) -> FeatureSchema:
    """Infer each feature's modality, using heuristics and declared categoricals.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer feature modalities from.
        feature_names: The names of the features.
        provided_categorical_indices: User-provided indices considered categorical.
        provided_date_indices: Indices already known to be dates, because they
            arrived as a datetime dtype rather than as strings that might parse
            as one. Tagged `DATE` outright: there is nothing for the string
            heuristics to guess at, and skipping them also skips their cost.
        min_samples_for_inference: Minimum samples required to auto-infer a
            feature not provided as categorical.
        max_unique_for_category: Max unique values for a feature to be categorical.
        min_unique_for_numerical: Min unique values for a feature to be numerical.
        min_cardinality_for_text: Unique-value count above which a candidate
            string column (not parsed as a number) is `TEXT` rather than
            `CATEGORICAL` -- independent of the two thresholds above.
        transform_dates: Whether a detected date column is left as `DATE` for the
            caller to expand, rather than demoted to `CATEGORICAL`/`TEXT`.

    Returns:
        The inferred `FeatureSchema`.
    """
    features: list[Feature] = []
    big_enough_n_to_infer_cat = len(X) > min_samples_for_inference
    unique_feature_names = build_input_feature_names(feature_names, X.shape[1])
    reported_dates = set(provided_date_indices or ())
    for i, index in enumerate(range(X.shape[1])):
        X_slice: np.ndarray = X[:, index]
        reported_categorical = index in (provided_categorical_indices or ())
        feature_name = unique_feature_names[i]
        if index in reported_dates and not reported_categorical:
            # A datetime dtype says outright what the string heuristics below
            # can only guess at, so they are not consulted -- unless the caller
            # also declared the column categorical, which overrides either way.
            feat_modality = FeatureModality.DATE
        else:
            feat_modality = _detect_feature_modality(
                s=pd.Series(X_slice, name=feature_name),
                reported_categorical=reported_categorical,
                max_unique_for_category=max_unique_for_category,
                min_unique_for_numerical=min_unique_for_numerical,
                min_cardinality_for_text=min_cardinality_for_text,
                big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
            )
        features.append(Feature(name=feature_name, modality=feat_modality))
    feature_schema = FeatureSchema(features=features)
    # Before dates are demoted, so a date isn't yet TEXT and is never counted here.
    _warn_on_texts(feature_schema, declared_cat_indices=provided_categorical_indices)

    if not transform_dates and feature_schema.indices_for(FeatureModality.DATE):
        feature_schema, demoted_date_columns = _demote_dates(
            feature_schema,
            X,
            min_cardinality_for_text=min_cardinality_for_text,
        )
        _warn_on_dates(demoted_date_columns)
    return feature_schema


def _demote_dates(
    feature_schema: FeatureSchema,
    X: np.ndarray,
    *,
    min_cardinality_for_text: int,
) -> tuple[FeatureSchema, list[str]]:
    """Demote every detected `DATE` feature to `CATEGORICAL`/`TEXT`.

    Only called when `transform_dates` is off, so every detected date is demoted,
    via the same cardinality rule `_classify_string_like_column` already
    applies to a same-shaped non-date string.

    Every column here is one nothing else claimed: a declared categorical never
    reaches `DATE` in the first place, so there is no declaration to respect,
    and every demoted column is one worth warning about.

    Returns:
        The updated schema, and the demoted column names to warn about.
    """
    features = list(feature_schema.features)
    demoted_columns = []
    for index in feature_schema.indices_for(FeatureModality.DATE):
        n_unique = _get_unique_with_sklearn_compatible_error(pd.Series(X[:, index]))
        if n_unique <= min_cardinality_for_text:
            demoted = FeatureModality.CATEGORICAL
        else:
            demoted = FeatureModality.TEXT
        features[index] = dataclasses.replace(features[index], modality=demoted)
        demoted_columns.append(features[index].name.removeprefix(INPUT_FEATURE_PREFIX))
    return FeatureSchema(features=features), demoted_columns


def _format_names_for_warning(names: list[str]) -> str:
    """Render column names for a warning, capped so it stays readable."""
    shown = names[:_MAX_TEXT_COLUMNS_IN_WARNING]
    printed = ", ".join(repr(name) for name in shown)
    if len(names) > len(shown):
        printed += f" (and {len(names) - len(shown)} more)"
    return printed


def _warn_on_texts(
    feature_schema: FeatureSchema,
    *,
    declared_cat_indices: Sequence[int] | None = None,
) -> None:
    """Warn when input columns look like free text rather than categoricals.

    Called before `_demote_dates`, while a date is still tagged `DATE` rather
    than `TEXT`, so it's never counted here.

    Args:
        feature_schema: The schema before any `DATE` column has been demoted.
        declared_cat_indices: Indices passed as `categorical_features_indices`;
            never reported, since declaring a column categorical means the user
            already intends its non-numeric values as categories.
    """
    declared = set(declared_cat_indices or ())
    text_names = [
        feature.name.removeprefix(INPUT_FEATURE_PREFIX)
        for index, feature in enumerate(feature_schema.features)
        if feature.modality is FeatureModality.TEXT and index not in declared
    ]
    if not text_names:
        return

    warnings.warn(
        f"These columns look like free text and are being ordinal-encoded as "
        f"high-cardinality categoricals, which usually adds noise rather than "
        f"signal: {_format_names_for_warning(text_names)}.\n"
        "If such a column holds numbers stored as strings, convert it to a numeric "
        "dtype. If it is a category rather than text, raise "
        '`inference_config={"MIN_CARDINALITY_FOR_TEXT": ...}` above its number of '
        "distinct values. If it holds genuine text, this package has no text "
        "handling -- consider the tabpfn-client API, which embeds text natively: "
        "https://github.com/PriorLabs/tabpfn-client \n"
        "To silence this for a column that is genuinely a high-cardinality category, "
        "pass its index in `categorical_features_indices`.",
        UserWarning,
        # stacklevel=6 reaches the `estimator.fit(X, y)` call site; pinned by the
        # `warning.filename` asserts in the tests.
        stacklevel=6,
    )


def _warn_on_dates(demoted_date_columns: list[str]) -> None:
    """Warn about dates `_demote_dates` demoted to a plain category or text.

    Empty whenever every demoted date was declared categorical.
    """
    if not demoted_date_columns:
        return
    warnings.warn(
        f"These columns hold dates, which are read as plain categories or "
        f"text: {_format_names_for_warning(demoted_date_columns)}.\n"
        'Raise `inference_config={"TRANSFORM_DATES": True}` to expand them into '
        "calendar features instead. To silence this for a column that should "
        "stay a plain category or text, pass its index in "
        "`categorical_features_indices`.",
        UserWarning,
        stacklevel=6,
    )


def _detect_feature_modality(
    s: pd.Series,
    *,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    min_cardinality_for_text: int,
    big_enough_n_to_infer_cat: bool,
) -> FeatureModality:
    """Decide a single column's modality; see `_demote_dates` for DATE."""
    # Early exit: once a prefix already clears every threshold below, the full
    # count would land in the same bucket, so skip scanning the rest.
    # min_cardinality_for_text is included since it can exceed the other two.
    decided_at = (
        max(
            max_unique_for_category,
            min_unique_for_numerical,
            min_cardinality_for_text,
            1,
        )
        + 1
    )
    n_unique = 0
    if len(s) > _EARLY_EXIT_PREFIX_ROWS:
        n_unique = _get_unique_with_sklearn_compatible_error(
            s.iloc[:_EARLY_EXIT_PREFIX_ROWS]
        )
    if n_unique < decided_at:
        n_unique = _get_unique_with_sklearn_compatible_error(s)

    if n_unique <= 1 and not reported_categorical:
        # All-missing or single-value. A declared-categorical column is exempt so
        # it still routes through the ordinal encoder instead of crashing as a
        # constant numeric column when predict sees an unseen string value.
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

    is_string_like = pd.api.types.is_string_dtype(s.dtype) or isinstance(
        s.dtype, pd.CategoricalDtype
    )
    if is_string_like:
        return _classify_string_like_column(
            n_unique=n_unique, min_cardinality_for_text=min_cardinality_for_text
        )
    raise TabPFNUserError(
        f"Column {s.name!r} has dtype {s.dtype}, which cannot be read as numbers, "
        f"categories, text or dates (it holds {s.nunique(dropna=False)} distinct "
        f"values).\n"
        "Convert it to something this can read: a numeric dtype for a quantity, "
        "a string/category dtype for a label, or a datetime dtype for a point in "
        "time (e.g. `df[col] = pd.to_datetime(df[col])`, or `.dt.to_timestamp()` "
        "for a period)."
    )


def _classify_string_like_column(
    *,
    n_unique: int,
    min_cardinality_for_text: int,
) -> FeatureModality:
    """Classify a string/categorical-dtype column as CATEGORICAL or TEXT.

    No content-based date guessing here: `DATE` is only ever assigned via
    `provided_date_indices`, i.e. a column that arrived as a genuine datetime
    dtype (see `detect_feature_modalities`). A string that merely looks like a
    date -- "2020-01-01" -- is just an ordinary string, classified by
    cardinality like any other.
    """
    if n_unique <= min_cardinality_for_text:
        return FeatureModality.CATEGORICAL
    return FeatureModality.TEXT


def _is_numeric_pandas_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return True
    if PANDAS_BELOW_3:
        return all(_is_numeric_or_missing_for_old_pandas(value) for value in s)
    # The generator above stops at the first non-numeric value; `pd.to_numeric`
    # coerces the whole column first, so reject on a prefix instead. Exact, not
    # approximate: one non-numeric value settles the all-or-nothing answer, and
    # a non-numeric column fails on its first value, which the prefix always
    # includes.
    if len(s) > _EARLY_EXIT_PREFIX_ROWS and not _all_numeric_or_missing(
        s.iloc[:_EARLY_EXIT_PREFIX_ROWS]
    ):
        return False
    return _all_numeric_or_missing(s)


def _all_numeric_or_missing(s: pd.Series) -> bool:
    """Whether every value in `s` is a number, a spelling of one, or missing."""
    coerced = pd.to_numeric(s, errors="coerce")
    is_numeric_or_missing = coerced.notna() | s.isna()
    return bool(is_numeric_or_missing.all())


def _is_numeric_or_missing_for_old_pandas(value: object) -> bool:
    # Below pandas 3.0, `pd.to_numeric` segfaults on a string whose scientific-notation
    # exponent lands in [2**31, 2**32), e.g. "8e2569614270" (pandas#63650), and a
    # segfault cannot be caught. Not vectorized, but no slower here: `pd.to_numeric`
    # also walks an object column value by value. Delete once the pandas floor is 3.0.
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # Not a number, so only a missing value still counts. `is_scalar` guards
        # `pd.isna`, which answers element-wise for a list or an array cell.
        return bool(pd.api.types.is_scalar(value) and pd.isna(value))
    # Anything else `float` accepted is already a number, not a spelling of one,
    # except a buffer: `float` reads any of them, pandas only `bytes`.
    if not isinstance(value, str):
        return not isinstance(value, (bytearray, memoryview))
    # Non-ASCII digits and spaces, e.g. "٣" and "\xa0 5".
    if not value.isascii():
        return False
    # PEP 515 digit separators, e.g. "1_000".
    if "_" in value:
        return False
    # The literal "nan", in any spelling: no other string parses to NaN.
    if math.isnan(parsed):
        return False
    # A finite literal too large for a float64, e.g. "1e400". Only a spelled-out
    # infinity counts as numeric.
    return not (math.isinf(parsed) and "inf" not in value.lower())


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
