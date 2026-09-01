#  Copyright (c) Prior Labs GmbH 2026.

"""Module to infer feature modalities: numerical, categorical, text, etc."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
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
    from tabpfn.constants import XType

_EARLY_EXIT_PREFIX_ROWS = 1024

#: Cap on how many column names the likely-text warning lists, so a wide frame of
#: text columns does not produce an unreadable multi-kilobyte message.
_MAX_TEXT_COLUMNS_IN_WARNING = 10


def resolve_datetime_columns(
    X: XType,
    *,
    categorical_features_indices: Sequence[int] | None,
) -> tuple[XType, list[int]]:
    """Cast genuine `datetime64` columns to numeric, and report their indices.

    Must run before `ensure_compatible_fit_inputs`: a `datetime64` dtype has no
    common numpy dtype with a plain numeric/bool/string column (only with
    `object`), so `check_array`'s `np.result_type` over the raw column dtypes
    crashes outright otherwise (a real, previously-unfixed bug -- see
    `test__classifier_fit__native_datetime_column__no_longer_crashes`).
    Casting unifies fine with any other numeric dtype, so validation proceeds
    normally afterward.

    Returns the (possibly cast) `X` and the indices of the columns cast, so the
    caller can tag them `DATE` via `detect_feature_modalities`'s
    `provided_date_indices`, instead of them silently reading as an ordinary
    numeric column.
    """
    date_indices = detect_datetime_columns(
        X, categorical_features_indices=categorical_features_indices
    )
    if not date_indices:
        return X, []

    date_set = set(date_indices)
    columns = [
        _datetime_column_to_numeric(X.iloc[:, i]) if i in date_set else X.iloc[:, i]
        for i in range(X.shape[1])
    ]
    # `.iloc[:, i] = ...` would try to preserve the column's existing datetime64
    # storage and reject the cast; rebuilding column-by-column changes its dtype
    # instead. Positional, so duplicate column labels are handled correctly.
    resolved = pd.concat(columns, axis=1)
    resolved.columns = X.columns
    return resolved, date_indices


def detect_datetime_columns(
    X: XType,
    *,
    categorical_features_indices: Sequence[int] | None,
) -> list[int]:
    """Indices of real `datetime64` columns in `X`, before validation runs.

    Must be computed off the raw input, before `ensure_compatible_fit_inputs`
    converts it: sklearn's own validation flattens a mixed-dtype DataFrame into
    one numpy array, so a genuine `datetime64` dtype is only visible here.

    A column already declared categorical is excluded -- the user's declared
    intent for it wins over treating it as a date.
    """
    if not isinstance(X, pd.DataFrame):
        return []
    declared = set(categorical_features_indices or ())
    return [
        i
        for i, dtype in enumerate(X.dtypes)
        if pd.api.types.is_datetime64_any_dtype(dtype) and i not in declared
    ]


def _datetime_column_to_numeric(column: pd.Series) -> pd.Series:
    """Cast one `datetime64` column to nanoseconds since the epoch.

    As `float64`, so a missing date (`NaT`) survives as `NaN` -- `NaT.astype`
    `("int64")` maps it to that dtype's huge sentinel value instead, which
    `float64` has no equivalent of. Exact enough for any realistic datetime
    column: nanosecond-since-epoch magnitudes are still ~256ns short of
    float64's precision limit for a present-day date, far below anything a
    tabular feature would need to distinguish.
    """
    is_missing = column.isna().to_numpy()
    as_ns = column.astype("int64").astype("float64")
    as_ns[is_missing] = np.nan
    return as_ns


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
) -> FeatureSchema:
    """Infer each feature's modality, using heuristics and declared categoricals.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer feature modalities from.
        feature_names: The names of the features.
        provided_categorical_indices: User-provided indices considered categorical.
        provided_date_indices: Indices already known to hold a real `datetime64`
            dtype before validation (see `resolve_datetime_columns`); tagged
            `DATE` outright, without running the heuristics below.
        min_samples_for_inference: Minimum samples required to auto-infer a
            feature not provided as categorical.
        max_unique_for_category: Max unique values for a feature to be categorical.
        min_unique_for_numerical: Min unique values for a feature to be numerical.
        min_cardinality_for_text: Unique-value count above which a candidate
            string column (not parsed as a number) is `TEXT` rather than
            `CATEGORICAL` -- independent of the two thresholds above.

    Returns:
        The inferred `FeatureSchema`.
    """
    features: list[Feature] = []
    big_enough_n_to_infer_cat = len(X) > min_samples_for_inference
    unique_feature_names = build_input_feature_names(feature_names, X.shape[1])
    date_indices = set(provided_date_indices or ())
    for i, index in enumerate(range(X.shape[1])):
        feature_name = unique_feature_names[i]
        if index in date_indices:
            features.append(Feature(name=feature_name, modality=FeatureModality.DATE))
            continue
        X_slice: np.ndarray = X[:, index]
        reported_categorical = index in (provided_categorical_indices or ())
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
    _warn_on_multimodal(
        feature_schema, declared_cat_indices=provided_categorical_indices
    )
    return _demote_dates_for_now(
        feature_schema,
        X,
        provided_categorical_indices=provided_categorical_indices,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        min_cardinality_for_text=min_cardinality_for_text,
        big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
    )


def _demote_dates_for_now(
    feature_schema: FeatureSchema,
    X: np.ndarray,
    *,
    provided_categorical_indices: Sequence[int] | None,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    min_cardinality_for_text: int,
    big_enough_n_to_infer_cat: bool,
) -> FeatureSchema:
    """Temporary: demote every detected `DATE` feature to `CATEGORICAL`/`TEXT`.

    Nothing expands a date into calendar features yet, so `DATE` isn't
    consumable downstream -- demoted the same way a same-shaped non-date
    string would be. A follow-up adding real date expansion should delete this
    function outright. Called after `_warn_on_multimodal`, while the schema
    still says `DATE`.
    """
    date_indices = feature_schema.indices_for(FeatureModality.DATE)
    if not date_indices:
        return feature_schema

    declared = set(provided_categorical_indices or ())
    features = list(feature_schema.features)
    for index in date_indices:
        n_unique = _get_unique_with_sklearn_compatible_error(pd.Series(X[:, index]))
        if _detect_numeric_as_categorical(
            n_unique=n_unique,
            reported_categorical=index in declared,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        ):
            demoted = FeatureModality.CATEGORICAL
        else:
            demoted = (
                FeatureModality.CATEGORICAL
                if n_unique <= min_cardinality_for_text
                else FeatureModality.TEXT
            )
        features[index] = dataclasses.replace(features[index], modality=demoted)
    return FeatureSchema(features=features)


def _format_names_for_warning(names: list[str]) -> str:
    """Render column names for a warning, capped so it stays readable."""
    shown = names[:_MAX_TEXT_COLUMNS_IN_WARNING]
    printed = ", ".join(repr(name) for name in shown)
    if len(names) > len(shown):
        printed += f" (and {len(names) - len(shown)} more)"
    return printed


def _warn_on_multimodal(
    feature_schema: FeatureSchema,
    *,
    declared_cat_indices: Sequence[int] | None = None,
) -> None:
    """Warn about detected dates, then about any remaining free-text columns.

    Called before `_demote_dates_for_now` acts on `DATE`, so a date column
    isn't `TEXT` yet and needs no special-casing to avoid a double warning.

    Args:
        feature_schema: The schema produced by detection, before `DATE` is acted on.
        declared_cat_indices: Indices passed as `categorical_features_indices`;
            never reported, since declaring a column categorical means the user
            already intends its non-numeric values as categories.
    """
    date_columns = [
        feature_schema.features[index].name.removeprefix(INPUT_FEATURE_PREFIX)
        for index in feature_schema.indices_for(FeatureModality.DATE)
    ]
    if date_columns:
        warnings.warn(
            f"These columns hold dates, which are not yet expanded into calendar "
            f"features, so they are read as plain categories or text: "
            f"{_format_names_for_warning(date_columns)}.",
            UserWarning,
            stacklevel=6,
        )

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


def _detect_feature_modality(
    s: pd.Series,
    *,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    min_cardinality_for_text: int,
    big_enough_n_to_infer_cat: bool,
) -> FeatureModality:
    """Decide a single column's modality via heuristics."""
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
        return (
            FeatureModality.CATEGORICAL
            if n_unique <= min_cardinality_for_text
            else FeatureModality.TEXT
        )
    raise TabPFNUserError(
        f"Unknown dtype: {s.dtype}, with {s.nunique(dropna=False)} unique values"
    )


def _is_numeric_pandas_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return True
    if PANDAS_BELOW_3:
        return all(_is_numeric_or_missing_for_old_pandas(value) for value in s)
    # The generator above stops at the first non-numeric value; `pd.to_numeric`
    # coerces the whole column first, so reject on a prefix instead: one
    # non-numeric value anywhere settles the answer, so a prefix that already
    # fails proves the full column does too.
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
