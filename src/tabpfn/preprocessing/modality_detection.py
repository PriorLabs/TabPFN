#  Copyright (c) Prior Labs GmbH 2026.

"""Module to infer feature modalities: numerical, categorical, text, etc."""

from __future__ import annotations

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
        min_cardinality_for_text: The number of distinct values above which a
            candidate string column (one that wasn't parsed as a number or a
            date) is `TEXT` rather than `CATEGORICAL`. A separate decision from
            `max_unique_for_category`/`min_unique_for_numerical`, which govern
            numerical-vs-categorical.

    Returns:
        A dictionary with the feature modalities as keys and the column as
        values.
    """
    features: list[Feature] = []
    date_columns: list[str] = []
    big_enough_n_to_infer_cat = len(X) > min_samples_for_inference
    unique_feature_names = build_input_feature_names(feature_names, X.shape[1])
    for i, index in enumerate(range(X.shape[1])):
        X_slice: np.ndarray = X[:, index]
        reported_categorical = index in (provided_categorical_indices or ())
        feature_name = unique_feature_names[i]
        feat_modality, is_date = _detect_feature_modality(
            s=pd.Series(X_slice, name=feature_name),
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            min_cardinality_for_text=min_cardinality_for_text,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        )
        if is_date:
            date_columns.append(feature_name.removeprefix(INPUT_FEATURE_PREFIX))
        features.append(Feature(name=feature_name, modality=feat_modality))
    feature_schema = FeatureSchema(features=features)
    _warn_about_text_or_dates(
        feature_schema,
        date_columns,
        declared_categorical_indices=provided_categorical_indices,
    )
    return feature_schema


def _format_names_for_warning(names: list[str]) -> str:
    """Render column names for a warning, capped so it stays readable."""
    shown = names[:_MAX_TEXT_COLUMNS_IN_WARNING]
    printed = ", ".join(repr(name) for name in shown)
    if len(names) > len(shown):
        printed += f" (and {len(names) - len(shown)} more)"
    return printed


def _warn_about_text_or_dates(
    feature_schema: FeatureSchema,
    date_columns: list[str],
    *,
    declared_categorical_indices: Sequence[int] | None = None,
) -> None:
    """Warn about date columns, then about any remaining free-text columns.

    Nothing expands a date into calendar features yet, so a recognized date is
    always read as a plain category or text, same as before it was recognized as
    one; naming it explicitly avoids reporting it as generic free text, which
    would suggest remedies that do not apply to a date.

    High-cardinality string columns are labelled `FeatureModality.TEXT` by
    `detect_feature_modalities` and are then swept into the same `OrdinalEncoder` as
    real categoricals, which selects columns by dtype (see `get_ordinal_encoder`).
    That turns near-unique text into near-unique integer codes, i.e. noise rather
    than signal, without any error to hint at it. A date column is excluded from
    this second warning since it was already reported above.

    Called by `detect_feature_modalities` while the schema still carries the TEXT
    labels, i.e. before the first preprocessing step that rebuilds it, since
    `FeatureSchema.from_only_categorical_indices` collapses TEXT into NUMERICAL.

    Args:
        feature_schema: The schema produced by `detect_feature_modalities`.
        date_columns: Names of columns recognized as date-like.
        declared_categorical_indices: Positional indices the caller passed as
            `categorical_features_indices`. These are never reported: declaring a
            column categorical states that the user already knows it holds
            non-numeric values and intends them as categories, so warning about it
            would be noise.
    """
    if date_columns:
        warnings.warn(
            f"These columns hold dates, which are not yet expanded into calendar "
            f"features, so they are read as plain categories or text: "
            f"{_format_names_for_warning(date_columns)}.",
            UserWarning,
            stacklevel=6,
        )

    declared = set(declared_categorical_indices or ())
    reported = set(date_columns)
    text_names = [
        name
        for index, feature in enumerate(feature_schema.features)
        if feature.modality is FeatureModality.TEXT
        and index not in declared
        and (name := feature.name.removeprefix(INPUT_FEATURE_PREFIX)) not in reported
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
    min_cardinality_for_text: int,
    big_enough_n_to_infer_cat: bool,
) -> tuple[FeatureModality, bool]:
    """Decide a single column's modality.

    Returns:
        The modality, and whether the column was recognized as a date. Nothing
        expands a date into calendar features yet, so this is always
        `CATEGORICAL`/`TEXT` in practice, exactly like a non-date string would
        get; the caller only uses the flag to warn about it distinctly from
        plain free text.
    """
    # Early exit for the distinct count: counts only grow with the number of
    # rows scanned, and every comparison below is against a threshold of at
    # most max(max_unique_for_category, min_unique_for_numerical,
    # min_cardinality_for_text). So once a small prefix of the column already
    # exceeds all of them, every decision equals the full-column one and
    # scanning the remaining rows is skipped — the common case for continuous
    # columns. Low-cardinality columns fall through to the exact full count,
    # paying one extra prefix pass. `min_cardinality_for_text` has to be
    # included here too: `_classify_string_like_column` compares against it as
    # well, and it can be set above the other two, in which case a prefix
    # count that clears only those two would stop early on an undercount and
    # land a long TEXT column on CATEGORICAL instead.
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
        # Either all values are missing, or all values are the same.
        # If there's a single value but also missing ones, it's not constant.
        # Columns the user explicitly declared categorical are kept categorical
        # (handled below) even when all-missing, so they route through the ordinal
        # encoder consistently between fit and predict instead of being treated as
        # a constant numeric column that crashes on string values seen at predict.
        return FeatureModality.CONSTANT, False

    if _is_numeric_pandas_series(s):
        if _detect_numeric_as_categorical(
            n_unique=n_unique,
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        ):
            return FeatureModality.CATEGORICAL, False
        return FeatureModality.NUMERICAL, False

    is_string_like = pd.api.types.is_string_dtype(s.dtype) or isinstance(
        s.dtype, pd.CategoricalDtype
    )
    if is_string_like:
        return _classify_string_like_column(
            s,
            n_unique=n_unique,
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            min_cardinality_for_text=min_cardinality_for_text,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        )
    raise TabPFNUserError(
        f"Unknown dtype: {s.dtype}, with {s.nunique(dropna=False)} unique values"
    )


def _classify_string_like_column(
    s: pd.Series,
    *,
    n_unique: int,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    min_cardinality_for_text: int,
    big_enough_n_to_infer_cat: bool,
) -> tuple[FeatureModality, bool]:
    """Classify a string/categorical-dtype column as CATEGORICAL or TEXT.

    A date-like column is recognized here too (see `_is_date_like_pandas_series`),
    but nothing expands a date into calendar features yet, so it is always
    demoted to whichever of CATEGORICAL/TEXT its cardinality implies -- the
    only difference recognizing it makes right now is that the caller can warn
    about it distinctly, instead of reporting it as plain free text.

    Two independent cardinality decisions are in play: `max_unique_for_category`/
    `min_unique_for_numerical` (shared with the numeric branch, via
    `_detect_numeric_as_categorical`) decide whether a low-cardinality date is
    just a plain category, the same call a low-cardinality *number* already
    gets; `min_cardinality_for_text` separately decides, for whatever is left a
    plain string, whether it is a category or text. A number being few enough
    to be a category says nothing about how much variety makes a string text,
    so the two thresholds are free to move independently.

    Returns the same `(modality, is_date)` pair as `_detect_feature_modality`.
    """
    is_date = _is_date_like_pandas_series(s)
    if is_date and _detect_numeric_as_categorical(
        n_unique=n_unique,
        reported_categorical=reported_categorical,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
    ):
        # Checked before the cardinality split below, so a low-cardinality date
        # (e.g. a handful of distinct quarter-end dates) gets exactly the
        # treatment a low-cardinality number already gets, rather than being
        # decided by the text threshold below.
        return FeatureModality.CATEGORICAL, True

    modality = (
        FeatureModality.CATEGORICAL
        if n_unique <= min_cardinality_for_text
        else FeatureModality.TEXT
    )
    return modality, is_date


def _is_numeric_pandas_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return True
    if PANDAS_BELOW_3:
        return all(_is_numeric_or_missing_for_old_pandas(value) for value in s)
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


def _is_date_like_pandas_series(s: pd.Series) -> bool:
    """Whether every non-null value in `s` parses as a date.

    All-or-nothing, the same style as `_is_numeric_pandas_series`: a column that is
    mostly dates but has a few non-date values is left as a string rather than
    guessed at. Only reached for a column that already failed the numeric check, so
    a column of plain numbers (including one that could also be read as a date,
    e.g. an 8-digit `"20240101"`) is never reclassified here.
    """
    non_null = s.dropna()
    if non_null.empty:
        return False
    try:
        with warnings.catch_warnings():
            # A mixed-format column makes `to_datetime` fall back to parsing one
            # value at a time and warn about it; this is only a probe, so the
            # warning would be noise for the caller regardless of the outcome.
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(non_null, errors="coerce")
    except (TypeError, ValueError):
        return False
    return bool(parsed.notna().all())


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
