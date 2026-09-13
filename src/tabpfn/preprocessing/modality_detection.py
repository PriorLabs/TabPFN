#  Copyright (c) Prior Labs GmbH 2026.

"""Module to infer feature modalities: numerical, categorical, text, etc."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

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
    """Infer each feature's modality, using heuristics and declared categoricals.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer feature modalities from.
        feature_names: The names of the features.
        provided_categorical_indices: User-provided indices considered categorical.
            A string column among them is `CATEGORICAL` at any cardinality, never
            `TEXT`; a numeric one is still subject to `max_unique_for_category`.
        min_samples_for_inference: Minimum samples required to auto-infer a
            feature not provided as categorical.
        max_unique_for_category: Max unique values for a feature to be categorical.
        min_unique_for_numerical: Min unique values for a feature to be numerical.
        min_cardinality_for_text: Unique-value count above which an undeclared
            string column (not parsed as a number) is `TEXT` rather than
            `CATEGORICAL` -- independent of the two thresholds above.

    Returns:
        The inferred `FeatureSchema`.
    """
    features: list[Feature] = []
    big_enough_n_to_infer_cat = len(X) > min_samples_for_inference
    unique_feature_names = build_input_feature_names(feature_names, X.shape[1])
    provided = set(provided_categorical_indices or ())
    decided_at = _decided_at(
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        min_cardinality_for_text=min_cardinality_for_text,
    )
    # A numeric array needs no per-column parsing: every column is numeric, so only
    # the distinct-value count decides, and that is counted for all columns at once.
    n_unique_per_column = _numeric_n_unique_per_column(X, decided_at=decided_at)
    for i, index in enumerate(range(X.shape[1])):
        feature_name = unique_feature_names[i]
        reported_categorical = index in provided
        if n_unique_per_column is not None:
            feat_modality = _numeric_modality(
                n_unique=int(n_unique_per_column[index]),
                reported_categorical=reported_categorical,
                max_unique_for_category=max_unique_for_category,
                min_unique_for_numerical=min_unique_for_numerical,
                big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
            )
        else:
            feat_modality = _detect_feature_modality(
                s=pd.Series(X[:, index], name=feature_name),
                reported_categorical=reported_categorical,
                max_unique_for_category=max_unique_for_category,
                min_unique_for_numerical=min_unique_for_numerical,
                min_cardinality_for_text=min_cardinality_for_text,
                big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
            )
        features.append(Feature(name=feature_name, modality=feat_modality))
    feature_schema = FeatureSchema(features=features)
    _warn_on_text(feature_schema)
    return feature_schema


def _format_names_for_warning(names: list[str]) -> str:
    """Render column names for a warning, capped so it stays readable."""
    shown = names[:_MAX_TEXT_COLUMNS_IN_WARNING]
    printed = ", ".join(repr(name) for name in shown)
    if len(names) > len(shown):
        printed += f" (and {len(names) - len(shown)} more)"
    return printed


def _warn_on_text(feature_schema: FeatureSchema) -> None:
    """Warn about any free-text columns.

    A column declared categorical is never `TEXT`, so it never shows up here.
    """
    text_names = [
        feature.name.removeprefix(INPUT_FEATURE_PREFIX)
        for feature in feature_schema.features
        if feature.modality is FeatureModality.TEXT
    ]
    if not text_names:
        return

    warnings.warn(
        f"These columns look like free text and are being ordinal-encoded as "
        f"high-cardinality categoricals, which usually adds noise rather than "
        f"signal: {_format_names_for_warning(text_names)}.\n"
        "If such a column holds numbers stored as strings, convert it to a numeric "
        "dtype. If it is a category rather than text, pass its index in "
        "`categorical_features_indices` or give it pandas' `category` dtype, and it "
        "is read as a categorical whatever its cardinality; or raise "
        '`inference_config={"MIN_CARDINALITY_FOR_TEXT": ...}` above its number of '
        "distinct values. If it holds genuine text, give it pandas' `string` dtype "
        'and set `inference_config={"TRANSFORM_TEXT": True}` to expand it into '
        "numeric features, or consider the tabpfn-client API, which embeds text "
        "natively: https://github.com/PriorLabs/tabpfn-client",
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
    assert not isinstance(s.dtype, pd.CategoricalDtype), (
        "Categorical dtype must be converted before modality detection; "
        "preserve its intent in provided_categorical_indices."
    )
    decided_at = _decided_at(
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        min_cardinality_for_text=min_cardinality_for_text,
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
        return _numeric_modality(
            n_unique=n_unique,
            reported_categorical=reported_categorical,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
        )

    # A pandas `category` column never arrives here as such: `X` is a numpy array
    # by now, and its intent travels in `provided_categorical_indices` instead.
    if pd.api.types.is_string_dtype(s.dtype):
        # A declared categorical is taken at face value: the cardinality cutoff
        # only sorts undeclared string columns into category or text.
        if reported_categorical or n_unique <= min_cardinality_for_text:
            return FeatureModality.CATEGORICAL
        return FeatureModality.TEXT
    raise TabPFNUserError(
        f"Unknown dtype: {s.dtype}, with {s.nunique(dropna=False)} unique values"
    )


def _decided_at(
    *,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    min_cardinality_for_text: int,
) -> int:
    """The distinct-value count at which every threshold below is cleared.

    Once a prefix of a column already holds this many distinct values, the full count
    would land in the same bucket, so the rest of the column need not be scanned.
    `min_cardinality_for_text` is included since it can exceed the other two.
    """
    return (
        max(
            max_unique_for_category,
            min_unique_for_numerical,
            min_cardinality_for_text,
            1,
        )
        + 1
    )


def _numeric_modality(
    *,
    n_unique: int,
    reported_categorical: bool,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    big_enough_n_to_infer_cat: bool,
) -> FeatureModality:
    """The modality of a numeric column with `n_unique` distinct values (NaN counted).

    A constant (or all-missing) column is `CONSTANT` unless declared categorical, so
    that it still routes through the ordinal encoder instead of crashing as a
    constant numeric column when predict sees an unseen value.
    """
    if n_unique <= 1 and not reported_categorical:
        return FeatureModality.CONSTANT
    if _detect_numeric_as_categorical(
        n_unique=n_unique,
        reported_categorical=reported_categorical,
        max_unique_for_category=max_unique_for_category,
        min_unique_for_numerical=min_unique_for_numerical,
        big_enough_n_to_infer_cat=big_enough_n_to_infer_cat,
    ):
        return FeatureModality.CATEGORICAL
    return FeatureModality.NUMERICAL


def _numeric_n_unique_per_column(
    X: np.ndarray, *, decided_at: int
) -> np.ndarray | None:
    """Distinct values per column of a numeric or bool array, NaN counted as a value.

    `None` for anything else (an object array is parsed column by column). Mirrors
    the per-column early exit: a column whose first `_EARLY_EXIT_PREFIX_ROWS` rows
    already hold `decided_at` distinct values keeps that prefix count, which lands
    in the same bucket as the full count; only the other columns are counted in
    full.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype.kind not in "biuf":
        return None
    n_rows, n_columns = X.shape
    if n_rows == 0:
        return np.zeros(n_columns, dtype=np.int64)
    if n_rows <= _EARLY_EXIT_PREFIX_ROWS:
        return _count_distinct_per_column(X)
    n_unique = _count_distinct_per_column(X[:_EARLY_EXIT_PREFIX_ROWS])
    undecided = np.flatnonzero(n_unique < decided_at)
    if len(undecided):
        n_unique[undecided] = _count_distinct_per_column(X[:, undecided])
    return n_unique


def _count_distinct_per_column(X: np.ndarray) -> np.ndarray:
    """`pd.Series(column).nunique(dropna=False)` per column of a numeric or bool array.

    A bool column holds one or two distinct values, told apart by `any` and `all`.
    For the other dtypes, sorting puts equal values next to each other and NaN
    last, so the count is one plus the number of adjacent unequal pairs, with NaN
    counted once when present. `-0.0` equals `0.0` and `inf` equals `inf` here as
    under `nunique`.
    """
    if X.dtype.kind == "b":
        return 1 + (X.any(axis=0) & ~X.all(axis=0)).astype(np.int64)
    values = np.sort(X, axis=0)
    if values.dtype.kind == "f":
        missing = np.isnan(values)
        differs = (values[1:] != values[:-1]) & ~missing[1:]
        return (
            differs.sum(axis=0)
            + (~missing).any(axis=0).astype(np.int64)
            + missing.any(axis=0).astype(np.int64)
        )
    return (values[1:] != values[:-1]).sum(axis=0) + 1


#: `pd.api.types.infer_dtype` kinds whose every non-missing value is a number. A
#: `string` or `mixed` column is not settled by them: a spelled-out number counts too.
_INFERRED_NUMERIC_KINDS = frozenset(
    {"integer", "floating", "mixed-integer-float", "boolean", "decimal", "empty"}
)


def _is_numeric_pandas_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s.dtype):
        return True
    # A numeric column stored as object is the common case: a frame with one
    # non-numeric column arrives as a single object array. `infer_dtype` settles it in
    # C rather than a Python-level walk over every value.
    if pd.api.types.infer_dtype(s, skipna=True) in _INFERRED_NUMERIC_KINDS:
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
