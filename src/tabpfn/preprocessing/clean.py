#  Copyright (c) Prior Labs GmbH 2026.

"""Module for cleaning the data.

These cleaning steps are performed before further preprocessing,
e.g. NaN mapping and dtype conversion.
"""

from __future__ import annotations

import typing
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabpfn.constants import NA_PLACEHOLDER
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from tabpfn.preprocessing.steps.preprocessing_helpers import (
        OrderPreservingColumnTransformer,
    )
    from tabpfn.preprocessing.torch import FeatureSchema

# https://numpy.org/doc/2.1/reference/arrays.dtypes.html#checking-the-data-type

NUMERIC_DTYPE_KINDS = "?bBiufm"
OBJECT_DTYPE_KINDS = "OV"
STRING_DTYPE_KINDS = "SaU"
UNSUPPORTED_DTYPE_KINDS = "cM"  # Not needed, just for completeness


def clean_data(
    X: np.ndarray,
    feature_schema: FeatureSchema,
    *,
    passthrough_inf: bool = False,
) -> tuple[np.ndarray, OrderPreservingColumnTransformer, FeatureSchema]:
    """Clean the data by converting dtypes and ordinally encoding categorical columns.

    Args:
        X: The data to clean.
        feature_schema: The feature schema corresponding to the data.
        passthrough_inf: If True, +/-inf values are carried through the ordinal
            encoding stage unchanged instead of crashing it (see
            `process_text_na_dataframe`).

    Returns:
        A tuple containing the cleaned data, the ordinal encoder, and the inferred
        feature modalities.
    """
    # Will convert inferred categorical indices to category dtype,
    # to be picked up by the ord_encoder, as well
    # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
    X_pandas: pd.DataFrame = fix_dtypes(
        X=X,
        cat_indices=feature_schema.indices_for(FeatureModality.CATEGORICAL),
    )

    # Ensure categories are ordinally encoded
    ord_encoder = get_ordinal_encoder()
    X_numpy = process_text_na_dataframe(
        X=X_pandas,
        ord_encoder=ord_encoder,
        fit_encoder=True,
        passthrough_inf=passthrough_inf,
    )

    return X_numpy, ord_encoder, feature_schema


def coerce_nullable_dtypes_to_numpy(X: pd.DataFrame) -> pd.DataFrame:
    """Convert numpy/nullable boolean and nullable numeric columns to float64.

    Runs *before* sklearn's ``validate_data``. Any boolean column (numpy ``bool`` or
    nullable ``boolean``) and any nullable numeric extension dtype
    (``Int64``/``Float64``) makes sklearn's ``check_array`` perform a whole-frame
    ``astype`` even with ``dtype=None``, which crashes when another column is a
    string-valued category (it cannot cast e.g. ``'0e63c0f0'`` to float). Coercing
    these columns up front removes that trigger.

    ``category``/``string``/``object`` columns are left untouched.
    """
    cols = [
        col
        for col, dtype in X.dtypes.items()
        if pd.api.types.is_bool_dtype(dtype)
        or (pd.api.types.is_extension_array_dtype(dtype) and dtype.kind in "iuf")
    ]
    if cols:
        X = X.copy()
        X[cols] = X[cols].astype("float64")
    return X


def fix_dtypes(  # noqa: D103, C901, PLR0912
    X: pd.DataFrame | np.ndarray,
    cat_indices: Sequence[int | str] | None,
    numeric_dtype: Literal["float32", "float64"] = "float64",
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        # This will help us get better dtype inference later
        convert_dtype = True
    elif isinstance(X, np.ndarray):
        if X.dtype.kind in NUMERIC_DTYPE_KINDS:
            # It's a numeric type, just wrap the array in pandas with the correct dtype
            X = pd.DataFrame(X, copy=False, dtype=numeric_dtype)
            convert_dtype = False
        elif X.dtype.kind in OBJECT_DTYPE_KINDS:
            # If numpy and object dtype, we rely on pandas to handle introspection
            # of columns and rows to determine the dtypes.
            X = pd.DataFrame(X, copy=True)
            convert_dtype = True
        elif X.dtype.kind in STRING_DTYPE_KINDS:
            raise ValueError(
                f"String dtypes are not supported. Got dtype: {X.dtype}",
            )
        else:
            raise ValueError(f"Invalid dtype for X: {X.dtype}")
    else:
        raise ValueError(f"Invalid type for X: {type(X)}")

    if cat_indices is not None:
        # So annoyingly, things like AutoML Benchmark may sometimes provide
        # numeric indices for categoricals, while providing named columns in the
        # dataframe. Equally, dataframes loaded from something like a csv may just have
        # integer column names, and so it makes sense to access them just like you would
        # string columns.
        # Hence, we check if the types match and decide whether to use `iloc` to select
        # columns, or use the indices as column names...
        is_numeric_indices = all(isinstance(i, (int, np.integer)) for i in cat_indices)
        columns_are_numeric = all(
            isinstance(col, (int, np.integer)) for col in X.columns.tolist()
        )
        use_col_names = is_numeric_indices and not columns_are_numeric
        if use_col_names:
            cat_col_names = [X.columns[i] for i in cat_indices]
            X[cat_col_names] = X[cat_col_names].astype("category")
        else:
            X[cat_indices] = X[cat_indices].astype("category")

    # Alright, pandas can have a few things go wrong.
    #
    # 1. Of course, object dtypes, `convert_dtypes()` will handle this for us if
    #   possible. This will raise later if can't convert.
    # 2. String dtypes can still exist, OrdinalEncoder will do something but
    #   it's not ideal. We should probably check unique counts at the expense of doing
    #   so.
    # 3. For all dtypes relating to timeseries and other _exotic_ types not supported by
    #   numpy, we leave them be and let the pipeline error out where it will.
    # 4. Pandas will convert dtypes to Int64Dtype/Float64Dtype, which include
    #   `pd.NA`. Sklearn's Ordinal encoder treats this differently than `np.nan`.
    #   We can fix this one by converting all numeric columns to float64, which uses
    #   `np.nan` instead of `pd.NA`.
    #
    if convert_dtype:
        X = X.convert_dtypes()
        # Columns still `object` after convert_dtypes (e.g. all-missing columns) are
        # typed as `string` so the ordinal encoder's dtype-based column selection is
        # consistent between fit and predict. Otherwise an all-missing column is
        # `object` at fit (-> passthrough) but `string` at predict; the frozen
        # passthrough then lets raw strings reach the float cast below and crash.
        object_columns = X.select_dtypes(include=["object"]).columns
        if len(object_columns) > 0:
            X[object_columns] = X[object_columns].astype("string")

    numerical_columns = X.select_dtypes(include=["number"]).columns
    if len(numerical_columns) > 0:
        X[numerical_columns] = X[numerical_columns].astype(numeric_dtype)
    return X


def _column_kind(dtype: Any) -> str:
    """Return a column's scalar dtype kind, unwrapping categorical dtypes."""
    if isinstance(dtype, pd.CategoricalDtype):
        return dtype.categories.dtype.kind
    return dtype.kind


def _align_columns_to_fitted_dtypes(
    X: pd.DataFrame, ord_encoder: OrderPreservingColumnTransformer
) -> pd.DataFrame:
    """Coerce each encoded column to the scalar dtype it had when the encoder was fit.

    Only the dtypes seen at fit are authoritative: the frozen ``OrdinalEncoder`` stored
    its ``categories_`` (and their dtype) at fit, so an incoming column is interpreted
    as that fit-time dtype at predict. Two mismatches are handled:

    * string at fit, numeric at predict -> the column is cast to ``string``. Otherwise
      sklearn's ``_check_unknown`` takes its numeric branch and compares float values
      against the string ``categories_``, raising a ``TypeError``.
    * numeric at fit, string at predict -> the column is cast to numeric via
      ``pd.to_numeric(..., errors="coerce")``. Numeric-looking strings match their fit
      category; non-numeric strings become ``NaN`` (treated as missing).

    Either way, values that do not match a fit category map to the encoder's unknown
    code. A dtype change between fit and predict usually signals an inconsistent feature
    pipeline, so we warn.
    """
    encoder = ord_encoder.named_transformers_.get("encoder")
    if encoder is None or not hasattr(encoder, "categories_"):
        return X
    selected = next(
        (cols for name, _, cols in ord_encoder.transformers_ if name == "encoder"),
        [],
    )
    to_string, to_numeric = [], []
    for col, categories in zip(selected, encoder.categories_, strict=True):
        fit_kind = categories.dtype.kind
        values_kind = _column_kind(X[col].dtype)
        if fit_kind in "OUS" and values_kind in "iufcb":
            to_string.append(col)
        elif fit_kind in "iuf" and values_kind in "OUS":
            to_numeric.append(col)

    if not to_string and not to_numeric:
        return X

    warnings.warn(
        f"Column(s) {to_string + to_numeric} have a dtype at predict time that differs "
        f"from fit time; only the fit-time dtype is treated as correct, so they are "
        f"coerced to it and values that don't match a fitted category are treated as "
        f"unseen or missing. This usually indicates an inconsistent feature pipeline "
        f"between fit and predict.",
        stacklevel=2,
    )
    X = X.copy()
    if to_string:
        X[to_string] = X[to_string].astype("string")
    for col in to_numeric:
        X[col] = pd.to_numeric(X[col].astype("object"), errors="coerce")
    return X


def process_text_na_dataframe(
    X: pd.DataFrame,
    placeholder: str = NA_PLACEHOLDER,
    ord_encoder: OrderPreservingColumnTransformer | None = None,
    *,
    fit_encoder: bool = False,
    passthrough_inf: bool = False,
) -> np.ndarray:
    """Convert `X` to float64, replacing NA with NaN in string cells.

    If `ord_encoder` is not None, then it will be used to encode `X` before the
    conversion to float64.

    If `passthrough_inf` is True, +/-inf in numeric columns would otherwise crash
    the ordinal encoder, so they are replaced with NaN before encoding and written
    back into the output at their original positions afterwards. The output columns
    align positionally with `X`'s columns, so the recorded positions stay valid.

    Note that this function sometimes mutates its input.
    """
    # TODO: Check if this step needs to be done as early as it is done here, or whether
    # it can be done later and include it in a main preprocessor object.

    # Record +/-inf positions (numeric columns only) and replace them with NaN so the
    # ordinal encoder doesn't crash; they are restored into the output further below.
    pos_inf = neg_inf = None

    if passthrough_inf:
        # use pandas to compute the mask in order to handle non-numeric dtypes
        # with infinite values:
        pos_inf = X == np.inf  # noqa: SIM300
        neg_inf = X == -np.inf  # noqa: SIM300
        # coerce columns to NaN:
        X[neg_inf | pos_inf] = np.nan
        # Coerce to a plain boolean array: comparing a `string` column yields a
        # nullable `boolean` mask, which would otherwise make `to_numpy()` return
        # an `object` array that cannot be used to index `X_encoded`. NA entries
        # (from string columns, which never hold true infinities) become False.
        pos_inf.fillna(value=False, inplace=True)  # noqa: PD002
        neg_inf.fillna(value=False, inplace=True)  # noqa: PD002
        pos_inf = pos_inf.to_numpy(dtype=bool)
        neg_inf = neg_inf.to_numpy(dtype=bool)

    # When transforming with a fitted encoder, coerce columns whose dtype drifted
    # between fit and predict back to their fit-time dtype, so the OrdinalEncoder is
    # consistent and does not crash. This must run before `string_cols` is computed so
    # the coerced columns get NA handling.
    if not fit_encoder and ord_encoder is not None:
        X = _align_columns_to_fitted_dtypes(X, ord_encoder)

    # Replace NAN values in X, for dtypes, which the OrdinalEncoder cannot handle
    # with placeholder NAN value. Later placeholder NAN values are transformed to np.nan
    string_cols = X.select_dtypes(include=["string", "object"]).columns
    if len(string_cols) > 0:
        X[string_cols] = X[string_cols].fillna(placeholder)

    if fit_encoder and ord_encoder is not None:
        X_encoded = ord_encoder.fit_transform(X)
    elif ord_encoder is not None:
        X_encoded = ord_encoder.transform(X)
    else:
        X_encoded = X.to_numpy()

    string_cols_ix = [X.columns.get_loc(col) for col in string_cols]
    placeholder_mask = X[string_cols] == placeholder
    X_encoded[:, string_cols_ix] = np.where(
        placeholder_mask,
        np.nan,
        X_encoded[:, string_cols_ix],
    )
    X_encoded = X_encoded.astype(np.float64)

    # Write the recorded +/-inf values back into their original numeric cells.
    if passthrough_inf and (pos_inf.any() or neg_inf.any()):
        X_encoded[pos_inf] = np.inf
        X_encoded[neg_inf] = -np.inf

    return typing.cast("np.ndarray", X_encoded)
