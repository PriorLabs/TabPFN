"""Module for cleaning the data.

These cleaning steps are performed before further preprocessing,
e.g. NaN mapping and dtype conversion.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabpfn.constants import NA_PLACEHOLDER
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from sklearn.compose import ColumnTransformer

    from tabpfn.preprocessing.torch import FeatureSchema

# https://numpy.org/doc/2.1/reference/arrays.dtypes.html#checking-the-data-type

NUMERIC_DTYPE_KINDS = "?bBiufm"
OBJECT_DTYPE_KINDS = "OV"
STRING_DTYPE_KINDS = "SaU"
UNSUPPORTED_DTYPE_KINDS = "cM"  # Not needed, just for completeness


def clean_data(
    X: np.ndarray,
    feature_schema: FeatureSchema,
) -> tuple[np.ndarray, ColumnTransformer, FeatureSchema]:
    """Clean the data by converting dtypes and ordinally encoding categorical columns.

    Args:
        X: The data to clean.
        feature_schema: The feature schema corresponding to the data.

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
        X=X_pandas, ord_encoder=ord_encoder, fit_encoder=True
    )

    return X_numpy, ord_encoder, feature_schema


def fix_dtypes(  # noqa: D103
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

    numerical_columns = X.select_dtypes(include=["number"]).columns
    if len(numerical_columns) > 0:
        X[numerical_columns] = X[numerical_columns].astype(numeric_dtype)
    return X


def process_text_na_dataframe(
    X: pd.DataFrame,
    placeholder: str = NA_PLACEHOLDER,
    ord_encoder: ColumnTransformer | None = None,
    *,
    fit_encoder: bool = False,
) -> np.ndarray:
    """Convert `X` to float64, replacing NA with NaN in string cells.

    If `ord_encoder` is not None, then it will be used to encode `X` before the
    conversion to float64.

    Note that this function sometimes mutates its input.
    """
    # TODO: Check if this step needs to be done as early as it is done here, or whether
    # it can be done later and include it in a main preprocessor object.

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
    return typing.cast("np.ndarray", X_encoded.astype(np.float64))
