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
from packaging.version import Version

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
# The subset of the above that numpy casts to float64 the same way pandas does, so
# a frame need not be built to convert it. Timedeltas ("m") are excluded: pandas
# converts those through its own units rather than numpy's raw integers.
FAST_CONVERTIBLE_DTYPE_KINDS = "?bBiuf"
OBJECT_DTYPE_KINDS = "OV"
STRING_DTYPE_KINDS = "SaU"
UNSUPPORTED_DTYPE_KINDS = "cM"  # Not needed, just for completeness
PANDAS_FASTER_THAN_MIXED_PATH = Version(pd.__version__) < Version("3.0.0")
# Before 3.0 `astype` copies every column by default, including the ones it is not
# casting; from 3.0 copy-on-write makes the keyword a no-op and passing it warns.
_ASTYPE_KEEPS_UNCAST_COLUMNS = (
    {"copy": False} if Version(pd.__version__) < Version("3.0.0") else {}
)

_FLOAT64 = np.dtype(np.float64)


def _cast_columns_share_a_block(
    X: pd.DataFrame,
    columns: pd.Index | Sequence[Any],
) -> bool:
    """Whether any of `columns` sits in a block that holds more than one column.

    That would make assigning a cast back expensive: the block a column is deleted
    from is rebuilt whole.
    Since columns are assigned one at a time on pandas < 3, casting `c` columns
    out of a block costs `c(c+1)/2` column copies.
    """
    try:
        blocks = X._mgr.blocks
        columns_in_block = np.zeros(X.shape[1], dtype=np.intp)
        for block in blocks:
            positions = block.mgr_locs.as_array
            columns_in_block[positions] = len(positions)
        cast_positions = X.columns.get_indexer_for(columns)
    except (AttributeError, TypeError, ValueError, IndexError):
        return True
    return (
        len(cast_positions) == 0
        or (cast_positions < 0).any().item()
        or (columns_in_block[cast_positions] > 1).any().item()
    )


def _cast_columns(
    X: pd.DataFrame,
    columns: pd.Index | Sequence[Any],
    dtype: Any,
) -> pd.DataFrame:
    """Efficiently cast `columns` in `X` to `dtype`."""
    if len(columns) == 0:
        return X

    if not _cast_columns_share_a_block(X, columns) and not X.columns.has_duplicates:
        # this path uses less memory when available
        # Copied shallowly to not copy the columns themselves:
        X = X.copy(deep=False)
        X[columns] = X[columns].astype(dtype)
        return X

    # NOTE: this path is there for pandas < 3 compatibility

    # fallback: never costly in time
    # cast only the columns that need to be:
    return X.astype(dict.fromkeys(columns, dtype), **_ASTYPE_KEEPS_UNCAST_COLUMNS)


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
    cat_indices = feature_schema.indices_for(FeatureModality.CATEGORICAL)

    # Ensure categories are ordinally encoded
    ord_encoder = get_ordinal_encoder()

    if (
        not cat_indices
        and isinstance(X, np.ndarray)
        and X.dtype.kind in FAST_CONVERTIBLE_DTYPE_KINDS
    ):
        # Nothing to encode and no dtype to infer, so the two steps below come out
        # as a single cast: `fix_dtypes` would wrap `X` in a float64 frame that
        # `process_text_na_dataframe` then copies straight back out, holding two
        # full-size float64 buffers to produce one. Convert once, into the array
        # that is returned.
        #
        # `passthrough_inf` makes no difference here: it records the +/-inf cells,
        # NaNs them so the encoder does not choke, and writes them back at the same
        # positions afterwards -- an exact round trip when nothing is encoded.
        #
        # The encoder is still fit, since the caller keeps it for predict, but on a
        # single row: with no column selected it learns nothing from the values,
        # only the column bookkeeping.
        ord_encoder.fit(fix_dtypes(X=X[:1], cat_indices=cat_indices))
        return (
            np.array(X, dtype=np.float64, order="F", copy=True),
            ord_encoder,
            feature_schema,
        )

    # Will convert inferred categorical indices to category dtype,
    # to be picked up by the ord_encoder, as well
    # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
    X_pandas: pd.DataFrame = fix_dtypes(X=X, cat_indices=cat_indices)

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
    return _cast_columns(X, cols, "float64")


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
            X = _cast_columns(X, cat_col_names, "category")
        else:
            X = _cast_columns(X, cat_indices, "category")

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
        X = _cast_columns(X, object_columns, "string")

    numerical_columns = X.select_dtypes(include=["number"]).columns
    # Assigning the numeric columns back is not free even when the cast is a no-op:
    # it rewrites them as one block per column, and a fragmented frame has to be
    # re-materialised (a full extra copy) by every later `to_numpy`. Skip it when
    # they already hold the target dtype -- the common case for a numeric ndarray
    # input, whose DataFrame was constructed with `numeric_dtype` above.
    if (
        len(numerical_columns) > 0
        and not (X.dtypes[numerical_columns] == np.dtype(numeric_dtype)).all()
    ):
        X = _cast_columns(X, numerical_columns, numeric_dtype)
    return X


def _column_kind(dtype: Any) -> str:
    """Return a column's scalar dtype kind, unwrapping categorical dtypes."""
    if isinstance(dtype, pd.CategoricalDtype):
        return dtype.categories.dtype.kind
    return dtype.kind


def _encoder_selection(ord_encoder: OrderPreservingColumnTransformer) -> list[Any]:
    """The columns a fitted encoder takes, in the order it holds them."""
    return next(
        (cols for name, _, cols in ord_encoder.transformers_ if name == "encoder"),
        [],
    )


def _is_single_float_block(X: pd.DataFrame) -> bool:
    """True if ``X`` is backed by a single contiguous numpy float block.

    For such frames pandas' vectorized ``X == inf`` runs directly on the one block
    and ``to_numpy()`` returns a view, which is faster than extracting the numeric
    columns into a fresh array (the per-block path below). A column-fragmented
    frame -- e.g. what ``fix_dtypes`` produces via per-column ``astype`` -- has many
    blocks and does not qualify. Defensively returns ``False`` if pandas' block
    internals are unavailable, falling back to the per-block path.
    """
    blocks = getattr(getattr(X, "_mgr", None), "blocks", ())
    # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
    return len(blocks) == 1 and blocks[0].values.dtype.kind == "f"  # noqa: PD011


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
    selected = _encoder_selection(ord_encoder)
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


def _inf_masks_pandas_only(
    X: pd.DataFrame,
    *,
    numeric_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Pandas inf detection.

    Generic but slow with both numeric and object dtypes, fastest in the
    all-numeric case.

    Args:
        X (pd.DataFrame): DataFrame to check for +/-infs.
        numeric_only (Any, optional): If True, skips checks for boolean mask NaNs.

    Returns:
        pos_inf, neg_inf: Boolean masks.
    """
    kwargs = {} if numeric_only else {"na_value": False}
    pos_inf = (X == np.inf).to_numpy(dtype=bool, **kwargs)  # noqa: SIM300
    neg_inf = (X == -np.inf).to_numpy(dtype=bool, **kwargs)  # noqa: SIM300
    return pos_inf, neg_inf


def numeric_columns(X: pd.DataFrame) -> np.ndarray:
    """Computes a mask for the numeric columns of a DataFrame."""
    return np.array(
        [pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes],
        dtype=bool,
    )


def _inf_masks_numpy_numeric_(
    X: pd.DataFrame,
    numeric_col_mask: np.ndarray,
    pos_inf: np.ndarray,
    neg_inf: np.ndarray,
) -> None:
    """Computes infinite masks for dataframes, with a fast numpy path for
    numeric columns.

    Args:
        X (pd.DataFrame): DataFrame to check for +/-infs.
        numeric_col_mask (np.ndarray): Numeric columns of X.
        pos_inf (np.ndarray): Boolean mask, modified in-place.
        neg_inf (np.ndarray): Boolean mask, modified in-place.
    """
    numeric_values = X.iloc[:, numeric_col_mask].to_numpy(dtype=np.float64)
    pos_inf[:, numeric_col_mask] = numeric_values == np.inf
    neg_inf[:, numeric_col_mask] = numeric_values == -np.inf


def _inf_masks_mixed(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Computes infinite masks for dataframes, with a fast numpy path for
    numeric columns.

    Args:
        X (pd.DataFrame): DataFrame to check for +/-infs.

    Returns:
        pos_inf, neg_inf: Boolean masks.
    """
    # Per-block path for fragmented / mixed frames. Numeric columns are the
    # common case and the element-wise pandas comparison over a fragmented
    # frame is slow, so test them directly with numpy and fall back to pandas
    # only for the (rare) non-numeric columns that may still hold python
    # float infinities.
    pos_inf = np.zeros(X.shape, dtype=bool)
    neg_inf = np.zeros(X.shape, dtype=bool)

    numeric_col_mask = numeric_columns(X)

    # Fast numpy path for numeric columns. `to_numpy(dtype=float64)` coerces
    # any nullable NA to NaN, which never matches +/-inf, so masks stay correct.
    if numeric_col_mask.any():
        _inf_masks_numpy_numeric_(X, numeric_col_mask, pos_inf, neg_inf)

    # Slow pandas path for the remaining (non-numeric) columns. Comparing a
    # `string` column yields a nullable `boolean` mask, so coerce to a plain
    # bool array; NA entries (never true infinities) become False.
    non_numeric_col_mask = ~numeric_col_mask
    if non_numeric_col_mask.any():
        other = X.iloc[:, non_numeric_col_mask]
        pos_inf[:, non_numeric_col_mask], neg_inf[:, non_numeric_col_mask] = (
            _inf_masks_pandas_only(other)
        )
    return pos_inf, neg_inf


def _inf_masks_dataframe(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Computes infinite masks for dataframes, with a fast path for numeric dtypes.

    Slower than `_inf_masks_pandas_only` on pandas < 3.0.0.
    Use `inf_masks_dataframe` in general.

    Args:
        X (pd.DataFrame): DataFrame to check for +/-infs.

    Returns:
        pos_inf, neg_inf: Boolean masks.
    """
    # Build the +/-inf masks (shape matches `X`).
    if _is_single_float_block(X):
        return _inf_masks_pandas_only(X, numeric_only=True)

    return _inf_masks_mixed(X)


if PANDAS_FASTER_THAN_MIXED_PATH:
    inf_masks_dataframe = _inf_masks_pandas_only
else:
    inf_masks_dataframe = _inf_masks_dataframe


def _encoding_is_identity(
    X: pd.DataFrame,
    ord_encoder: OrderPreservingColumnTransformer | None,
    *,
    fit_encoder: bool,
) -> bool:
    """Whether the ordinal-encoding step provably cannot change any value.

    The encoder selects columns by dtype (``category``/``string``), so a frame of
    plain float64 columns leaves it with nothing to select: the transformer reduces
    to its passthrough remainder and the whole step is the float64 cast it ends
    with. Plain float64 specifically, not merely numeric -- a nullable ``Float64``
    holds ``pd.NA``, which only survives the trip through pandas.

    A *frozen* encoder does not re-select, it reuses the columns it saw at fit, so
    that selection has to be empty as well -- and the frame has to be one sklearn
    would have accepted, since skipping `transform` skips its checks along with it.
    """
    return (
        # only passthrough non-nullable all-fp64 dataframes
        all(dtype == _FLOAT64 for dtype in X.dtypes)
        and (
            # trainable encoders are ok
            fit_encoder
            or ord_encoder is None
            or (
                # condition 1: the ordinal encoder has been fitted and needs to
                # be reducible to its passthrough remainder
                all(
                    len(columns) == 0
                    for name, _, columns in ord_encoder.transformers_
                    if name != "remainder"
                )
                # condition 2: X needs to line up with what the encoder was fitted on
                # condition 2.1: encoder needs to have the same input feature shape
                and getattr(ord_encoder, "n_features_in_", None) == X.shape[1]
                # condition 2.2: either no fitted feature names, or they match 1:1
                and (
                    getattr(ord_encoder, "feature_names_in_", None) is None
                    or not all(isinstance(col, str) for col in X.columns)
                    # compared as plain lists to be dtype-insensitive:
                    or list(X.columns) == list(ord_encoder.feature_names_in_)  # ty: ignore[unresolved-attribute]
                )
            )
        )
    )


def _to_numpy_may_alias(X: pd.DataFrame) -> bool:
    """Whether ``X.to_numpy()`` can hand back a view of the frame's own buffer.

    A single-block frame is handed out as that block: read-only under copy-on-write,
    writeable and aliasing whatever the frame was built from without it (pandas < 3),
    which for a numeric ndarray input is the caller's own array. Anything wider has
    to be materialised into a new array first, so what comes back is private.

    Defensively ``True`` when the block internals are unavailable, so an unrecognised
    layout is copied rather than handed out.
    """
    blocks = getattr(getattr(X, "_mgr", None), "blocks", None)
    return blocks is None or len(blocks) <= 1


def _owned_float64_values(X: pd.DataFrame) -> np.ndarray:
    """`X`'s values as a writeable float64 array that the caller owns.

    One full-size allocation, whichever pandas is installed. What it cannot be is
    `to_numpy`'s result handed straight back, for a different reason per shape:

    * A single-block frame is handed out as that block, so `to_numpy` allocates
      nothing at all. That one is copied -- see `_to_numpy_may_alias`.
    * A frame of many blocks pandas 3 materialises into a private array, which could
      be taken as it is. Earlier pandas instead consolidates the frame *in place*
      first and returns a view of the block it just built: taking that would write
      through into the frame, and copying it costs a second full-size buffer on top
      of the consolidation. Measured at 200,000 x 300 float64, that pair peaks at
      twice the frame's size on pandas 1.4 against once on pandas 3.

    So a many-block frame is assembled column by column into an array preallocated
    here, which nothing else has ever referenced and which costs one buffer on every
    version. It also leaves the frame's own blocks alone, where `to_numpy` would
    consolidate them out from under a caller still holding it.

    Column-major because that is what the encoder path has always returned -- its
    `hstack` builds an F-ordered array and the closing `astype` preserves layout --
    and downstream preprocessing is column-wise. Pinning the order here keeps this a
    change of cost only, not of what callers receive.
    """
    if _to_numpy_may_alias(X):
        return np.array(X.to_numpy(dtype=np.float64, copy=False), order="F", copy=True)

    out = np.empty(X.shape, dtype=np.float64, order="F")
    for position in range(X.shape[1]):
        # Positional throughout: a duplicate column name makes `X[label]` a frame.
        out[:, position] = X.iloc[:, position].to_numpy(dtype=np.float64, copy=False)
    return out


def _can_write_encoded_columns(X: pd.DataFrame, selected: list[Any]) -> bool:
    """Whether the encoded array can be assembled column by column.

    Two things have to hold. Every column the encoder does *not* take must already be
    plain float64, so writing it into a float64 output is a copy and not a conversion
    that could differ from the one `ColumnTransformer` would have done -- notably an
    `object` column, which the encoder's dtype selector skips and the old path carried
    through as objects until the closing cast. And the column names must be unique,
    since each column is placed by name.
    """
    if X.columns.has_duplicates:
        return False
    taken = set(selected)
    return all(
        dtype == _FLOAT64
        for column, dtype in zip(X.columns, X.dtypes, strict=True)
        if column not in taken
    )


def _encode_into_preallocated(
    X: pd.DataFrame,
    ord_encoder: OrderPreservingColumnTransformer,
    selected: list[Any],
) -> np.ndarray:
    """Assemble the encoded array by writing each column straight to its final place.

    `ColumnTransformer` gets the same result in three full-size arrays: one per
    transformer, a second from stacking them, and a third from the gather
    `_preserve_order` needs to undo the stacking's column order. A mixed-column clean
    reaches its RAM peak twice inside that -- once in the stack, once in the gather --
    which is why removing either one alone changes nothing. Writing into the output
    costs one array, plus the block of codes the encoder returns.

    Column order is the frame's own, so no reordering is needed afterwards, and the
    layout is column-major to match what the stack used to produce.
    """
    positions = {column: index for index, column in enumerate(X.columns)}
    taken = set(selected)
    out = np.empty(X.shape, dtype=np.float64, order="F")

    if selected:
        # One call rather than one per column: the encoder validates against the
        # column count it was fitted on, so its codes come as a single block.
        codes = ord_encoder.named_transformers_["encoder"].transform(X[selected])
        out[:, [positions[column] for column in selected]] = codes
        del codes

    # Per column, so no full-width temporary is built for the passthrough half; each
    # write is a copy out of the frame's block, which `_can_write_encoded_columns` has
    # established is already float64.
    for column, index in positions.items():
        if column not in taken:
            out[:, index] = X[column].to_numpy(dtype=np.float64, copy=False)
    return out


def _apply_ordinal_encoder(
    X: pd.DataFrame,
    ord_encoder: OrderPreservingColumnTransformer | None,
    *,
    fit_encoder: bool,
) -> np.ndarray:
    """Run the ordinal-encoding step, or skip it where it cannot change anything.

    Every branch returns an array the caller owns, which is what lets it write the
    placeholder and +/-inf cells in place and cast to float64 with ``copy=False``.
    Three of the four allocate outright -- the copy, the preallocated output, the
    encoder's own hstack. The fourth, `X.to_numpy()`, does not: for a single-block
    frame pandas hands back the block itself, read-only under copy-on-write and
    aliasing the caller's ndarray without it.

    What keeps that branch honest is the identity check above: it takes every frame
    whose columns are all plain float64, so the frames that reach `to_numpy()` are
    never float64 throughout and the caller's `astype` has real work to do, which
    allocates. Widen `_encoding_is_identity` to accept a dtype it does not convert --
    a nullable ``Float64``, say -- and a view starts escaping. The caller asserts on
    it rather than leaving that to be noticed downstream.
    """
    if _encoding_is_identity(X, ord_encoder, fit_encoder=fit_encoder):
        if fit_encoder and ord_encoder is not None:
            # Fitting still has to happen -- the caller keeps the encoder for
            # predict -- but with no column selected it learns nothing from the
            # values, so a single row settles the column bookkeeping and spares us
            # the transform this branch exists to avoid.
            ord_encoder.fit(X.iloc[:1])
        return _owned_float64_values(X)
    if fit_encoder and ord_encoder is not None:
        # `ColumnTransformer.fit` is implemented as `fit_transform`, so fitting on the
        # whole frame would run the very transform the assembly below replaces. Fit on
        # one row for the column bookkeeping instead -- widths and output indices do
        # not depend on the row count for a one-to-one encoder -- and then teach the
        # inner encoder its categories from every row.
        ord_encoder.fit(X.iloc[:1])
        selected = _encoder_selection(ord_encoder)
        if not _can_write_encoded_columns(X, selected):
            # Bail before learning any categories: `fit_transform` learns them again
            # from scratch, so doing it first would be a wasted pass over the data.
            # Only the one-row fit above is lost, which costs no pass at all.
            return ord_encoder.fit_transform(X)
        if selected:
            ord_encoder.named_transformers_["encoder"].fit(X[selected])
        return _encode_into_preallocated(X, ord_encoder, selected)
    if ord_encoder is not None:
        # Left on sklearn deliberately. `transform` also validates the frame against
        # the one seen at fit -- column count, names, order -- and the assembly above
        # does not, so using it here would trade a real check for memory the wrapper
        # has already validated by other means.
        return ord_encoder.transform(X)
    return X.to_numpy()


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
    X_input = X

    if passthrough_inf:
        pos_inf, neg_inf = inf_masks_dataframe(X)

        X = X.copy()
        # coerce columns to NaN:
        X[neg_inf | pos_inf] = np.nan

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
        if X is X_input:
            X = X.copy()
        X[string_cols] = X[string_cols].fillna(placeholder)

    X_encoded = _apply_ordinal_encoder(X, ord_encoder, fit_encoder=fit_encoder)
    # Everything below writes into this array and then hands it to the caller, so it
    # has to be one no one else holds. Read-only means pandas handed back a view of a
    # frame's block instead: see `_apply_ordinal_encoder` for how that is kept from
    # happening, and note that on pandas 2 such a view is writeable and would pass
    # this while quietly writing through to whatever the frame was built from.
    assert X_encoded.flags.writeable, (
        "the ordinal-encoding step returned an array it does not own"
    )

    string_cols_ix = [X.columns.get_loc(col) for col in string_cols]
    placeholder_mask = X[string_cols] == placeholder
    X_encoded[:, string_cols_ix] = np.where(
        placeholder_mask,
        np.nan,
        X_encoded[:, string_cols_ix],
    )
    # `copy=False` because the cast has nothing to do whenever the step above already
    # produced float64.
    # Safe to hand back uncopied because every branch above allocates its own array.
    X_encoded = X_encoded.astype(np.float64, copy=False)

    # Write the recorded +/-inf values back into their original numeric cells.
    if passthrough_inf and (pos_inf.any() or neg_inf.any()):
        X_encoded[pos_inf] = np.inf
        X_encoded[neg_inf] = -np.inf

    return typing.cast("np.ndarray", X_encoded)
