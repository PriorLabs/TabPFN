#  Copyright (c) Prior Labs GmbH 2026.

"""Module for cleaning the data.

These cleaning steps are performed before further preprocessing,
e.g. NaN mapping and dtype conversion.
"""

from __future__ import annotations

import contextlib
import contextvars
import ctypes
import dataclasses
import mmap
import sys
import threading
import typing
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from packaging.version import Version

try:
    from numpy.lib.array_utils import byte_bounds
except ImportError:  # numpy < 2
    from numpy import byte_bounds  # type: ignore[attr-defined,no-redef]

from tabpfn.constants import NA_PLACEHOLDER
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.pipeline_interface import _usable_cpus
from tabpfn.preprocessing.steps.preprocessing_helpers import (
    get_ordinal_encoder,
    to_numpy_may_alias,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import Any, Literal

    from tabpfn.constants import XType
    from tabpfn.preprocessing.steps.preprocessing_helpers import (
        OrderPreservingColumnTransformer,
    )
    from tabpfn.preprocessing.torch import FeatureSchema

# An array's dtype is told apart by numpy's one-letter dtype kind:
# https://numpy.org/doc/2.1/reference/arrays.dtypes.html#checking-the-data-type

# `?` bool, `b` signed byte, `B` unsigned byte, `i` signed integer, `u` unsigned
# integer, `f` float, `m` timedelta.
NUMERIC_DTYPE_KINDS = "?bBiufm"
# The subset of the above that numpy casts to float64 the same way pandas does, so
# a frame need not be built to convert it. Timedeltas ("m") are excluded: pandas
# converts those through its own units rather than numpy's raw integers.
FAST_CONVERTIBLE_DTYPE_KINDS = "?bBiuf"
# `O` object, `V` void (structured records), `U` fixed-width unicode strings, `T`
# variable-width unicode strings (numpy 2's `StringDType`): pandas reads the cells to
# work out each column's dtype.
OBJECT_OR_STRING_DTYPE_KINDS = "OVUT"
# `S` fixed-width byte strings and `a`, its legacy alias: refused.
BYTES_DTYPE_KINDS = "Sa"
# `c` complex, `M` datetime64. Not needed, just for completeness.
UNSUPPORTED_DTYPE_KINDS = "cM"
PANDAS_BELOW_3 = Version(pd.__version__) < Version("3.0.0")
# Below this version, deleting a column of a frame copies the rest of its block.
DELETING_COLUMNS_COPIES = Version(pd.__version__) < Version("2.0.0")
# Before 3.0 `astype` copies every column by default, including the ones it is not
# casting; from 3.0 copy-on-write makes the keyword a no-op and passing it warns.
_ASTYPE_KEEPS_UNCAST_COLUMNS = {"copy": False} if PANDAS_BELOW_3 else {}


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


def delete_columns(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Delete the columns at `positions` from `X` in place, in one call.

    By position, so a label that repeats loses only the positions given. The kept
    columns are not copied.
    """
    labels = X.columns[list(positions)]
    X._mgr = X._mgr.idelete(positions)
    # As `del X[label]` does: a cached column keeps its block's memory alive.
    cache = getattr(X, "_item_cache", None)
    if cache is not None:
        for label in labels:
            cache.pop(label, None)


_FILL_BLOCK_BYTES = 1 << 24
"""Bytes of rows `_with_constants` copies at a time. Each block's constant columns are
written while it is in cache: larger blocks fall out of it, smaller ones pay more
per-block overhead."""


def _with_constants(X: np.ndarray, constant_values: dict[int, Any]) -> np.ndarray:
    """A C-ordered copy of `X`, with `constant_values` in their columns."""
    out = np.empty(X.shape, dtype=X.dtype)
    columns = list(constant_values)
    values = np.array(list(constant_values.values()), dtype=X.dtype)
    rows = max(1, _FILL_BLOCK_BYTES // max(1, X.shape[1] * out.itemsize))
    for start in range(0, X.shape[0], rows):
        block = out[start : start + rows]
        block[...] = X[start : start + rows]
        block[:, columns] = values
    return out


def _is_plain_numeric_array(X: np.ndarray | pd.DataFrame) -> bool:
    """Whether `X` is a numpy array of a dtype that casts straight to float64.

    Such an array carries no dtype to infer, no category to map and no string
    cell to NA-handle, so cleaning it is a single cast (`_cast_to_float64_table`)
    provided nothing is encoded either.
    """
    return isinstance(X, np.ndarray) and X.dtype.kind in FAST_CONVERTIBLE_DTYPE_KINDS


def _encoder_selects_nothing(
    ord_encoder: OrderPreservingColumnTransformer | None,
) -> bool:
    """Whether a *fitted* encoder would leave every column alone.

    Asked of the encoder rather than of the incoming frame's categorical indices,
    because the two can disagree: a column detected as text is encoded at fit but
    never appears in `indices_for(FeatureModality.CATEGORICAL)`, and a column that
    was string at fit can arrive numeric at predict. Only the encoder knows what
    it was fitted to touch.
    """
    return ord_encoder is None or not ord_encoder.selected_columns()


def clean_data_transform(
    X: XType,
    *,
    cat_indices: Sequence[int] | None,
    ord_encoder: OrderPreservingColumnTransformer | ColumnRangeEncoders | None,
    passthrough_inf: bool = False,
    validate: Callable[[XType], np.ndarray] | None = None,
) -> np.ndarray:
    """Clean data being predicted on, the way `clean_data` cleaned the training data.

    The predict-time half of `clean_data`: the same dtype fixing and ordinal
    encoding, driven by the encoder `clean_data` fitted rather than fitting one,
    and taking the same single-cast shortcut when there is nothing to encode.

    Args:
        X: The data to clean.
        cat_indices: Indices of the columns the encoder was fitted on.
        ord_encoder: The `ordinal_encoder_` of a fitted estimator. This is the
            encoder that `clean_data` returned at fit time, or the
            `ColumnRangeEncoders` of a fit input cleaned one range of columns at a
            time.
        passthrough_inf: If True, +/-inf values are carried through the ordinal
            encoding stage unchanged instead of crashing it.
        validate: Validates `X`, or each column range of `X`, before cleaning. None
            if `X` is already validated.

    Returns:
        For `ColumnRangeEncoders`, a new table of their `dtype`. Otherwise `X`, or a
        copy of it with the fit-time constant columns restored, when `X` is a numeric
        array the encoder encodes no column of; a new float32, C-ordered array when
        `X` is a float32 array whose encoded columns all held numbers at fit, and
        which holds no infinity if `passthrough_inf`; otherwise a new float64 array.
    """
    if isinstance(ord_encoder, ColumnRangeEncoders):
        return ord_encoder.transform(
            X,
            cat_indices=cat_indices or (),
            passthrough_inf=passthrough_inf,
            validate=validate,
            validate_runs=False,
        )
    if validate is not None:
        X = validate(X)
    given = X
    if constant_values := getattr(ord_encoder, "constant_values_", None):
        # A column constant at fit gets its fit-time value back, so it is cleaned
        # exactly as at fit, whatever it holds at predict.
        numeric = all(
            isinstance(v, (int, float, np.number)) for v in constant_values.values()
        )
        # A copy either way: the caller's array must not be written to.
        if numeric and X.dtype == np.float32:
            # The float32 paths below read a C-ordered table.
            X = _with_constants(X, constant_values)
        else:
            if not numeric:
                X = X.astype(object)
            else:
                X = X.astype(X.dtype if X.dtype.kind in "fO" else np.float64)
            for index, value in constant_values.items():
                X[:, index] = value

    if _is_plain_numeric_array(X) and _encoder_selects_nothing(ord_encoder):
        # `passthrough_inf` makes no difference here: it records the +/-inf cells,
        # NaNs them so the encoder does not choke, and writes them back at the same
        # positions afterwards -- an exact round trip when nothing is encoded.
        return X

    if (
        ord_encoder is not None
        and cat_indices is not None
        and _encodes_categoricals_only(X, cat_indices, passthrough_inf=passthrough_inf)
        and _fitted_on_numbers(ord_encoder)
    ):
        # Restoring the constant columns above made `X` a C-ordered copy already.
        owned = X is not given
        X_encoded = ord_encoder.transform_into(X if owned else X.copy(order="C"))
        if X_encoded is not None:
            return X_encoded

    return process_text_na_dataframe(
        X=fix_dtypes(X=X, cat_indices=cat_indices),
        ord_encoder=ord_encoder,
        passthrough_inf=passthrough_inf,
    )


def clean_data(
    X: np.ndarray,
    feature_schema: FeatureSchema,
    *,
    passthrough_inf: bool = False,
    overwrite_input: bool = False,
) -> tuple[np.ndarray, OrderPreservingColumnTransformer, FeatureSchema]:
    """Clean the data by converting dtypes and ordinally encoding categorical columns.

    Args:
        X: The data to clean.
        feature_schema: The feature schema corresponding to the data.
        passthrough_inf: If True, +/-inf values are carried through the ordinal
            encoding stage unchanged instead of crashing it (see
            `process_text_na_dataframe`).
        overwrite_input: If True, `X` is the caller's own copy, and the categorical
            codes of a float32 `X` may be written into it.

    Returns:
        A tuple containing the cleaned data, the ordinal encoder, and the inferred
        feature modalities.
    """
    cat_indices = feature_schema.indices_for(FeatureModality.CATEGORICAL)

    # Ensure categories are ordinally encoded
    ord_encoder = get_ordinal_encoder()
    # Restored by `clean_data_transform`, so these columns carry no information at
    # predict and never meet a value, e.g. a string, that fit did not see.
    ord_encoder.constant_values_ = {  # type: ignore[attr-defined]
        index: X[0, index]
        for index in feature_schema.indices_for(FeatureModality.CONSTANT)
    }

    if not cat_indices and _is_plain_numeric_array(X):
        # A numeric array holds no string cell, so with no categorical column
        # declared there is nothing for the encoder to select.
        #
        # The encoder is still fit, since the caller keeps it for predict, but on a
        # single row: with no column selected it learns nothing from the values,
        # only the column bookkeeping.
        ord_encoder.fit(fix_dtypes(X=X[:1], cat_indices=cat_indices))
        return X, ord_encoder, feature_schema

    if _encodes_categoricals_only(X, cat_indices, passthrough_inf=passthrough_inf):
        X_encoded = _encode_categoricals(
            X, cat_indices, ord_encoder, in_place=overwrite_input
        )
        if X_encoded is not None:
            return X_encoded, ord_encoder, feature_schema

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


_INF_SCAN_COLUMNS = 64
"""Columns `_encodes_categoricals_only` scans for infinities at a time."""


def _encodes_categoricals_only(
    X: np.ndarray, cat_indices: Sequence[int], *, passthrough_inf: bool
) -> bool:
    """Whether cleaning can encode a float32 array's categorical columns alone.

    The general path makes a float64 frame of the whole array, and a float64 output
    from it. Encoding only the categorical columns into a float32 copy gives the same
    values and, at fit, the same fitted encoder. With `passthrough_inf`, an array
    holding an infinity takes the general path, which carries it through the encoder.
    """
    if not (
        isinstance(X, np.ndarray)
        and X.dtype == np.float32
        and X.ndim == 2
        and all(isinstance(i, (int, np.integer)) for i in cat_indices)
    ):
        return False
    if not passthrough_inf:
        return True
    for start in range(0, X.shape[1], _INF_SCAN_COLUMNS):
        if np.isinf(X[:, start : start + _INF_SCAN_COLUMNS]).any():
            return False
    return True


def _fitted_on_numbers(ord_encoder: OrderPreservingColumnTransformer) -> bool:
    """Whether every column the fitted encoder encodes held numbers at fit.

    The general path casts a column that held strings at fit back to strings, which
    a float32 array cannot hold.
    """
    categories = ord_encoder.named_transformers_["encoder"].categories_
    return all(c.dtype.kind in "fiu" for c in categories)


def _with_categoricals(X: np.ndarray, cat_indices: Sequence[int]) -> pd.DataFrame:
    """`X` as a frame over its own data, with categorical columns as `fix_dtypes` has.

    `fix_dtypes` makes them categories of the float64 values.
    """
    frame = pd.DataFrame(X, copy=False).copy(deep=False)
    for index in cat_indices:
        frame[frame.columns[index]] = pd.Categorical(X[:, index].astype(np.float64))
    return frame


def _encode_categoricals(
    X: np.ndarray,
    cat_indices: Sequence[int],
    ord_encoder: OrderPreservingColumnTransformer,
    *,
    in_place: bool,
) -> np.ndarray | None:
    """Fit `ord_encoder` as `clean_data` does; `X` with its categorical columns coded.

    A float32 array: `X` itself when `in_place`, else a copy. The values are those of
    the float64 array the general path returns: a float32 value widens exactly, and a
    code is a small integer. None, with the encoder left to be fitted again, when its
    layout cannot be written column by column.
    """
    frame = _with_categoricals(X, cat_indices)
    out = ord_encoder.fit_transform_into(frame, out=X if in_place else X.copy())
    if out is None or not ord_encoder.selected_columns():
        return None
    if out is X:
        mark_fit_memory(X, overwritten=True)
    return out


@dataclasses.dataclass
class ColumnRangeEncoders:
    """The fitted cleaning of a fit input cleaned one range of columns at a time.

    `clean_data` fitted one encoder on each range. Predict input is cleaned with the
    same ranges into a table of `dtype`.
    """

    ranges: list[tuple[int, int, OrderPreservingColumnTransformer]]
    """`(start, stop, encoder)` per range of columns, in column order."""
    dtype: np.dtype

    def transform(
        self,
        X: XType,
        *,
        cat_indices: Sequence[int],
        passthrough_inf: bool,
        validate: Callable[[XType], np.ndarray] | None,
        validate_runs: bool,
    ) -> np.ndarray:
        """`clean_data_transform` of `X` by these ranges.

        With `validate_runs`, each of `_runs(X)` is validated at once rather than
        each range: one fixed validation cost per run, but a run's converted copy
        held at once.
        """
        if validate is not None and (
            not isinstance(X, (pd.DataFrame, np.ndarray)) or X.ndim != 2
        ):
            # Only a frame or a 2-D array can be cut into ranges. Validation converts
            # any other input to an array or refuses it.
            X = validate(X)
        table = np.empty((X.shape[0], self.ranges[-1][1]), dtype=self.dtype)
        runs = self._runs(X) if validate_runs else [[r] for r in self.ranges]
        for run in runs:
            first, last = run[0][0], run[-1][1]
            part = column_range(X, first, last)
            if validate is not None:
                part = validate(part)
            for start, stop, encoder in run:
                table[:, start:stop] = clean_data_transform(
                    column_range(part, start - first, stop - first),
                    cat_indices=[i - start for i in cat_indices if start <= i < stop],
                    ord_encoder=encoder,
                    passthrough_inf=passthrough_inf,
                )
        return table

    def _runs(
        self, X: pd.DataFrame | np.ndarray
    ) -> list[list[tuple[int, int, OrderPreservingColumnTransformer]]]:
        """The ranges, grouped into runs of adjacent ranges whose columns share one
        dtype that validates the same however many columns of it are validated
        together. A range with other columns is a run of its own.
        """
        dtypes = (
            list(X.dtypes) if isinstance(X, pd.DataFrame) else [X.dtype] * X.shape[1]
        )
        runs: list[list[tuple[int, int, OrderPreservingColumnTransformer]]] = []
        run_dtype = None
        for start, stop, encoder in self.ranges:
            own = set(dtypes[start:stop])
            dtype = own.pop() if len(own) == 1 else None
            # `run_dtype` is checked too: `np.dtype(np.float64) == None` holds.
            if (
                _validates_alike_in_runs(dtype)
                and _validates_alike_in_runs(run_dtype)
                and dtype == run_dtype
            ):
                runs[-1].append((start, stop, encoder))
            else:
                runs.append([(start, stop, encoder)])
                run_dtype = dtype
        return runs


def _validates_alike_in_runs(dtype: Any) -> bool:
    """Whether columns of `dtype` validate to the same array whether validated one by
    one or together.

    Not so for a categorical column, which converts to its categories' dtype alone
    but to objects next to another one.
    """
    return isinstance(dtype, (np.dtype, pd.StringDtype)) or _coerced_to_float64(dtype)


def column_range(X: pd.DataFrame | np.ndarray, start: int, stop: int) -> XType:
    """The columns `start:stop` of `X`, in the same container type."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, start:stop]
    return X[:, start:stop]


_OVERWRITE_FIT_INPUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tabpfn_overwrite_fit_input", default=False
)


@dataclasses.dataclass(frozen=True)
class _FitMemory:
    """A byte range of an array's memory, recorded while the array owning it lives."""

    owner: weakref.ref[Any]
    start: int
    stop: int


_FIT_MEMORY_LOCK = threading.Lock()
"""Guards both registries against fits on other threads."""
_OVERWRITTEN_FIT_MEMORY: list[_FitMemory] = []
"""The memory that a `fit` wrote into or released."""
_KEPT_FIT_MEMORY: list[_FitMemory] = []
"""The memory that a fitted model reads without a copy."""
_EMPTIED_FIT_FRAMES: weakref.WeakValueDictionary[int, pd.DataFrame] = (
    weakref.WeakValueDictionary()
)
"""The live frames a `fit` emptied, by `id`."""


@contextlib.contextmanager
def overwrite_fit_input() -> Iterator[None]:
    """Let a `fit` inside this block write into the array it is given.

    For a caller that hands over a large, writeable float32 array it no longer
    needs: its categorical columns are then coded in place rather than in a copy,
    and preprocessing that writes nothing keeps the array rather than a copy. A
    read-only array's categorical columns are coded in a copy, as outside the block.

    The caller gives up each input that it passes to a `fit` inside this block. Pass
    each input, and anything that shares its memory, to one `fit` only. After that
    fit, the caller must not read or write them. Their contents are undefined, and the
    fitted model may keep them.

    When `fit` cleans a given-up input into a new table:

    - A frame is emptied after its targets are validated. Its blocks are freed as
      their columns are copied into the table. If pandas can delete a column without
      copying its block, the memory of the largest block is released to the system
      a slice of rows at a time.
    - For a float32 table, a writeable numeric array of a wider dtype is cast to
      float32 in its own memory, and the rest of that memory is released.

    Released memory reads as zeros through any array or frame that shares it. This
    includes frames the input was derived from without a copy, e.g. under pandas
    copy-on-write.

    A later `fit` raises a `ValueError` for an input whose memory a fit wrote into or
    released, and for a frame that a fit emptied. A fit never writes into or releases
    memory that a read-only array holds, that maps a file (`np.memmap`, `mmap`), or
    that another fitted model reads. It copies instead.
    """
    token = _OVERWRITE_FIT_INPUT.set(True)
    try:
        yield
    finally:
        _OVERWRITE_FIT_INPUT.reset(token)


def overwriting_fit_input() -> bool:
    """Whether the caller is inside `overwrite_fit_input()`."""
    return _OVERWRITE_FIT_INPUT.get()


def check_fit_input(given: Any) -> None:
    """Refuse a fit input whose memory an earlier `fit` wrote into or released, or a
    frame that an earlier `fit` emptied.

    Raises:
        ValueError: If `given` is such an input.
    """
    if (
        isinstance(given, pd.DataFrame) and _EMPTIED_FIT_FRAMES.get(id(given)) is given
    ) or _shares_memory(_arrays_of(given), _OVERWRITTEN_FIT_MEMORY):
        raise ValueError(
            "This input was overwritten by an earlier fit inside "
            "`overwrite_fit_input()` and no longer holds its original data. Pass "
            "each input to one fit only, or fit on a copy."
        )


def may_overwrite_fit_input(X: Any, given: Any) -> bool:
    """Whether `fit` may write into `X`, the array it made from the caller's `given`.

    The caller has already refused, with `check_fit_input`, a `given` that an earlier
    fit overwrote.
    """
    if X is given:
        return isinstance(X, np.ndarray) and may_write_fit_memory(X)
    return is_private_copy(X, given)


def may_write_fit_memory(X: np.ndarray) -> bool:
    """Whether a `fit` may write into or release the memory of `X`, the caller's array.

    True inside `overwrite_fit_input()` if `X` is writeable, maps no file, and no
    fitted model reads its memory. `check_fit_input` has already refused memory a fit
    overwrote.
    """
    return (
        _OVERWRITE_FIT_INPUT.get()
        and X.flags.writeable
        and not _maps_a_file(X)
        and not _shares_memory([X], _KEPT_FIT_MEMORY)
    )


def _maps_a_file(X: np.ndarray) -> bool:
    """Whether the memory of `X` is a memory map, whose writes can reach a file."""
    owner: Any = X
    while isinstance(owner, np.ndarray):
        if isinstance(owner, np.memmap):
            return True
        owner = owner.base
    if isinstance(owner, memoryview):
        owner = owner.obj
    return isinstance(owner, mmap.mmap)


def mark_fit_memory(X: np.ndarray, *, overwritten: bool) -> None:
    """Record that a `fit` writes into or releases the memory of `X` (`overwritten`),
    or that a fitted model reads it without a copy.
    """
    start, stop = byte_bounds(X)
    entry = _FitMemory(_owner_ref(X), start, stop)
    registry = _OVERWRITTEN_FIT_MEMORY if overwritten else _KEPT_FIT_MEMORY
    with _FIT_MEMORY_LOCK:
        # An entry whose owner is gone no longer describes live memory.
        registry[:] = [memory for memory in registry if memory.owner() is not None]
        registry.append(entry)


def _owner_ref(X: np.ndarray) -> weakref.ref[Any]:
    """A weak reference to the object that owns the memory of `X`, as far as the
    chain of bases leads to one that takes weak references.

    The memory lives as long as its owner. An array made from a torch tensor or a
    buffer has a base that is not an array; for a tensor, its storage owns the
    memory.
    """
    owner: Any = X
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    base = owner.base
    if isinstance(base, memoryview):
        base = base.obj
    if callable(getattr(base, "untyped_storage", None)):
        # `Tensor.numpy()` makes an alias tensor that dies with the array. The
        # storage is shared and lives as long as the memory.
        base = base.untyped_storage()
    try:
        return weakref.ref(base)
    except TypeError:
        # No base, or one without weak references (e.g. `bytes`): the owning array
        # is as far as the chain can be followed.
        return weakref.ref(owner)


def _arrays_of(given: Any) -> list[np.ndarray]:
    """The arrays that hold the values of `given`: the array, a frame's blocks, or a
    CPU tensor's memory.
    """
    if isinstance(given, np.ndarray):
        return [given]
    if isinstance(given, torch.Tensor):
        return [given.detach().numpy()] if given.device.type == "cpu" else []
    if isinstance(given, pd.DataFrame):
        # `.values` is the pandas Block array accessor here, not a Series/DataFrame.
        blocks = [block.values for block in given._mgr.blocks]  # noqa: PD011
        return [values for values in blocks if isinstance(values, np.ndarray)]
    return []


def _shares_memory(arrays: list[np.ndarray], registry: list[_FitMemory]) -> bool:
    """Whether the bytes of any of `arrays` overlap live memory in `registry`."""
    with _FIT_MEMORY_LOCK:
        held = [memory for memory in registry if memory.owner() is not None]
    for array in arrays:
        start, stop = byte_bounds(array)
        if any(memory.start < stop and start < memory.stop for memory in held):
            return True
    return False


_NARROW_BLOCK_BYTES = 1 << 26
"""Bytes of rows that `narrow_fit_input_in_place` casts at a time. Each block is cast
into a temporary array first. Larger blocks need a larger temporary array. Smaller
blocks pay the per-block overhead more often."""

_MADV_DONTNEED = 4
"""Linux's `madvise` advice to drop a range of pages."""


def narrow_fit_input_in_place(X: np.ndarray) -> np.ndarray:
    """Return the values of `X` as a float32 table over the start of the memory of `X`.

    `X` must be a given-up, writeable, C-ordered numeric array with at least four
    bytes per value. The rest of its memory is released to the system. Afterwards `X`
    holds no meaningful values.
    """
    n_rows, n_columns = X.shape
    table = X.reshape(-1).view(np.float32)[: X.size].reshape(n_rows, n_columns)
    rows = max(1, _NARROW_BLOCK_BYTES // max(1, n_columns * X.itemsize))
    ratio = X.itemsize // table.itemsize

    def narrow(start: int, stop: int) -> None:
        # `astype` reads the whole block into a temporary array before the write, so
        # the write can overlap the block's own input.
        table[start:stop] = X[start:stop].astype(np.float32)

    # Output rows `[w, ratio * w)` overwrite the memory of input rows `[w / ratio, w)`.
    # Earlier waves have read those input rows, so the blocks of one wave can run in
    # any order. The first block overwrites its own input and runs alone. If
    # `ratio == 1`, each row overwrites only itself, so one wave holds all rows.
    waves = [(0, n_rows)] if ratio == 1 else [(0, min(rows, n_rows))]
    while waves[-1][1] < n_rows:
        start = waves[-1][1]
        waves.append((start, min(ratio * start, n_rows)))
    # Mark before the first write, so that a later fit refuses `X` even if this fit
    # fails midway.
    mark_fit_memory(X, overwritten=True)
    with ThreadPoolExecutor(max_workers=_usable_cpus()) as pool:
        for wave_start, wave_stop in waves:
            starts = range(wave_start, wave_stop, rows)
            stops = [min(start + rows, wave_stop) for start in starts]
            list(pool.map(narrow, starts, stops))
    _release_pages(X, table.nbytes)
    return table


def release_rows(X: np.ndarray, start: int, stop: int) -> None:
    """Release the whole pages of rows `start:stop` of `X` to the system.

    `X` must be a given-up, C-ordered array.
    """
    row_bytes = X.strides[0]
    _release_pages(X, start * row_bytes, stop * row_bytes)


def _release_pages(X: np.ndarray, start: int, stop: int | None = None) -> None:
    """Release the whole pages of the memory of `X` from byte `start` to byte `stop`
    (default: the end) to the system.

    This works on Linux only. On other systems the pages stay, which costs memory but
    changes no value. A released page reads as zeros when it is read again.
    """
    if not sys.platform.startswith("linux"):
        return
    page = mmap.PAGESIZE
    stop = X.nbytes if stop is None else stop
    first = -(-(X.ctypes.data + start) // page) * page
    last = (X.ctypes.data + stop) // page * page
    if last <= first:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    if libc.madvise(first, last - first, _MADV_DONTNEED) != 0:
        # Locked or huge pages cannot be released. They only cost memory, and the fit
        # has already written its input, so it continues.
        warnings.warn(
            f"madvise could not release fit input memory (errno "
            f"{ctypes.get_errno()}), so it stays in use.",
            stacklevel=2,
        )


def empty_fit_frame(X: pd.DataFrame) -> None:
    """Drop every column of `X`, a given-up frame, so that its memory is freed when
    nothing else holds it.
    """
    # Drop in place: the caller's frame object must stop referencing its blocks.
    X.drop(columns=X.columns, inplace=True)  # noqa: PD002
    _EMPTIED_FIT_FRAMES[id(X)] = X


def is_private_copy(X: Any, given: Any) -> bool:
    """Whether the array `X`, made from the caller's `given`, shares no memory with it.

    A frame's blocks and an array are compared by their memory bounds, so a doubt
    counts as shared.
    """
    if not isinstance(X, np.ndarray) or X is given:
        return False
    if isinstance(given, np.ndarray):
        return not np.may_share_memory(X, given)
    if isinstance(given, pd.DataFrame):
        try:
            blocks = [block.values for block in given._mgr.blocks]  # noqa: PD011
        except AttributeError:
            return False
        return not any(
            isinstance(values, np.ndarray) and np.may_share_memory(X, values)
            for values in blocks
        )
    return X.base is None and X.flags.owndata


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
    dtypes = X.dtypes
    # Decided once per distinct dtype rather than once per column: a wide frame has
    # thousands of columns and a handful of dtypes.
    dtypes_to_cast = {dtype for dtype in dtypes.unique() if _coerced_to_float64(dtype)}
    cols = [col for col, dtype in dtypes.items() if dtype in dtypes_to_cast]
    return _cast_columns(X, cols, "float64")


def _coerced_to_float64(dtype: Any) -> bool:
    """Whether `coerce_nullable_dtypes_to_numpy` casts a column of `dtype`."""
    return pd.api.types.is_bool_dtype(dtype) or (
        pd.api.types.is_extension_array_dtype(dtype) and dtype.kind in "iuf"
    )


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
        elif X.dtype.kind in OBJECT_OR_STRING_DTYPE_KINDS:
            # For an object or string array we rely on pandas to introspect the
            # cells and determine each column's dtype.
            X = pd.DataFrame(X, copy=True)
            convert_dtype = True
        elif X.dtype.kind in BYTES_DTYPE_KINDS:
            raise ValueError(
                f"Byte string dtypes are not supported. Got dtype: {X.dtype}",
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
    selected = ord_encoder.selected_columns()
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


inf_masks_dataframe = _inf_masks_pandas_only if PANDAS_BELOW_3 else _inf_masks_dataframe


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

    if ord_encoder is None:
        # No encoding step at all, so the frame's own values are the output. Copied
        # where `to_numpy` may hand back one of the frame's blocks rather than a private
        # array, since the writes below are the caller's to make; taken as it is
        # otherwise, which costs the one full-size buffer either way.
        X_encoded = X.to_numpy()
        if to_numpy_may_alias(X):
            X_encoded = X_encoded.copy(order="K")
    elif fit_encoder:
        X_encoded = ord_encoder.fit_transform(X)
    else:
        X_encoded = ord_encoder.transform(X)

    # Everything below writes into this array and then hands it to the caller, so it
    # has to be one no one else holds. Read-only means pandas handed back a view of a
    # frame's block instead, and note that on pandas 2 such a view is writeable and
    # would pass this while quietly writing through to whatever the frame was built
    # from.
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
