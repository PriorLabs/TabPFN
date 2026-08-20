#  Copyright (c) Prior Labs GmbH 2026.

"""Feature Preprocessing Transformer Step."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import Self, override

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import get_config
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    check_is_fitted,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tabpfn.constants import DEFAULT_NUMPY_PREPROCESSING_DTYPE

if TYPE_CHECKING:
    from typing import Literal

    from tabpfn.classifier import XType, YType

_FLOAT64 = np.dtype(np.float64)


def _input_columns(X: XType) -> pd.Index | range:
    """The input's column keys, in input order."""
    return X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])


def _head(X: XType, rows: int) -> XType:
    """The input's leading rows, as the same kind of container."""
    return X.iloc[:rows] if isinstance(X, pd.DataFrame) else X[:rows]


def _column_subset(X: XType, columns: list[Any]) -> XType:
    """The given columns, as the same kind of container."""
    return X[columns] if isinstance(X, pd.DataFrame) else X[:, columns]


def _columns_at(X: XType, positions: list[int]) -> XType:
    """The columns at the given positions, as the same kind of container."""
    return X.iloc[:, positions] if isinstance(X, pd.DataFrame) else X[:, positions]


def _column_values(X: XType, position: int) -> np.ndarray:
    """One column, by position, as an array -- a view wherever that is possible."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, position].to_numpy(copy=False)
    return X[:, position]


def _hands_columns_through(remainder: Any) -> bool:
    """Whether `remainder` returns the columns it is handed, untouched.

    `"passthrough"` says so outright. A `FunctionTransformer` with no `func` is
    the identity too, and is what `get_ordinal_encoder` configures.
    """
    if isinstance(remainder, str):
        return remainder == "passthrough"
    return type(remainder) is FunctionTransformer and remainder.func is None


class EfficientColumnTransformer(ColumnTransformer):
    """A `ColumnTransformer` that assembles its output into one preallocated array.

    `ColumnTransformer` reaches its result through three full-size arrays: one per
    transformer, a second from stacking them, and (when preserving column order)
    a third from the gather that reorders. A wide input reaches its RAM peak twice
    inside that, once in the stack and once in the gather, which is why removing either
    one alone changes nothing. This class writes every column straight to its final
    place instead, so only the output and the named transformer's own block are ever
    held at once.

    That is only possible for the narrow shape this codebase needs: at most one named
    transformer, one-to-one, over a subset of the columns, plus a remainder that hands
    the rest through untouched. Anything else -- a transformer that expands its columns
    or is fed a `y`, a sparse result, a `set_output` asking for a frame, a frame whose
    passthrough columns are not already float64 -- falls back to `ColumnTransformer`,
    so this stays a drop-in replacement for it.

    `fit` goes one further and builds no output at all, where `ColumnTransformer.fit`
    -- implemented as `fit_transform` -- pays for a full-size result and discards it.

    `transform` is deliberately left on sklearn. It validates the input against the
    one seen at fit -- column count, names, order -- which the assembly does not, so
    replacing it would trade a real check for memory the fit path has already saved.
    """

    # Whether the output keeps the input's column order rather than
    # `ColumnTransformer`'s `[transformed, remainder]` one. A class attribute
    # rather than a constructor parameter, so sklearn's parameter introspection --
    # and with it `clone` and `get_params` -- stays exactly its parent's.
    preserves_column_order: ClassVar[bool] = False

    @override
    def fit(self, X: XType, y: YType = None, **params: Any) -> Self:
        """Fit without building the transformed array `ColumnTransformer.fit` builds.

        All the fitted state consists of is the column bookkeeping and the named
        transformer. The bookkeeping comes from one row -- widths and output positions
        do not depend on the row count for a one-to-one transformer -- and the
        transformer then learns from every row, in one pass, with no full-size result
        stacked or discarded on the way.

        Decided on the same conditions as the assembly, so that the state left here is
        either sklearn's own or one those conditions vouch for.
        """
        if not self._may_assemble(X, y, params):
            return super().fit(X, y, **params)
        probe = self._fit_column_bookkeeping(X)
        name, selected = self._named_selection()
        if not self._can_assemble(X, probe, selected):
            return super().fit(X, y, **params)
        if selected:
            self.named_transformers_[name].fit(_column_subset(X, selected))
        return self

    @override
    def fit_transform(self, X: XType, y: YType = None, **params: Any) -> XType:
        """Fit and transform, writing each column straight to its final place."""
        original_columns = _input_columns(X)
        if not self._may_assemble(X, y, params):
            return self._in_input_order(
                super().fit_transform(X, y, **params), original_columns
            )

        probe = self._fit_column_bookkeeping(X)
        name, selected = self._named_selection()
        if not self._can_assemble(X, probe, selected):
            # Bail before the named transformer has learned anything from the values:
            # the fallback learns them again from scratch, so doing it first would be a
            # wasted pass over the data. Only the one-row fit is lost, which costs no
            # pass at all.
            return self._in_input_order(
                super().fit_transform(X, y, **params), original_columns
            )

        codes = None
        if selected:
            # `fit_transform` rather than a fit and then a transform: it is one pass
            # over one slice of the selected columns, where the second pass would have
            # to hold that slice for the whole of it.
            codes = self.named_transformers_[name].fit_transform(
                _column_subset(X, selected)
            )
            # What `_hstack` does with a sparse block whose density beat
            # `sparse_threshold`, and what a `set_output` on the transformer itself
            # would otherwise leave here.
            codes = np.asarray(codes.toarray() if sparse.issparse(codes) else codes)
        return self._assemble(X, name, codes, selected)

    @override
    def transform(self, X: XType, **params: Any) -> XType:
        """Left to `ColumnTransformer`; only the column order is restored after it."""
        original_columns = _input_columns(X)
        return self._in_input_order(super().transform(X, **params), original_columns)

    def selected_columns(self) -> list[Any]:
        """The columns the named transformer holds, in the order it holds them.

        Empty when it selected nothing, when only a remainder is configured, or when
        the selection is not a list of column keys at all.
        """
        _, selected = self._named_selection()
        return selected or []

    def _may_assemble(self, X: XType, y: YType, params: dict[str, Any]) -> bool:
        """Whether the transformers as configured could be written into one array.

        Read off the specification rather than a fitted state, so that a transformer
        this rejects never sees the one-row fit below -- which for one that cannot be
        fitted on a single row, a quantile transform say, is the difference between a
        fallback and a crash.

        A `y` or routed metadata is declined outright: the one-row fit would have to
        be given something to match it, and no transformer with the shape assembled here
        has any use for either. So is an input that is neither an array nor a frame:
        a sparse one would be densified by the array written into here, which is not
        what `ColumnTransformer` hands back.
        """
        if y is not None or params or not isinstance(X, (np.ndarray, pd.DataFrame)):
            return False
        named = [
            (name, transformer)
            for name, transformer, _ in self.transformers
            if name != "remainder"
        ]
        output_config = getattr(self, "_sklearn_output_config", {}).get(
            "transform", get_config()["transform_output"]
        )
        return (
            len(named) <= 1
            and all(isinstance(t, OneToOneFeatureMixin) for _, t in named)
            and _hands_columns_through(self.remainder)
            and output_config == "default"
        )

    def _fit_column_bookkeeping(self, X: XType) -> XType:
        """Fit everything but the values, and return the one-row result that took.

        `ColumnTransformer.fit` is implemented as `fit_transform`, so fitting on
        the whole input would run the very transform this class replaces. Called on the
        parent because `self.fit` would land back here.
        """
        return ColumnTransformer.fit_transform(self, _head(X, 1), None)

    def _named_selection(self) -> tuple[str | None, list[Any] | None]:
        """The named transformer's name and the columns it holds, in its own order.

        `None` for the columns when the selection is not a list of keys -- a `slice`
        or a bare label -- which cannot be placed one column at a time. Both entries are
        empty when only a remainder is configured.
        """
        for name, _, columns in self.transformers_:
            if name == "remainder":
                continue
            if isinstance(columns, (list, np.ndarray, pd.Index)):
                return name, list(columns)
            return name, None
        return None, []

    def _can_assemble(self, X: XType, probe: XType, selected: list[Any] | None) -> bool:
        """Whether the layout just fitted is one that can be written column by column.

        The one-row `probe` is the fit's own answer on the output's width, so
        one-to-oneness is checked rather than taken on the mixin's word: every input
        column has to reach exactly one output column, or the array the assembly
        allocates is the wrong shape and the bookkeeping around it is wrong too.

        A frame carries one more condition. Every column the transformer does *not*
        take has to be plain float64 already, so writing it into the float64 output is a
        copy and not a conversion that could differ from the one `ColumnTransformer`
        would have done -- notably an `object` column, which a dtype selector skips
        and the stacking path carries through as objects until the caller's closing
        cast. And the column keys have to be unique, since each column is placed by key.
        """
        if selected is None or self.sparse_output_ or probe.shape[1] != X.shape[1]:
            return False
        if not isinstance(X, pd.DataFrame):
            return True
        if X.columns.has_duplicates:
            return False
        taken = set(selected)
        return all(
            dtype == _FLOAT64
            for column, dtype in zip(X.columns, X.dtypes, strict=True)
            if column not in taken
        )

    def _assemble(
        self,
        X: XType,
        name: str | None,
        codes: np.ndarray | None,
        selected: list[Any],
    ) -> np.ndarray:
        """Write the transformer's block and every other column to its final place."""
        destination, passthrough = self._output_positions(X, name, selected)
        out = np.empty(
            X.shape,
            dtype=self._assembled_dtype(X, codes, passthrough),
            order=self._assembled_order(X, codes, passthrough),
        )
        if codes is not None:
            out[:, destination] = codes
        # Per column, so the passthrough half never needs a full-width temporary of its
        # own; each write is a copy out of the container's own buffer.
        for position, source in passthrough:
            out[:, position] = _column_values(X, source)
        return out

    def _output_positions(
        self, X: XType, name: str | None, selected: list[Any]
    ) -> tuple[Any, list[tuple[int, int]]]:
        """Where the transformer's block, and each column it left, land in the output.

        Two layouts. Preserving the input order, every column stays where it came from,
        so the block is scattered back over the positions it was taken from. Otherwise
        it is `ColumnTransformer`'s, read off the fit's own `output_indices_`: the
        block takes its slice, and the columns the transformer did not take follow in
        input order.
        """
        positions = {column: index for index, column in enumerate(_input_columns(X))}
        taken = set(selected)
        remainder = [
            index for column, index in positions.items() if column not in taken
        ]
        if self.preserves_column_order:
            return (
                [positions[column] for column in selected],
                [(index, index) for index in remainder],
            )
        destination = slice(0, 0) if name is None else self.output_indices_[name]
        start = self.output_indices_["remainder"].start
        return destination, list(enumerate(remainder, start=start))

    def _assembled_dtype(
        self, X: XType, codes: np.ndarray | None, passthrough: list[tuple[int, int]]
    ) -> np.dtype:
        """The dtype `ColumnTransformer` would have stacked its way to.

        `np.concatenate` promotes across the blocks it stacks, so the columns handed
        through only weigh in when there are any: an input whose every column is
        transformed comes out as the transformer's own dtype. A frame weighs in as
        float64, which `_can_assemble` has established every one of its passthrough
        columns already is.
        """
        input_dtype = _FLOAT64 if isinstance(X, pd.DataFrame) else X.dtype
        if codes is None:
            return input_dtype
        if not passthrough:
            return codes.dtype
        return np.result_type(codes.dtype, input_dtype)

    def _assembled_order(
        self, X: XType, codes: np.ndarray | None, passthrough: list[tuple[int, int]]
    ) -> Literal["C", "F"]:
        """The memory layout `ColumnTransformer` would have arrived at.

        Not cosmetic: the sklearn SVD that can run downstream of these steps converges
        to a different basis on a C- than on a Fortran-contiguous input, so a drift here
        would quietly rotate those features.

        Preserving the input order, the layout is the one the gather in
        `_preserve_order` produced, which numpy makes column-major for any output
        wider than the single column where the two orders coincide. Otherwise it is
        `np.concatenate`'s: Fortran only when every block it stacks already is, and
        row-major otherwise -- so the blocks decide. The passthrough block's layout is
        read off a two-row slice rather than the full-size block this exists not to
        build; column selection gives the same answer at either height.
        """
        if self.preserves_column_order:
            return "F"
        blocks = [] if codes is None else [codes]
        if passthrough:
            sources = [source for _, source in passthrough]
            blocks.append(np.asarray(_columns_at(_head(X, 2), sources)))
        return "F" if all(block.flags.f_contiguous for block in blocks) else "C"

    def _in_input_order(self, Xt: XType, original_columns: pd.Index | range) -> XType:
        """`Xt` with the input's column order restored, if this class preserves it."""
        if not self.preserves_column_order:
            return Xt
        return self._preserve_order(X=Xt, original_columns=original_columns)

    def _preserve_order(
        self, X: XType, original_columns: list | range | pd.Index
    ) -> XType:
        check_is_fitted(self)
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D (shape={X.shape})"
        for name, _, col_subset in reversed(self.transformers_):
            if (
                len(col_subset) > 0
                and len(col_subset) < X.shape[-1]
                and name != "remainder"
            ):
                col_subset_list = list(col_subset)
                # Map original columns to indices in the transformed array
                transformed_columns = col_subset_list + [
                    c for c in original_columns if c not in col_subset_list
                ]
                indices = [transformed_columns.index(c) for c in original_columns]
                # restore the column order from before the transfomer has been applied
                X = X.iloc[:, indices] if isinstance(X, pd.DataFrame) else X[:, indices]
        return X


class OrderPreservingColumnTransformer(EfficientColumnTransformer):
    """An EfficientColumnTransformer that preserves the column order after transform."""

    preserves_column_order: ClassVar[bool] = True

    def __init__(
        self,
        transformers: Sequence[
            tuple[
                str,
                BaseEstimator,
                str
                | int
                | slice
                | Iterable[str | int]
                | Callable[[Any], Iterable[str | int]],
            ]
        ],
        **kwargs: Any,
    ):
        """Implementation base on https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html.

        Parameters
        ----------
        transformers : sequence of (name, transformer, columns) tuples
            List of (name, transformer, columns) tuples specifying the transformers.
        **kwargs : additional keyword arguments
            Passed to sklearn.compose.ColumnTransformer.
        """
        super().__init__(transformers=transformers, **kwargs)

        # Check if there is a single transformer, of subtype OneToOneFeatureMixin
        assert all(
            isinstance(t, OneToOneFeatureMixin)
            for name, t, _ in transformers
            if name != "remainder"
        ), (
            "OrderPreservingColumnTransformer currently only supports transformers "
            "that are instances of OneToOneFeatureMixin."
        )

        assert len([t for name, _, t in transformers if name != "remainder"]) <= 1, (
            "OrderPreservingColumnTransformer only supports up to one transformer."
        )


def get_ordinal_encoder(
    *,
    numpy_dtype: np.floating = DEFAULT_NUMPY_PREPROCESSING_DTYPE,  # type: ignore
) -> OrderPreservingColumnTransformer:
    """Create a ColumnTransformer that ordinally encodes string/category columns."""
    oe = OrdinalEncoder(
        # TODO: Could utilize the categorical dtype values directly instead of "auto"
        categories="auto",
        dtype=numpy_dtype,  # type: ignore
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,  # Missing stays missing
    )
    # Documentation of sklearn, deferring to pandas is misleading here. It's done
    # using a regex on the type of the column, and using `object`, `"object"` and
    # `np.object` will not pick up strings.
    to_convert = ["category", "string"]
    return OrderPreservingColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
