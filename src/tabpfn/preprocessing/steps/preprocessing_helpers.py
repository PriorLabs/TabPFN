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
from tabpfn.preprocessing.steps.utils import is_identity_transformer

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


def to_numpy_may_alias(X: pd.DataFrame) -> bool:
    """Whether `X.to_numpy()` can hand back a view of the frame's own buffer.

    A single-block frame is handed out as that block: read-only under copy-on-write,
    writeable and aliasing whatever the frame was built from without it (pandas < 3),
    which for a numeric ndarray input is the caller's own array. Anything wider has to
    be materialised into a new array first, so what comes back is private.

    Defensively `True` when the block internals are unavailable, so an unrecognised
    layout is copied rather than handed out.
    """
    blocks = getattr(getattr(X, "_mgr", None), "blocks", None)
    return blocks is None or len(blocks) <= 1


def _converts_in_one_call(X: XType) -> bool:
    """Whether `X` can hand out every column at once more cheaply than one at a time.

    An array can, trivially. A frame only when `to_numpy` would hand back a block it
    already holds, so that copying it is the single allocation -- not when it would
    have to materialise one, and above all not when it would consolidate the frame in
    place to do so, which is what pandas before 3 does and what costs a second
    full-size buffer on top.
    """
    return not isinstance(X, pd.DataFrame) or to_numpy_may_alias(X)


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

    `transform` stays on sklearn, which validates the input against the one seen at fit
    -- column count, names, order -- where the assembly makes no such check. The one
    exception is the input for which those checks are settled in advance and the fitted
    transformer provably changes nothing: there the step is a cast, and the cast is
    assembled.
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
        if not (y is None and self._is_one_to_one and self._may_assemble(X, params)):
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
        if not (y is None and self._is_one_to_one and self._may_assemble(X, params)):
            return self._maybe_in_input_order(
                super().fit_transform(X, y, **params), original_columns
            )

        probe = self._fit_column_bookkeeping(X)
        name, selected = self._named_selection()
        if not self._can_assemble(X, probe, selected):
            # Bail before the named transformer has learned anything from the values:
            # the fallback learns them again from scratch, so doing it first would be a
            # wasted pass over the data. Only the one-row fit is lost, which costs no
            # pass at all.
            return self._maybe_in_input_order(
                super().fit_transform(X, y, **params), original_columns
            )

        return self._assemble(X, name, selected)

    @override
    def transform(self, X: XType, **params: Any) -> XType:
        """Left to `ColumnTransformer`, unless it provably cannot change a value.

        Its checks against the input seen at fit -- width, names, order -- are why it is
        left there at all: the assembly makes none of them. So the one case taken here
        is the one where they are settled in advance, by `_changes_no_value`.
        """
        if self._changes_no_value(X, params):
            return self._assemble(X, None, [])
        original_columns = _input_columns(X)
        return self._maybe_in_input_order(
            super().transform(X, **params), original_columns
        )

    def selected_columns(self) -> list[Any]:
        """The columns the named transformer holds, in the order it holds them.

        Empty when it selected nothing, when only a remainder is configured, or when
        the selection is not a list of column keys at all.
        """
        _, selected = self._named_selection()
        return selected or []

    def _may_assemble(self, X: XType, params: dict[str, Any]) -> bool:
        """Whether an output of this shape could be written into one array at all.

        Routed metadata is declined outright: nothing with the shape assembled here has
        any use for it. So is an input that is neither an array nor a frame -- a sparse
        one would be densified by the array written into here, which is not what
        `ColumnTransformer` hands back -- and a `set_output` asking for a frame, which
        this returns none of.
        """
        if params or not isinstance(X, (np.ndarray, pd.DataFrame)):
            return False
        output_config = getattr(self, "_sklearn_output_config", {}).get(
            "transform", get_config()["transform_output"]
        )
        return is_identity_transformer(self.remainder) and output_config == "default"

    @property
    def _is_one_to_one(self) -> bool:
        """Whether at most one transformer is configured, and it maps column to column.

        Read off the specification rather than a fitted state, so that a transformer
        this rejects never sees the one-row fit below -- which for one that cannot be
        fitted on a single row, a quantile transform say, is the difference between a
        fallback and a crash.
        """
        named = [t for name, t, _ in self.transformers if name != "remainder"]
        return len(named) <= 1 and all(
            isinstance(transformer, OneToOneFeatureMixin) for transformer in named
        )

    def _changes_no_value(self, X: XType, params: dict[str, Any]) -> bool:
        """Whether transforming `X` provably leaves every value where it was.

        Three things have to hold. The fitted transformer must have selected nothing, so
        the step is its passthrough remainder and there is no code to compute. `X` must
        be one the columns can be written out of. And it must line up with the input
        seen at fit -- same width, same names in the same order -- because sklearn lines
        a reordered frame back up with the fit-time order where the assembly would keep
        the input's own, and because skipping `transform` skips the checks that would
        have caught a frame not lining up at all.
        """
        if not hasattr(self, "transformers_"):
            # Unfitted. `transform` is the one that should say so.
            return False
        # An empty list, not merely a falsy one: a selection this cannot read is one
        # whose columns might well be transformed.
        if self._named_selection()[1] != []:
            return False
        if (
            not self._may_assemble(X, params)
            or self.sparse_output_
            or self.n_features_in_ != X.shape[1]
            or not self._can_place_columns(X, [])
        ):
            return False
        names = getattr(self, "feature_names_in_", None)
        return (
            names is None
            or not isinstance(X, pd.DataFrame)
            # Compared as plain lists to be dtype-insensitive.
            or list(X.columns) == list(names)
        )

    def _fit_column_bookkeeping(self, X: XType) -> XType:
        """Fit everything but the values, and return the one-row result that took.

        `ColumnTransformer.fit` is implemented as `fit_transform`, so fitting on the
        whole input would run the very transform this class replaces. Taken up the chain
        rather than through `self.fit`, which would land back in the override above.
        """
        return super().fit_transform(_head(X, 1), None)

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
            # slow path: can't identify the column order
            return name, None
        # fast path: nothing gets transformed
        return None, []

    def _can_assemble(self, X: XType, probe: XType, selected: list[Any] | None) -> bool:
        """Whether the layout just fitted is one that can be written column by column.

        The one-row `probe` is the fit's own answer on the output's width, so
        one-to-oneness is checked rather than taken on the mixin's word: every input
        column has to reach exactly one output column, or the array the assembly
        allocates is the wrong shape and the bookkeeping around it is wrong too.
        """
        return (
            selected is not None
            and not self.sparse_output_
            and probe.shape[1] == X.shape[1]
            and self._can_place_columns(X, selected)
        )

    def _can_place_columns(self, X: XType, selected: list[Any]) -> bool:
        """Whether `X`'s columns can be written verbatim into the output.

        Args:
            X (XType): Input.
            selected (list[Any]): Columns to be processed by the transformer.

        Returns:
            bool
        """
        if not isinstance(X, pd.DataFrame):
            # arrays are always ok
            return True
        if X.columns.has_duplicates:
            # can't assign columns by key if multiple share the same key
            return False
        taken = set(selected)
        # columns the transformer does *not* take need to already be float64,
        # otherwise an assignment into the output would be a conversion that could
        # differ from the one ColumnTransformer would have done
        return all(
            dtype == _FLOAT64
            for column, dtype in zip(X.columns, X.dtypes, strict=True)
            if column not in taken
        )

    def _transformed_block(
        self, X: XType, name: str | None, selected: list[Any]
    ) -> np.ndarray | None:
        """The named transformer's own output over the columns it selected.

        `None` when there is nothing selected, without touching the transformer -- which
        is the only case `transform` reaches this through, so nothing is fitted there.

        `fit_transform` rather than a fit and then a transform: it is one pass over one
        slice of the selected columns, where a second pass would have to hold that slice
        for the whole of it.
        """
        if not selected:
            return None
        codes = self.named_transformers_[name].fit_transform(
            _column_subset(X, selected)
        )
        # What `_hstack` does with a sparse block whose density beat `sparse_threshold`,
        # and what a `set_output` on the transformer itself would otherwise leave here.
        return np.asarray(codes.toarray() if sparse.issparse(codes) else codes)

    def _assemble(self, X: XType, name: str | None, selected: list[Any]) -> np.ndarray:
        """Write the transformer's block and every other column to its final place.

        The array returned is one nothing else has ever referenced, which is what lets a
        caller write into what it gets back. Notably it is never `to_numpy`'s result as
        it stands: pandas 3 materialises a many-block frame into a private array, which
        could be taken as it is, but earlier versions consolidate the frame *in place*
        first and return a view of the block they just built -- taking that would write
        through into the frame, and copying it costs a second full-size buffer on top of
        the consolidation. Going column by column costs one buffer on every version, and
        leaves the frame's own blocks alone.

        The transformer's block is computed here rather than handed in so that it can be
        dropped the moment it is written. `np.empty` faults a page in only when it is
        written to, so the output appears in RSS as its columns are filled: a caller
        still holding the block while the passthrough half fills the rest would have the
        whole output *and* the block resident at once, which the block dying first
        avoids.
        """
        destination, passthrough = self._output_positions(X, name, selected)
        codes = self._transformed_block(X, name, selected)
        dtype = self._assembled_dtype(X, codes, passthrough)
        order = self._assembled_order(X, codes, passthrough)

        if codes is None and _converts_in_one_call(X):
            # Nothing was transformed, so the output is the whole input converted, in
            # input order -- which both layouts agree on when the selection is empty.
            # One call for all of it where that is the cheaper way to reach the same
            # single allocation. What it saves is per-column overhead, so it grows with
            # the column count: measured against the loop below, interleaved so that
            # neither order is favoured, nothing at 200,000 x 300, 12% of the wall time
            # at 50,000 x 2,000 and 26% at 20,000 x 5,000.
            values = (
                X.to_numpy(dtype=dtype, copy=False)
                if isinstance(X, pd.DataFrame)
                else X
            )
            return np.array(values, dtype=dtype, order=order, copy=True)

        out = np.empty(X.shape, dtype=dtype, order=order)
        if codes is not None:
            out[:, destination] = codes
            del codes
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

    def _maybe_in_input_order(
        self, Xt: XType, original_columns: pd.Index | range
    ) -> XType:
        """`Xt` with the input's column order restored, if this class preserves it."""
        if not self.preserves_column_order:
            return Xt
        return self._in_input_order(X=Xt, original_columns=original_columns)

    def _in_input_order(
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
                # Where each input column landed: the transformer's block first, in the
                # order it selected, then the columns it left, in input order. So one
                # lookup over the selection answers it -- what the selection does not
                # hold follows in the order it is read here, which is the order those
                # columns were handed through in.
                rank = {column: index for index, column in enumerate(col_subset)}
                next_index = len(col_subset)
                indices = []
                for column in original_columns:
                    index = rank.get(column)
                    if index is None:
                        index, next_index = next_index, next_index + 1
                    indices.append(index)
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
