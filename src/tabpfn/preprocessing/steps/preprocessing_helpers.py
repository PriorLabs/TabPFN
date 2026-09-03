#  Copyright (c) Prior Labs GmbH 2026.

"""Feature Preprocessing Transformer Step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import Self, override

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import get_config
from sklearn.base import (
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

    `True` when the frame's layout cannot be read, so that an unknown one is copied.
    """
    # a single-block frame is handed out as that block -- writeable and pointing at
    # whatever the frame was built from before pandas 3, which for a numeric array
    # input is the caller's own. Anything wider is built fresh, so it is private.
    blocks = getattr(getattr(X, "_mgr", None), "blocks", None)
    return blocks is None or len(blocks) <= 1


class EfficientColumnTransformer(ColumnTransformer):
    """A `ColumnTransformer` that assembles its output into one preallocated array.

    Saves the two extra full-size arrays its parent reaches a result through -- the
    stack and, when preserving column order, the reorder -- by writing every column
    straight to its final place. `fit` goes further and builds no output at all.

    Only the narrow shape this codebase needs can be assembled: at most one named
    transformer, one-to-one, over a subset of the columns, plus
    a remainder that hands the rest through untouched. Anything else falls back to
    `ColumnTransformer`, so this stays a drop-in replacement for it.

    `transform` is left to `ColumnTransformer` for its checks against the input seen at
    fit, except where those are settled in advance and there is provably nothing to do.
    """

    # Whether the output keeps the input's column order rather than
    # `ColumnTransformer`'s `[transformed, remainder]` one. A class attribute rather
    # than a constructor parameter, so sklearn's `clone` and `get_params` stay exactly
    # its parent's.
    preserves_column_order: ClassVar[bool] = False

    @override
    def fit(self, X: XType, y: YType = None, **params: Any) -> Self:
        """Fit without building the transformed array `ColumnTransformer.fit` builds.

        Decided on the same conditions as the assembly, so the state left here is either
        sklearn's own or one those conditions vouch for.
        """
        if not self._may_assemble_a_fit(X, y, params):
            return super().fit(X, y, **params)
        # widths and output positions do not depend on the row count here, so one row
        # settles all of the bookkeeping
        probe = self._fit_column_bookkeeping(X)
        name, selected = self._named_selection()
        if not self._can_assemble(X, probe, selected):
            return super().fit(X, y, **params)
        if selected:
            # the values are all that is left to learn, in one pass, nothing stacked
            self.named_transformers_[name].fit(_column_subset(X, selected))
        return self

    @override
    def fit_transform(self, X: XType, y: YType = None, **params: Any) -> XType:
        """Fit and transform, writing each column straight to its final place."""
        original_columns = _input_columns(X)
        if not self._may_assemble_a_fit(X, y, params):
            return self._maybe_in_input_order(
                super().fit_transform(X, y, **params), original_columns
            )

        probe = self._fit_column_bookkeeping(X)
        name, selected = self._named_selection()
        if not self._can_assemble(X, probe, selected):
            # Bail before the named transformer has learned anything from the values:
            # the fallback learns them again anyway, so a pass over the data here would
            # be wasted. Only the one-row fit is lost, which costs no pass at all.
            return self._maybe_in_input_order(
                super().fit_transform(X, y, **params), original_columns
            )

        return self._assemble(X, name, selected)

    @override
    def transform(self, X: XType, **params: Any) -> XType:
        """Left to `ColumnTransformer`, unless it provably cannot change a value.

        Its checks against the input seen at fit -- width, names, order -- are why it is
        left there at all: the assembly makes none of them.
        """
        if self._changes_no_value(X, params):
            return self._assemble(X, None, [])
        original_columns = _input_columns(X)
        return self._maybe_in_input_order(
            super().transform(X, **params), original_columns
        )

    def selected_columns(self) -> list[Any]:
        """The columns the named transformer holds, in the order it holds them.

        Empty when it holds none, when only a remainder is configured, or when the
        selection is not a list of column keys at all.
        """
        _, selected = self._named_selection()
        return selected or []

    def _may_assemble_a_fit(self, X: XType, y: YType, params: dict[str, Any]) -> bool:
        """Whether a fit over this input can be assembled instead of stacked."""
        return (
            y is None
            and self._is_one_to_one
            and self._selects_the_same_from_one_row
            and self._may_assemble(X, params)
        )

    @property
    def _selects_the_same_from_one_row(self) -> bool:
        """Whether every column selection can be made based on a single row."""
        return all(
            not callable(columns) or isinstance(columns, make_column_selector)
            for name, _, columns in self.transformers
            if name != "remainder"
        )

    def _may_assemble(self, X: XType, params: dict[str, Any]) -> bool:
        """Whether an output of this shape could be written into one array at all."""
        if (
            # extra fit/predict params would be ignored:
            params
            # no transformer weight support:
            or self.transformer_weights
            # input can't be metadata:
            or not isinstance(X, (np.ndarray, pd.DataFrame))
        ):
            return False
        output_config = getattr(self, "_sklearn_output_config", {}).get(
            "transform", get_config()["transform_output"]
        )
        # a `set_output` asking for a frame gets none from here
        return is_identity_transformer(self.remainder) and output_config == "default"

    @property
    def _is_one_to_one(self) -> bool:
        """Whether at most one transformer is configured, and it is one-to-one."""
        # Being one-to-one does not make the probing one-row `transformer.fit` correct,
        # but the fast path refits on every row, so nothing the probe learned survives
        # into the output.
        named = [t for name, t, _ in self.transformers if name != "remainder"]
        return len(named) <= 1 and all(
            isinstance(transformer, OneToOneFeatureMixin) for transformer in named
        )

    def _changes_no_value(self, X: XType, params: dict[str, Any]) -> bool:
        """Whether transforming `X` provably leaves every value where it was."""
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
        # The names too, in order: sklearn lines a reordered frame back up with the
        # fit-time order where the assembly would keep the input's own, and skipping
        # `transform` skips the check that would reject a frame not lining up at all.
        names = getattr(self, "feature_names_in_", None)
        return (
            names is None
            or not isinstance(X, pd.DataFrame)
            # Compared as plain lists to be dtype-insensitive.
            or list(X.columns) == list(names)
        )

    def _fit_column_bookkeeping(self, X: XType) -> XType:
        """Fit everything but the values, and return the one-row result that took."""
        # Up the chain, not through `self.fit`: `ColumnTransformer.fit` is implemented
        # as `fit_transform`, so fitting on the whole input would run the very transform
        # this class replaces, and `self.fit` would land back in the override above.
        return super().fit_transform(_head(X, 1), None)

    def _named_selection(self) -> tuple[str | None, list[Any] | None]:
        """The named transformer's name and the columns it holds, in its own order.

        `None` for the columns when the selection is not a list of keys -- a `slice`
        or a bare label -- which cannot be placed one column at a time. Both entries
        are empty when only a remainder is configured.
        """
        for name, _, columns in self.transformers_:
            if name == "remainder":
                continue
            if isinstance(columns, (list, np.ndarray, pd.Index)):
                return name, list(columns)
            return name, None
        # fast path: nothing gets transformed
        return None, []

    def _can_assemble(self, X: XType, probe: XType, selected: list[Any] | None) -> bool:
        """Whether the layout just fitted can be written out column by column."""
        return (
            selected is not None
            and not self.sparse_output_
            # the one-row `probe` is the fit's own answer on the output's width: every
            # input column has to reach exactly one output column, or the array
            # allocated below is the wrong shape and the bookkeeping around it wrong too
            and probe.shape[1] == X.shape[1]
            and self._can_place_columns(X, selected)
        )

    def _can_place_columns(self, X: XType, selected: list[Any]) -> bool:
        """Whether `X`'s columns can be written verbatim into the output."""
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
        """
        if not selected:
            return None
        # fit_transform rather than a fit and then a transform: one pass over one slice
        # of the selected columns, where a second pass would hold that slice throughout
        transformed = self.named_transformers_[name].fit_transform(
            _column_subset(X, selected)
        )
        # What `_hstack` does with a sparse block whose density beat `sparse_threshold`,
        # and what a `set_output` on the transformer itself would otherwise leave here.
        return np.asarray(
            transformed.toarray() if sparse.issparse(transformed) else transformed
        )

    def _assemble(self, X: XType, name: str | None, selected: list[Any]) -> np.ndarray:
        """Write the transformed columns and all the others to their final place.

        The array returned is one nothing else has ever referenced, which is what lets a
        caller write into what it gets back.
        """
        destination, passthrough = self._output_positions(X, name, selected)
        # computed here rather than handed in, so it can be dropped once written
        transformed = self._transformed_block(X, name, selected)
        dtype = self._assembled_dtype(X, transformed, passthrough)
        order = self._assembled_order(X, transformed, passthrough)

        if transformed is None and (
            not isinstance(X, pd.DataFrame) or to_numpy_may_alias(X)
        ):
            # Nothing was transformed, so the output is the whole input converted, in
            # input order -- which both layouts agree on when the selection is empty.
            # Copied because `to_numpy` can hand back a view of a block the frame
            # itself still holds.
            values = (
                X.to_numpy(dtype=dtype, copy=False)
                if isinstance(X, pd.DataFrame)
                else X
            )
            return np.array(values, dtype=dtype, order=order, copy=True)

        out = np.empty(X.shape, dtype=dtype, order=order)
        if transformed is not None:
            out[:, destination] = transformed
            # `np.empty` takes memory only as it is written to, so dropping this here
            # keeps the peak at one full-size array rather than one plus a block
            del transformed
        # Per column, so the passthrough half never needs a full-width temporary of its
        # own, and the frame's own blocks are left alone -- `to_numpy` would rearrange
        # them in place before pandas 3, and hand back a view of what it just built.
        for position, source in passthrough:
            out[:, position] = _column_values(X, source)
        return out

    def _output_positions(
        self, X: XType, name: str | None, selected: list[Any]
    ) -> tuple[Any, list[tuple[int, int]]]:
        """Where the transformed columns, and all the others, land in the output.

        Returns:
            destination: Where the transformed columns go, in the order the transformer
                holds them -- a `slice` of the output, or the list of positions to
                scatter them over. Selects nothing when nothing was selected.
            passthrough: One `(output position, input position)` pair for every column
                the transformer did not take, in input order.
        """
        positions = {column: index for index, column in enumerate(_input_columns(X))}
        taken = set(selected)
        remainder = [
            index for column, index in positions.items() if column not in taken
        ]
        if self.preserves_column_order:
            # every column stays where it came from, so the transformed ones are
            # scattered back over the positions they were taken from
            return (
                [positions[column] for column in selected],
                [(index, index) for index in remainder],
            )
        # `ColumnTransformer`'s layout, read off the fit's own `output_indices_`: the
        # transformed columns take their slice, and the rest follow in input order
        destination = slice(0, 0) if name is None else self.output_indices_[name]
        start = self.output_indices_["remainder"].start
        return destination, list(enumerate(remainder, start=start))

    def _assembled_dtype(
        self,
        X: XType,
        transformed: np.ndarray | None,
        passthrough: list[tuple[int, int]],
    ) -> np.dtype:
        """The dtype `ColumnTransformer` would have stacked its way to."""
        # a frame weighs in as float64, which `_can_assemble` has established every one
        # of its passthrough columns already is
        input_dtype = _FLOAT64 if isinstance(X, pd.DataFrame) else X.dtype
        if transformed is None:
            return input_dtype
        if not passthrough:
            # every column was transformed, so nothing else weighs in
            return transformed.dtype
        # `np.concatenate` promotes across the blocks it stacks
        return np.result_type(transformed.dtype, input_dtype)

    def _assembled_order(
        self,
        X: XType,
        transformed: np.ndarray | None,
        passthrough: list[tuple[int, int]],
    ) -> Literal["C", "F"]:
        """The memory layout `ColumnTransformer` would have arrived at.

        Not cosmetic: the sklearn SVD that can run downstream of these steps settles on
        a different answer per layout, so a drift here would quietly rotate features.
        """
        if self.preserves_column_order:
            # what the reorder in `_in_input_order` leaves, for any output wider than
            # the single column where the two layouts coincide
            return "F"
        blocks = [] if transformed is None else [transformed]
        # A frame's passthrough block is column-major
        if passthrough and not isinstance(X, pd.DataFrame):
            sources = [source for _, source in passthrough]
            # read off two rows rather than the full-size block this exists not to
            # build; selecting columns gives the same layout at either height
            blocks.append(np.asarray(_columns_at(_head(X, 2), sources)))
        # `np.concatenate`'s answer: column-major only when every block it stacks is
        return "F" if all(block.flags.f_contiguous for block in blocks) else "C"

    def _maybe_in_input_order(
        self, Xt: XType, original_columns: pd.Index | range
    ) -> XType:
        """`Xt` with the input's column order restored, if this class preserves it."""
        if not self.preserves_column_order:
            return Xt
        return self._in_input_order(X=Xt, original_columns=original_columns)

    def _stacked_positions(
        self, original_columns: list | range | pd.Index
    ) -> list[int]:
        """Where each input column sits in the output `ColumnTransformer` stacked.

        Returns:
            indices (list[int]): `indices[i]` is the column index in the output
                corresponding to input column i.
        """
        name, selected = self._named_selection()
        if selected is None:
            raise TypeError(
                f"The {name!r} transformer selects its columns by something other than "
                "a list of keys, which cannot be placed back in the input's order."
            )
        # Every column is somewhere in `ColumnTransformer`'s two blocks: at its rank in
        # the selection, or, unselected, at the next place after that block.
        rank = {column: index for index, column in enumerate(selected)}
        next_index = len(selected)
        indices = []
        for column in original_columns:
            index = rank.get(column)
            if index is None:
                # column wasn't input to the transformer
                index, next_index = next_index, next_index + 1
            indices.append(index)
        return indices

    def _in_input_order(
        self, X: XType, original_columns: list | range | pd.Index
    ) -> XType:
        """`X` with the columns back where the input had them."""
        check_is_fitted(self)
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D (shape={X.shape})"
        indices = self._stacked_positions(original_columns)
        # Nothing moved, and the gather below is a full-size copy -- but only skipped
        # where it would also leave the layout alone, since it hands back a
        # Fortran-contiguous array whatever it was given and `_assembled_order` says
        # what reads that. A frame has no such layout to keep.
        if (not isinstance(X, np.ndarray) or X.flags.f_contiguous) and all(
            index == position for position, index in enumerate(indices)
        ):
            return X
        # restore the column order from before the transformer has been applied
        return X.iloc[:, indices] if isinstance(X, pd.DataFrame) else X[:, indices]


class OrderPreservingColumnTransformer(EfficientColumnTransformer):
    """An EfficientColumnTransformer that preserves the column order after transform.

    Its parameters are `ColumnTransformer`'s, narrowed to what restoring that order
    needs: at most one transformer, one-to-one, over a list of column keys or a callable
    returning one.
    """

    preserves_column_order: ClassVar[bool] = True

    @override
    def _validate_transformers(self) -> None:
        """What restoring the input order needs of the transformers, checked at fit.

        Each of these, unchecked, has the reorder returning columns in the wrong order
        rather than failing.
        """
        # At fit rather than in `__init__`: `set_params` writes `transformers` straight
        # onto the estimator, so a contract checked at construction is one every sklearn
        # caller that tunes a parameter steps around without seeing it.
        super()._validate_transformers()
        named = [
            (name, transformer, columns)
            for name, transformer, columns in self.transformers
            if name != "remainder"
        ]
        if len(named) > 1:
            raise ValueError(
                "OrderPreservingColumnTransformer only supports up to one transformer, "
                f"got {[name for name, _, _ in named]}."
            )
        for name, transformer, columns in named:
            if not isinstance(transformer, OneToOneFeatureMixin):
                raise ValueError(
                    "OrderPreservingColumnTransformer only supports transformers that "
                    f"are instances of OneToOneFeatureMixin, which {name!r} "
                    f"({type(transformer).__name__}) is not."
                )
            # the reorder places one column at a time, by key
            if not callable(columns) and not isinstance(
                columns, (list, np.ndarray, pd.Index)
            ):
                raise ValueError(
                    "OrderPreservingColumnTransformer only supports selecting columns "
                    "by a list of keys, or a callable returning one -- not by a slice "
                    f"or a single label, as {name!r} selects by {columns!r}."
                )

    @override
    def _validate_remainder(self, X: XType) -> None:
        """The remainder has to hand back every column it holds, checked at fit.

        The reorder puts each input column back where it came from, so one that never
        reached the output has no place to go: the gather runs off the end of the
        result, or -- where the two orders happen to coincide -- silently returns a
        narrower array as though it were in input order.
        """
        super()._validate_remainder(X)
        # One-to-one for the same reason the named transformer is, plus the identity
        # `FunctionTransformer` this codebase passes, which is not one.
        if not (
            is_identity_transformer(self.remainder)
            or isinstance(self.remainder, OneToOneFeatureMixin)
        ):
            raise ValueError(
                "OrderPreservingColumnTransformer only supports a remainder that hands "
                "every column it holds back, and "
                f"{self.remainder!r} does not. Note that `ColumnTransformer`'s default "
                "is 'drop': pass `remainder=FunctionTransformer()` to keep them."
            )

    @override
    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """The names of the output's columns, in the order the output has them."""
        names = super().get_feature_names_out(input_features)
        # the input's own keys, which is what the selection is expressed in
        original_columns = getattr(self, "feature_names_in_", None)
        if original_columns is None:
            original_columns = range(self.n_features_in_)
        return names[self._stacked_positions(list(original_columns))]


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
