#  Copyright (c) Prior Labs GmbH 2026.

"""Expand a DataFrame's text columns into numeric features before validation.

With `TRANSFORM_TEXT` on, a text column becomes tf-idf features over its
character n-grams, reduced by a truncated SVD (`skrub.StringEncoder`). A column
is text by its dtype and its values: a `string` column (the default for strings
from pandas 3.0), in any storage, or a pyarrow string column, with more than a
cutoff of distinct values that do not all parse as numbers. A column declared in
`categorical_features_indices` is never text,
nor is an `object` or `category` column, whatever it holds. Every other column
passes through unchanged, as does everything with the flag off. Only `DataFrame`
columns are inspected: any other input passes through unchanged.

At predict, a string is encoded by the character n-grams it shares with the fit
column, so an unseen sentence in the same language lands near its neighbours. A
missing value is encoded as the empty string, which shares none, and so becomes
an all-zero row, like an ID or a string in another script with no n-gram in
common. The model never sees a missing value in an expanded column, then, and
cannot tell a missing string from one with nothing in common.

Only `TabPFNClassifier` and `TabPFNRegressor` run this, right after
`DateTransformer`. The fine-tuning estimators validate their input directly.

Column handling is positional throughout: labels are the caller's and can repeat.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.exceptions import NotFittedError
from skrub import StringEncoder

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import make_names_unique
from tabpfn.preprocessing.modality_detection import (
    _get_unique_with_sklearn_compatible_error,
    _is_numeric_pandas_series,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType

__all__ = ["TextTransformer"]


@dataclasses.dataclass
class _FittedTextColumn:
    """One input column's fitted encoder and its features' names."""

    encoder: StringEncoder
    output_names: list[str]


class TextTransformer:
    """Expands each text column into numeric features, when asked to.

    Used like `DateTransformer`, and right after it: `fit_transform` once at fit
    time, keep the instance as `text_transformer_`, `transform` at predict time.

    Args:
        categorical_indices: Indices the caller declared categorical, as positions
            in the `X` handed here. Never text, whatever they hold.
        transform_text: Whether a `string` text column is expanded into tf-idf
            features over its character n-grams, reduced by a truncated SVD.
            Off, every column passes through unchanged.
        min_cardinality_for_text: Distinct-value count above which a string
            column counts as text.
        n_components: Features a text column is expanded into, at most: fewer
            when its n-gram vocabulary is smaller.

    Attributes:
        fitted_columns_: Input position -> the encoder fitted on that column and
            the names of the features it makes. Empty when nothing was expanded.
        feature_names_out_: The transformed frame's column labels as strings, or
            `None` when the input was not a `DataFrame` and so has no labels.
    """

    fitted_columns_: dict[int, _FittedTextColumn]
    feature_names_out_: list[str] | None

    def __init__(
        self,
        *,
        categorical_indices: Sequence[int] | None = None,
        transform_text: bool = False,
        min_cardinality_for_text: int = 30,
        n_components: int = 30,
    ) -> None:
        self._declared_categorical = set(categorical_indices or ())
        self._transform_text = transform_text
        self._min_cardinality_for_text = min_cardinality_for_text
        self._n_components = n_components

    @property
    def expanded_indices(self) -> list[int]:
        """Input positions that were expanded into text features, ascending."""
        self._check_is_fitted()
        return sorted(self.fitted_columns_)

    def fit(self, X: XType) -> TextTransformer:
        """Fit one encoder per text column in `X`, if `transform_text` is on.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            Itself, fitted.
        """
        self._fit(X)
        return self

    def fit_transform(self, X: XType) -> XType:
        """`fit(X).transform(X)`, encoding each text column only once.

        Expansion changes the column count: `feature_names_out_` reports the
        resulting labels and `output_indices` moves input indices to match.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            `X`, converted as `transform` would.
        """
        blocks = self._fit(X)
        if not isinstance(X, pd.DataFrame) or not blocks:
            return X
        return _drop_and_append(X.reset_index(drop=True), self.expanded_indices, blocks)

    def transform(self, X: XType) -> XType:
        """Reapply the expansion `fit` decided on, so the width holds.

        Each expanded position is read as strings whatever its dtype now: a
        `string` column at fit can arrive as `object` at predict on an older
        pandas, or as `category`. Only a column holding numbers is refused; one
        that is all missing arrives as float, and is read as missing strings.

        Args:
            X: The data, before any dtype fixing.

        Raises:
            NotFittedError: If `fit` has not run yet.
            TabPFNValidationError: If a position `fit` expanded holds numbers now,
                or if `fit` expanded columns and `X` is not a `DataFrame`, the
                only input that can carry them.
        """
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            _refuse_array_after_expansion(X, self.expanded_indices)
            return X
        if not self.expanded_indices:
            return X
        _refuse_numeric(
            X,
            [
                i
                for i in self.expanded_indices
                if pd.api.types.is_numeric_dtype(X.dtypes.iloc[i])
                and X.iloc[:, i].notna().any()
            ],
        )
        blocks = [
            self._apply_one(X.iloc[:, i], self.fitted_columns_[i])
            for i in self.expanded_indices
        ]
        return _drop_and_append(X.reset_index(drop=True), self.expanded_indices, blocks)

    def output_indices(self, indices: Sequence[int] | None) -> list[int] | None:
        """Where each of `indices`, input positions, sits in the transformed frame.

        A kept column shifts down by however many expanded columns sat ahead of
        it. An expanded position is never asked for: the one caller passes
        declared-categorical indices, which are never expanded.

        Args:
            indices: Input positions, or `None` for none declared.

        Returns:
            The same positions in the transformed frame, or `None` for `None`.
        """
        self._check_is_fitted()
        if indices is None:
            return None
        expanded = self.expanded_indices
        return [i - sum(1 for j in expanded if j < i) for i in indices]

    def _check_is_fitted(self) -> None:
        # By hand: sklearn's `check_is_fitted` requires a `BaseEstimator`.
        if not hasattr(self, "fitted_columns_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call "
                "`fit` before using `transform`."
            )

    def _fit(self, X: XType) -> list[pd.DataFrame]:
        """Fit the encoders; return each expanded column's features, in order."""
        # Cleared first, so refitting on an input with nothing to expand still
        # forgets the last fit.
        self.fitted_columns_ = {}
        self.feature_names_out_ = None
        if not isinstance(X, pd.DataFrame):
            return []

        positions: list[int] = []
        if self._transform_text:
            positions = _text_positions(
                X,
                declared=self._declared_categorical,
                min_cardinality=self._min_cardinality_for_text,
            )
        kept_names = [
            str(column) for i, column in enumerate(X.columns) if i not in set(positions)
        ]
        expanded_names: list[str] = []
        blocks: list[pd.DataFrame] = []
        for position in positions:
            column = _as_strings(X.iloc[:, position]).rename(str(X.columns[position]))
            block, fitted = self._fit_one(
                column, kept_names + expanded_names, self._n_components
            )
            self.fitted_columns_[position] = fitted
            expanded_names += fitted.output_names
            blocks.append(block)
        self.feature_names_out_ = kept_names + expanded_names
        return blocks

    @staticmethod
    def _fit_one(
        column: pd.Series,
        existing_names: Sequence[str],
        n_components: int,
    ) -> tuple[pd.DataFrame, _FittedTextColumn]:
        """Fit an encoder on one column, naming its output after that column.

        skrub keeps fewer than `n_components` features when the column's n-gram
        vocabulary is smaller, so how many there are is settled here, which is why
        `transform` reuses this encoder rather than fitting a fresh one.
        """
        # Seeded on its own: the features a column turns into are a property of
        # the data, and should not move with the estimator's seed.
        encoder = StringEncoder(n_components=n_components, random_state=0)
        with warnings.catch_warnings():
            # skrub warns when it keeps fewer features; the caller did not choose
            # the count and cannot act on it.
            warnings.filterwarnings(
                "ignore", message=".*truncated SVD.*", category=UserWarning
            )
            encoded = pd.DataFrame(encoder.fit_transform(column))
        output_names = make_names_unique(
            [str(name) for name in encoded.columns], existing=existing_names
        )
        return (
            encoded.set_axis(output_names, axis=1).reset_index(drop=True),
            _FittedTextColumn(encoder=encoder, output_names=output_names),
        )

    @staticmethod
    def _apply_one(column: pd.Series, fitted: _FittedTextColumn) -> pd.DataFrame:
        """Reapply one fitted encoder, naming its features as at fit."""
        encoded = pd.DataFrame(fitted.encoder.transform(_as_strings(column)))
        return encoded.set_axis(fitted.output_names, axis=1).reset_index(drop=True)


def _is_string_dtype(dtype: object) -> bool:
    """Whether `dtype` holds strings and nothing else: pandas' `string` dtype in
    any storage, or a pyarrow string dtype. Not `object`, which may hold anything.
    """
    return isinstance(dtype, pd.api.extensions.ExtensionDtype) and dtype.type is str


def _as_strings(column: pd.Series) -> pd.Series:
    """`column` as pandas' `string` dtype, whatever string dtype it arrived in.

    The encoder and the numeric check read that one alike; a pyarrow string
    column parses differently, since its `NaN` counts as a value, not a missing.
    """
    return column.astype("string")


def _text_positions(
    X: pd.DataFrame, *, declared: set[int], min_cardinality: int
) -> list[int]:
    """Positions of `X`'s text columns: a string dtype, not declared categorical,
    more distinct values than the cutoff, and not all numbers.
    """
    positions = []
    for i, dtype in enumerate(X.dtypes):
        if i in declared or not _is_string_dtype(dtype):
            continue
        column = _as_strings(X.iloc[:, i])
        if _get_unique_with_sklearn_compatible_error(column) <= min_cardinality:
            continue
        if _is_numeric_pandas_series(column):
            continue
        positions.append(i)
    return positions


def _drop_and_append(
    frame: pd.DataFrame, expanded: Sequence[int], blocks: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    """Drop the `expanded` positions and append `blocks` after the kept columns.

    skrub's output is default-indexed, so the kept columns must be too, or
    `concat` aligns the two by label instead of position.
    """
    keep = [i for i in range(frame.shape[1]) if i not in set(expanded)]
    return pd.concat([frame.iloc[:, keep], *blocks], axis=1)


def _name_columns(X: pd.DataFrame, positions: Sequence[int]) -> str:
    """Name each of `positions` by index and label, e.g. `1 ('review')`."""
    return ", ".join(f"{i} ({X.columns[i]!r})" for i in positions)


def _refuse_numeric(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on `positions` that `fit` expanded as text but that hold numbers now."""
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns held text when `fit` ran but hold numbers now: "
        f"{_name_columns(X, positions)}. A column encoded as text at fit needs "
        "strings at predict too: pass it with the dtype it had at fit."
    )


def _refuse_array_after_expansion(X: XType, expanded: Sequence[int]) -> None:
    """Raise on a non-`DataFrame` predict input once `fit` expanded columns.

    Its raw width may match `n_features_in_`, so the shape check upstream let it
    through, but nothing here can widen an array to the expanded layout.
    """
    if not expanded:
        return
    raise TabPFNValidationError(
        f"`fit` expanded the text columns at positions {list(expanded)} into "
        "numeric features, so predict input has to be a DataFrame carrying those "
        f"columns as strings; got {type(X).__name__}."
    )
