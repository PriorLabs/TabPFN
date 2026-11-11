#  Copyright (c) Prior Labs GmbH 2026.

"""Replace an input's group-key columns with one encoded categorical feature.

`group_col` names the columns that say which group a row belongs to, e.g. a
patient or a site. Their joint value is one composite key, encoded as an integer
code fitted on the training rows: a key `fit` did not see becomes `NaN`. The
encoded feature takes the place right before the first column declared
categorical, or the last place if there is none.

A key is a value to look up, never a number to compare: keys match by equality,
not by closeness, so two floats that differ in the last bit are two groups.

Only `TabPFNClassifier` and `TabPFNRegressor` run this, before `DateTransformer`,
so a group column is read as the caller passed it. Columns are selected by label
here, unlike everywhere else in the preprocessing, because that is what
`group_col` names. A label that repeats is refused.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import (
    FeatureModality,
    FeatureSchema,
    make_names_unique,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType

__all__ = [
    "GROUP_FEATURE_NAME",
    "GroupIdEncoder",
    "as_group_feature",
    "reject_group_col",
]

#: Label of the feature the group columns are replaced with, made unique against
#: the labels that stay.
GROUP_FEATURE_NAME = "group_id"


class GroupIdEncoder:
    """Encodes the composite group key of each row as one categorical feature.

    Used like `DateTransformer`, and right before it: `fit_transform` once at fit
    time, keep the instance as `group_encoder_`, `transform` at predict time.
    With no column selected every input passes through unchanged, whatever its
    type.

    Args:
        group_col: The label of the group column, several labels whose joint
            value is one composite key, or `None` for no grouping. An empty
            sequence selects nothing, like `None`.

    Attributes:
        group_columns_: The selected labels, in the order given. Empty when
            nothing is selected.
        positions_: Where each of `group_columns_` sat in the fit input.
        keys_: The fitted mapping: the training rows' composite keys,
            deduplicated, each one's position being its code.
        feature_name_: The label of the encoded feature.
        feature_position_: Where the encoded feature sits in the replaced frame:
            right before the first declared categorical column that stays, or
            last if there is none.
    """

    group_columns_: list[str]
    positions_: list[int]
    keys_: pd.MultiIndex | None
    feature_name_: str | None
    feature_position_: int | None

    def __init__(self, group_col: str | Sequence[str] | None = None) -> None:
        self.group_col = group_col

    def fit(self, X: XType) -> GroupIdEncoder:
        """Give every composite key in `X` a code, and return itself."""
        self._fit(X, None)
        return self

    def fit_transform(
        self, X: XType, categorical_indices: Sequence[int] | None = None
    ) -> XType:
        """`fit(X).transform(X)`, reading the group columns only once.

        Args:
            X: The training input.
            categorical_indices: The positions in `X` declared categorical, which
                place the encoded feature; `None` for none.

        Returns:
            `X` with the group columns replaced by the encoded feature, placed at
            `feature_position_`. `X` itself when no column is selected.

        Raises:
            TabPFNValidationError: If a column is selected and `X` is not a
                `DataFrame`, or does not hold every selected label exactly once.
        """
        codes = self._fit(X, categorical_indices)
        if codes is None:
            return X
        return self._replace(X, codes)

    def transform(self, X: XType) -> XType:
        """Encode `X`'s keys with the mapping `fit` built, so the width holds.

        A key the training rows did not hold has no code and becomes `NaN`; the
        mapping itself is left as `fit` built it.

        Raises:
            NotFittedError: If `fit` has not run yet.
            TabPFNValidationError: As `fit_transform`.
        """
        self._check_is_fitted()
        if self.keys_ is None:
            return X
        _, keys = self._keys(X)
        indexer = self.keys_.get_indexer(keys)
        codes = indexer.astype(np.float64)
        codes[indexer < 0] = np.nan
        return self._replace(X, codes)

    def output_indices(self, indices: Sequence[int] | None) -> list[int] | None:
        """Where each of `indices`, input positions, sits in the replaced frame.

        A kept column shifts down by however many group columns sat ahead of it,
        and up by one if it lands at or after the encoded feature. A selected
        position is dropped rather than moved: its column is gone. The encoded
        feature is not added either: the estimator sets its modality itself,
        around cleaning.
        """
        self._check_is_fitted()
        if indices is None:
            return None
        if self.keys_ is None:
            return list(indices)
        assert self.feature_position_ is not None
        kept = self._kept_indices(indices)
        return [i + (i >= self.feature_position_) for i in kept]

    def feature_index(self, feature_names: Sequence[str] | None) -> int | None:
        """The encoded feature's position among `feature_names`, `None` if none."""
        self._check_is_fitted()
        if self.feature_name_ is None:
            return None
        assert feature_names is not None, "the fit input was not a DataFrame"
        return list(feature_names).index(self.feature_name_)

    def _check_is_fitted(self) -> None:
        # By hand: sklearn's `check_is_fitted` requires a `BaseEstimator`.
        if not hasattr(self, "keys_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call "
                "`fit` before using `transform`."
            )

    def _kept_indices(self, indices: Sequence[int]) -> list[int]:
        """`indices` without the group columns, shifted down past the dropped ones."""
        dropped = set(self.positions_)
        return [
            i - sum(1 for p in self.positions_ if p < i)
            for i in indices
            if i not in dropped
        ]

    def _fit(
        self, X: XType, categorical_indices: Sequence[int] | None
    ) -> np.ndarray | None:
        """Fit the mapping; return the training rows' codes, `None` for no grouping."""
        # Cleared first, so refitting without a grouping still forgets the last fit.
        self.group_columns_ = _resolve_group_col(self.group_col)
        self.positions_ = []
        self.keys_ = None
        self.feature_name_ = None
        self.feature_position_ = None
        if not self.group_columns_:
            return None

        self.positions_, keys = self._keys(X)
        kept = [
            str(label)
            for i, label in enumerate(X.columns)
            if i not in set(self.positions_)
        ]
        self.feature_name_ = make_names_unique([GROUP_FEATURE_NAME], existing=kept)[0]
        self.feature_position_ = min(
            self._kept_indices(categorical_indices or []), default=len(kept)
        )
        codes, self.keys_ = pd.factorize(keys)
        return codes.astype(np.float64)

    def _keys(self, X: XType) -> tuple[list[int], pd.MultiIndex]:
        """Where the group columns sit in `X`, and the composite key of every row."""
        if not isinstance(X, pd.DataFrame):
            raise TabPFNValidationError(
                f"`group_col` selects the columns {self.group_columns_} by label, so "
                f"fit and predict input has to be a DataFrame; got {type(X).__name__}."
            )
        positions = _locate(X, self.group_columns_)
        return positions, pd.MultiIndex.from_frame(X.loc[:, self.group_columns_])

    def _replace(self, X: pd.DataFrame, codes: np.ndarray) -> pd.DataFrame:
        """Drop the group columns and insert the encoded feature, leaving `X` alone."""
        out = X.drop(columns=self.group_columns_)
        # Positional, so the inserted values keep the caller's row order whatever
        # the frame's own index holds.
        out.insert(self.feature_position_, self.feature_name_, codes)
        return out


def as_group_feature(
    feature_schema: FeatureSchema, index: int | None, modality: FeatureModality
) -> FeatureSchema:
    """`feature_schema` with the feature at `index` marked as the group feature."""
    if index is None:
        return feature_schema
    features = list(feature_schema.features)
    features[index] = dataclasses.replace(
        features[index], modality=modality, is_group_id=True
    )
    return FeatureSchema(features=features)


def reject_group_col(group_col: str | Sequence[str] | None, path: str) -> None:
    """Raise if `group_col` selects a column, for a `path` that does not encode it."""
    if _resolve_group_col(group_col):
        raise TabPFNValidationError(
            f"`group_col` is not supported with {path}; set `group_col=None`."
        )


def _resolve_group_col(group_col: str | Sequence[str] | None) -> list[str]:
    """Read `group_col` as the list of labels it selects, refusing anything else."""
    if group_col is None:
        return []
    if isinstance(group_col, str):
        return [group_col]
    if not isinstance(group_col, (list, tuple)) or not all(
        isinstance(label, str) for label in group_col
    ):
        raise TabPFNValidationError(
            f"`group_col` must be a column label, a list of them, or None; got "
            f"{group_col!r} ({type(group_col).__name__})."
        )
    labels = list(group_col)
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise TabPFNValidationError(
            f"`group_col` names these columns more than once: {repeated}. A column "
            "adds nothing to the composite key twice over; name each one once."
        )
    return labels


def _locate(X: pd.DataFrame, labels: Sequence[str]) -> list[int]:
    """The position of each of `labels` in `X`, refusing a missing or repeated one."""
    counts = X.columns.value_counts()
    missing = [label for label in labels if counts.get(label, 0) == 0]
    if missing:
        raise TabPFNValidationError(
            f"`group_col` selects columns that are not in the input: {missing}. "
            f"The input holds {[str(label) for label in X.columns]}."
        )
    repeated = [label for label in labels if counts[label] > 1]
    if repeated:
        raise TabPFNValidationError(
            f"`group_col` selects column labels that the input holds more than "
            f"once: {repeated}. Rename them so each one names a single column."
        )
    return [X.columns.get_loc(label) for label in labels]
