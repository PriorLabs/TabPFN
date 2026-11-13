#  Copyright (c) Prior Labs GmbH 2026.

"""Append per-group aggregates of the features when each group has one label."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabpfn.preprocessing.datamodel import make_names_unique

if TYPE_CHECKING:
    from tabpfn.constants import YType

__all__ = ["GroupAggregator"]

_NUMERIC_AGGREGATES = ("mean", "std", "min", "max", "last")
_OTHER_AGGREGATES = ("count", "last", "nunique")


class GroupAggregator:
    """Appends per-group aggregates of the features, for one label per group.

    Numeric columns get the mean, std, min, max and last value of their group;
    other columns the count, last value and number of unique values. Of these,
    the `max_features` with the largest variance over the training rows are
    kept. The aggregates are computed from the rows passed in, at fit and at
    predict alike, so the rows of a group must be predicted in one call.

    Attributes:
        aggregates_: The kept `(column, aggregate)` pairs. Empty when `y` is not
            constant within every group, or when no group has more than one row.
        names_: The labels of the appended columns.
    """

    aggregates_: list[tuple[str, str]]
    names_: list[str]

    def __init__(self, max_features: int = 50) -> None:
        self.max_features = max_features

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: YType,
        groups: np.ndarray | None,
        *,
        exclude: str | None,
    ) -> pd.DataFrame:
        """Select the aggregates to keep, then `transform(X, groups)`.

        Args:
            X: The training rows.
            y: The training labels.
            groups: Each row's group as a code in `0..n_groups - 1`, or `None`
                without grouping.
            exclude: A column of `X` not to aggregate.
        """
        self.aggregates_, self.names_ = [], []
        if groups is None or not _one_label_per_group(y, groups):
            return X

        sizes = np.bincount(groups)
        spread: dict[tuple[str, str], float] = {}
        for column in X.columns[X.columns != exclude]:
            numeric = pd.api.types.is_numeric_dtype(X[column])
            aggregates = _NUMERIC_AGGREGATES if numeric else _OTHER_AGGREGATES
            table = X[column].groupby(groups).agg(list(aggregates))
            for aggregate in aggregates:
                values = table[aggregate]
                if not pd.api.types.is_numeric_dtype(values):
                    codes = values.astype("category").cat.codes
                    values = codes.astype(float).where(codes >= 0)
                spread[column, aggregate] = _row_variance(values.to_numpy(float), sizes)

        # Rounded so that equal spreads tie on the name, not on summation order.
        # A NaN spread, from at most one row with a value, ranks last.
        ranked = sorted(
            spread,
            key=lambda key: (
                np.nan_to_num(-float(f"{spread[key]:.12g}"), nan=np.inf),
                f"{key[0]}_{key[1]}",
            ),
        )
        self.aggregates_ = ranked[: self.max_features]
        self.names_ = make_names_unique(
            [f"{column}_{aggregate}" for column, aggregate in self.aggregates_],
            existing=map(str, X.columns),
        )
        return self.transform(X, groups)

    def transform(self, X: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
        """Append the kept aggregates, each computed over its group's rows in `X`."""
        if not self.aggregates_:
            return X
        columns = {
            name: X[column].groupby(groups).transform(aggregate).to_numpy()
            for (column, aggregate), name in zip(
                self.aggregates_, self.names_, strict=True
            )
        }
        return pd.concat([X, pd.DataFrame(columns, index=X.index)], axis=1)


def _one_label_per_group(y: YType, groups: np.ndarray) -> bool:
    """Whether `y` is constant within every group and some group has two rows."""
    labels = pd.Series(np.asarray(y)).groupby(groups).nunique(dropna=False)
    return bool(labels.max() <= 1 and np.bincount(groups).max() > 1)


def _row_variance(values: np.ndarray, sizes: np.ndarray) -> float:
    """Sample variance of each group's value repeated once per row, NaN skipped."""
    keep = ~np.isnan(values)
    values, sizes = values[keep], sizes[keep]
    n = sizes.sum()
    if n <= 1:
        return float("nan")
    mean = (sizes * values).sum() / n
    return float((sizes * (values - mean) ** 2).sum() / (n - 1))
