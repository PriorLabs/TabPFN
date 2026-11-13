#  Copyright (c) Prior Labs GmbH 2026.

"""Per-group aggregates and pooled predictions, for one label per group."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from tabpfn.preprocessing.datamodel import make_names_unique

if TYPE_CHECKING:
    from tabpfn.constants import YType

__all__ = ["GroupAggregator", "pool_rows"]

_NUMERIC_AGGREGATES = ("mean", "std", "min", "max", "last")
_OTHER_AGGREGATES = ("count", "last", "nunique")


class GroupAggregator:
    """Appends per-group aggregates of the features, for one label per group.

    Numeric columns get the mean, std, min, max and last value of their group;
    other columns the count, last value and number of unique values. Of these,
    the `max_features` with the largest variance over the training rows are
    kept. The aggregates are computed from the rows passed in, at fit and at
    predict alike.

    Args:
        append: Whether to append the aggregates. Off, `fit_transform` only
            detects one label per group.
        max_features: How many aggregates to keep at most.

    Attributes:
        one_label_per_group_: Whether `y` is constant within every training
            group and some group has more than one row.
        aggregates_: The kept `(column, aggregate)` pairs. Empty unless
            `append` and `one_label_per_group_`.
        names_: The labels of the appended columns.
    """

    one_label_per_group_: bool
    aggregates_: list[tuple[str, str]]
    names_: list[str]

    def __init__(self, *, append: bool, max_features: int = 50) -> None:
        self.append = append
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
        self.one_label_per_group_ = groups is not None and _one_label_per_group(
            y, groups
        )
        if not (self.append and self.one_label_per_group_):
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


def pool_rows(values: torch.Tensor, groups: np.ndarray | None) -> torch.Tensor:
    """Replace each row, along dim -2, by the mean over its group's rows."""
    if groups is None:
        return values
    index = torch.as_tensor(groups, device=values.device)
    shape = (*values.shape[:-2], int(index.max()) + 1, values.shape[-1])
    sums = values.new_zeros(shape).index_add_(-2, index, values)
    counts = torch.bincount(index).to(values.dtype)
    return (sums / counts[:, None]).index_select(-2, index)


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
