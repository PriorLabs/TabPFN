#  Copyright (c) Prior Labs GmbH 2026.

"""Per-group aggregates and pooled predictions, for one label per group."""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import torch

from tabpfn.preprocessing.datamodel import make_names_unique

if TYPE_CHECKING:
    from tabpfn.constants import YType

__all__ = ["GroupAggregator", "pool_rows"]

logger = logging.getLogger(__name__)

_NUMERIC_AGGREGATES = ("mean", "std", "min", "max", "last")
_OTHER_AGGREGATES = ("count", "last", "nunique")


class GroupAggregator:
    """Appends per-group aggregates of the features, and decides on pooling.

    Numeric columns get the mean, std, min, max and last value of their group;
    other columns the count, last value and number of unique values. Of these,
    the `max_features` with the largest variance over the training rows are
    kept. The aggregates are computed from the rows passed in, at fit and at
    predict alike.

    Args:
        append: Whether to append the aggregates. They need a group with more
            than one row.
        pool: Whether to pool the predictions per group. This needs one label
            per group too: `y` constant within every training group.
        max_features: How many aggregates to keep at most.
        n_jobs: The number of threads that select the aggregates at fit, as
            in `joblib.Parallel`.
        device: A device to select the aggregates of the numeric columns on,
            with torch in float64. `None` selects them with pandas.

    Attributes:
        aggregates_: The kept `(column, aggregate)` pairs. Empty when none are
            appended.
        names_: The labels of the appended columns.
        pool_: Whether the predictions of a group's rows are to be pooled.
    """

    aggregates_: list[tuple[str, str]]
    names_: list[str]
    pool_: bool

    def __init__(
        self,
        *,
        append: bool,
        pool: bool,
        max_features: int = 64,
        n_jobs: int = 1,
        device: torch.device | None = None,
    ) -> None:
        self.append = append
        self.pool = pool
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.device = device

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: YType,
        groups: np.ndarray | None,
        *,
        exclude: str | None,
    ) -> pd.DataFrame:
        """Select the aggregates to keep, then `transform(X, groups)`.

        A requested mode that the data does not allow is skipped with a warning.

        Args:
            X: The training rows.
            y: The training labels.
            groups: Each row's group as a code in `0..n_groups - 1`, or `None`
                without grouping.
            exclude: A column of `X` not to aggregate.
        """
        self.aggregates_, self.names_, self.pool_ = [], [], False
        if groups is None:
            return X
        several_rows = bool(np.bincount(groups).max() > 1)
        one_label = several_rows and _one_label_per_group(y, groups)
        if one_label:
            logger.info("`group_col`: every training group has one label.")
        self.pool_ = self.pool and one_label
        if self.pool and not one_label:
            warnings.warn(
                "`POOL_GROUP_PREDICTIONS` is ignored: it needs one label per group, "
                "but the target varies within a training group or no group has "
                "more than one row.",
                stacklevel=2,
            )
        if self.append and not several_rows:
            warnings.warn(
                "`APPEND_GROUP_AGGREGATES` is ignored: no training group has more "
                "than one row.",
                stacklevel=2,
            )
        if not (self.append and several_rows):
            return X

        spread = self._spread(X, groups, exclude)

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
        logger.info(
            "Appending %d per-group aggregates. They are computed over the rows "
            "passed in, so predict all rows of a group in one call.",
            len(self.names_),
        )
        return self.transform(X, groups)

    def _spread(
        self, X: pd.DataFrame, groups: np.ndarray, exclude: str | None
    ) -> dict[tuple[str, str], float]:
        """The `_row_variance` of every aggregate of every column."""
        sizes = np.bincount(groups)
        grouped = X.groupby(groups)

        def column_spread(column: str) -> dict[tuple[str, str], float]:
            numeric = pd.api.types.is_numeric_dtype(X[column])
            aggregates = _NUMERIC_AGGREGATES if numeric else _OTHER_AGGREGATES
            table = grouped[column].agg(list(aggregates))
            spread = {}
            for aggregate in aggregates:
                values = table[aggregate]
                if not pd.api.types.is_numeric_dtype(values):
                    codes = values.astype("category").cat.codes
                    values = codes.astype(float).where(codes >= 0)
                spread[column, aggregate] = _row_variance(values.to_numpy(float), sizes)
            return spread

        columns = X.columns[X.columns != exclude]
        spread: dict[tuple[str, str], float] = {}
        if self.device is not None:
            numeric = np.array(
                [pd.api.types.is_numeric_dtype(X[column]) for column in columns],
                dtype=bool,
            )
            spread = _numeric_spread(X, columns[numeric], groups, sizes, self.device)
            columns = columns[~numeric]
        if len(columns):
            # The first column runs alone: it fills the lazy caches of `grouped`,
            # which the threads then only read. pandas releases the GIL in the
            # numeric aggregations.
            spread.update(column_spread(columns[0]))
        for column_result in joblib.Parallel(n_jobs=self.n_jobs, prefer="threads")(
            joblib.delayed(column_spread)(column) for column in columns[1:]
        ):
            spread.update(column_result)
        return spread

    def transform(self, X: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
        """Append the kept aggregates, each computed over its group's rows in `X`."""
        if not self.aggregates_:
            return X
        grouped = X.groupby(groups)
        # Columns are added to a shallow copy one by one: `pd.concat` would hold
        # two more copies of `X` at its peak.
        out = X.copy(deep=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for (column, aggregate), name in zip(
                self.aggregates_, self.names_, strict=True
            ):
                # A group's aggregate sits at its code, so indexing by `groups`
                # repeats it once per row.
                out[name] = grouped[column].agg(aggregate).to_numpy()[groups]
        return out


def pool_rows(
    values: torch.Tensor, groups: np.ndarray | None, *, logits: bool, geometric: bool
) -> torch.Tensor:
    """Replace each row, along dim -2, by the mean distribution of its group.

    The arithmetic mean averages the rows' probabilities; the `geometric` one
    averages their log-probabilities and normalises the result. `logits` says
    whether `values` holds logits or probabilities; the result holds the same.
    `groups=None` returns `values` unchanged.
    """
    if groups is None:
        return values
    if values.dtype == torch.float16:
        values = values.float()
    if geometric:
        # Floored: a probability that is zero in some row would rule the class
        # out for the group, and rows with disjoint support would leave no class.
        values = values.log_softmax(dim=-1) if logits else values.log()
        values = values.clamp_min(math.log(torch.finfo(values.dtype).tiny))
    elif logits:
        values = values.softmax(dim=-1)
    index = torch.as_tensor(groups, device=values.device)
    shape = (*values.shape[:-2], int(index.max()) + 1, values.shape[-1])
    sums = values.new_zeros(shape).index_add_(-2, index, values)
    counts = torch.bincount(index).to(values.dtype)
    pooled = (sums / counts[:, None]).index_select(-2, index)
    if geometric:
        pooled = pooled.log_softmax(dim=-1)
        return pooled if logits else pooled.exp()
    return pooled.log() if logits else pooled


def _one_label_per_group(y: YType, groups: np.ndarray) -> bool:
    """Whether `y` is constant within every group."""
    # `y` is not validated yet: a column vector is flattened, and a wrong length
    # is left for the validation to report.
    y = np.ravel(y)
    if len(y) != len(groups):
        return False
    return bool(pd.Series(y).groupby(groups).nunique(dropna=False).max() <= 1)


def _numeric_spread(
    X: pd.DataFrame,
    columns: pd.Index,
    groups: np.ndarray,
    sizes: np.ndarray,
    device: torch.device,
) -> dict[tuple[str, str], float]:
    """The `_row_variance` of every numeric aggregate of `columns`, with torch."""
    index = torch.as_tensor(groups, device=device)[:, None]
    weights = torch.as_tensor(sizes, dtype=torch.float64, device=device)[:, None]
    # Bounds the device memory to a few GB: the temporaries are about 4 tensors
    # of rows x columns and about 20 of groups x columns, in float64.
    step = max(1, 2**26 // (len(X) + 4 * len(sizes)))
    spread = {}
    for start in range(0, len(columns), step):
        chunk = columns[start : start + step]
        values = torch.as_tensor(
            X[chunk].to_numpy(np.float64, na_value=np.nan), device=device
        )
        tables = _numeric_aggregates(values, index.expand_as(values), len(sizes))
        keep = ~tables.isnan()
        n = (weights * keep).sum(dim=1)
        tables = tables.where(keep, 0)
        mean = (weights * tables).sum(dim=1, keepdim=True) / n[:, None]
        variance = (weights * keep * (tables - mean) ** 2).sum(dim=1) / (n - 1)
        variance = variance.where(n > 1, torch.nan).tolist()
        for aggregate, of_columns in zip(_NUMERIC_AGGREGATES, variance, strict=True):
            for column, value in zip(chunk, of_columns, strict=True):
                spread[column, aggregate] = value
    return spread


def _numeric_aggregates(
    values: torch.Tensor, index: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """The `_NUMERIC_AGGREGATES` of each column of `values`, as pandas computes them.

    NaN is skipped, the std has one degree of freedom less than values, and a
    group without a value gives NaN.

    Args:
        values: The rows, of shape `(n_rows, n_columns)`.
        index: Each value's group, of the shape of `values`.
        n_groups: The number of groups.

    Returns:
        A tensor of shape `(n_aggregates, n_groups, n_columns)`.
    """

    def per_group(source: torch.Tensor, reduce: str, initial: float) -> torch.Tensor:
        out = source.new_full((n_groups, source.shape[1]), initial)
        return out.scatter_reduce_(0, index, source, reduce)

    valid = ~values.isnan()
    count = per_group(valid.double(), "sum", 0)
    mean = per_group(values.where(valid, 0), "sum", 0) / count
    # Deviations from the group's mean: a difference of sums would cancel.
    deviation = (values - mean.gather(0, index)).where(valid, 0)
    std = (per_group(deviation**2, "sum", 0) / (count - 1)).sqrt()
    rows = torch.arange(len(values), device=values.device)[:, None]
    last = per_group(rows.where(valid, -1), "amax", -1).clamp(min=0)
    return torch.stack(
        [
            mean,
            std.where(count > 1, torch.nan),
            per_group(values.where(valid, torch.inf), "amin", torch.inf),
            per_group(values.where(valid, -torch.inf), "amax", -torch.inf),
            values.gather(0, last),
        ]
    ).where(count > 0, torch.nan)


def _row_variance(values: np.ndarray, sizes: np.ndarray) -> float:
    """Sample variance of each group's value repeated once per row, NaN skipped."""
    keep = ~np.isnan(values)
    values, sizes = values[keep], sizes[keep]
    n = sizes.sum()
    if n <= 1:
        return float("nan")
    mean = (sizes * values).sum() / n
    return float((sizes * (values - mean) ** 2).sum() / (n - 1))
