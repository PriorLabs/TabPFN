#  Copyright (c) Prior Labs GmbH 2026.

"""Decisions the ensemble takes from the fit input, taken before it is narrowed."""

from __future__ import annotations

import dataclasses

import numpy as np

from tabpfn.preprocessing.steps.remove_constant_features_step import kept_by_fit


@dataclasses.dataclass
class SourceDecisions:
    """What the ensemble members decide from the cleaned fit input, decided on it in
    its own dtype before a float32 table holds it.

    Values that differ can round to one float32: a column can then look constant to a
    member, or the feature-importance model see fewer distinct values, where on the
    cleaned input they would not.
    """

    member_rows: list[np.ndarray] | None
    """Each member's rows, or `None` when every member reads all of them."""

    kept: np.ndarray
    """Per member, or once for all rows, per column: whether the member's
    `RemoveConstantFeaturesStep` keeps it."""

    importance_rows: np.ndarray | None
    """The rows the feature-importance model is fit on, `None` without one."""

    importance_values: np.ndarray | None
    """The cleaned values of `importance_rows`, in float64."""

    @classmethod
    def for_rows(
        cls,
        member_rows: list[np.ndarray] | None,
        importance_rows: np.ndarray | None,
        n_columns: int,
    ) -> SourceDecisions:
        """Decisions for these rows, to be taken a range of columns at a time."""
        return cls(
            member_rows=member_rows,
            kept=np.zeros((len(member_rows or [None]), n_columns), dtype=bool),
            importance_rows=importance_rows,
            importance_values=None
            if importance_rows is None
            else np.empty((len(importance_rows), n_columns), dtype=np.float64),
        )

    def importance_source(self) -> tuple[np.ndarray, np.ndarray]:
        """The feature-importance model's rows and their values."""
        if self.importance_rows is None or self.importance_values is None:
            raise RuntimeError("No rows were read for a feature-importance model.")
        return self.importance_rows, self.importance_values

    def take(self, start: int, cleaned: np.ndarray) -> None:
        """Take the decisions for the cleaned columns from `start` on."""
        stop = start + cleaned.shape[1]
        for member, rows in enumerate(self.member_rows or [None]):
            self.kept[member, start:stop] = kept_by_fit(
                cleaned if rows is None else cleaned[rows]
            )
        if self.importance_values is not None:
            self.importance_values[:, start:stop] = cleaned[self.importance_rows]
