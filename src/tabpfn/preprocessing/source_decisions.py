#  Copyright (c) Prior Labs GmbH 2026.

"""Decisions that the ensemble takes from the fit input before its cast to float32."""

from __future__ import annotations

import dataclasses

import numpy as np

from tabpfn.preprocessing.steps.remove_constant_features_step import kept_by_fit


@dataclasses.dataclass
class SourceDecisions:
    """What the ensemble members decide from the cleaned fit input, decided in the
    input's own dtype before the cast to float32.

    Different values can round to the same float32. Then a column can look constant
    to a member, or the feature-importance model can see fewer distinct values than
    the cleaned input has.
    """

    member_rows: list[np.ndarray] | None
    """Each member's rows, or `None` when every member reads all of them."""

    kept: np.ndarray
    """Whether the `RemoveConstantFeaturesStep` of a member keeps a column. One row
    per member, or a single row when every member reads all rows."""

    importance: tuple[np.ndarray, np.ndarray] | None
    """The rows the feature-importance model is fit on and their cleaned values in
    float64, `None` without that model."""

    @classmethod
    def for_rows(
        cls,
        member_rows: list[np.ndarray] | None,
        importance_rows: np.ndarray | None,
        n_columns: int,
    ) -> SourceDecisions:
        """Empty decisions for these rows, for `take` to fill one range at a time."""
        return cls(
            member_rows=member_rows,
            kept=np.zeros((len(member_rows or [None]), n_columns), dtype=bool),
            importance=None
            if importance_rows is None
            else (
                importance_rows,
                np.empty((len(importance_rows), n_columns), dtype=np.float64),
            ),
        )

    def take(self, start: int, cleaned: np.ndarray) -> None:
        """Take the decisions for `cleaned`, the columns from `start` on."""
        stop = start + cleaned.shape[1]
        for member, rows in enumerate(self.member_rows or [None]):
            self.kept[member, start:stop] = kept_by_fit(
                cleaned if rows is None else cleaned[rows]
            )
        if self.importance is not None:
            rows, values = self.importance
            values[:, start:stop] = cleaned[rows]
