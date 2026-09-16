#  Copyright (c) Prior Labs GmbH 2026.

"""What `fit` saw of its input, kept so the reading of each column can be explained.

Fit converts, expands, validates and encodes its input before any model sees it,
and afterwards only the outcome is left. The record keeps the little that is lost
on the way: each input column as received, and the evidence behind each modality
decision. Everything in it is read off state fit computes anyway, except for a
few example values per column drawn from the first rows.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType
    from tabpfn.preprocessing.modality_detection import ModalityDecision

#: How many example values are kept per input column.
MAX_EXAMPLES = 5
#: Characters an example value is cut to, so a free-text value stays a glimpse.
MAX_EXAMPLE_LENGTH = 40
#: Rows the examples are drawn from, so that drawing them costs nothing noticeable.
_EXAMPLE_ROWS = 1000


@dataclasses.dataclass(frozen=True)
class InputRecord:
    """What `fit` saw of its input, before any conversion, and what it decided.

    Attributes:
        labels: Each input column's label as text, or `None` for an input without
            labels.
        dtypes: Each input column's dtype as text, or `None` for an input that
            carries none.
        examples: Per input column, up to `MAX_EXAMPLES` distinct present values
            from the first `_EXAMPLE_ROWS` rows, rendered as text and cut to
            `MAX_EXAMPLE_LENGTH` characters; strings keep their quotes.
        category_dtype_positions: Input positions declared categorical through
            pandas' `category` dtype.
        decisions: One modality decision per column of the expanded, validated
            input, in its column order.
    """

    labels: tuple[str, ...] | None
    dtypes: tuple[str, ...] | None
    examples: tuple[tuple[str, ...], ...]
    category_dtype_positions: tuple[int, ...]
    decisions: tuple[ModalityDecision, ...]


def record_input(X: XType, *, decisions: Sequence[ModalityDecision]) -> InputRecord:
    """Record the fit input `X` as the caller passed it, with the decisions taken on it.

    Args:
        X: The fit input, before any conversion or expansion.
        decisions: The modality decisions for the expanded, validated input.

    Returns:
        The record.
    """
    is_frame = isinstance(X, pd.DataFrame)
    head = X.iloc[:_EXAMPLE_ROWS] if is_frame else X[:_EXAMPLE_ROWS]
    values = _as_array(head)
    missing = pd.isna(values)
    return InputRecord(
        labels=tuple(str(label) for label in X.columns) if is_frame else None,
        dtypes=_dtypes(X, n_columns=values.shape[1]),
        examples=tuple(
            _examples(values[~missing[:, i], i]) for i in range(values.shape[1])
        ),
        category_dtype_positions=tuple(
            i
            for i, dtype in enumerate(X.dtypes)
            if isinstance(dtype, pd.CategoricalDtype)
        )
        if is_frame
        else (),
        decisions=tuple(decisions),
    )


def _as_array(head: XType) -> np.ndarray:
    """The first rows as one array, numeric input as it is and anything else as objects.

    Cells of a frame with other dtypes are taken as objects so that each renders as
    its own scalar does; numeric input is left in its dtype since the cast would
    cost more than everything else here.
    """
    if isinstance(head, np.ndarray) and head.dtype.kind in "biuf":
        return head
    if isinstance(head, pd.DataFrame) and all(
        isinstance(dtype, np.dtype) and dtype.kind in "biuf" for dtype in head.dtypes
    ):
        return head.to_numpy()
    return np.asarray(head, dtype=object)


def _dtypes(X: XType, *, n_columns: int) -> tuple[str, ...] | None:
    """Each column's dtype as text, `None` for an input that carries none."""
    if isinstance(X, pd.DataFrame):
        return tuple(str(dtype) for dtype in X.dtypes)
    if isinstance(X, np.ndarray):
        return (str(X.dtype),) * n_columns
    return None


def _examples(present: np.ndarray) -> tuple[str, ...]:
    """Up to `MAX_EXAMPLES` distinct values, in order of first appearance."""
    return tuple(
        (repr(value) if isinstance(value, str) else str(value))[:MAX_EXAMPLE_LENGTH]
        for value in pd.unique(present)[:MAX_EXAMPLES]
    )


__all__ = ["MAX_EXAMPLES", "MAX_EXAMPLE_LENGTH", "InputRecord", "record_input"]
