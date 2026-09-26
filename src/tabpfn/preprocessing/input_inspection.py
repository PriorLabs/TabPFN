#  Copyright (c) Prior Labs GmbH 2026.

"""What `fit` saw of its input, kept so the reading of each column can be explained.

Fit converts, expands, validates and encodes its input before any model sees it,
and afterwards only the outcome is left. The inspection keeps the little that is
lost on the way: each input column as received and how it was declared, the
columns expanded into generated features and what they became, the evidence
behind each modality decision, and which columns were ordinal-encoded.
Everything in it is read off state fit computes anyway, except for a few example
values per column drawn from the first rows.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType
    from tabpfn.preprocessing.datetimes import DateTransformer
    from tabpfn.preprocessing.modality_detection import ModalityDecision
    from tabpfn.preprocessing.steps.preprocessing_helpers import (
        OrderPreservingColumnTransformer,
    )
    from tabpfn.preprocessing.text import TextTransformer

#: How many example values are kept per input column.
MAX_EXAMPLES = 5
#: Characters an example value is cut to, so a free-text value stays a glimpse.
MAX_EXAMPLE_LENGTH = 40
#: Rows the examples are drawn from, so that drawing them costs nothing noticeable.
_EXAMPLE_ROWS = 1000


@dataclasses.dataclass(frozen=True)
class ColumnExpansion:
    """One input column expanded into generated features before validation.

    Attributes:
        position: The column's position in the input.
        kind: `"dates"` for a point in time expanded into calendar features,
            `"text"` for a text column expanded into features of its character
            n-grams.
        features: The generated features' names, as the transformer named them.
    """

    position: int
    kind: Literal["dates", "text"]
    features: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class InputInspection:
    """What `fit` saw of its input, before any conversion, and what it decided.

    The input is read in stages and the inspection follows them: the columns as
    received and as declared, the columns expanded before validation, and the
    decisions on the expanded input. The expanded input holds the kept input
    columns in their order, then the features of each expansion in the order of
    `expansions`; `position_after_expansion` follows an input column there.

    Attributes:
        labels: Each input column's label as text, or `None` for an input without
            labels.
        dtypes: Each input column's dtype as text, or `None` for an input that
            carries none.
        examples: Per input column, up to `MAX_EXAMPLES` distinct present values
            from the first `_EXAMPLE_ROWS` rows, rendered as text and cut to
            `MAX_EXAMPLE_LENGTH` characters; strings keep their quotes.
        declared_positions: Input positions listed in `categorical_features_indices`.
        category_dtype_positions: Input positions declared categorical through
            pandas' `category` dtype.
        expansions: The input columns expanded into generated features, dates
            first, each kind in input order.
        decisions: One modality decision per column of the expanded, validated
            input, in its column order.
        ordinal_encoded_positions: Positions in the expanded input whose values
            were ordinal-encoded.
    """

    labels: tuple[str, ...] | None
    dtypes: tuple[str, ...] | None
    examples: tuple[tuple[str, ...], ...]
    declared_positions: tuple[int, ...]
    category_dtype_positions: tuple[int, ...]
    expansions: tuple[ColumnExpansion, ...]
    decisions: tuple[ModalityDecision, ...]
    ordinal_encoded_positions: tuple[int, ...]

    def position_after_expansion(self, position: int) -> int | None:
        """Where the input column at `position` sits in the expanded input.

        `None` for an expanded column, which is not there as such. A kept column
        moves down by the expanded columns ahead of it, since each of those is
        dropped and its features appended after every kept column.
        """
        expanded = [expansion.position for expansion in self.expansions]
        if position in expanded:
            return None
        return position - sum(1 for ahead in expanded if ahead < position)


def build_input_inspection(
    X: XType,
    *,
    declared_positions: Sequence[int] | None,
    date_transformer: DateTransformer,
    text_transformer: TextTransformer,
    decisions: Sequence[ModalityDecision],
    ordinal_encoder: OrderPreservingColumnTransformer,
) -> InputInspection:
    """Inspect the fit input `X` as received, with everything fit decided on it.

    Args:
        X: The fit input, before any conversion or expansion.
        declared_positions: The caller's `categorical_features_indices`.
        date_transformer: Fitted on `X`.
        text_transformer: Fitted on the date-expanded `X`.
        decisions: The modality decisions for the expanded, validated input.
        ordinal_encoder: Fitted on the expanded, validated input.

    Returns:
        The inspection.
    """
    is_frame = isinstance(X, pd.DataFrame)
    head = X.iloc[:_EXAMPLE_ROWS] if is_frame else X[:_EXAMPLE_ROWS]
    values = _as_array(head)
    missing = pd.isna(values)
    n_columns = values.shape[1]
    return InputInspection(
        labels=tuple(str(label) for label in X.columns) if is_frame else None,
        dtypes=_dtypes(X, n_columns=n_columns),
        examples=tuple(_examples(values[~missing[:, i], i]) for i in range(n_columns)),
        declared_positions=tuple(int(i) for i in declared_positions or ()),
        category_dtype_positions=tuple(
            i
            for i, dtype in enumerate(X.dtypes)
            if isinstance(dtype, pd.CategoricalDtype)
        )
        if is_frame
        else (),
        expansions=_expansions(date_transformer, text_transformer, n_columns=n_columns),
        decisions=tuple(decisions),
        ordinal_encoded_positions=tuple(
            int(i) for i in ordinal_encoder.selected_columns()
        ),
    )


def _expansions(
    date_transformer: DateTransformer,
    text_transformer: TextTransformer,
    *,
    n_columns: int,
) -> tuple[ColumnExpansion, ...]:
    """Each expanded column by its input position, dates first.

    The text transformer ran on the date-expanded input, where its positions
    count the columns the date transformer kept, so each is mapped back to the
    kept input column of the same rank.
    """
    kept = [i for i in range(n_columns) if i not in date_transformer.fitted_columns_]
    dates = tuple(
        ColumnExpansion(position=i, kind="dates", features=tuple(fitted.output_names))
        for i, fitted in sorted(date_transformer.fitted_columns_.items())
    )
    text = tuple(
        ColumnExpansion(
            position=kept[j], kind="text", features=tuple(fitted.output_names)
        )
        for j, fitted in sorted(text_transformer.fitted_columns_.items())
    )
    return dates + text


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


__all__ = [
    "MAX_EXAMPLES",
    "MAX_EXAMPLE_LENGTH",
    "ColumnExpansion",
    "InputInspection",
    "build_input_inspection",
]
