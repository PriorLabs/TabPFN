#  Copyright (c) Prior Labs GmbH 2026.

"""Positional column surgery on a `DataFrame`, with no knowledge of dates.

Positional throughout, because the labels are the caller's and so can repeat
(the same duplicate-name case `build_input_feature_names` exists for), which
makes anything driven by a label ambiguous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


def replace_columns_positionally(
    X: pd.DataFrame,
    replacements: dict[int, pd.Series],
) -> pd.DataFrame:
    """Return `X` with the given column positions replaced, leaving `X` untouched.

    Positional via a temporary integer column axis: numbering the axis makes
    every label unique and equal to its own position, so a plain assignment is
    unambiguous, and the caller's labels go back afterwards.

    The copy is shallow and the frame handed in is never written through: each
    assignment replaces a whole column rather than any value inside one.
    """
    if not replacements:
        return X
    out = X.copy(deep=False)
    original_columns = out.columns
    out.columns = pd.RangeIndex(out.shape[1])
    for position, values in replacements.items():
        out[position] = values.to_numpy()
    out.columns = original_columns
    return out


def drop_and_append(
    frame: pd.DataFrame,
    expanded: Sequence[int],
    blocks: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Drop every expanded column and append its calendar features instead.

    Positional, not `frame.drop(columns=...)`, which would take the wrong column
    when labels repeat. Every expanded column's output lands after all the kept
    ones, in original-position order, which is what makes the appended block's
    positions predictable.
    """
    keep = [i for i in range(frame.shape[1]) if i not in set(expanded)]
    return pd.concat([frame.iloc[:, keep], *blocks], axis=1)
