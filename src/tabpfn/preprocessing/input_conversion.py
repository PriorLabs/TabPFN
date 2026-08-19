#  Copyright (c) Prior Labs GmbH 2026.

"""Convert columns to the type their contents imply, before validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from skrub import DatetimeEncoder, TableVectorizer

    from tabpfn.constants import XType

#: Above this many distinct values a string column is treated as high cardinality.
#: Both cardinality branches are passed through, so this only decides which one a
#: column takes, not what happens to it.
CARDINALITY_THRESHOLD = 40


def make_datetime_encoder() -> DatetimeEncoder:
    """Build the encoder that turns a datetime column into numeric features.

    Returns:
        An encoder producing the year, the day of year, the seconds since epoch and
        the cyclical month, day and weekday pairs, plus the time of day when the
        column carries one.
    """
    # Imported here rather than at module scope: skrub depends on matplotlib, and
    # importing it eagerly would pull a plotting stack into every `import tabpfn`.
    from skrub import DatetimeEncoder  # noqa: PLC0415

    return DatetimeEncoder(
        resolution="second",
        add_weekday=True,
        add_day_of_year=True,
        periodic_encoding="circular",
    )


class InputTypeConverter:
    """Give each column the type its contents imply, deciding on train.

    Categorical and numeric columns are passed through, so the estimator keeps its
    own categorical detection and encoding. Datetime columns, which the rest of the
    pipeline cannot represent, are expanded into numeric calendar features.

    Input that is not a dataframe is returned untouched, which leaves plain arrays,
    lists and tensors on their existing path.

    Attributes:
        vectorizer_: The fitted vectorizer, or None when the converter was fitted
            on something other than a dataframe.
    """

    def __init__(self) -> None:
        self.vectorizer_: TableVectorizer | None = None

    def fit_transform(self, X: XType) -> XType:
        """Decide the conversions on `X` and apply them.

        Args:
            X: The training input, in whatever form the caller passed it.

        Returns:
            The converted frame, or `X` unchanged when it is not a dataframe.
        """
        if not isinstance(X, pd.DataFrame):
            return X
        from skrub import TableVectorizer  # noqa: PLC0415

        self.vectorizer_ = TableVectorizer(
            low_cardinality="passthrough",
            high_cardinality="passthrough",
            numeric="passthrough",
            cardinality_threshold=CARDINALITY_THRESHOLD,
            datetime=make_datetime_encoder(),
        )
        return self.vectorizer_.fit_transform(X)

    def transform(self, X: XType) -> XType:
        """Apply the conversions decided at fit.

        Args:
            X: The input to convert.

        Returns:
            The converted frame, or `X` unchanged when there is nothing to apply.
        """
        if self.vectorizer_ is None or not isinstance(X, pd.DataFrame):
            return X
        return self.vectorizer_.transform(X)
