#  Copyright (c) Prior Labs GmbH 2026.

"""Convert columns to the type their contents imply, before validation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from skrub import DatetimeEncoder, TableVectorizer

    from tabpfn.constants import XType

#: Number of distinct values above which a string column is treated as text rather
#: than as a category. Set to the same value as
#: `MAX_UNIQUE_FOR_CATEGORICAL_FEATURES`, so that by default a string column is
#: either a category or text with nothing in between: raising it above that value
#: leaves the columns in the gap being ordinal-encoded as high-cardinality
#: categoricals instead.
DEFAULT_TEXT_CARDINALITY_THRESHOLD = 30

#: Number of features a text column is encoded into. skrub's default.
DEFAULT_TEXT_N_COMPONENTS = 30

#: Cap on how many column names a warning lists, so a wide frame does not produce
#: an unreadable multi-kilobyte message.
_MAX_COLUMNS_IN_WARNING = 10


def _format_names(columns: list[str]) -> str:
    """Render column names for a warning, capped so the message stays readable."""
    shown = columns[:_MAX_COLUMNS_IN_WARNING]
    names = ", ".join(repr(name) for name in shown)
    if len(columns) > len(shown):
        names += f" (and {len(columns) - len(shown)} more)"
    return names


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


def make_text_encoder(n_components: int) -> Any:
    """Build the encoder that turns a text column into numeric features.

    Args:
        n_components: Number of features to encode each text column into.

    Returns:
        An encoder applying tf-idf over character n-grams followed by a truncated
        SVD, producing `n_components` columns.
    """
    from skrub import StringEncoder  # noqa: PLC0415

    # Seeded independently of the estimator: giving a column its type is a property
    # of the data, so it should not move when the ensemble seed does.
    return StringEncoder(n_components=n_components, random_state=0)


class InputTypeConverter:
    """Give each column the type its contents imply, deciding on train.

    Numeric columns and low-cardinality strings are passed through, so the
    estimator keeps its own categorical detection and encoding. Datetime columns,
    which the rest of the pipeline cannot represent, are expanded into numeric
    calendar features. Text columns are encoded into numeric features.

    Input that is not a dataframe is returned untouched, which leaves plain arrays,
    lists and tensors on their existing path.

    Args:
        use_dates: Whether to expand datetime columns into calendar features. When
            False, date columns are left exactly as they arrived, and a datetime
            column reaches a pipeline that cannot represent it.
        use_text: Whether to encode text columns into numeric features. When False,
            they are left as strings and reach the estimator's ordinal encoder.
        text_cardinality_threshold: Number of distinct values above which a string
            column is treated as text rather than as a category.
        text_n_components: Number of features each text column is encoded into.

    Attributes:
        vectorizer_: The fitted vectorizer, or None when the converter was fitted
            on something other than a dataframe.
    """

    def __init__(
        self,
        *,
        use_dates: bool = True,
        use_text: bool = True,
        text_cardinality_threshold: int = DEFAULT_TEXT_CARDINALITY_THRESHOLD,
        text_n_components: int = DEFAULT_TEXT_N_COMPONENTS,
    ) -> None:
        self.use_dates = use_dates
        self.use_text = use_text
        self.text_cardinality_threshold = text_cardinality_threshold
        self.text_n_components = text_n_components
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

        # Categoricals are deliberately left to this package: `low_cardinality`
        # passes through untouched rather than being encoded here, so
        # `MAX_UNIQUE_FOR_CATEGORICAL_FEATURES` and the ordinal encoder keep
        # deciding what a category is and how it is represented. That leaves two
        # cardinality thresholds in the codebase, this one and that one, and
        # coupling them may well be the right move; see the matching note in
        # `modality_detection._detect_feature_modality`.
        self.vectorizer_ = TableVectorizer(
            low_cardinality="passthrough",
            high_cardinality=make_text_encoder(self.text_n_components)
            if self.use_text
            else "passthrough",
            numeric="passthrough",
            datetime=make_datetime_encoder() if self.use_dates else "passthrough",
            cardinality_threshold=self.text_cardinality_threshold,
        )
        out = self.vectorizer_.fit_transform(X)
        self._warn_about_columns()
        return self._restore_dates(out, X)

    def transform(self, X: XType) -> XType:
        """Apply the conversions decided at fit.

        Args:
            X: The input to convert.

        Returns:
            The converted frame, or `X` unchanged when there is nothing to apply.
        """
        if self.vectorizer_ is None:
            return X
        frame = self._as_fitted_frame(X)
        if frame is None:
            return X
        return self._restore_dates(self.vectorizer_.transform(frame), frame)

    def _as_fitted_frame(self, X: XType) -> pd.DataFrame | None:
        """Present `X` as a frame the fitted vectorizer can transform.

        Fitting on a named frame and predicting with a bare array is a supported
        combination, so an array of the width seen at fit is given the fit-time
        column names rather than being left unconverted, which would disagree with
        `n_features_in_` once a column has been expanded.

        Args:
            X: The input to present as a frame.

        Returns:
            A frame with the fit-time columns, or None when `X` cannot be one, in
            which case validation reports the mismatch.
        """
        if isinstance(X, pd.DataFrame):
            return X
        names = list(self.vectorizer_.feature_names_in_)  # type: ignore[union-attr]
        try:
            frame = pd.DataFrame(X, copy=False)
        except (TypeError, ValueError):
            return None
        if frame.shape[1] != len(names):
            return None
        frame.columns = names
        return frame

    def _restore_dates(self, out: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Put date columns back as they arrived when `use_dates` is False.

        Parsing a date is part of inferring types and happens whether or not the
        result is wanted, so with `use_dates` off the parsed column is written back
        from the input. Without this, a column of date strings would arrive as a
        dtype the rest of the pipeline cannot represent, which is worse than the
        string it started as.

        Args:
            out: The converted frame.
            X: The frame the conversion was applied to.

        Returns:
            `out`, with any date column replaced by its original values.
        """
        if self.use_dates:
            return out
        restored = [
            col
            for col in out.columns
            if col in X.columns and pd.api.types.is_datetime64_any_dtype(out[col])
        ]
        if not restored:
            return out
        out = out.copy()
        for col in restored:
            out[col] = X[col]
        return out

    def _columns_of_kind(self, kind: str) -> list[str]:
        """Names of the input columns the vectorizer sorted into `kind`."""
        if self.vectorizer_ is None:
            return []
        return [
            str(column)
            for column, column_kind in self.vectorizer_.column_to_kind_.items()
            if column_kind == kind
        ]

    def _warn_about_columns(self) -> None:
        """Report what was read as text, and what was read as a date but left."""
        self._warn_about_encoded_text()
        self._warn_about_unused_dates()

    def _warn_about_unused_dates(self) -> None:
        """Say which columns hold dates that `use_dates` told us to leave alone.

        Without this the column is reported by the free-text warning downstream,
        which suggests remedies that make no sense for a date.
        """
        if self.use_dates:
            return
        date_columns = self._columns_of_kind("datetime")
        if not date_columns:
            return
        warnings.warn(
            f"These columns hold dates, and `use_dates` is off, so they are left as "
            f"they arrived: {_format_names(date_columns)}.\n"
            "A column already of a datetime dtype cannot be represented further "
            "down the pipeline and will raise; one holding date strings is treated "
            "as a high-cardinality category, which discards the ordering a date "
            "has. Set `use_dates=True` to expand them into calendar features "
            "instead.",
            UserWarning,
            stacklevel=2,
        )

    def _warn_about_encoded_text(self) -> None:
        """Say which columns were read as text, since it is easy to get wrong."""
        if not self.use_text:
            return
        text_columns = self._columns_of_kind("high_cardinality")
        if not text_columns:
            return

        names = _format_names(text_columns)
        warnings.warn(
            f"These columns hold more than {self.text_cardinality_threshold} distinct "
            f"values and were encoded as text, into "
            f"{self.text_n_components} numeric features each: {names}.\n"
            "If such a column is a category rather than text, raise "
            "`text_cardinality_threshold` above its number of distinct values, or "
            "pass its index in `categorical_features_indices`.\n"
            "This encoding is character-level and carries no meaning of the words. "
            "For text where the meaning matters, consider the tabpfn-client API, "
            "which embeds text natively: "
            "https://github.com/PriorLabs/tabpfn-client",
            UserWarning,
            stacklevel=2,
        )
