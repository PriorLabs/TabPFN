#  Copyright (c) Prior Labs GmbH 2026.

"""Build the encoders `clean.expand_date_and_text_features` applies to a `DATE` or
`TEXT` column once `detect_feature_modalities` has already decided it is one.

Both are used as plain single-column transformers (`.fit(series)`), not through
`skrub.TableVectorizer`: type detection is this package's own, not skrub's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skrub import DatetimeEncoder

#: Number of features a text column is encoded into. skrub's default.
DEFAULT_TEXT_N_COMPONENTS = 30


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
