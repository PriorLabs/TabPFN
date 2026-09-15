#  Copyright (c) Prior Labs GmbH 2026.

"""The input report: how each column of a fit input was read.

For every column the caller passed, the report states what became of it before
any model saw it: its dtype as received, a few of its values, how many are
missing, whether it was declared categorical, how many distinct values it holds,
the modality it was read as or the features it was expanded into, whether the
ordinal encoder replaced its values by codes, the rule that decided, and the
setting that changes that rule. Everything but the missing-value count is read
off state that fit computes anyway.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from tabpfn.preprocessing.datamodel import FeatureModality

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType
    from tabpfn.inference_config import InferenceConfig
    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.datetimes import DateTransformer
    from tabpfn.preprocessing.modality_detection import ModalityDecision
    from tabpfn.preprocessing.steps.preprocessing_helpers import (
        OrderPreservingColumnTransformer,
    )
    from tabpfn.preprocessing.text import TextTransformer

#: The `inference_config` fields that steer how a column is read.
INPUT_SETTINGS = (
    "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES",
    "MIN_UNIQUE_FOR_NUMERICAL_FEATURES",
    "MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE",
    "MIN_CARDINALITY_FOR_TEXT",
    "TRANSFORM_DATES",
    "TRANSFORM_TEXT",
    "TEXT_N_COMPONENTS",
)
#: How many example values a column report holds at most.
MAX_EXAMPLES = 3
#: Rows the examples are drawn from, so that drawing them costs nothing noticeable.
_EXAMPLE_ROWS = 100


@dataclasses.dataclass(frozen=True)
class ColumnReport:
    """How one column of the fit input was read.

    Attributes:
        index: The column's position in the fit input.
        name: Its label, or `f{index}` for an input without labels.
        dtype: Its dtype as received, or `None` for an input that carries none.
        examples: Up to `MAX_EXAMPLES` distinct present values from the first rows,
            rendered as text; strings keep their quotes.
        n_missing: How many of its values are missing.
        declared_categorical: Whether it was declared categorical, by position in
            `categorical_features_indices` or through pandas' `category` dtype.
        n_unique: Distinct values, a missing value counting as one of them.
            `None` for an expanded column, whose values are not counted.
        n_unique_is_exact: False when `n_unique` is only a lower bound: counting
            stops once the first rows have cleared every cardinality threshold.
        modality: The modality it was read as, or `None` for an expanded column.
        expanded_by: `"dates"` or `"text"` when the column was replaced by generated
            features, else `None`.
        generated_features: The names of those features; empty otherwise.
        ordinal_encoded: Whether the ordinal encoder replaced its values by codes.
        reason: The rule and threshold that decided.
        how_to_change: The setting or argument that would decide otherwise.
    """

    index: int
    name: str
    dtype: str | None
    examples: tuple[str, ...]
    n_missing: int
    declared_categorical: bool
    n_unique: int | None
    n_unique_is_exact: bool
    modality: FeatureModality | None
    expanded_by: Literal["dates", "text"] | None
    generated_features: tuple[str, ...]
    ordinal_encoded: bool
    reason: str
    how_to_change: str

    @property
    def read_as(self) -> str:
        """The modality as text, or `expanded` for an expanded column."""
        return "expanded" if self.modality is None else self.modality.value


@dataclasses.dataclass(frozen=True, repr=False)
class InputReport:
    """How every column of a fit input was read, before the model saw any of it.

    `print(report)` shows one line per input column; `to_frame` gives the same
    as a DataFrame, `to_dict` as plain values for `json.dumps`, `to_markdown` as
    a pipe table.

    Attributes:
        columns: One entry per column of the fit input, in input order.
        n_samples: Rows in the fit input.
        n_features_out: Columns after expansion, as detection and encoding saw them.
        settings: The `inference_config` fields that steer the reading, as applied.
    """

    columns: tuple[ColumnReport, ...]
    n_samples: int
    n_features_out: int
    settings: dict[str, Any]

    def columns_for(self, modality: FeatureModality) -> list[ColumnReport]:
        """The columns read as `modality`."""
        return [column for column in self.columns if column.modality is modality]

    def to_frame(self) -> pd.DataFrame:
        """One row per input column, one column per `ColumnReport` field."""
        return pd.DataFrame([_plain(column) for column in self.columns])

    def to_dict(self) -> dict[str, Any]:
        """The report as plain Python values, ready for `json.dumps`."""
        return {
            "n_samples": self.n_samples,
            "n_features_out": self.n_features_out,
            "settings": dict(self.settings),
            "columns": [_plain(column) for column in self.columns],
        }

    def to_markdown(self) -> str:
        """The table `print(report)` shows, as a Markdown pipe table."""
        cells = self._display_frame().astype(str).replace(r"\|", r"\\|", regex=True)
        header = "| " + " | ".join(cells.columns) + " |"
        separator = "|" + "|".join(" --- " for _ in cells.columns) + "|"
        rows = ["| " + " | ".join(row) + " |" for row in cells.to_numpy().tolist()]
        return "\n".join([header, separator, *rows])

    def __str__(self) -> str:
        header = (
            f"InputReport: {len(self.columns)} input columns, {self.n_samples} rows, "
            f"{self.n_features_out} columns after expansion"
        )
        settings = ", ".join(f"{name}={value}" for name, value in self.settings.items())
        return f"{header}\nsettings: {settings}\n{_left_aligned(self._display_frame())}"

    def __repr__(self) -> str:
        return str(self)

    def _display_frame(self) -> pd.DataFrame:
        """The table shown by `__str__`, one short column per fact."""
        return pd.DataFrame(
            {
                "#": [column.index for column in self.columns],
                "column": [column.name for column in self.columns],
                "dtype": [column.dtype or "" for column in self.columns],
                "examples": [", ".join(column.examples) for column in self.columns],
                "missing": [column.n_missing for column in self.columns],
                "declared": [
                    "yes" if column.declared_categorical else "no"
                    for column in self.columns
                ],
                "distinct": [_render_count(column) for column in self.columns],
                "read as": [column.read_as for column in self.columns],
                "encoded": [
                    "ordinal" if column.ordinal_encoded else ""
                    for column in self.columns
                ],
                "why": [column.reason for column in self.columns],
                "how to change": [column.how_to_change for column in self.columns],
            }
        )


def _plain(column: ColumnReport) -> dict[str, Any]:
    """`column` as plain values: the enum as its string, tuples as lists."""
    values = dataclasses.asdict(column)
    values["modality"] = None if column.modality is None else column.modality.value
    values["examples"] = list(column.examples)
    values["generated_features"] = list(column.generated_features)
    return values


def _left_aligned(frame: pd.DataFrame) -> str:
    """`frame` as text, each column left-aligned and as wide as its widest cell."""
    cells = frame.astype(str)
    widths = [
        max(len(str(name)), *(len(value) for value in cells[name]))
        for name in cells.columns
    ]
    rows = [[str(name) for name in cells.columns], *cells.to_numpy().tolist()]
    return "\n".join(
        "  ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    )


def _render_count(column: ColumnReport) -> str:
    if column.n_unique is None:
        return ""
    return f"{'' if column.n_unique_is_exact else '>='}{column.n_unique}"


def input_settings(inference_config: InferenceConfig) -> dict[str, Any]:
    """The values `inference_config` holds for `INPUT_SETTINGS`, by field name."""
    return {name: getattr(inference_config, name) for name in INPUT_SETTINGS}


def build_input_report(
    X: XType,
    *,
    declared: Sequence[int] | None,
    date_transformer: DateTransformer,
    text_transformer: TextTransformer,
    decisions: Sequence[ModalityDecision],
    feature_schema: FeatureSchema,
    ordinal_encoder: OrderPreservingColumnTransformer,
    settings: dict[str, Any],
) -> InputReport:
    """Assemble the report from what fit computed while reading `X`.

    Each input position is followed through the two expansions: a position the
    date transformer expanded is reported from its record, else it is shifted to
    the date-expanded frame, where the text transformer may have expanded it, else
    it is shifted once more to the validated frame, where its decision and its
    encoding live. Expanded positions must be caught before shifting, since the
    shift is only defined for kept columns.

    Args:
        X: The fit input as the caller passed it, before any conversion.
        declared: Positions in `X` declared categorical, or `None` for none.
        date_transformer: Fitted on `X`.
        text_transformer: Fitted on the date-expanded `X`.
        decisions: One per column of the expanded, validated input, in order.
        feature_schema: The schema `clean_data` returned for that input.
        ordinal_encoder: The encoder `clean_data` fitted on it.
        settings: `input_settings` of the config the reading ran with.

    Returns:
        The report, one `ColumnReport` per column of `X`.
    """
    frame = _as_frame(X)
    n_samples, n_columns = frame.shape
    names = _column_names(X, n_columns)
    dtypes = _column_dtypes(X, n_columns)
    missing_counts = frame.isna().sum().tolist()
    encoded_positions = set(ordinal_encoder.selected_columns())
    declared_positions = set(declared or ())

    columns: list[ColumnReport] = []
    for i in range(n_columns):
        facts = _ColumnFacts(
            index=i,
            name=names[i],
            dtype=dtypes[i],
            examples=_examples(frame.iloc[:_EXAMPLE_ROWS, i]),
            n_missing=int(missing_counts[i]),
            declared_categorical=i in declared_positions,
        )
        if i in date_transformer.expansions:
            columns.append(
                _expanded_column(
                    facts,
                    expanded_by="dates",
                    generated_features=tuple(date_transformer.expansions[i]),
                    settings=settings,
                )
            )
            continue
        j = date_transformer.output_indices([i])[0]
        if j in text_transformer.expansions:
            columns.append(
                _expanded_column(
                    facts,
                    expanded_by="text",
                    generated_features=tuple(text_transformer.expansions[j]),
                    settings=settings,
                )
            )
            continue
        k = text_transformer.output_indices([j])[0]
        columns.append(
            _decided_column(
                facts,
                decision=decisions[k],
                ordinal_encoded=k in encoded_positions,
                n_samples=n_samples,
                settings=settings,
            )
        )

    return InputReport(
        columns=tuple(columns),
        n_samples=n_samples,
        n_features_out=feature_schema.num_columns,
        settings=dict(settings),
    )


@dataclasses.dataclass(frozen=True)
class _ColumnFacts:
    """What is known about an input column before its outcome is looked up."""

    index: int
    name: str
    dtype: str | None
    examples: tuple[str, ...]
    n_missing: int
    declared_categorical: bool


def _expanded_column(
    facts: _ColumnFacts,
    *,
    expanded_by: Literal["dates", "text"],
    generated_features: tuple[str, ...],
    settings: dict[str, Any],
) -> ColumnReport:
    count = len(generated_features)
    if expanded_by == "dates":
        reason = (
            f"{facts.dtype} column, expanded into {count} calendar features because "
            "TRANSFORM_DATES=True."
        )
        how_to_change = (
            'Set inference_config={"TRANSFORM_DATES": False} to have fit refuse it, '
            "or convert it before fit."
        )
    else:
        reason = (
            "pandas `string` column, not declared categorical, with more than "
            f"MIN_CARDINALITY_FOR_TEXT={settings['MIN_CARDINALITY_FOR_TEXT']} distinct "
            f"values and not all numbers; expanded into {count} text features "
            "because TRANSFORM_TEXT=True."
        )
        how_to_change = (
            "List its position in `categorical_features_indices` to keep it as one "
            "category; raise MIN_CARDINALITY_FOR_TEXT; set TRANSFORM_TEXT=False; "
            "TEXT_N_COMPONENTS sets how many features it becomes."
        )
    return ColumnReport(
        index=facts.index,
        name=facts.name,
        dtype=facts.dtype,
        examples=facts.examples,
        n_missing=facts.n_missing,
        declared_categorical=facts.declared_categorical,
        n_unique=None,
        n_unique_is_exact=True,
        modality=None,
        expanded_by=expanded_by,
        generated_features=generated_features,
        ordinal_encoded=False,
        reason=reason,
        how_to_change=how_to_change,
    )


def _decided_column(
    facts: _ColumnFacts,
    *,
    decision: ModalityDecision,
    ordinal_encoded: bool,
    n_samples: int,
    settings: dict[str, Any],
) -> ColumnReport:
    reason, how_to_change = _explain(
        decision,
        declared=facts.declared_categorical,
        dtype=facts.dtype,
        ordinal_encoded=ordinal_encoded,
        all_missing=facts.n_missing == n_samples,
        n_samples=n_samples,
        settings=settings,
    )
    if facts.dtype is not None and facts.dtype.startswith("timedelta"):
        reason = (
            f"timedelta converted to seconds first; {reason[0].lower()}{reason[1:]}"
        )
    return ColumnReport(
        index=facts.index,
        name=facts.name,
        dtype=facts.dtype,
        examples=facts.examples,
        n_missing=facts.n_missing,
        declared_categorical=facts.declared_categorical,
        n_unique=decision.n_unique,
        n_unique_is_exact=decision.n_unique_is_exact,
        modality=decision.modality,
        expanded_by=None,
        generated_features=(),
        ordinal_encoded=ordinal_encoded,
        reason=reason,
        how_to_change=how_to_change,
    )


def _explain(  # noqa: PLR0911
    decision: ModalityDecision,
    *,
    declared: bool,
    dtype: str | None,
    ordinal_encoded: bool,
    all_missing: bool,
    n_samples: int,
    settings: dict[str, Any],
) -> tuple[str, str]:
    """The rule that decided a kept column's modality, and what would change it."""
    max_unique = settings["MAX_UNIQUE_FOR_CATEGORICAL_FEATURES"]
    min_unique = settings["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"]
    min_samples = settings["MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE"]
    text_cutoff = settings["MIN_CARDINALITY_FOR_TEXT"]
    n = decision.n_unique
    count = f"{'' if decision.n_unique_is_exact else 'at least '}{n}"
    declare = "List its position in `categorical_features_indices`"

    if decision.modality is FeatureModality.CONSTANT:
        if all_missing:
            return (
                "Every value is missing; dropped as constant.",
                f"{declare} to keep it.",
            )
        return (
            f"{count} distinct value, a missing value counting as one; dropped as "
            "constant.",
            f"{declare} to keep it.",
        )

    if decision.modality is FeatureModality.CATEGORICAL:
        if declared:
            within = (
                f", with {count} distinct values within "
                f"MAX_UNIQUE_FOR_CATEGORICAL_FEATURES={max_unique}"
                if decision.numeric_like
                else ""
            )
            return (
                "Declared categorical through `categorical_features_indices` or the "
                f"`category` dtype{within}.",
                "Remove the declaration to have it inferred.",
            )
        if decision.numeric_like:
            return (
                f"Numbers with {count} distinct values, below "
                f"MIN_UNIQUE_FOR_NUMERICAL_FEATURES={min_unique}, in more than "
                f"MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE={min_samples} rows.",
                f"Lower MIN_UNIQUE_FOR_NUMERICAL_FEATURES to {n} or below to read it "
                "as numerical.",
            )
        return (
            f"Values do not all parse as numbers; {count} distinct values, at most "
            f"MIN_CARDINALITY_FOR_TEXT={text_cutoff}.",
            f"Lower MIN_CARDINALITY_FOR_TEXT below {n} to read it as text; cast it "
            "to a numeric dtype if it holds numbers.",
        )

    if decision.modality is FeatureModality.NUMERICAL:
        if ordinal_encoded:
            return (
                "Every value is a number stored as a string; counted as numeric, but "
                "ordinal-encoded by dtype, so the numeric order is lost.",
                "Cast the column to a numeric dtype, e.g. with `pd.to_numeric`.",
            )
        if declared:
            target = (
                f"{n} or above" if decision.n_unique_is_exact else "its distinct count"
            )
            return (
                f"Declared categorical, but {count} distinct values exceed "
                f"MAX_UNIQUE_FOR_CATEGORICAL_FEATURES={max_unique}, so read as "
                "numerical.",
                f"Raise MAX_UNIQUE_FOR_CATEGORICAL_FEATURES to {target}.",
            )
        if n_samples <= min_samples:
            return (
                f"Numbers; {n_samples} rows is not more than "
                f"MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE={min_samples}, so no "
                "category is inferred.",
                f"{declare}, or lower MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE.",
            )
        return (
            f"Numbers with {count} distinct values, at least "
            f"MIN_UNIQUE_FOR_NUMERICAL_FEATURES={min_unique}.",
            f"{declare} (honoured up to MAX_UNIQUE_FOR_CATEGORICAL_FEATURES distinct "
            "values), or raise MIN_UNIQUE_FOR_NUMERICAL_FEATURES above its distinct "
            "count.",
        )

    not_expanded = (
        f" Not expanded: only pandas' `string` dtype is, and this column's dtype is "
        f"{dtype}."
        if settings["TRANSFORM_TEXT"]
        else ""
    )
    return (
        f"Values do not all parse as numbers; {count} distinct values exceed "
        f"MIN_CARDINALITY_FOR_TEXT={text_cutoff}; ordinal-encoded as a "
        f"high-cardinality category.{not_expanded}",
        f"{declare}; raise MIN_CARDINALITY_FOR_TEXT; or give it pandas' `string` "
        "dtype and set TRANSFORM_TEXT=True to expand it into numeric features.",
    )


def _as_frame(X: XType) -> pd.DataFrame:
    """`X` as a DataFrame, without copying an array."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    return pd.DataFrame(np.asarray(X, dtype=object))


def _column_names(X: XType, n_columns: int) -> list[str]:
    """Input labels as text, or positional `f{i}` names for an input without any."""
    if isinstance(X, pd.DataFrame):
        return [str(column) for column in X.columns]
    return [f"f{i}" for i in range(n_columns)]


def _column_dtypes(X: XType, n_columns: int) -> list[str | None]:
    """Each input column's dtype as text, `None` for an input that has none."""
    if isinstance(X, pd.DataFrame):
        return [str(dtype) for dtype in X.dtypes]
    if isinstance(X, np.ndarray):
        return [str(X.dtype)] * n_columns
    return [None] * n_columns


def _examples(values: pd.Series) -> tuple[str, ...]:
    """Up to `MAX_EXAMPLES` distinct present values, in order of first appearance."""
    present = values[values.notna()].to_numpy(dtype=object)
    distinct = pd.unique(present)[:MAX_EXAMPLES]
    return tuple(
        repr(value) if isinstance(value, str) else str(value) for value in distinct
    )


__all__ = [
    "INPUT_SETTINGS",
    "MAX_EXAMPLES",
    "ColumnReport",
    "InputReport",
    "build_input_report",
    "input_settings",
]
