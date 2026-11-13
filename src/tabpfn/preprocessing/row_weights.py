#  Copyright (c) Prior Labs GmbH 2026.

"""Validate the per-row weights `fit` takes, and how the model applies them.

`sample_weight` says how many rows a training row stands for. Only its relative
values matter, so it is scaled to mean 1 over each estimator's train rows.
`label_reliability` is the probability that a row's label is correct. The model
receives both as one `(rows, 2)` array, `(sample_weight, label_reliability)` per
row, where 1 is neutral in each column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from tabpfn.errors import TabPFNValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.architectures.interface import Architecture
    from tabpfn.constants import TaskType

__all__ = [
    "SampleWeightMode",
    "resolve_sample_weight_mode",
    "row_weights_for_estimator",
    "validate_row_weights",
]

SampleWeightMode: TypeAlias = Literal["learned", "head"]
"""How the model applies `sample_weight`.

`"learned"` feeds the weights to the model as an input it was trained to read.
`"head"` scales each training row's vote in the decoder, so it exists for
classification only. `label_reliability` is always a model input.
"""


def validate_row_weights(
    sample_weight: object | None,
    label_reliability: object | None,
    *,
    n_samples: int,
    y_encoded: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> np.ndarray | None:
    """Check the per-row weights and stack them into a `(n_samples, 2)` array.

    Args:
        sample_weight: One non-negative weight per training row, or `None`.
        label_reliability: One value in `[0, 1]` per training row, or `None`.
        n_samples: The number of training rows.
        y_encoded: The class index of each row, for a classifier. Every class
            must keep a positive total weight.
        classes: The class labels `y_encoded` indexes, named in the error.

    Returns:
        The stacked float32 weights, `sample_weight` divided by its maximum, or
        `None` when neither argument is given.
    """
    if sample_weight is None and label_reliability is None:
        return None
    row_weights = np.ones((n_samples, 2), dtype=np.float32)

    if sample_weight is not None:
        weights = _per_row_values(sample_weight, "sample_weight", n_samples)
        if (weights < 0).any():
            raise TabPFNValidationError("`sample_weight` must not be negative.")
        if not weights.any():
            raise TabPFNValidationError(
                "`sample_weight` is zero for every row; at least one weight must "
                "be positive."
            )
        # Dividing by the maximum before the float32 cast keeps any finite input
        # in range, and bounds the per-estimator sum by the number of rows.
        row_weights[:, 0] = weights / weights.max()
        if y_encoded is not None:
            _raise_if_a_class_has_zero_weight(row_weights[:, 0], y_encoded, classes)

    if label_reliability is not None:
        reliability = _per_row_values(label_reliability, "label_reliability", n_samples)
        if ((reliability < 0) | (reliability > 1)).any():
            raise TabPFNValidationError(
                "`label_reliability` must lie in [0, 1]: it is the probability "
                "that a row's label is correct."
            )
        row_weights[:, 1] = reliability

    return row_weights


def row_weights_for_estimator(
    row_weights: np.ndarray, rows: np.ndarray | None
) -> np.ndarray:
    """The weights of one estimator's train rows, `sample_weight` at mean 1.

    Args:
        row_weights: The output of `validate_row_weights`.
        rows: The indices of the train rows the estimator keeps, or `None` for all.
    """
    selected = row_weights if rows is None else row_weights[rows]
    total = selected[:, 0].sum()
    if total == 0:
        raise TabPFNValidationError(
            "Row subsampling left an estimator only rows with `sample_weight` 0. "
            "Raise `SUBSAMPLE_SAMPLES` in the inference config, or give fewer rows "
            "weight 0."
        )
    estimator_weights = selected.copy()
    estimator_weights[:, 0] *= len(selected) / total
    return estimator_weights


def resolve_sample_weight_mode(
    models: Sequence[Architecture],
    mode: SampleWeightMode | None,
    *,
    uses_label_reliability: bool,
    task_type: TaskType,
) -> SampleWeightMode:
    """Check that every model can apply the weights `fit` was given.

    Args:
        models: The loaded models of the ensemble.
        mode: `SAMPLE_WEIGHT_MODE` from the inference config.
        uses_label_reliability: Whether `fit` was given `label_reliability`,
            which needs the `"learned"` mode in the model.
        task_type: The task of the estimator.

    Returns:
        `mode`, once every check has passed.
    """
    if mode is None:
        raise TabPFNValidationError(
            "This model does not support `sample_weight` or `label_reliability`: "
            "its inference config sets no `SAMPLE_WEIGHT_MODE`. Use a checkpoint "
            "trained with per-row weights, or fit without them."
        )
    if mode == "head" and task_type == "regression":
        raise TabPFNValidationError(
            "SAMPLE_WEIGHT_MODE='head' scales class votes and exists for "
            "classification only. Use SAMPLE_WEIGHT_MODE='learned' for regression."
        )
    for model in models:
        supported = model.get_supported_sample_weight_modes()
        if mode not in supported:
            raise TabPFNValidationError(
                f"SAMPLE_WEIGHT_MODE={mode!r} is not supported by this model, "
                f"which supports {list(supported)}."
            )
        if uses_label_reliability and "learned" not in supported:
            raise TabPFNValidationError(
                "This model does not support `label_reliability`: it needs a "
                "model that supports SAMPLE_WEIGHT_MODE='learned'."
            )
    return mode


def _per_row_values(values: object, name: str, n_samples: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TabPFNValidationError(f"`{name}` must be numeric.") from e
    if array.shape != (n_samples,):
        raise TabPFNValidationError(
            f"`{name}` must have shape ({n_samples},), one value per training "
            f"row, but has shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise TabPFNValidationError(f"`{name}` must be finite.")
    return array


def _raise_if_a_class_has_zero_weight(
    weights: np.ndarray, y_encoded: np.ndarray, classes: np.ndarray | None
) -> None:
    class_indices = np.asarray(y_encoded).astype(np.int64)
    present = np.bincount(class_indices) > 0
    totals = np.bincount(class_indices, weights=weights, minlength=len(present))
    empty = np.flatnonzero(present & (totals == 0))
    if len(empty) == 0:
        return
    names = (classes[empty] if classes is not None else empty).tolist()
    raise TabPFNValidationError(
        f"`sample_weight` is zero for every row of class(es) {names}. Each class "
        "in `y` needs a positive total weight; drop the class's rows instead."
    )
