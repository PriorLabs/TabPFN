#  Copyright (c) Prior Labs GmbH 2026.

"""The usage event a finished call is logged as.

Each event is a dict in the format of the telemetry API's events, with the same
field names, so what is built here is what is sent.
"""

from __future__ import annotations

import functools
import platform
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

import torch
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import _num_features, _num_samples

from tabpfn.telemetry.parameters import (
    checkpoint_of,
    config_of,
    embed_params_of,
    label_of,
    predict_params_of,
)

EventKind = Literal["fit", "predict", "embed"]


def usage_event(
    estimator: Any,
    arguments: Mapping[str, Any],
    *,
    kind: EventKind,
    batched: bool,
    method: str,
    succeeded: bool,
    started_at: datetime,
    duration_ms: int,
) -> dict[str, Any] | None:
    """The usage event a finished call is logged as, or None if it is not logged.

    Every call is one event. A batched call's event covers all of its datasets,
    see `_batched_event`. A fit or embedding call that failed before the size of
    its data could be read is not logged: the telemetry API requires that size.
    """
    fitted = _fitted(estimator)
    fields = _call_fields(
        estimator,
        fitted,
        kind=kind,
        method=method,
        succeeded=succeeded,
        started_at=started_at,
        duration_ms=duration_ms,
    )
    if batched:
        return _batched_event(arguments, fields)

    return _single_event(fitted, arguments, fields, kind=kind, succeeded=succeeded)


def _single_event(
    fitted: Any,
    arguments: Mapping[str, Any],
    call: Mapping[str, Any],
    *,
    kind: EventKind,
    succeeded: bool,
) -> dict[str, Any] | None:
    """The event of a call, sized from its `X` argument.

    None for a call the telemetry API would reject: a fit or embedding call that
    failed before the size of its data could be read, or an embedding call
    with a data source the API does not accept.
    """
    num_rows, num_columns = _shape(arguments.get("X"))
    if num_rows is None and kind != "predict":
        return None

    fields: dict[str, Any] = {"num_rows": num_rows, "num_columns": num_columns}

    # After a failed fit, the fitted state may still describe an earlier fit.
    if succeeded or kind != "fit":
        fields["num_classes"] = getattr(fitted, "n_classes_", None)
        fields["actual_estimators"] = getattr(fitted, "n_estimators_", None)

    # Add the fields specific to the kind of call.
    if kind == "predict":
        fields["fit_num_rows"] = getattr(fitted, "n_train_samples_", None)
        fields["predict_params"] = predict_params_of(arguments)
    elif kind == "embed":
        fields["fit_num_rows"] = getattr(fitted, "n_train_samples_", None)
        fields.update(embed_params_of(arguments))
        if "data_source" not in fields:
            return None

    return _create_event(kind, call, **fields)


def _batched_event(
    arguments: Mapping[str, Any], call: Mapping[str, Any]
) -> dict[str, Any]:
    """The event of a batched call, which predicts for several datasets at once.

    The call fits and predicts each dataset in `X_train_list` and `X_test_list`
    on an internal copy of the estimator, and only accepts datasets that share
    one train and one test shape. Its event has that shape, per dataset, and the
    number of datasets. A call given datasets of different shapes fails before
    predicting, so its event has no shape. The estimator's fitted state, which
    the call leaves unchanged, is not read.
    """
    train_shapes = {_shape(X) for X in arguments["X_train_list"]}
    test_shapes = {_shape(X) for X in arguments["X_test_list"]}
    fields: dict[str, Any] = {
        # An empty list fails the call, and the API only accepts a positive count.
        "num_datasets": len(arguments["X_train_list"]) or None,
        "predict_params": predict_params_of(arguments),
    }

    if len(train_shapes) == 1 and len(test_shapes) == 1:
        fit_num_rows, num_columns = next(iter(train_shapes))
        num_rows, _ = next(iter(test_shapes))
        fields.update(
            num_rows=num_rows, fit_num_rows=fit_num_rows, num_columns=num_columns
        )
    return _create_event("predict", call, **fields)


def _call_fields(
    estimator: Any,
    fitted: Any,
    *,
    kind: EventKind,
    method: str,
    succeeded: bool,
    started_at: datetime,
    duration_ms: int,
) -> dict[str, Any]:
    """The fields every event of a call has in common."""
    model_path, model_version = checkpoint_of(fitted)
    if model_version is None:
        # A fine-tuned model is held in memory, not in a checkpoint file. Its
        # version is the one that was fine-tuned.
        model_version = label_of(getattr(estimator, "finetune_model_version", None))

    is_classifier = isinstance(estimator, ClassifierMixin)
    devices = getattr(fitted, "devices_", None)
    return {
        "timestamp": started_at.isoformat(),
        "python_version": platform.python_version(),
        "tabpfn_version": _tabpfn_version(),
        "gpu_type": _gpu_type(devices[0]) if devices else None,
        "model_path": model_path,
        "model_version": model_version,
        "task": "classification" if is_classifier else "regression",
        "method": method,
        # A fit logs the settings it was asked for, such as the epochs of
        # fine-tuning. A prediction logs those of the estimator that made it.
        "config": config_of(estimator if kind == "fit" else fitted),
        "status": "success" if succeeded else "failed",
        "duration_ms": duration_ms,
    }


def _fitted(estimator: Any) -> Any:
    """The estimator that holds the fitted state of `estimator`.

    A fine-tuned estimator predicts with an ordinary one that it fits at the end
    of fine-tuning, and which holds its fitted state.
    """
    for name in ("finetuned_inference_classifier_", "finetuned_inference_regressor_"):
        inner = getattr(estimator, name, None)
        if inner is not None:
            return inner
    return estimator


def _create_event(
    kind: EventKind, call: Mapping[str, Any], **fields: Any
) -> dict[str, Any]:
    # The id is generated once and kept when the event is sent again, so the API
    # can tell a resent event from a new one.
    return {"event": f"{kind}_called", "event_id": str(uuid.uuid4()), **call, **fields}


def _shape(X: Any) -> tuple[int | None, int | None]:
    """Read rows and columns with the helpers TabPFN's input validation uses.

    They raise for inputs TabPFN rejects, such as a 1D array. Those dimensions
    are None instead, so a call that fails on such an input is still logged.
    """
    try:
        num_rows = _num_samples(X)
    except TypeError:
        return None, None
    try:
        return num_rows, _num_features(X)
    except TypeError:
        return num_rows, None


@functools.lru_cache
def _gpu_type(device: torch.device) -> str | None:
    """The GPU a call ran on, or None if it ran on the CPU."""
    if device.type == "mps":
        return "mps"
    if device.type != "cuda":
        return None
    try:
        return label_of(torch.cuda.get_device_name(device))
    except (RuntimeError, AssertionError):
        # CUDA cannot start here, as in a process forked after it had started.
        return None


@functools.cache
def _tabpfn_version() -> str:
    # Imported here: this module is itself imported while `tabpfn` loads.
    import tabpfn  # noqa: PLC0415

    # Without a local build suffix, such as "+g8ed2398d" from a source checkout.
    return tabpfn.__version__.split("+", maxsplit=1)[0]
