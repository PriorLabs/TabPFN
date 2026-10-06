#  Copyright (c) Prior Labs GmbH 2026.

"""Which of an estimator's parameters are logged, and how values are logged."""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import torch
from pydantic import (
    BeforeValidator,
    Field,
    NonNegativeInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from tabpfn.constants import ModelVersion
from tabpfn.model_loading import ModelType, _get_model_source, resolve_model_version

Label = Annotated[str, StringConstraints(pattern=r"^[\w .:+()\-]{1,128}$")]
"""A string the analytics API accepts: short, and without the "/", "\\" or "@"
that paths, URLs and email addresses need."""


def _torch_name(value: Any) -> Any:
    # TabPFN takes dtypes and devices as torch objects, the API takes their names.
    return str(value) if isinstance(value, (torch.dtype, torch.device)) else value


TorchName = Annotated[Label, BeforeValidator(_torch_name)]
"""A `Label`, which may also be given as a torch dtype or device."""

# The values the analytics API accepts in an event, declared as gapi declares
# them in `TabPFNConfig`, `PredictParams` and `EmbedCalled`; keep them in sync.
# The API rejects any other name and any value of another type, such as the
# "auto" default of `n_estimators`, so those are left out.
_CONFIG_FIELDS: dict[str, TypeAdapter[Any]] = {
    "n_estimators": TypeAdapter(NonNegativeInt),
    "auto_scale_n_estimators": TypeAdapter(bool),
    "softmax_temperature": TypeAdapter(float),
    "balance_probabilities": TypeAdapter(bool),
    "average_before_softmax": TypeAdapter(bool),
    "device": TypeAdapter(TorchName | list[TorchName]),
    "ignore_pretraining_limits": TypeAdapter(bool),
    "inference_precision": TypeAdapter(TorchName),
    "fit_mode": TypeAdapter(Label),
    "memory_saving_mode": TypeAdapter(bool | float | Label),
    "keep_cache_on_device": TypeAdapter(bool),
    "kv_cache_precision": TypeAdapter(Label),
    "n_preprocessing_jobs": TypeAdapter(int),
    "differentiable_input": TypeAdapter(bool),
    "eval_metric": TypeAdapter(Label),
    "epochs": TypeAdapter(NonNegativeInt),
    "time_limit": TypeAdapter(NonNegativeInt),
}
_PREDICT_PARAMS: dict[str, TypeAdapter[Any]] = {
    "output_type": TypeAdapter(Label),
    "quantiles": TypeAdapter(list[Annotated[float, Field(ge=0, le=1)]]),
}
_EMBED_PARAMS: dict[str, TypeAdapter[Any]] = {
    "data_source": TypeAdapter(Label),
}
_LABEL: TypeAdapter[str] = TypeAdapter(Label)


def config_of(estimator: Any) -> dict[str, Any]:
    """The estimator's settings that the analytics API accepts.

    They are read from the estimator's constructor parameters with `get_params`,
    so a renamed or removed parameter is no longer logged.
    """
    return _accepted(_CONFIG_FIELDS, estimator.get_params(deep=False))


def predict_params_of(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments of a prediction that the analytics API accepts."""
    return _accepted(_PREDICT_PARAMS, arguments)


def embed_params_of(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments of an embedding call that the analytics API accepts."""
    return _accepted(_EMBED_PARAMS, arguments)


def label_of(value: Any) -> str | None:
    """The value as a string the analytics API accepts, or None if it is not one."""
    try:
        return _LABEL.validate_python(value)
    except ValidationError:
        return None


def checkpoint_of(estimator: Any) -> tuple[str | None, str | None]:
    """The checkpoint's file name and model version.

    Both are logged only for a published checkpoint, which is what
    `create_default_for_version` uses, and for "auto". Any other file is logged as
    "other", without a version: its name could identify the user's own files,
    and the version TabPFN infers from a name is a guess.
    """
    model_path = getattr(estimator, "model_path", None)
    if model_path == "auto":
        return "auto", resolve_model_version(None).value
    if not isinstance(model_path, (str, Path)):
        return None, None
    name = Path(model_path).name
    if name not in _published_checkpoint_names():
        return "other", None
    return name, resolve_model_version(model_path).value


def _accepted(
    fields: Mapping[str, TypeAdapter[Any]], values: Mapping[str, Any]
) -> dict[str, Any]:
    """The values the analytics API accepts for these fields, as it accepts them.

    A missing value, or one the API would reject, is left out.
    """
    accepted: dict[str, Any] = {}
    for name, field in fields.items():
        with contextlib.suppress(ValidationError):
            accepted[name] = field.validate_python(values.get(name))
    return accepted


@functools.cache
def _published_checkpoint_names() -> frozenset[str]:
    names: set[str] = set()
    for version in ModelVersion:
        for model_type in ModelType:
            try:
                names.update(_get_model_source(version, model_type).filenames)
            except ValueError:
                # This version has no checkpoint for this model type.
                continue
    return frozenset(names)
