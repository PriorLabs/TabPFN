#  Copyright (c) Prior Labs GmbH 2026.

"""Utilities for GPU preprocessing pipeline construction."""

from __future__ import annotations

from typing import Literal

from tabpfn.preprocessing.datamodel import GPUTransformType
from tabpfn.preprocessing.steps.adaptive_quantile_transformer import (
    compute_effective_n_quantiles as _compute_effective_n_quantiles,
    get_user_n_quantiles_for_preset,
)

_GPU_ELIGIBLE_QUANTILE_TRANSFORMS: dict[str, Literal["uniform", "normal"]] = {
    "quantile_uni": "uniform",
    "quantile_uni_coarse": "uniform",
    "quantile_uni_fine": "uniform",
    "quantile_uni_extrapolate": "uniform",
    "quantile_norm": "normal",
    "quantile_norm_coarse": "normal",
    "quantile_norm_fine": "normal",
}

_GPU_ELIGIBLE_SQUASHING_SCALER_TRANSFORMS: dict[str, float] = {
    "squashing_scaler_default": 3.0,
    "squashing_scaler_max10": 10.0,
}


def is_gpu_quantile_eligible(transform_name: str) -> bool:
    """Check if a quantile transform name can be accelerated on GPU."""
    return transform_name in _GPU_ELIGIBLE_QUANTILE_TRANSFORMS


def get_quantile_output_distribution(
    transform_name: str,
) -> Literal["uniform", "normal"]:
    """Return the output distribution of a GPU-eligible quantile preset."""
    try:
        return _GPU_ELIGIBLE_QUANTILE_TRANSFORMS[transform_name]
    except KeyError as err:
        raise ValueError(
            f"Unknown GPU-eligible quantile preset: {transform_name!r}"
        ) from err


def is_gpu_squashing_scaler_eligible(transform_name: str) -> bool:
    """Check if a squashing-scaler transform name can be accelerated on GPU."""
    return transform_name in _GPU_ELIGIBLE_SQUASHING_SCALER_TRANSFORMS


def get_gpu_transform_type(transform_name: str) -> GPUTransformType | None:
    """Return the GPU transform that replaces a reshape preset, if any."""
    if is_gpu_quantile_eligible(transform_name):
        return GPUTransformType.QUANTILE
    if is_gpu_squashing_scaler_eligible(transform_name):
        return GPUTransformType.SQUASHING_SCALER
    if transform_name == "safepower":
        return GPUTransformType.SAFE_POWER
    return None


def get_squashing_scaler_max_absolute_value(transform_name: str) -> float:
    """Return the ``max_absolute_value`` for a GPU-eligible squashing scaler preset."""
    try:
        return _GPU_ELIGIBLE_SQUASHING_SCALER_TRANSFORMS[transform_name]
    except KeyError as err:
        raise ValueError(
            f"Unknown GPU-eligible squashing scaler preset: {transform_name!r}"
        ) from err


def compute_effective_n_quantiles(transform_name: str, n_samples: int) -> int:
    """Compute effective n_quantiles for a named quantile preset."""
    user_n_quantiles = get_user_n_quantiles_for_preset(transform_name, n_samples)
    return _compute_effective_n_quantiles(user_n_quantiles, n_samples)
