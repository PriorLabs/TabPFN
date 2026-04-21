#  Copyright (c) Prior Labs GmbH 2025.

"""Utilities for GPU preprocessing pipeline construction."""

from __future__ import annotations

from tabpfn.preprocessing.steps.adaptive_quantile_transformer import (
    compute_effective_n_quantiles as _compute_effective_n_quantiles,
    get_user_n_quantiles_for_preset,
)

_GPU_ELIGIBLE_QUANTILE_TRANSFORMS = frozenset(
    {
        "quantile_uni",
        "quantile_uni_coarse",
        "quantile_uni_fine",
    }
)


def is_gpu_quantile_eligible(transform_name: str) -> bool:
    """Check if a transform name can be accelerated on GPU.

    Currently, only the quantile_uni* transforms are eligible for GPU acceleration.
    """
    return transform_name in _GPU_ELIGIBLE_QUANTILE_TRANSFORMS


def compute_effective_n_quantiles(transform_name: str, n_samples: int) -> int:
    """Compute effective n_quantiles for a named quantile preset."""
    user_n_quantiles = get_user_n_quantiles_for_preset(transform_name, n_samples)
    return _compute_effective_n_quantiles(user_n_quantiles, n_samples)
