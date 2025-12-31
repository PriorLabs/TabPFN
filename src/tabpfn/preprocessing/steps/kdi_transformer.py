"""KDI Transformer with NaN."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer

try:
    from kditransform import KDITransformer

    # This import fails on some systems, due to problems with numba
except ImportError:
    KDITransformer = PowerTransformer  # fallback to avoid error

# Track whether we've warned the user about missing kditransform
_warned_about_missing_kditransform = False

ALPHAS = (
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
    1.8,
    2.0,
    2.5,
    3.0,
    5.0,
)


class KDITransformerWithNaN(KDITransformer):
    """KDI transformer that can handle NaN values.

    It performs KDI with NaNs replaced by mean values and then fills the NaN values
    with NaNs after the transformation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        output_distribution: str = "uniform",
        *,
        standardize: bool = True,
        copy: bool = True,
    ) -> None:
        # ``kditransform`` exposes ``alpha`` and ``output_distribution`` but the
        # PowerTransformer fallback does not. To keep compatibility across both
        # backends, only pass the parameters that are supported by the active
        # base class.
        if KDITransformer is PowerTransformer:
            self.alpha = alpha
            self.output_distribution = output_distribution
            super().__init__(standardize=standardize, copy=copy)
        else:
            self.standardize = standardize
            super().__init__(
                alpha=alpha,
                output_distribution=output_distribution,
                copy=copy,
            )

    def _more_tags(self) -> dict:
        return {"allow_nan": True}

    def fit(
        self,
        X: torch.Tensor | np.ndarray,
        y: Any | None = None,
    ) -> KDITransformerWithNaN:
        """Fit the transformer."""
        global _warned_about_missing_kditransform  # noqa: PLW0603
        if (
            KDITransformer is PowerTransformer
            and not _warned_about_missing_kditransform
        ):
            warnings.warn(
                "Cannot use KDITransformer because `kditransform` is not installed. "
                "Using `PowerTransformer` as fallback. This warning is only shown "
                "once per Python interpreter instance.",
                UserWarning,
                stacklevel=2,
            )
            _warned_about_missing_kditransform = True

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # If all-nan or empty, nanmean returns nan.
        self.imputation_values_ = np.nan_to_num(np.nanmean(X, axis=0), nan=0)
        X = np.nan_to_num(X, nan=self.imputation_values_)

        return super().fit(X, y)  # type: ignore

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        """Transform the data."""
        # if tensor convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Calculate the NaN mask for the current dataset
        nan_mask = np.isnan(X)

        # Replace NaNs with the mean of columns
        X = np.nan_to_num(X, nan=self.imputation_values_)

        # Apply the transformation
        X = super().transform(X)

        # Reintroduce NaN values based on the current dataset's mask
        X[nan_mask] = np.nan

        return X  # type: ignore


def get_all_kdi_transformers() -> dict[str, KDITransformerWithNaN]:
    """Get all KDI transformers."""
    try:
        all_preprocessors = {
            "kdi": KDITransformerWithNaN(alpha=1.0, output_distribution="normal"),
            "kdi_uni": KDITransformerWithNaN(
                alpha=1.0,
                output_distribution="uniform",
            ),
        }
        for alpha in ALPHAS:
            all_preprocessors[f"kdi_alpha_{alpha}"] = KDITransformerWithNaN(
                alpha=alpha,
                output_distribution="normal",
            )
            all_preprocessors[f"kdi_alpha_{alpha}_uni"] = KDITransformerWithNaN(
                alpha=alpha,
                output_distribution="uniform",
            )
        return all_preprocessors
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "KDITransformerWithNaN",
    "get_all_kdi_transformers",
]
