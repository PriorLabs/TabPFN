#  Copyright (c) Prior Labs GmbH 2026.

"""Preprocessing steps for passing infinite values through to the model.

The standard preprocessing pipeline cannot handle non-finite values, so when
``passthrough_inf`` is enabled :class:`InfToNanStep` runs first to record the
location and sign of any infinities and replace them with NaN (which the
pipeline tolerates). After all other steps run, :class:`RestoreInfStep` writes
the original infinities back so they reach the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema


class InfToNanStep(PreprocessingStep):
    """Replace infinities with NaN, recording them for later restoration.

    Stores the infinite values (with sign) on each affected feature's
    ``inf_mask`` so :class:`RestoreInfStep` can put them back after the rest of
    the pipeline has run.
    """

    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        """No-op fit; the schema is returned unchanged."""
        return feature_schema

    @override
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        """Record infinite values on the feature schema and replace with NaN.

        Args:
            X: 2d array of shape (n_samples, n_features).
            is_test: Whether this is a test-time transform. Unused.

        Returns:
            Tuple of (X with infinities replaced by NaN, None, None).
        """
        bool_mask = np.isinf(X)
        features = self.feature_schema_updated_.features
        for idx, feat in enumerate(features):
            feature_bool_mask = bool_mask[:, idx]
            if np.any(feature_bool_mask):
                feat.inf_mask = np.where(feature_bool_mask, X[:, idx], 0)
        X[bool_mask] = np.nan
        return X, None, None


class RestoreInfStep(PreprocessingStep):
    """Restore infinities recorded by :class:`InfToNanStep`.

    Writes each feature's stored ``inf_mask`` values back into ``X`` so the
    original infinities reach the model.
    """

    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        """No-op fit; the schema is returned unchanged."""
        return feature_schema

    @override
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        """Write recorded infinities back into the transformed array.

        Args:
            X: 2d array of shape (n_samples, n_features).
            is_test: Whether this is a test-time transform. Unused.

        Returns:
            Tuple of (X with infinities restored, None, None).
        """
        features = self.feature_schema_updated_.features
        for idx, feat in enumerate(features):
            if feat.inf_mask is not None:
                # TODO: should we store the bool mask or recompute it?
                bool_mask = np.isinf(feat.inf_mask)
                X[bool_mask, idx] = feat.inf_mask[bool_mask]
        return X, None, None


__all__ = [
    "InfToNanStep",
    "RestoreInfStep",
]
