#  Copyright (c) Prior Labs GmbH 2026.

"""Preprocessing steps for passing infinite values through to the model.

The standard preprocessing pipeline cannot handle non-finite values, so when
``passthrough_inf`` is enabled :class:`InfToNanStep` runs first to record the
location and sign of any infinities and replace them with NaN (which the
pipeline tolerates). After all other steps run, :class:`RestoreInfStep` writes
the original infinities back so they reach the model.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from tabpfn.preprocessing.datamodel import FeatureSchema
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
    PreprocessingStepResult,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureModality


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
    def transform(
        self, X: np.ndarray, *, is_test: bool = True
    ) -> PreprocessingStepResult:
        """Record infinite values and replace them with NaN.

        Overrides :meth:`PreprocessingStep.transform` instead of implementing
        ``_transform`` so that ``self.feature_schema_updated_`` is **not**
        mutated in place: the recorded ``inf_mask`` is written onto a fresh copy
        of the schema which is then returned. The same schema (and its
        ``Feature`` objects) is shared across ensemble members preprocessed in
        parallel, so mutating it in place would be a data race.

        Args:
            X: 2d array of shape (n_samples, n_features).
            is_test: Whether this is a test-time transform. Unused.

        Returns:
            PreprocessingStepResult with infinities replaced by NaN and a copy
            of the feature schema carrying the recorded ``inf_mask``.
        """
        del is_test
        bool_mask = np.isinf(X)
        new_features = []
        for idx, feat in enumerate(self.feature_schema_updated_.features):
            feature_bool_mask = bool_mask[:, idx]
            if np.any(feature_bool_mask):
                new_features.append(
                    dataclasses.replace(
                        feat, inf_mask=np.where(feature_bool_mask, X[:, idx], 0)
                    )
                )
            else:
                new_features.append(feat)
        X[bool_mask] = np.nan

        self._validate_added_data(X_added=None, modality_added=None)
        return PreprocessingStepResult(
            X=X,
            feature_schema=FeatureSchema(features=new_features),
            X_added=None,
            modality_added=None,
        )

    @override
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        """Not used; :meth:`transform` is overridden directly.

        Raising here guarantees no caller reaches the default ``transform``
        path, which would return ``self.feature_schema_updated_`` and reopen the
        in-place-mutation race this step avoids.
        """
        raise NotImplementedError(
            "InfToNanStep overrides transform() directly; _transform() must not "
            "be called."
        )


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
    def transform(
        self, X: np.ndarray, *, is_test: bool = True
    ) -> PreprocessingStepResult:
        """Write recorded infinities back into the transformed array.

        Overrides :meth:`PreprocessingStep.transform` directly (rather than
        implementing ``_transform``) so it stays consistent with
        :class:`InfToNanStep`. This step only reads ``inf_mask`` and writes to
        ``X``; it leaves ``self.feature_schema_updated_`` untouched and returns
        it unchanged.

        Args:
            X: 2d array of shape (n_samples, n_features).
            is_test: Whether this is a test-time transform. Unused.

        Returns:
            PreprocessingStepResult with infinities restored.
        """
        del is_test
        for idx, feat in enumerate(self.feature_schema_updated_.features):
            if feat.inf_mask is not None:
                # TODO: should we store the bool mask or recompute it?
                bool_mask = np.isinf(feat.inf_mask)
                X[bool_mask, idx] = feat.inf_mask[bool_mask]

        self._validate_added_data(X_added=None, modality_added=None)
        return PreprocessingStepResult(
            X=X,
            feature_schema=self.feature_schema_updated_,
            X_added=None,
            modality_added=None,
        )

    @override
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        """Not used; :meth:`transform` is overridden directly.

        Raising keeps the step's contract explicit: callers must go through the
        overridden ``transform``.
        """
        raise NotImplementedError(
            "RestoreInfStep overrides transform() directly; _transform() must "
            "not be called."
        )


__all__ = [
    "InfToNanStep",
    "RestoreInfStep",
]
