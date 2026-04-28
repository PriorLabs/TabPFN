"""Append duplicate copies of selected columns as additional features."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.pipeline_interface import PreprocessingStep

if TYPE_CHECKING:
    import numpy as np

    from tabpfn.preprocessing.datamodel import FeatureSchema


class DuplicateImportantFeaturesStep(PreprocessingStep):
    """Append exact duplicates of the columns at ``local_indices``.

    The duplicates are emitted via ``X_added`` so the pipeline framework
    concatenates them as a contiguous tail at the end of the transformed
    output. Used to encourage the per-feature transformer's contiguous feature
    grouping to co-locate important features.

    ``num_added_features`` returns 0 by design: this is augmentation, not
    subsampling, so the upstream subsampling budget should not shrink to
    accommodate the duplicates. The model will see ``input_features + K``
    columns.
    """

    def __init__(self, local_indices: list[int] | None = None):
        super().__init__()
        self.local_indices: list[int] = (
            list(local_indices) if local_indices is not None else []
        )

    @override
    def num_added_features(self, n_samples: int, feature_schema: FeatureSchema) -> int:
        del n_samples, feature_schema
        return 0

    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        n_features = X.shape[1]
        for idx in self.local_indices:
            if not 0 <= idx < n_features:
                raise ValueError(
                    f"DuplicateImportantFeaturesStep got local index {idx} "
                    f"but input has only {n_features} columns."
                )
        return feature_schema

    @override
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        if not self.local_indices:
            return X, None, None
        return X, X[:, self.local_indices], FeatureModality.NUMERICAL
