"""Remove Constant Features Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
import torch

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.pipeline_interfaces import (
    PreprocessingStep,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import (
    filter_modalities_by_kept_indices,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureModality


class RemoveConstantFeaturesStep(PreprocessingStep):
    """Remove features that are constant in the training data."""

    def __init__(self) -> None:
        super().__init__()
        self.sel_: list[bool] | None = None

    @override
    def _fit(  # type: ignore
        self,
        X: np.ndarray | torch.Tensor,
        feature_modalities: dict[FeatureModality, list[int]],
    ) -> dict[FeatureModality, list[int]]:
        if isinstance(X, torch.Tensor):
            sel_ = torch.max(X[0:1, :] != X, dim=0)[0].cpu()
        else:
            sel_ = ((X[0:1, :] == X).mean(axis=0) < 1.0).tolist()

        if not any(sel_):
            raise TabPFNValidationError(
                "All features are constant and would have been removed!"
                " Unable to predict using TabPFN.",
            )
        self.sel_ = sel_

        # Get indices of kept features and remap all modalities
        kept_indices = list(np.where(sel_)[0])
        return filter_modalities_by_kept_indices(feature_modalities, kept_indices)

    @override
    def _transform(
        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
    ) -> np.ndarray:
        assert self.sel_ is not None, "You must call fit first"
        return X[:, self.sel_]
