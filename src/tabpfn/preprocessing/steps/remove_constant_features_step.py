"""Remove Constant Features Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
import torch

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema


class RemoveConstantFeaturesStep(PreprocessingStep):
    """Remove features that are constant in the training data."""

    def __init__(self) -> None:
        super().__init__()
        self.sel_: list[bool] | None = None

    @override
    def _fit(  # type: ignore
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
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

        # Get indices of removed features and update schema
        removed_indices = list(np.where(~np.array(sel_))[0])
        return feature_schema.remove_columns(removed_indices)

    @override
    def _transform(
        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        assert self.sel_ is not None, "You must call fit first"
        return X[:, self.sel_], None, None
