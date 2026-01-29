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

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import ColumnMetadata, FeatureModality


class RemoveConstantFeaturesStep(PreprocessingStep):
    """Remove features that are constant in the training data."""

    def __init__(self) -> None:
        super().__init__()
        self.sel_: list[bool] | None = None

    @override
    def _fit(  # type: ignore
        self,
        X: np.ndarray | torch.Tensor,
        metadata: ColumnMetadata,
    ) -> ColumnMetadata:
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

        # Get indices of removed features and update metadata
        removed_indices = list(np.where(~np.array(sel_))[0])
        return metadata.remove_columns(removed_indices)

    # TODO: Add test for it and make it useable with modality assignment
    # in pipeline registration.
    @override
    def _transform(
        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        assert self.sel_ is not None, "You must call fit first"
        return X[:, self.sel_], None, None
