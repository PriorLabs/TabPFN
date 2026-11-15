#  Copyright (c) Prior Labs GmbH 2026.

"""Remove Constant Features Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
import torch

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.pipeline_interface import (
    _SCAN_ROWS,
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
        forced = [feat.non_constant_with_inf for feat in feature_schema.features]
        if isinstance(X, torch.Tensor):
            sel_ = (torch.max(X[0:1, :] != X, dim=0)[0] & ~X.isnan().all(dim=0)).cpu()
            if any(forced):
                sel_ = sel_ | torch.tensor(forced, dtype=torch.bool)
        else:
            # A column is constant if no row differs from the first; NaN differs from
            # everything. Scanned a block of rows at a time so the masks stay small.
            differs = np.zeros(X.shape[1], dtype=bool)
            all_nan = np.ones(X.shape[1], dtype=bool)
            for start in range(0, X.shape[0], _SCAN_ROWS):
                block = X[start : start + _SCAN_ROWS]
                differs |= (block != X[0:1, :]).any(axis=0)
                all_nan &= np.isnan(block).all(axis=0)
            sel_ = np.logical_and(differs, ~all_nan).tolist()
            if any(forced):
                sel_ = [bool(keep or f) for keep, f in zip(sel_, forced, strict=False)]

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
        if self._keeps_every_column():
            # Selecting every column with a boolean mask still builds a full copy of
            # the array
            return X, None, None
        return X[:, self.sel_], None, None

    def _keeps_every_column(self) -> bool:
        """Whether the fitted selection drops nothing."""
        if isinstance(self.sel_, torch.Tensor):
            return bool(torch.all(self.sel_))
        return all(self.sel_)  # type: ignore[arg-type]
