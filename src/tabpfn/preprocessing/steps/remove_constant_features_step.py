#  Copyright (c) Prior Labs GmbH 2026.

"""Remove Constant Features Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
import torch

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
    any_per_column,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema


class RemoveConstantFeaturesStep(PreprocessingStep):
    """Remove features that are constant in the training data."""

    no_arithmetic = True

    source_selection: list[bool] | None = None
    """The columns to keep, decided on the fit input in a wider dtype than this step
    reads. Rounding to the narrower dtype can make a column look constant."""

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
        if self.source_selection is not None:
            if len(self.source_selection) != X.shape[1]:
                raise ValueError(
                    f"The selection has {len(self.source_selection)} columns, but "
                    f"the input has {X.shape[1]}."
                )
            sel_ = list(self.source_selection)
        elif isinstance(X, torch.Tensor):
            sel_ = (torch.max(X[0:1, :] != X, dim=0)[0] & ~X.isnan().all(dim=0)).cpu()
            if any(forced):
                sel_ = sel_ | torch.tensor(forced, dtype=torch.bool)
        else:
            sel_ = _non_constant_columns(X).tolist()
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
        if isinstance(X, np.ndarray):
            return X.compress(np.asarray(self.sel_, dtype=bool), axis=1), None, None
        return X[:, self.sel_], None, None

    def _keeps_every_column(self) -> bool:
        """Whether the fitted selection drops nothing."""
        if isinstance(self.sel_, torch.Tensor):
            return bool(torch.all(self.sel_))
        return all(self.sel_)  # type: ignore[arg-type]


def kept_by_fit(X: np.ndarray) -> np.ndarray:
    """Per column of `X`, whether this step keeps it after a pipeline fits it on `X`.

    The pipeline passes infinities to the step as NaN. The pipeline also keeps a
    column whose infinities vary.
    """
    infinite = np.isinf(X)
    if not infinite.any():
        return _non_constant_columns(X)
    signs = np.where(infinite, X, 0)
    return _non_constant_columns(np.where(infinite, np.nan, X)) | (
        signs != signs[:1]
    ).any(axis=0)


def _non_constant_columns(X: np.ndarray) -> np.ndarray:
    """Per column of `X`, whether its rows are not all equal and not all NaN.

    A NaN equals nothing, so a column with a NaN and a number is not constant.
    """
    first = X[0:1]
    differs, has_number = any_per_column(
        X, [lambda block: block != first, lambda block: ~np.isnan(block)]
    )
    return differs & has_number
