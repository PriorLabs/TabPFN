#  Copyright (c) Prior Labs GmbH 2025.

"""Pipeline step wrappers for torch preprocessing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from tabpfn.preprocessing.torch.pipeline_interface import TorchPreprocessingStep
from tabpfn.preprocessing.torch.torch_remove_outliers import TorchRemoveOutliers
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler

if TYPE_CHECKING:
    import torch

    from tabpfn.preprocessing.torch.datamodel import FeatureModality


class TorchStandardScalerStep(TorchPreprocessingStep):
    """Pipeline step wrapper for TorchStandardScaler."""

    def __init__(self) -> None:
        """Initialize the standard scaler step."""
        super().__init__()
        self._scaler = TorchStandardScaler()

    @override
    def _fit(self, x: torch.Tensor) -> None:
        """Fit the scaler on the selected columns."""
        self._scaler.fit(x)

    @override
    def _transform(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform columns using the fitted scaler."""
        return self._scaler.transform(x), None, None


class TorchRemoveOutliersStep(TorchPreprocessingStep):
    """Pipeline step wrapper for TorchRemoveOutliers."""

    def __init__(self, n_sigma: float = 4.0) -> None:
        """Initialize the outlier removal step.

        Args:
            n_sigma: Number of standard deviations to use for outlier threshold.
        """
        super().__init__()
        self._outlier_remover = TorchRemoveOutliers(n_sigma=n_sigma)

    @override
    def _fit(self, x: torch.Tensor) -> None:
        """Fit the outlier remover on the selected columns."""
        self._outlier_remover.fit(x)

    @override
    def _transform(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform columns using the fitted outlier remover."""
        return self._outlier_remover.transform(x), None, None
