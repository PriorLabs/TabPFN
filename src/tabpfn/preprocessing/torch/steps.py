#  Copyright (c) Prior Labs GmbH 2025.

"""Pipeline step wrappers for torch preprocessing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from .datamodel import TransformResult
from .pipeline_interface import TorchPreprocessingStep
from .torch_remove_outliers import TorchRemoveOutliers
from .torch_standard_scaler import TorchStandardScaler

if TYPE_CHECKING:
    import torch


class TorchStandardScalerStep(TorchPreprocessingStep):
    """Pipeline step wrapper for TorchStandardScaler."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self._scaler = TorchStandardScaler()

    @override
    def fit(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
    ) -> None:
        """Fit the scaler on training data for the specified columns.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should fit on.
            num_train_rows: Number of training rows (fit on x[:num_train_rows]).
        """
        x_cols = x[:num_train_rows, :, column_indices]
        self._scaler.fit(x_cols)

    @override
    def transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
    ) -> TransformResult:
        """Transform the specified columns using the fitted scaler.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should transform.

        Returns:
            TransformResult with the transformed tensor.
        """
        x_cols = x[:, :, column_indices]
        transformed = self._scaler.transform(x_cols)
        x = x.clone()
        x[:, :, column_indices] = transformed
        return TransformResult(x=x)


class TorchRemoveOutliersStep(TorchPreprocessingStep):
    """Pipeline step wrapper for TorchRemoveOutliers.

    This wraps the TorchRemoveOutliers to implement the TorchPreprocessingStep
    interface, allowing it to be used in the preprocessing pipeline.
    """

    def __init__(self, n_sigma: float = 4.0) -> None:
        """Initialize the outlier removal step.

        Args:
            n_sigma: Number of standard deviations to use for outlier threshold.
        """
        super().__init__()
        self._outlier_remover = TorchRemoveOutliers(n_sigma=n_sigma)

    @override
    def fit(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
    ) -> None:
        """Fit the outlier remover on training data for the specified columns.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should fit on.
            num_train_rows: Number of training rows (fit on x[:num_train_rows]).
        """
        x_cols = x[:num_train_rows, :, column_indices]
        self._outlier_remover.fit(x_cols)

    @override
    def transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
    ) -> TransformResult:
        """Transform the specified columns using the fitted outlier remover.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should transform.

        Returns:
            TransformResult with the transformed tensor.
        """
        x_cols = x[:, :, column_indices]
        transformed = self._outlier_remover.transform(x_cols)
        x = x.clone()
        x[:, :, column_indices] = transformed
        return TransformResult(x=x)
