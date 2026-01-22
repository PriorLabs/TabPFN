#  Copyright (c) Prior Labs GmbH 2025.

"""Interfaces for torch preprocessing pipeline."""

from __future__ import annotations

import abc
from typing_extensions import override

import torch

from tabpfn.preprocessing.torch.datamodel import (
    ColumnMetadata,
    FeatureModality,
    TorchPreprocessingPipelineOutput,
    TorchPreprocessingStepResult,
)


class TorchPreprocessingStep(abc.ABC):
    """Base class for preprocessing steps that operate on specific columns.

    Subclasses should implement `_fit` and `_transform` to define the actual
    transformation logic. The base class handles column selection, tensor
    cloning, and reassignment.
    """

    def fit(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
    ) -> None:
        """Fit on training data for the specified columns.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should fit on.
            num_train_rows: Number of training rows (fit on x[:num_train_rows]).
        """
        x_cols = x[:num_train_rows, :, column_indices]
        self._fit(x_cols)

    def transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
    ) -> TorchPreprocessingStepResult:
        """Transform the specified columns.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should transform.

        Returns:
            TorchPreprocessingStepResult with the transformed tensor and any
            added columns.
        """
        x_cols = x[:, :, column_indices]
        transformed, added_columns, added_modality = self._transform(x_cols)
        x = x.clone()
        x[:, :, column_indices] = transformed
        return TorchPreprocessingStepResult(
            x=x,
            added_columns=added_columns,
            added_modality=added_modality,
        )

    @override
    def __repr__(self) -> str:
        """Return a string representation of the step."""
        return f"{self.__class__.__name__}"

    @abc.abstractmethod
    def _fit(self, x: torch.Tensor) -> None:
        """Fit on the selected columns (training rows only).

        Args:
            x: Tensor of selected columns [num_train_rows, batch_size, num_cols].
        """
        ...

    @abc.abstractmethod
    def _transform(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform the selected columns.

        Args:
            x: Tensor of selected columns [num_rows, batch_size, num_cols].

        Returns:
            Tuple of (transformed_columns, added_columns, added_modality).
            added_columns and added_modality can be None if no columns are added.
        """
        ...


class TorchPreprocessingPipeline:
    """Modality-aware preprocessing pipeline.

    This pipeline applies a sequence of preprocessing steps to a tensor,
    where each step targets specific feature modalities. Steps can target
    multiple modalities at once (e.g., StandardScaler for both NUMERICAL
    and CATEGORICAL features).
    """

    def __init__(
        self,
        steps: list[tuple[TorchPreprocessingStep, set[FeatureModality]]],
    ) -> None:
        """Initialize with list of (step, target_modalities) pairs.

        Args:
            steps: List of (step, modalities) where modalities is a set of
                FeatureModality values the step should be applied to.
        """
        super().__init__()
        self.steps = steps

    def __call__(
        self,
        x: torch.Tensor,
        metadata: ColumnMetadata,
        num_train_rows: int | None = None,
    ) -> TorchPreprocessingPipelineOutput:
        """Apply all steps to the input tensor.

        Args:
            x: Input tensor [num_rows, batch_size, num_columns] or
                [num_rows, num_columns]. If 2D, a batch dimension is added
                and removed after processing.
            metadata: Column modality information.
            num_train_rows: If provided, fit steps on x[:num_train_rows]. If
                not provided, fits on the entire input tensor.

        Returns:
            PipelineOutput with transformed tensor and updated metadata.
        """
        squeeze_batch = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze_batch = True

        num_columns = x.shape[-1]
        if num_columns != metadata.num_columns:
            raise ValueError(
                f"Number of columns in input tensor ({num_columns}) does not match "
                f"number of columns in metadata ({metadata.num_columns})"
            )

        for step, modalities in self.steps:
            indices = metadata.indices_for_modalities(modalities)
            if not indices:
                continue

            if num_train_rows is None:
                num_train_rows = x.shape[0]

            step.fit(x, column_indices=indices, num_train_rows=num_train_rows)
            result = step.transform(x, column_indices=indices)
            x = result.x

            if result.added_columns is not None:
                x = torch.cat([x, result.added_columns], dim=-1)
                metadata = metadata.add_columns(
                    result.added_modality or FeatureModality.NUMERICAL,
                    result.added_columns.shape[-1],
                )

        if squeeze_batch:
            x = x.squeeze(1)

        return TorchPreprocessingPipelineOutput(x=x, metadata=metadata)

    @override
    def __repr__(self) -> str:
        """Return a string representation of the pipeline."""
        return f"TorchPreprocessingPipeline(steps={self.steps})"
