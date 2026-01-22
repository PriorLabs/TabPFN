#  Copyright (c) Prior Labs GmbH 2025.

"""Interfaces for torch preprocessing pipeline."""

from __future__ import annotations

import abc

import torch

from .datamodel import (
    ColumnMetadata,
    FeatureModality,
    PipelineOutput,
    TransformResult,
)


class TorchPreprocessingStep(abc.ABC):
    """Base class for preprocessing steps that operate on specific columns.

    Subclasses should create any state that depends on the train set in
    `fit` and use it in `transform`. This allows fitting on data first
    and doing inference later without refitting.
    """

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
    def transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
    ) -> TransformResult:
        """Transform the specified columns.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should transform.

        Returns:
            TransformResult with the transformed tensor and any added columns.
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
    ) -> PipelineOutput:
        """Apply all steps to the input tensor.

        Args:
            x: Input tensor [num_rows, batch_size, num_columns] or
                [num_rows, num_columns]. If 2D, a batch dimension is added
                and removed after processing.
            metadata: Column modality information.
            num_train_rows: If provided, fit steps on x[:num_train_rows].

        Returns:
            PipelineOutput with transformed tensor and updated metadata.
        """
        squeeze_batch = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze_batch = True

        for step, modalities in self.steps:
            indices = metadata.indices_for_modalities(modalities)
            if not indices:
                continue

            # TODO: Probably we always want to fit fow now.
            # Fit if we have training data
            if num_train_rows is not None and num_train_rows > 0:
                step.fit(x, indices, num_train_rows)

            result = step.transform(x, indices)
            x = result.x

            if result.added_columns is not None:
                x = torch.cat([x, result.added_columns], dim=-1)
                metadata = metadata.add_columns(
                    result.added_modality or FeatureModality.NUMERICAL,
                    result.added_columns.shape[-1],
                )

        if squeeze_batch:
            x = x.squeeze(1)

        return PipelineOutput(x=x, metadata=metadata)
