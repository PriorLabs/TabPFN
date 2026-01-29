"""Interfaces for creating preprocessing pipelines."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing_extensions import TypeAlias

import numpy as np

from tabpfn.preprocessing.datamodel import ColumnMetadata, FeatureModality

if TYPE_CHECKING:
    import torch


StepWithModalities: TypeAlias = tuple["PreprocessingStep", set[FeatureModality]]


@dataclasses.dataclass
class PreprocessingStepResult:
    """Result of a feature preprocessing step.

    Attributes:
        X: Transformed array. For steps registered with specific modalities,
            this is only the transformed columns (not the full array).
            The shape should match the input shape unless columns are removed.
        column_metadata: Column metadata for the columns this step processed.
            Contains 0-based indices relative to the step's input.
            Should NOT include added_columns - the pipeline handles that.
        X_added: Optional new features to append (e.g., fingerprint features).
            These are handled by the pipeline, which concatenates them and
            updates the metadata accordingly. Steps should NOT concatenate
            these internally.
        modality_added: Modality for the added features. Required if X_added
            is provided.
    """

    X: np.ndarray | torch.Tensor
    column_metadata: ColumnMetadata
    X_added: np.ndarray | torch.Tensor | None = None
    modality_added: FeatureModality | None = None

    def __post_init__(self) -> None:
        """Validate that modality_added is provided when X_added is set."""
        if self.X_added is not None and self.modality_added is None:
            raise ValueError("modality_added must be provided when X_added is not None")


@dataclasses.dataclass
class PreprocessingPipelineResult:
    """Result from the preprocessing pipeline.

    Attributes:
        X: The transformed array.
        column_metadata: Updated column metadata (may have new columns added).
    """

    X: np.ndarray | torch.Tensor
    column_metadata: ColumnMetadata


class PreprocessingStep:
    """Base class for feature preprocessing steps.

    Steps can be registered with specific feature modalities, and the pipeline
    will handle slicing the data to only pass the relevant columns to the step.

    Subclasses should implement `_fit` and `_transform` methods. The `_fit` method
    receives the sliced data and metadata, and should return the metadata after
    transformation (for the transformed columns only, NOT including added_columns).

    The `_transform` method receives the sliced data and returns the transformed
    array plus new columns and new modality separately. The pipeline handles
    concatenation.

    Design principle: Steps should NOT internally handle passthrough of columns
    they don't transform. The pipeline handles column slicing and reassembly.
    """

    metadata_after_transform_: ColumnMetadata

    @abstractmethod
    def _fit(
        self,
        X: np.ndarray,
        metadata: ColumnMetadata,
    ) -> ColumnMetadata:
        """Underlying method of the preprocessor to implement by subclasses.

        Args:
            X: 2d array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            metadata: Column metadata for the input columns.

        Returns:
            Column metadata after the transform.
        """
        ...

    @abstractmethod
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        """Underlying method of the preprocessor to implement by subclasses.

        Args:
            X: array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            is_test: Whether this is test data (used for AddFingerPrint step).

        Returns:
            Tuple of (transformed_columns, added_columns, added_modality).
            added_columns and added_modality can be None if no columns are added.
        """
        ...

    def fit_transform(
        self,
        X: np.ndarray,
        metadata: ColumnMetadata | dict[FeatureModality, list[int]],
    ) -> PreprocessingStepResult:
        """Fits the preprocessor and transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
            metadata: Column metadata or dictionary of feature modalities.

        Returns:
            PreprocessingStepResult with transformed data and updated metadata.
        """
        if isinstance(metadata, dict):
            metadata = ColumnMetadata.from_dict(metadata)

        self.metadata_after_transform_ = self._fit(X, metadata)
        return self.transform(X, is_test=False)

    def transform(
        self,
        X: np.ndarray,
        *,
        is_test: bool = True,
    ) -> PreprocessingStepResult:
        """Transforms the data.

        Args:
            X: array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            is_test: Whether this is test data (used for AddFingerPrint step).

        Returns:
            PreprocessingStepResult with transformed data and metadata.
        """
        result, X_added, modality_added = self._transform(X, is_test=is_test)

        return PreprocessingStepResult(
            X=result,
            column_metadata=self.metadata_after_transform_,
            X_added=X_added,
            modality_added=modality_added,
        )


class PreprocessingPipeline:
    """Modality-aware preprocessing pipeline that handles column slicing.

    This pipeline applies a sequence of preprocessing steps to an array,
    where each step can be registered to target specific feature modalities.
    The pipeline handles slicing columns based on registered modalities,
    passing only relevant columns to each step, reassembling data after each
    step, and tracking metadata updates.

    Steps can be registered as (step, modalities) tuples where the step receives
    only columns matching the specified modalities, or as bare steps that receive
    all columns.
    """

    def __init__(
        self,
        steps: list[PreprocessingStep | StepWithModalities],
    ) -> None:
        """Initialize the pipeline with preprocessing steps.

        Args:
            steps: List of preprocessing steps. Each can be a PreprocessingStep
                (receives all columns) or a tuple of (PreprocessingStep,
                set[FeatureModality]) where the step receives only columns
                matching the specified modalities.
        """
        super().__init__()
        self._raw_steps = steps
        self.steps = self._validate_steps(steps)
        self.final_metadata_: ColumnMetadata | None = None
        self.initial_metadata_: ColumnMetadata | None = None

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        metadata: ColumnMetadata | dict[FeatureModality, list[int]],
    ) -> PreprocessingPipelineResult:
        """Fit and transform the data using the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
            metadata: Column metadata or dictionary of feature modalities.

        Returns:
            PreprocessingPipelineResult with transformed data and updated metadata.
        """
        if isinstance(metadata, dict):
            metadata = ColumnMetadata.from_dict(metadata)

        self.initial_metadata_ = metadata
        X, metadata = self._process_steps(X, metadata, is_fitting=True)
        self.final_metadata_ = metadata
        return PreprocessingPipelineResult(X=X, column_metadata=metadata)

    def transform(self, X: np.ndarray | torch.Tensor) -> PreprocessingPipelineResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).

        Returns:
            PreprocessingPipelineResult with transformed data and metadata.
        """
        assert self.initial_metadata_ is not None, (
            "The pipeline must be fit before it can be used to transform."
        )
        assert self.final_metadata_ is not None

        X, _ = self._process_steps(X, self.initial_metadata_, is_fitting=False)
        return PreprocessingPipelineResult(X=X, column_metadata=self.final_metadata_)

    def _process_steps(
        self,
        X: np.ndarray | torch.Tensor,
        metadata: ColumnMetadata,
        *,
        is_fitting: bool,
    ) -> tuple[np.ndarray | torch.Tensor, ColumnMetadata]:
        """Process all pipeline steps.

        Args:
            X: Input array of shape (n_samples, n_features).
            metadata: Column metadata.
            is_fitting: If True, call fit_transform on steps; otherwise transform.

        Returns:
            Tuple of (transformed array, updated metadata).
        """
        for step, modalities in self.steps:
            if modalities:
                indices = metadata.indices_for_modalities(modalities)
                if not indices:
                    continue

                X_slice = X[:, indices]
                result = (
                    step.fit_transform(X_slice, metadata.slice_for_indices(indices))
                    if is_fitting
                    else step.transform(X_slice)
                )

                if result.X.shape[1] != len(indices):
                    raise ValueError(
                        f"Step {step.__class__.__name__} registered with modalities "
                        f"{modalities} received {len(indices)} columns but returned "
                        f"{result.X.shape[1]} columns. Steps registered with "
                        f"modalities must return the same number of columns."
                    )

                X = X.copy() if isinstance(X, np.ndarray) else X.clone()
                X[:, indices] = result.X

                X, metadata = self._append_added_columns(X, metadata, result)
                metadata = metadata.update_from_step_result(
                    indices, result.column_metadata
                )
            else:
                # We still have preprocessing steps that don't change the columns
                # internally (will be deprecated going forward). For backwards
                # compatibility, we still handle these here.
                result = (
                    step.fit_transform(X, metadata) if is_fitting else step.transform(X)
                )
                X = result.X
                metadata = result.column_metadata
                X, metadata = self._append_added_columns(X, metadata, result)

        return X, metadata

    def _append_added_columns(
        self,
        X: np.ndarray | torch.Tensor,
        metadata: ColumnMetadata,
        result: PreprocessingStepResult,
    ) -> tuple[np.ndarray | torch.Tensor, ColumnMetadata]:
        """Append added columns from a step result and update metadata."""
        if result.X_added is not None:
            X = np.concatenate([X, result.X_added], axis=1)
            metadata = metadata.add_columns(
                result.modality_added or FeatureModality.NUMERICAL,
                result.X_added.shape[1],
            )
        return X, metadata

    def _validate_steps(
        self,
        steps: list[PreprocessingStep | StepWithModalities],
    ) -> list[StepWithModalities]:
        """Convert steps to normalized (step, modalities) format."""
        normalized: list[StepWithModalities] = []
        if len(steps) == 0:
            raise ValueError("The pipeline must have at least one step.")
        for step in steps:
            if isinstance(step, tuple):
                if len(step) != 2:
                    raise ValueError(
                        f"Step tuple must be (step, modalities), got {step}"
                    )
                normalized.append(step)
            else:
                normalized.append((step, set()))
        return normalized

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)
