"""Interfaces for creating preprocessing pipelines."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing_extensions import Self

import numpy as np

from tabpfn.preprocessing.datamodel import ColumnMetadata, FeatureModality

if TYPE_CHECKING:
    import torch


@dataclasses.dataclass
class PreprocessingStepResult:
    """Result of a feature preprocessing step.

    Attributes:
        X: Transformed array. For steps registered with specific modalities,
            this is only the transformed columns (not the full array).
            The shape should match the input shape unless columns are removed.
        metadata: Column metadata for the columns this step processed.
            Contains 0-based indices relative to the step's input.
            Should NOT include added_columns - the pipeline handles that.
        added_columns: Optional new columns to append (e.g., fingerprint features).
            These are handled by the pipeline, which concatenates them and
            updates the metadata accordingly. Steps should NOT concatenate
            these internally.
        added_modality: Modality for the added columns. Required if added_columns
            is provided.
    """

    X: np.ndarray | torch.Tensor
    metadata: ColumnMetadata
    added_columns: np.ndarray | torch.Tensor | None = None
    added_modality: FeatureModality | None = None

    def __post_init__(self) -> None:
        """Validate that added_modality is provided when added_columns is set."""
        if self.added_columns is not None and self.added_modality is None:
            raise ValueError(
                "added_modality must be provided when added_columns is not None"
            )

    # TODO: Remove once all tests are updated to use the new API
    @property
    def feature_modalities(self) -> dict[FeatureModality, list[int]]:
        """Get feature modalities as a dictionary (for backward compatibility)."""
        return self.metadata.to_dict()


class PreprocessingStep:
    """Base class for feature preprocessing steps.

    Steps can be registered with specific feature modalities, and the pipeline
    will handle slicing the data to only pass the relevant columns to the step.

    Subclasses should implement `_fit` and `_transform` methods. The `_fit` method
    receives the sliced data and metadata, and should return the metadata after
    transformation (for the transformed columns only, NOT including added_columns).

    The `_transform` method receives the sliced data and returns the transformed
    array. Steps that add columns should override `_get_added_columns` to return
    the new columns separately - the pipeline handles concatenation.

    Design principle: Steps should NOT internally handle passthrough of columns
    they don't transform. The pipeline handles column slicing and reassembly.
    """

    metadata_after_transform_: ColumnMetadata

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

        self.fit(X, metadata)
        result = self._transform(X, is_test=False)

        # Check if step has added columns
        added_columns, added_modality = self._get_added_columns()

        return PreprocessingStepResult(
            X=result,
            metadata=self.metadata_after_transform_,
            added_columns=added_columns,
            added_modality=added_modality,
        )

    def _get_added_columns(
        self,
    ) -> tuple[np.ndarray | torch.Tensor | None, FeatureModality | None]:
        """Return any columns added by this step.

        Override this method in subclasses that add new columns (e.g., fingerprint
        features, SVD components). The pipeline will handle concatenating these
        columns and updating the metadata.

        Returns:
            Tuple of (added_columns, modality) or (None, None) if no columns added.
        """
        return None, None

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
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray,
        metadata: ColumnMetadata | dict[FeatureModality, list[int]],
    ) -> Self:
        """Fits the preprocessor.

        Args:
            X: 2d array of shape (n_samples, n_features).
            metadata: Column metadata or dictionary of feature modalities.
        """
        if isinstance(metadata, dict):
            metadata = ColumnMetadata.from_dict(metadata)

        self.metadata_after_transform_ = self._fit(X, metadata)
        assert self.metadata_after_transform_ is not None, (
            "_fit should have returned ColumnMetadata after the transform."
        )
        return self

    @abstractmethod
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        """Underlying method of the preprocessor to implement by subclasses.

        Args:
            X: 2d array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            is_test: Whether this is test data (used for AddFingerPrint step).

        Returns:
            2d np.ndarray of shape (n_samples, new n_features).
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> PreprocessingStepResult:
        """Transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).

        Returns:
            PreprocessingStepResult with transformed data and metadata.
        """
        result = self._transform(X, is_test=True)

        # Check if step has added columns
        added_columns, added_modality = self._get_added_columns()

        return PreprocessingStepResult(
            X=result,
            metadata=self.metadata_after_transform_,
            added_columns=added_columns,
            added_modality=added_modality,
        )


@dataclasses.dataclass
class PreprocessingPipelineResult:
    """Result from the preprocessing pipeline.

    Attributes:
        X: The transformed array.
        metadata: Updated column metadata (may have new columns added).
    """

    X: np.ndarray | torch.Tensor
    metadata: ColumnMetadata

    # TODO: Remove once classifier updated.
    @property
    def feature_modalities(self) -> dict[FeatureModality, list[int]]:
        """Get feature modalities as a dictionary (for backward compatibility)."""
        return self.metadata.to_dict()


# Type alias for step registration
StepWithModalities = tuple[PreprocessingStep, set[FeatureModality]]


class PreprocessingPipeline:
    """Modality-aware preprocessing pipeline that handles column slicing.

    This pipeline applies a sequence of preprocessing steps to an array,
    where each step can be registered to target specific feature modalities.
    The pipeline handles:
    - Slicing columns based on registered modalities
    - Passing only relevant columns to each step
    - Reassembling data after each step
    - Tracking metadata updates (added columns, modality changes)

    Steps can be registered in two ways:
    1. As (step, modalities) tuples: step receives only the specified columns
    2. As bare steps: step receives all columns (for backward compatibility)
    """

    def __init__(
        self,
        steps: list[PreprocessingStep | StepWithModalities],
    ) -> None:
        """Initialize the pipeline with preprocessing steps.

        Args:
            steps: List of preprocessing steps. Each can be:
                - A PreprocessingStep (receives all columns)
                - A tuple of (PreprocessingStep, set[FeatureModality]) where the
                  step receives only columns matching the specified modalities.
        """
        super().__init__()
        self._raw_steps = steps
        self.steps = self._normalize_steps(steps)
        self.metadata_: ColumnMetadata | None = None

    def _normalize_steps(
        self,
        steps: list[PreprocessingStep | StepWithModalities],
    ) -> list[StepWithModalities]:
        """Convert steps to normalized (step, modalities) format.

        Bare steps are registered for all modalities (None means all).
        """
        normalized: list[StepWithModalities] = []
        for step in steps:
            if isinstance(step, tuple):
                if len(step) != 2:
                    raise ValueError(
                        f"Step tuple must be (step, modalities), got {step}"
                    )
                normalized.append(step)
            else:
                # Bare step - use empty set to indicate "all columns"
                normalized.append((step, set()))
        return normalized

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)

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

        for step, modalities in self.steps:
            # Determine which columns this step should process
            if modalities:
                # Step registered for specific modalities
                indices = metadata.indices_for_modalities(modalities)
                if not indices:
                    # No columns match the modalities, skip this step
                    continue

                # Slice columns for this step
                X_slice = X[:, indices]
                metadata_slice = metadata.slice_for_indices(indices)

                # Fit and transform on the slice
                result = step.fit_transform(X_slice, metadata_slice)

                # Validate that the step returned the expected number of columns
                # Steps registered with modalities must return same column count
                # (use added_columns for new columns, or register as bare step)
                if result.X.shape[1] != len(indices):
                    step_name = step.__class__.__name__
                    raise ValueError(
                        f"Step {step_name} registered with modalities {modalities} "
                        f"received {len(indices)} columns but returned "
                        f"{result.X.shape[1]} columns. Steps registered with "
                        f"modalities must return the same number of columns. "
                        f"Use 'added_columns' for new columns, or register as a "
                        f"bare step (no modalities) if the step changes column count."
                    )

                # Reassemble: update X with transformed columns
                if isinstance(X, np.ndarray):
                    X = X.copy()
                X[:, indices] = result.X

                # Handle added columns
                if result.added_columns is not None:
                    X = np.concatenate([X, result.added_columns], axis=1)
                    metadata = metadata.add_columns(
                        result.added_modality or FeatureModality.NUMERICAL,
                        result.added_columns.shape[1],
                    )

                # Update metadata with any modality changes from the step
                metadata = metadata.update_from_step_result(indices, result.metadata)
            else:
                # Bare step - receives all columns
                result = step.fit_transform(X, metadata)
                X = result.X
                metadata = result.metadata

                # Handle added columns
                if result.added_columns is not None:
                    X = np.concatenate([X, result.added_columns], axis=1)
                    metadata = metadata.add_columns(
                        result.added_modality or FeatureModality.NUMERICAL,
                        result.added_columns.shape[1],
                    )

        self.metadata_ = metadata
        return PreprocessingPipelineResult(X=X, metadata=metadata)

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        metadata: ColumnMetadata | dict[FeatureModality, list[int]],
    ) -> Self:
        """Fit all the steps in the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
            metadata: Column metadata or dictionary of feature modalities.
        """
        assert len(self) > 0, "The pipeline must have at least one step."
        self.fit_transform(X, metadata)
        return self

    def transform(self, X: np.ndarray | torch.Tensor) -> PreprocessingPipelineResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).

        Returns:
            PreprocessingPipelineResult with transformed data and metadata.
        """
        assert len(self.steps) > 0, "The pipeline must have at least one step."
        assert self.metadata_ is not None, (
            "The pipeline must be fit before it can be used to transform."
        )

        # We need to track metadata during transform as well
        # Start fresh and rebuild through steps
        X_out: np.ndarray | torch.Tensor = X
        metadata: ColumnMetadata | None = None
        for step, modalities in self.steps:
            if modalities:
                if metadata is None:
                    # First step with modalities - we don't know the initial metadata
                    # This is a limitation; for now, use all columns
                    result = step.transform(X_out)  # type: ignore[arg-type]
                    X_out = result.X
                    metadata = result.metadata
                else:
                    indices = metadata.indices_for_modalities(modalities)
                    if not indices:
                        continue

                    X_slice = X_out[:, indices]
                    result = step.transform(X_slice)  # type: ignore[arg-type]

                    # Validate column count (same check as in fit_transform)
                    if result.X.shape[1] != len(indices):
                        step_name = step.__class__.__name__
                        raise ValueError(
                            f"Step {step_name} registered with modalities "
                            f"{modalities} received {len(indices)} columns but "
                            f"returned {result.X.shape[1]} columns."
                        )

                    if isinstance(X_out, np.ndarray):
                        X_out = X_out.copy()
                    X_out[:, indices] = result.X  # type: ignore[index]

                    if result.added_columns is not None:
                        X_out = np.concatenate([X_out, result.added_columns], axis=1)
                        metadata = metadata.add_columns(
                            result.added_modality or FeatureModality.NUMERICAL,
                            result.added_columns.shape[1],
                        )

                    metadata = metadata.update_from_step_result(
                        indices, result.metadata
                    )
            else:
                result = step.transform(X_out)  # type: ignore[arg-type]
                X_out = result.X
                metadata = result.metadata

                if result.added_columns is not None:
                    X_out = np.concatenate([X_out, result.added_columns], axis=1)
                    metadata = metadata.add_columns(
                        result.added_modality or FeatureModality.NUMERICAL,
                        result.added_columns.shape[1],
                    )

        # Use stored metadata as it should match
        return PreprocessingPipelineResult(X=X_out, metadata=self.metadata_)
