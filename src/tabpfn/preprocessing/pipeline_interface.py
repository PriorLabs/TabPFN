"""Interfaces for creating preprocessing pipelines."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing_extensions import TypeAlias

import numpy as np
import torch

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema

StepWithModalities: TypeAlias = tuple["PreprocessingStep", set[FeatureModality]]


@dataclasses.dataclass
class PreprocessingStepResult:
    """Result of a feature preprocessing step.

    Attributes:
        X: Transformed array. For steps registered with specific modalities,
            this is only the transformed columns (not the full array).
            The shape should match the input shape unless columns are removed.
        feature_schema: Feature schema for the columns this step processed.
            Contains 0-based indices relative to the step's input.
            Should NOT include added_columns - the pipeline handles that.
        X_added: Optional new features to append (e.g., fingerprint features).
            These are handled by the pipeline, which concatenates them and
            updates the schema accordingly. Steps should NOT concatenate
            these internally.
        modality_added: Modality for the added features. Required if X_added
            is provided.
    """

    X: np.ndarray | torch.Tensor
    feature_schema: FeatureSchema
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
        feature_schema: Updated feature schema (may have new columns added).
    """

    X: np.ndarray | torch.Tensor
    feature_schema: FeatureSchema


class PreprocessingStep:
    """Base class for feature preprocessing steps.

    Steps can be registered with specific feature modalities, and the pipeline
    will handle slicing the data to only pass the relevant columns to the step.

    Subclasses should implement `_fit` and `_transform` methods. The `_fit` method
    receives the sliced data and schema, and should return the schema after
    transformation (for the transformed columns only, NOT including added_columns).

    The `_transform` method receives the sliced data and returns the transformed
    array plus new columns and new modality separately. The pipeline handles
    concatenation.

    Design principle: Steps should NOT internally handle passthrough of columns
    they don't transform. The pipeline handles column slicing and reassembly.
    """

    feature_schema_updated_: FeatureSchema
    """Schema describing this step's *main* transformed output.

    This schema corresponds to the `X` returned by `_transform(...)` (i.e. the
    transformed columns for this step) and **must not** include any features
    produced via the optional `X_added` / `modality_added` return values.

    Expected usage patterns (for backwards compatibility):
    - If the step adds features via `X_added` (e.g. fingerprint features), then
      `_fit(...)` should describe only the main output `X` with potentially permuted
      features. The pipeline will append `X_added` and update the overall
      feature schema itself.
    - If the step changes the column semantics/shape of its main output `X`
      (e.g. encoding, reordering, dropping columns) and does **not** rely on the
      `X_added` pathway, then `_fit(...)` should update this schema to match that
      main output.
    """

    n_added_columns_: int | None
    """Number of added columns from `_transform`. Set during first transform call."""

    modality_added_: FeatureModality | None
    """Modality of added columns from `_transform`. Set during first transform call."""

    @abstractmethod
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        """Underlying method of the preprocessor to implement by subclasses.

        Args:
            X: 2d array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            feature_schema: feature schema for the input columns.

        Returns:
            Feature schema after the transform.
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
        feature_schema: FeatureSchema,
    ) -> PreprocessingStepResult:
        """Fits the preprocessor and transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
            feature_schema: feature schema.

        Returns:
            PreprocessingStepResult with transformed data and updated feature schema.
        """
        self.feature_schema_updated_ = self._fit(X, feature_schema)
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
            PreprocessingStepResult with transformed data and feature schema.
        """
        result, X_added, modality_added = self._transform(X, is_test=is_test)
        self._validate_added_data(X_added=X_added, modality_added=modality_added)

        return PreprocessingStepResult(
            X=result,
            feature_schema=self.feature_schema_updated_,
            X_added=X_added,
            modality_added=modality_added,
        )

    def _validate_added_data(
        self,
        X_added: np.ndarray | None,
        modality_added: FeatureModality | None,
    ) -> None:
        """Validate consistency of added columns across train/test transforms."""
        n_added = X_added.shape[1] if X_added is not None else None

        store_expected_values_on_first_call = not hasattr(self, "n_added_columns_")
        if store_expected_values_on_first_call:
            self.n_added_columns_ = n_added
            self.modality_added_ = modality_added
        else:
            if n_added != self.n_added_columns_:
                raise ValueError(
                    f"Inconsistent number of added columns: expected "
                    f"{self.n_added_columns_}, got {n_added}"
                )
            if modality_added != self.modality_added_:
                raise ValueError(
                    f"Inconsistent modality for added columns: expected "
                    f"{self.modality_added_}, got {modality_added}"
                )


class PreprocessingPipeline:
    """Modality-aware preprocessing pipeline that handles column slicing.

    This pipeline applies a sequence of preprocessing steps to data,
    where each step can be registered to target specific feature modalities.
    The pipeline handles slicing columns based on registered modalities,
    passing only relevant columns to each step, reassembling data after each
    step, and tracking feature schema updates.

    For backwards compatibility, steps can be registered as (step, modalities)
    tuples where the step receives only columns matching the specified modalities,
    or as bare steps that receive all columns. In the latter case, the column modalities
    need to be tracked by the step itself. Going forward, only the former format will be
    supported.
    """

    initial_feature_schema_: FeatureSchema | None = None
    """The feature schema before the pipeline has been fit."""

    final_feature_schema_: FeatureSchema | None = None
    """The feature schema after the pipeline has been fit."""

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

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
    ) -> PreprocessingPipelineResult:
        """Fit and transform the data using the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
            feature_schema: feature schema.

        Returns:
            PreprocessingPipelineResult with transformed data and updated schema.
        """
        self.initial_feature_schema_ = feature_schema
        X, feature_schema = self._process_steps(X, feature_schema, is_fitting=True)
        self.final_feature_schema_ = feature_schema
        return PreprocessingPipelineResult(X=X, feature_schema=feature_schema)

    def transform(self, X: np.ndarray | torch.Tensor) -> PreprocessingPipelineResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).

        Returns:
            PreprocessingPipelineResult with transformed data and feature schema.
        """
        assert self.initial_feature_schema_ is not None, (
            "The pipeline must be fit before it can be used to transform."
        )
        assert self.final_feature_schema_ is not None

        X, _ = self._process_steps(X, self.initial_feature_schema_, is_fitting=False)
        return PreprocessingPipelineResult(
            X=X, feature_schema=self.final_feature_schema_
        )

    def _process_steps(
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
        *,
        is_fitting: bool,
    ) -> tuple[np.ndarray | torch.Tensor, FeatureSchema]:
        """Process all pipeline steps.

        Args:
            X: Input array of shape (n_samples, n_features).
            feature_schema: Feature schema.
            is_fitting: If True, call fit_transform on steps; otherwise transform.

        Returns:
            Tuple of (transformed array, updated feature schema).
        """
        # Single copy to preserve immutability for the caller, avoiding N copies
        # inside the loop for steps that target specific modalities.
        X = X.copy() if isinstance(X, np.ndarray) else X.clone()

        for step, modalities in self.steps:
            if modalities:
                indices = feature_schema.indices_for_modalities(modalities)
                if not indices:
                    continue

                X_slice = X[:, indices]
                result = (
                    step.fit_transform(
                        X_slice, feature_schema.slice_for_indices(indices)
                    )
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

                assert X.dtype == result.X.dtype
                X[:, indices] = result.X

                X, feature_schema = self._maybe_append_added_columns(
                    X, feature_schema, result
                )
                feature_schema = feature_schema.update_from_preprocessing_step_result(
                    indices, result.feature_schema
                )
            else:
                # We still have preprocessing steps that don't change the columns
                # internally (will be deprecated going forward). For backwards
                # compatibility, we still handle these here.
                result = (
                    step.fit_transform(X, feature_schema)
                    if is_fitting
                    else step.transform(X)
                )
                X = result.X
                feature_schema = result.feature_schema
                X, feature_schema = self._maybe_append_added_columns(
                    X, feature_schema, result
                )

        return X, feature_schema

    def _maybe_append_added_columns(
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
        result: PreprocessingStepResult,
    ) -> tuple[np.ndarray | torch.Tensor, FeatureSchema]:
        """Append added columns from a step result and update schema."""
        if result.X_added is not None:
            if isinstance(X, np.ndarray):
                X = np.concatenate([X, result.X_added], axis=1)
            else:
                assert isinstance(result.X_added, torch.Tensor)
                X = torch.cat([X, result.X_added], dim=1)
            feature_schema = feature_schema.append_columns(
                result.modality_added or FeatureModality.NUMERICAL,
                result.X_added.shape[1],
            )
        return X, feature_schema

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
