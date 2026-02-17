"""Data model for the preprocessing pipeline."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class FeatureModality(str, Enum):
    """The modality of a feature.

    This denotes what the column actually represents, not how it is stored. For
    instance, a numerical dtype could represent numerical features
    or categorical features, while a string could represent categorical
    or text features.
    """

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    CONSTANT = "constant"


@dataclasses.dataclass
class Feature:
    """A single feature with its name and modality.

    Attributes:
        name: The name of the feature.
        modality: The modality (type) of the feature.
    """

    name: str | None
    modality: FeatureModality


@dataclasses.dataclass
class FeatureSchema:
    """Metadata about the features in the dataset.

    Uses a single list of Feature objects to track the features in the dataset, where
    position in the list corresponds to column index. Provides utilities
    for tracking which columns represent which modality, and for updating
    this mapping as preprocessing steps transform the data.

    Attributes:
        features: List of Feature objects where index = column position.
    """

    features: list[Feature] = dataclasses.field(default_factory=list)

    @classmethod
    def from_only_categorical_indices(
        cls,
        categorical_indices: list[int],
        num_columns: int,
    ) -> FeatureSchema:
        """Create FeatureSchema from only categorical indices.

        This is used for backwards compatibility with the old preprocessing pipeline
        that only tracked categorical indices. All columns that are not categorical
        are assumed to be numerical.
        """
        numerical_indices = [
            i for i in range(num_columns) if i not in categorical_indices
        ]
        if not numerical_indices and not categorical_indices:
            return cls(features=[])

        features: list[Feature | None] = [None] * num_columns
        for idx in categorical_indices:
            features[idx] = Feature(name=None, modality=FeatureModality.CATEGORICAL)
        for idx in numerical_indices:
            features[idx] = Feature(name=None, modality=FeatureModality.NUMERICAL)

        return cls(features=features)  # type: ignore[arg-type]

    @property
    def feature_names(self) -> list[str | None]:
        """Get list of feature names (derived from features list)."""
        return [f.name for f in self.features]

    @property
    def num_columns(self) -> int:
        """Get the total number of columns."""
        return len(self.features)

    def indices_for(self, modality: FeatureModality) -> list[int]:
        """Get column indices for a single modality."""
        return [i for i, f in enumerate(self.features) if f.modality == modality]

    def indices_for_modalities(
        self, modalities: Iterable[FeatureModality]
    ) -> list[int]:
        """Get combined column indices for multiple modalities (sorted)."""
        modality_set = set(modalities)
        return sorted(
            i for i, f in enumerate(self.features) if f.modality in modality_set
        )

    def append_columns(
        self,
        modality: FeatureModality,
        num_new: int,
        names: list[str] | None = None,
    ) -> FeatureSchema:
        """Return new schema with additional columns appended.

        Args:
            modality: The modality for the new columns.
            num_new: Number of new columns to add.
            names: Names for the new columns. If None, uses "added_0", "added_1", ...

        Returns:
            New FeatureSchema instance with added features.
        """
        chosen_names = [None] * num_new if names is None else names
        if len(chosen_names) != num_new:
            raise ValueError(f"Expected {num_new} names, got {len(chosen_names)}")

        new_features = self.features + [
            Feature(name=name, modality=modality) for name in chosen_names
        ]
        return FeatureSchema(features=new_features)

    def slice_for_indices(self, indices: list[int]) -> FeatureSchema:
        """Create schema for a subset of columns, remapping to 0-based indices.

        When slicing columns from an array, this method creates new schema
        where the selected columns are remapped to positions 0, 1, 2, etc.

        Args:
            indices: The column indices being selected (in original indexing).

        Returns:
            New FeatureSchema with features at the selected indices.
        """
        return FeatureSchema(features=[self.features[i] for i in indices])

    def update_from_preprocessing_step_result(
        self,
        original_indices: list[int],
        new_schema: FeatureSchema,
    ) -> FeatureSchema:
        """Update schema after a step has transformed selected columns.

        This method merges the step's output schema back into the full schema.
        The new_schema contains features for the columns it processed (0-based),
        which are mapped back to the original column positions.

        Args:
            original_indices: The column indices that were passed to the step.
            new_schema: The schema returned by the step (0-based indices).

        Returns:
            New FeatureSchema with updated modalities for the processed columns.
        """
        # Copy features and update the processed ones
        new_features = list(self.features)
        for step_idx, original_idx in enumerate(original_indices):
            step_feature = new_schema.features[step_idx]
            new_features[original_idx] = Feature(
                name=step_feature.name,
                modality=step_feature.modality,
            )
        return FeatureSchema(features=new_features)

    def remove_columns(self, indices_to_remove: list[int]) -> FeatureSchema:
        """Return new schema with specified columns removed."""
        remove_set = set(indices_to_remove)
        return FeatureSchema(
            features=[f for i, f in enumerate(self.features) if i not in remove_set]
        )

    def apply_permutation(self, permutation: list[int]) -> FeatureSchema:
        """Apply a column permutation to the schema."""
        _validate_permutation(permutation)
        return FeatureSchema(features=[self.features[i] for i in permutation])


def _validate_permutation(permutation: list[int]) -> None:
    """Ensure a permutation is valid."""
    if len(permutation) != len(set(permutation)):
        raise ValueError("Permutation is not valid: contains duplicates.")
    if any(i < 0 or i >= len(permutation) for i in permutation):
        raise ValueError("Permutation is not valid: contains indices out of range.")
