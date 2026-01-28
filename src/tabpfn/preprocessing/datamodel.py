"""Data model for the preprocessing pipeline."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

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


@dataclasses.dataclass(frozen=True)
class DatasetView:
    """A view of a dataset split by feature modalities."""

    X: pd.DataFrame
    columns_by_modality: dict[FeatureModality, list[int]]

    @property
    def feature_names(self) -> list[str]:
        """Returns the feature names as a list of strings."""
        return self.X.columns.tolist()

    @property
    def x_num(self) -> pd.DataFrame:
        """Returns the numerical features as a pd.DataFrame."""
        return self._get_modality(FeatureModality.NUMERICAL)

    @property
    def x_cat(self) -> pd.DataFrame:
        """Returns the categorical features as a pd.DataFrame."""
        return self._get_modality(FeatureModality.CATEGORICAL)

    @property
    def x_num_and_cat(self) -> pd.DataFrame:
        """Returns the numerical and categorical features as a pd.DataFrame."""
        return pd.concat([self.x_num, self.x_cat], axis=1)

    @property
    def x_txt(self) -> pd.DataFrame:
        """Returns the text features as a pd.DataFrame."""
        return self._get_modality(FeatureModality.TEXT)

    def _get_modality(self, modality: FeatureModality) -> pd.DataFrame:
        return self.X.iloc[:, self.columns_by_modality[modality]]


@dataclasses.dataclass
class ColumnMetadata:
    """Maps feature modalities to column indices in the tensor/array.

    This class provides utilities for tracking which columns belong to which
    modality, and for updating this mapping as preprocessing steps transform
    the data.
    """

    indices_by_modality: dict[FeatureModality, list[int]] = dataclasses.field(
        default_factory=dict
    )

    # TODO: Remove because this was only for backwards compatibility
    @classmethod
    def from_dict(
        cls, feature_modalities: dict[FeatureModality, list[int]]
    ) -> ColumnMetadata:
        """Create ColumnMetadata from a feature_modalities dictionary.

        Args:
            feature_modalities: Dictionary mapping modality to list of column indices.

        Returns:
            New ColumnMetadata instance.
        """
        return cls(
            indices_by_modality={
                mod: list(indices) for mod, indices in feature_modalities.items()
            }
        )

    # TODO: Remove because this was only for backwards compatibility
    def to_dict(self) -> dict[FeatureModality, list[int]]:
        """Convert to a feature_modalities dictionary.

        Returns:
            Dictionary mapping modality to list of column indices.
        """
        return {mod: list(indices) for mod, indices in self.indices_by_modality.items()}

    @property
    def num_columns(self) -> int:
        """Get the total number of columns."""
        return sum(len(indices) for indices in self.indices_by_modality.values())

    def indices_for(self, modality: FeatureModality) -> list[int]:
        """Get column indices for a single modality."""
        return self.indices_by_modality.get(modality, [])

    def indices_for_modalities(
        self, modalities: Iterable[FeatureModality]
    ) -> list[int]:
        """Get combined column indices for multiple modalities (sorted)."""
        indices: list[int] = []
        for modality in modalities:
            indices.extend(self.indices_by_modality.get(modality, []))
        return sorted(set(indices))

    def add_columns(self, modality: FeatureModality, num_new: int) -> ColumnMetadata:
        """Return new metadata with additional columns appended.

        Args:
            modality: The modality for the new columns.
            num_new: Number of new columns to add.

        Returns:
            New ColumnMetadata instance with updated indices.
        """
        new_indices_by_modality = {
            mod: list(indices) for mod, indices in self.indices_by_modality.items()
        }

        new_column_indices = list(range(self.num_columns, self.num_columns + num_new))
        if modality in new_indices_by_modality:
            new_indices_by_modality[modality].extend(new_column_indices)
        else:
            new_indices_by_modality[modality] = new_column_indices

        return ColumnMetadata(
            indices_by_modality=new_indices_by_modality,
        )

    def slice_for_indices(self, indices: list[int]) -> ColumnMetadata:
        """Create metadata for a subset of columns, remapping to 0-based indices.

        When slicing columns from an array, this method creates new metadata
        where the selected columns are remapped to positions 0, 1, 2, etc.

        Args:
            indices: The column indices being selected (in original indexing).

        Returns:
            New ColumnMetadata with remapped indices for the selected columns.
        """
        indices_set = set(indices)
        # Create mapping: old_idx -> new_idx
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        new_indices_by_modality: dict[FeatureModality, list[int]] = {}
        for modality, mod_indices in self.indices_by_modality.items():
            remapped = [old_to_new[idx] for idx in mod_indices if idx in indices_set]
            if remapped:
                new_indices_by_modality[modality] = sorted(remapped)

        return ColumnMetadata(indices_by_modality=new_indices_by_modality)

    def update_from_step_result(
        self,
        original_indices: list[int],
        step_metadata: ColumnMetadata,
    ) -> ColumnMetadata:
        """Update metadata after a step has transformed selected columns.

        This method merges the step's output metadata back into the full metadata.
        The step_metadata contains 0-based indices for the columns it processed,
        which are mapped back to the original column positions.

        Args:
            original_indices: The column indices that were passed to the step.
            step_metadata: The metadata returned by the step (0-based indices).

        Returns:
            New ColumnMetadata with updated modalities for the processed columns.
        """
        # Create mapping: new_idx (0-based in step) -> old_idx (original position)
        new_to_old = dict(enumerate(original_indices))

        # Start with modalities for columns NOT processed by this step
        processed_set = set(original_indices)
        new_indices_by_modality: dict[FeatureModality, list[int]] = {}

        for modality, mod_indices in self.indices_by_modality.items():
            unprocessed = [idx for idx in mod_indices if idx not in processed_set]
            if unprocessed:
                new_indices_by_modality[modality] = unprocessed

        # Add back the processed columns with their new modalities
        for modality, step_indices in step_metadata.indices_by_modality.items():
            original_positions = [new_to_old[idx] for idx in step_indices]
            if modality in new_indices_by_modality:
                new_indices_by_modality[modality].extend(original_positions)
                new_indices_by_modality[modality] = sorted(
                    new_indices_by_modality[modality]
                )
            else:
                new_indices_by_modality[modality] = sorted(original_positions)

        return ColumnMetadata(indices_by_modality=new_indices_by_modality)

    def remove_columns(self, indices_to_remove: list[int]) -> ColumnMetadata:
        """Return new metadata with specified columns removed and indices remapped."""
        remove_set = set(indices_to_remove)
        # Get all current indices sorted
        all_indices = sorted(
            idx for indices in self.indices_by_modality.values() for idx in indices
        )
        # Keep only indices not being removed
        kept_indices = [idx for idx in all_indices if idx not in remove_set]
        # Create mapping: old_idx -> new_idx
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}

        new_indices_by_modality: dict[FeatureModality, list[int]] = {}
        for modality, mod_indices in self.indices_by_modality.items():
            remapped = [old_to_new[idx] for idx in mod_indices if idx not in remove_set]
            if remapped:
                new_indices_by_modality[modality] = sorted(remapped)

        return ColumnMetadata(indices_by_modality=new_indices_by_modality)

    def apply_permutation(self, permutation: list[int]) -> ColumnMetadata:
        """Apply a column permutation to the metadata.

        Args:
            permutation: The permutation where permutation[new_idx] = old_idx.

        Returns:
            New ColumnMetadata with updated indices.
        """
        # Create reverse mapping: old_idx -> new_idx
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(permutation)}

        new_indices_by_modality: dict[FeatureModality, list[int]] = {}
        for modality, mod_indices in self.indices_by_modality.items():
            remapped = sorted(old_to_new[idx] for idx in mod_indices)
            new_indices_by_modality[modality] = remapped

        return ColumnMetadata(indices_by_modality=new_indices_by_modality)
