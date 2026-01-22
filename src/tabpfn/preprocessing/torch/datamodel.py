#  Copyright (c) Prior Labs GmbH 2025.

"""Data models for the torch preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class FeatureModality(str, Enum):
    """The modality of a feature.

    Here we move between the way the data is stored, and what it actually
    represents. For instance, a numerical dtype could represent numerical
    and categorical features, while a string could represent categorical
    or text features.
    """

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    CONSTANT = "constant"


@dataclass
class ColumnMetadata:
    """Maps feature modalities to column indices in the tensor."""

    indices_by_modality: dict[FeatureModality, list[int]] = field(default_factory=dict)

    @property
    def num_columns(self) -> int:
        """Get the total number of columns."""
        return sum(len(indices) for indices in self.indices_by_modality.values())

    def indices_for(self, modality: FeatureModality) -> list[int]:
        """Get column indices for a single modality."""
        return self.indices_by_modality.get(modality, [])

    def indices_for_modalities(self, modalities: set[FeatureModality]) -> list[int]:
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


@dataclass
class TorchPreprocessingStepResult:
    """Result from a preprocessing step's transform.

    Attributes:
        x: Full tensor with columns modified in-place.
        added_columns: Optional new columns to append (e.g., NaN indicators).
        added_modality: Modality for the added columns.
    """

    x: torch.Tensor
    added_columns: torch.Tensor | None = None
    added_modality: FeatureModality | None = None


@dataclass
class TorchPreprocessingPipelineOutput:
    """Output from the preprocessing pipeline.

    Attributes:
        x: The transformed tensor.
        metadata: Updated column metadata (may have new columns added).
    """

    x: torch.Tensor
    metadata: ColumnMetadata
