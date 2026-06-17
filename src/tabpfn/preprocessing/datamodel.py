#  Copyright (c) Prior Labs GmbH 2026.

"""Data model for the preprocessing pipeline."""

from __future__ import annotations

import dataclasses
from copy import copy
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


#: Prefix applied to feature names derived from the input data (DataFrame columns
#: or positional ``f{i}`` names for plain arrays). Namespacing input-derived names
#: keeps them from colliding with names generated for features that preprocessing
#: steps add (e.g. ``svd_0``, ``fingerprint``), so the full schema can stay unique
#: by construction without runtime checks.
INPUT_FEATURE_PREFIX = "input_"


def make_names_unique(
    names: Iterable[str],
    *,
    existing: Iterable[str] = (),
) -> list[str]:
    """Return ``names`` made unique w.r.t. each other and ``existing``.

    Collisions are resolved deterministically by appending ``_1``, ``_2``, ...
    until the name is free. This is the single mechanism used to guarantee
    feature-name uniqueness by construction (see module-level naming docs).

    Args:
        names: The candidate names, in order.
        existing: Names already in use that the result must not collide with.

    Returns:
        A list the same length as ``names`` with all entries distinct and
        distinct from ``existing``.
    """
    seen: set[str] = set(existing)
    out: list[str] = []
    for name in names:
        candidate = name
        suffix = 1
        while candidate in seen:
            candidate = f"{name}_{suffix}"
            suffix += 1
        seen.add(candidate)
        out.append(candidate)
    return out


def default_input_feature_names(num_features: int) -> list[str]:
    """Build default input feature names for unnamed features.

    Args:
        num_features (int): Number of names to generate.

    Returns:
        ["f0", "f1", ... ,"f{num_features-1}"]
    """
    return [f"f{i}" for i in range(num_features)]


def build_input_feature_names(
    feature_names: list[str] | None,
    num_features: int,
) -> list[str]:
    """Build unique names for the original input features.

    Args:
        feature_names: Names of the input columns (e.g. from a DataFrame), or
            ``None`` when the input was a plain array with no column names.
        num_features: Number of input features.

    Returns:
        For DataFrame input, each column name prefixed with
        :data:`INPUT_FEATURE_PREFIX` and de-duplicated (pandas allows duplicate
        column names). For array input, positional ``f0``, ``f1``, ... names.
    """
    if feature_names is None:
        return default_input_feature_names(num_features)
    prefixed = [f"{INPUT_FEATURE_PREFIX}{name}" for name in feature_names]
    return make_names_unique(prefixed)


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


class GPUTransformType(str, Enum):
    """GPU transform types that a feature column can be marked for.

    Used to flag columns during CPU preprocessing so the GPU pipeline
    knows which transforms to apply.
    """

    QUANTILE = "quantile"
    SQUASHING_SCALER = "squashing_scaler"


@dataclasses.dataclass
class Feature:
    """A single feature with its name and modality.

    Warning: features are computed/updated at `fit()`-time only.
    Existing/transformed columns carry over their features, using
    `dataclasses.replace()` if needed instead of implicitly copying with
    `Feature()`.

    Attributes:
        name: The name of the feature. Should be unique inside any given FeatureSchema.
        modality: The modality (type) of the feature.
        scheduled_gpu_transform: When set, indicates that this column still
            needs the specified GPU transform.  Set by CPU preprocessing
            steps (e.g. :class:`ReshapeFeatureDistributionsStep`) and
            cleared by the GPU pipeline after the transform has been applied.
        ancestor: Name of the feature that this feature is derived from, if applicable.
    """

    name: str
    modality: FeatureModality
    scheduled_gpu_transform: GPUTransformType | None = None
    ancestor: str | None = None


@dataclasses.dataclass
class FeatureSchema:
    """Metadata about the features in the dataset.

    Uses a single list of Feature objects to track the features in the dataset, where
    position in the list corresponds to column index. Provides utilities
    for tracking which columns represent which modality, and for updating
    this mapping as preprocessing steps transform the data.

    Feature-name uniqueness is an invariant of any schema, guaranteed by
    construction (names come from the input columns or from a transform plus an
    index; see ``build_input_feature_names`` and ``append_columns``). Prefer the
    methods that return new instances (``append_columns``, ``remove_columns``,
    ``apply_permutation``, ...) over mutating ``features`` in place, so the
    invariant stays easy to verify at construction time in tests.

    Attributes:
        features: List of Feature objects where index = column position.
    """

    features: list[Feature] = dataclasses.field(default_factory=list)

    @classmethod
    def from_only_categorical_indices(
        cls,
        categorical_indices: list[int],
        num_columns: int,
        names: list[str] | None = None,
    ) -> FeatureSchema:
        """Create FeatureSchema from only categorical indices.

        This is used for backwards compatibility with the old preprocessing pipeline
        that only tracked categorical indices. All columns that are not categorical
        are assumed to be numerical.

        Args:
            categorical_indices: Output column indices that are categorical.
            num_columns: Total number of output columns.
            names: Names for the output columns (in output order). If ``None``,
                positional ``f{i}`` names are generated. Callers that reorder
                columns are responsible for passing names already in output order.
        """
        numerical_indices = [
            i for i in range(num_columns) if i not in categorical_indices
        ]
        if not numerical_indices and not categorical_indices:
            return cls(features=[])

        chosen_names = names or default_input_feature_names(num_columns)
        if len(chosen_names) != num_columns:
            raise ValueError(f"Expected {num_columns} names, got {len(chosen_names)}")

        features: list[Feature | None] = [None] * num_columns
        for idx in categorical_indices:
            features[idx] = Feature(
                name=chosen_names[idx], modality=FeatureModality.CATEGORICAL
            )
        for idx in numerical_indices:
            features[idx] = Feature(
                name=chosen_names[idx], modality=FeatureModality.NUMERICAL
            )

        return cls(features=features)  # type: ignore[arg-type]

    @property
    def feature_names(self) -> list[str | None]:
        """Get list of feature names (derived from features list)."""
        return [f.name for f in self.features]

    @property
    def feature_name_set(self) -> set[str]:
        """Set of non-``None`` feature names currently in use."""
        return {f.name for f in self.features if f.name is not None}

    @property
    def num_columns(self) -> int:
        """Get the total number of columns."""
        return len(self.features)

    def indices_for(self, modality: FeatureModality) -> list[int]:
        """Get column indices for a single modality."""
        return [i for i, f in enumerate(self.features) if f.modality == modality]

    def get_indices_marked_for_gpu_quantile_transform(self) -> list[int]:
        """Get column indices marked for GPU quantile transform."""
        return [
            i
            for i, f in enumerate(self.features)
            if f.scheduled_gpu_transform == GPUTransformType.QUANTILE
        ]

    def get_indices_marked_for_gpu_squashing_scaler_transform(self) -> list[int]:
        """Get column indices marked for GPU squashing scaler transform."""
        return [
            i
            for i, f in enumerate(self.features)
            if f.scheduled_gpu_transform == GPUTransformType.SQUASHING_SCALER
        ]

    def clear_gpu_transform_marks(self) -> FeatureSchema:
        """Return a new schema with all GPU transform marks cleared.

        Called by the GPU pipeline after transforms have been applied.
        """
        if not any(f.scheduled_gpu_transform for f in self.features):
            return self
        return FeatureSchema(
            features=[
                dataclasses.replace(f, scheduled_gpu_transform=None)
                if f.scheduled_gpu_transform
                else f
                for f in self.features
            ],
        )

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
        *,
        name_prefix: str | None = None,
    ) -> FeatureSchema:
        """Return new schema with additional columns appended.

        Appended names are made unique against the names already in this schema,
        so features added by preprocessing steps never collide with input
        features or with each other.

        Args:
            modality: The modality for the new columns.
            num_new: Number of new columns to add.
            names: Explicit names for the new columns. If ``None``, names are
                generated as ``"{name_prefix}_{i}"``.
            name_prefix: Prefix used to generate names when ``names`` is ``None``.
                Defaults to ``"added"``. Should identify the producing transform
                (e.g. ``"svd"``, ``"fingerprint"``) so generated names are
                self-describing.

        Returns:
            New FeatureSchema instance with added features.
        """
        if names is None:
            prefix = name_prefix or "added"
            names = [f"{prefix}_{i}" for i in range(num_new)]
        if len(names) != num_new:
            raise ValueError(f"Expected {num_new} names, got {len(names)}")

        unique_names = make_names_unique(names, existing=self.feature_name_set)
        new_features = self.features + [
            Feature(name=name, modality=modality) for name in unique_names
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
            new_features[original_idx] = copy(step_feature)
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
