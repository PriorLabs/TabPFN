"""Feature Preprocessing Transformer Step."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    check_is_fitted,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tabpfn.constants import DEFAULT_NUMPY_PREPROCESSING_DTYPE
from tabpfn.preprocessing.datamodel import FeatureModality

if TYPE_CHECKING:
    from tabpfn.classifier import XType, YType


def apply_permutation_to_modalities(
    feature_modalities: dict[FeatureModality, list[int]],
    permutation: list[int] | np.ndarray,
) -> dict[FeatureModality, list[int]]:
    """Apply an index permutation to all feature modalities.

    When features are shuffled/permuted, we need to update all modality indices
    to reflect the new positions. This function maps old indices to new indices
    for all modalities in the dictionary.

    Args:
        feature_modalities: Dictionary mapping modality to list of column indices.
        permutation: The permutation applied to features, where
            permutation[new_idx] = old_idx.

    Returns:
        New dictionary with updated indices for all modalities.
    """
    # Create reverse mapping: old_idx -> new_idx
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(permutation)}

    return {
        modality: sorted(old_to_new[old_idx] for old_idx in indices)
        for modality, indices in feature_modalities.items()
    }


def filter_modalities_by_kept_indices(
    feature_modalities: dict[FeatureModality, list[int]],
    kept_indices: list[int] | np.ndarray,
) -> dict[FeatureModality, list[int]]:
    """Filter feature modalities to only include kept indices, remapped to new
    positions.

    When features are removed (e.g., constant features), we need to:
    1. Remove the deleted indices from the respective modalities
    2. Remap remaining indices to their new positions

    Args:
        feature_modalities: Dictionary mapping modality to list of column indices.
        kept_indices: The indices of features that are kept (in original indexing).

    Returns:
        New dictionary with updated indices for all modalities.
    """
    kept_set = set(kept_indices)
    # Create mapping: old_idx -> new_idx for kept features
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}

    return {
        modality: sorted(
            old_to_new[old_idx] for old_idx in indices if old_idx in kept_set
        )
        for modality, indices in feature_modalities.items()
    }


def get_categorical_indices(
    feature_modalities: dict[FeatureModality, list[int]],
) -> list[int]:
    """Extract categorical indices from feature modalities dictionary.

    Args:
        feature_modalities: Dictionary mapping modality to list of column indices.

    Returns:
        List of categorical feature indices.
    """
    return feature_modalities.get(FeatureModality.CATEGORICAL, [])


def update_categorical_indices(  # noqa: C901
    feature_modalities: dict[FeatureModality, list[int]],
    new_categorical_indices: list[int],
    n_features: int,
) -> dict[FeatureModality, list[int]]:
    """Update modality indices when categorical columns are moved to new positions.

    This handles column reordering (e.g., from ColumnTransformer) by:
    1. Building a mapping from old positions to new positions based on:
       - Categorical columns maintaining their relative order
       - Non-categorical columns maintaining their relative order
    2. Applying this mapping to all modalities
    3. Assigning any new indices (from expanded features) to NUMERICAL

    Args:
        feature_modalities: Dictionary mapping modality to list of column indices.
        new_categorical_indices: The new positions of categorical features.
        n_features: Total number of features after the transformation.

    Returns:
        New dictionary with updated modality indices for all modalities.
    """
    old_cat_indices = sorted(feature_modalities.get(FeatureModality.CATEGORICAL, []))
    new_cat_indices = sorted(new_categorical_indices)

    # Determine all old indices from the modalities
    all_old_indices: set[int] = set()
    for indices in feature_modalities.values():
        all_old_indices.update(indices)

    # Get old non-categorical indices in sorted order
    old_non_cat = sorted(all_old_indices - set(old_cat_indices))

    # The new non-categorical positions are those not occupied by new categoricals
    new_non_cat_positions = sorted(set(range(n_features)) - set(new_cat_indices))

    # Build old_to_new mapping
    old_to_new: dict[int, int] = {}

    # Map old categorical indices to new categorical positions
    # (preserving relative order)
    for i, old_idx in enumerate(old_cat_indices):
        if i < len(new_cat_indices):
            old_to_new[old_idx] = new_cat_indices[i]

    # Map old non-categorical indices to new non-categorical positions
    # (preserving relative order)
    for i, old_idx in enumerate(old_non_cat):
        if i < len(new_non_cat_positions):
            old_to_new[old_idx] = new_non_cat_positions[i]

    # Apply mapping to all modalities
    result: dict[FeatureModality, list[int]] = {}
    for modality, indices in feature_modalities.items():
        if modality == FeatureModality.CATEGORICAL:
            # Categorical indices are explicitly provided
            result[modality] = sorted(new_categorical_indices)
        else:
            # Remap other modalities using the old_to_new mapping
            result[modality] = sorted(
                old_to_new[idx] for idx in indices if idx in old_to_new
            )

    # Ensure CATEGORICAL key exists
    if FeatureModality.CATEGORICAL not in result:
        result[FeatureModality.CATEGORICAL] = sorted(new_categorical_indices)

    # Handle new indices (not mapped from old) - assign to NUMERICAL
    all_mapped_new = set(old_to_new.values()) | set(new_cat_indices)
    orphan_indices = set(range(n_features)) - all_mapped_new

    # Make categoricals that are not categoricals anymore numerical
    if FeatureModality.NUMERICAL not in result:
        result[FeatureModality.NUMERICAL] = sorted(orphan_indices)
    elif orphan_indices:
        result[FeatureModality.NUMERICAL] = sorted(
            set(result[FeatureModality.NUMERICAL]) | orphan_indices
        )

    return result


def append_numerical_features(
    feature_modalities: dict[FeatureModality, list[int]],
    current_n_features: int,
    n_new_features: int,
) -> dict[FeatureModality, list[int]]:
    """Append new numerical feature indices to the feature modalities.

    Used when new numerical columns are added at the end (e.g., fingerprint,
    polynomial features).

    Args:
        feature_modalities: Dictionary mapping modality to list of column indices.
        current_n_features: Number of features before adding new ones.
        n_new_features: Number of new numerical features to add.

    Returns:
        New dictionary with new numerical indices appended.
    """
    result = dict(feature_modalities)
    current_numerical = result.get(FeatureModality.NUMERICAL, [])
    new_indices = list(range(current_n_features, current_n_features + n_new_features))
    result[FeatureModality.NUMERICAL] = current_numerical + new_indices
    return result


class OrderPreservingColumnTransformer(ColumnTransformer):
    """An ColumnTransformer that preserves the column order after transformation."""

    def __init__(
        self,
        transformers: Sequence[
            tuple[
                str,
                BaseEstimator,
                str
                | int
                | slice
                | Iterable[str | int]
                | Callable[[Any], Iterable[str | int]],
            ]
        ],
        **kwargs: Any,
    ):
        """Implementation base on https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html.

        Parameters
        ----------
        transformers : sequence of (name, transformer, columns) tuples
            List of (name, transformer, columns) tuples specifying the transformers.
        **kwargs : additional keyword arguments
            Passed to sklearn.compose.ColumnTransformer.
        """
        super().__init__(transformers=transformers, **kwargs)

        # Check if there is a single transformer, of subtype OneToOneFeatureMixin
        assert all(
            isinstance(t, OneToOneFeatureMixin)
            for name, t, _ in transformers
            if name != "remainder"
        ), (
            "OrderPreservingColumnTransformer currently only supports transformers "
            "that are instances of OneToOneFeatureMixin."
        )

        assert len([t for name, _, t in transformers if name != "remainder"]) <= 1, (
            "OrderPreservingColumnTransformer only supports up to one transformer."
        )

    @override
    def transform(self, X: XType, **kwargs: dict[str, Any]) -> XType:
        original_columns = (
            X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
        )
        X_t = super().transform(X, **kwargs)
        return self._preserve_order(X=X_t, original_columns=original_columns)

    @override
    def fit_transform(
        self, X: XType, y: YType = None, **kwargs: dict[str, Any]
    ) -> XType:
        original_columns = (
            X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
        )
        X_t = super().fit_transform(X, y, **kwargs)
        return self._preserve_order(X=X_t, original_columns=original_columns)

    def _preserve_order(
        self, X: XType, original_columns: list | range | pd.Index
    ) -> XType:
        check_is_fitted(self)
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D (shape={X.shape})"
        for name, _, col_subset in reversed(self.transformers_):
            if (
                len(col_subset) > 0
                and len(col_subset) < X.shape[-1]
                and name != "remainder"
            ):
                col_subset_list = list(col_subset)
                # Map original columns to indices in the transformed array
                transformed_columns = col_subset_list + [
                    c for c in original_columns if c not in col_subset_list
                ]
                indices = [transformed_columns.index(c) for c in original_columns]
                # restore the column order from before the transfomer has been applied
                X = X.iloc[:, indices] if isinstance(X, pd.DataFrame) else X[:, indices]
        return X


def get_ordinal_encoder(
    *,
    numpy_dtype: np.floating = DEFAULT_NUMPY_PREPROCESSING_DTYPE,  # type: ignore
) -> ColumnTransformer:
    """Create a ColumnTransformer that ordinally encodes string/category columns."""
    oe = OrdinalEncoder(
        # TODO: Could utilize the categorical dtype values directly instead of "auto"
        categories="auto",
        dtype=numpy_dtype,  # type: ignore
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,  # Missing stays missing
    )

    # Documentation of sklearn, deferring to pandas is misleading here. It's done
    # using a regex on the type of the column, and using `object`, `"object"` and
    # `np.object` will not pick up strings.
    to_convert = ["category", "string"]

    # When using a ColumnTransformer with inner transformers applied to only a subset of
    # columns, the original column order of the data is not preserved. Because we do not
    # update the categorical indices after encoding, these indices may no longer align
    # with the true categorical columns.

    # Subsequent components rely on these categorical indices. For instance:
    # - QuantileTransformer should only be applied to numerical features.
    # - EncodeCategoricalFeaturesStep should be applied to all categorical features.

    # Despite the column shuffling introduced by the vanilla ColumnTransformer, we
    # observed better overall performance when using it. Therefore, we keep it.

    return ColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
