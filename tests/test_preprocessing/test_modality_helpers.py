"""Tests for feature modality helper functions."""

from __future__ import annotations

import pytest

from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.steps.preprocessing_helpers import (
    append_numerical_features,
    apply_permutation_to_modalities,
    filter_modalities_by_kept_indices,
    get_categorical_indices,
    update_categorical_indices,
)


@pytest.fixture
def basic_modalities() -> dict[FeatureModality, list[int]]:
    """Basic modalities: 5 features, 2 categorical (1, 3), 3 numerical (0, 2, 4)."""
    return {
        FeatureModality.NUMERICAL: [0, 2, 4],
        FeatureModality.CATEGORICAL: [1, 3],
    }


class TestGetCategoricalIndices:
    """Tests for get_categorical_indices."""

    def test__get_categorical_indices__basic(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test extracting categorical indices from modalities dict."""
        result = get_categorical_indices(basic_modalities)
        assert result == [1, 3]

    def test__get_categorical_indices__empty(self):
        """Test with no categorical features."""
        modalities = {FeatureModality.NUMERICAL: [0, 1, 2]}
        result = get_categorical_indices(modalities)
        assert result == []

    def test__get_categorical_indices__missing_key(self):
        """Test when CATEGORICAL key is missing entirely."""
        modalities: dict[FeatureModality, list[int]] = {
            FeatureModality.NUMERICAL: [0, 1]
        }
        result = get_categorical_indices(modalities)
        assert result == []


class TestApplyPermutationToModalities:
    """Tests for apply_permutation_to_modalities."""

    def test__apply_permutation__identity(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test identity permutation doesn't change indices."""
        permutation = [0, 1, 2, 3, 4]  # Identity
        result = apply_permutation_to_modalities(basic_modalities, permutation)
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4]
        assert result[FeatureModality.CATEGORICAL] == [1, 3]

    def test__apply_permutation__reverse(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test reverse permutation correctly remaps indices."""
        # Reverse: new position 0 has old column 4, new 1 has old 3, etc.
        permutation = [4, 3, 2, 1, 0]
        result = apply_permutation_to_modalities(basic_modalities, permutation)
        # Old indices: NUM [0, 2, 4] -> new positions [4, 2, 0] -> sorted [0, 2, 4]
        # Old indices: CAT [1, 3] -> new positions [3, 1] -> sorted [1, 3]
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4]
        assert result[FeatureModality.CATEGORICAL] == [1, 3]

    def test__apply_permutation__rotate(self):
        """Test rotation permutation."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 1],
            FeatureModality.CATEGORICAL: [2],
        }
        # Rotate right by 1: [2, 0, 1] means new[0]=old[2], new[1]=old[0], new[2]=old[1]
        permutation = [2, 0, 1]
        result = apply_permutation_to_modalities(modalities, permutation)
        # Old CAT [2] -> new position 0
        # Old NUM [0, 1] -> new positions [1, 2]
        assert result[FeatureModality.NUMERICAL] == [1, 2]
        assert result[FeatureModality.CATEGORICAL] == [0]

    def test__apply_permutation__swap_two(self):
        """Test swapping two adjacent columns."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 2],
            FeatureModality.CATEGORICAL: [1],
        }
        # Swap columns 0 and 1: permutation[0]=1, permutation[1]=0, permutation[2]=2
        permutation = [1, 0, 2]
        result = apply_permutation_to_modalities(modalities, permutation)
        # Old NUM [0, 2] -> new positions [1, 2]
        # Old CAT [1] -> new position 0
        assert result[FeatureModality.NUMERICAL] == [1, 2]
        assert result[FeatureModality.CATEGORICAL] == [0]


class TestFilterModalitiesByKeptIndices:
    """Tests for filter_modalities_by_kept_indices."""

    def test__filter__keep_all(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test keeping all indices doesn't change anything."""
        kept = [0, 1, 2, 3, 4]
        result = filter_modalities_by_kept_indices(basic_modalities, kept)
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4]
        assert result[FeatureModality.CATEGORICAL] == [1, 3]

    def test__filter__remove_one_numerical(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test removing one numerical feature remaps indices correctly."""
        # Remove column 2 (numerical), keep [0, 1, 3, 4]
        kept = [0, 1, 3, 4]
        result = filter_modalities_by_kept_indices(basic_modalities, kept)
        # Old NUM [0, 2, 4]: 0 stays at 0, 2 removed, 4 -> new index 3
        # Old CAT [1, 3]: 1 -> new index 1, 3 -> new index 2
        assert result[FeatureModality.NUMERICAL] == [0, 3]
        assert result[FeatureModality.CATEGORICAL] == [1, 2]

    def test__filter__remove_one_categorical(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test removing one categorical feature remaps indices correctly."""
        # Remove column 1 (categorical), keep [0, 2, 3, 4]
        kept = [0, 2, 3, 4]
        result = filter_modalities_by_kept_indices(basic_modalities, kept)
        # Old NUM [0, 2, 4]: 0 -> 0, 2 -> 1, 4 -> 3
        # Old CAT [1, 3]: 1 removed, 3 -> 2
        assert result[FeatureModality.NUMERICAL] == [0, 1, 3]
        assert result[FeatureModality.CATEGORICAL] == [2]

    def test__filter__remove_multiple(self):
        """Test removing multiple features."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 1, 4, 5],
            FeatureModality.CATEGORICAL: [2, 3],
        }
        # Keep only [1, 3, 5]
        kept = [1, 3, 5]
        result = filter_modalities_by_kept_indices(modalities, kept)
        # Old NUM [0, 1, 4, 5]: 0 removed, 1 -> 0, 4 removed, 5 -> 2
        # Old CAT [2, 3]: 2 removed, 3 -> 1
        assert result[FeatureModality.NUMERICAL] == [0, 2]
        assert result[FeatureModality.CATEGORICAL] == [1]

    def test__filter__empty_result(self):
        """Test filtering to empty categorical."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 1],
            FeatureModality.CATEGORICAL: [2],
        }
        # Keep only numerical features
        kept = [0, 1]
        result = filter_modalities_by_kept_indices(modalities, kept)
        assert result[FeatureModality.NUMERICAL] == [0, 1]
        assert result[FeatureModality.CATEGORICAL] == []


class TestUpdateCategoricalIndices:
    """Tests for update_categorical_indices.

    This function handles column reordering by computing a permutation:
    - Categorical columns map to new categorical positions (preserving relative order)
    - Non-categorical columns map to remaining positions (preserving relative order)
    """

    def test__update__same_indices(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test updating with same indices doesn't change anything."""
        result = update_categorical_indices(basic_modalities, [1, 3], n_features=5)
        assert result[FeatureModality.CATEGORICAL] == [1, 3]
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4]

    def test__update__column_transformer_reorder(self):
        """Test ColumnTransformer-style reordering where categoricals move to front.

        Original: [num_0, cat_1, num_2, cat_3, num_4]
        After ColumnTransformer: [cat_1, cat_3, num_0, num_2, num_4]
        New categorical positions: [0, 1]
        """
        modalities = {
            FeatureModality.NUMERICAL: [0, 2, 4],
            FeatureModality.CATEGORICAL: [1, 3],
        }
        # Categoricals moved to positions [0, 1], numericals shifted to [2, 3, 4]
        result = update_categorical_indices(modalities, [0, 1], n_features=5)
        assert result[FeatureModality.CATEGORICAL] == [0, 1]
        assert result[FeatureModality.NUMERICAL] == [2, 3, 4]

    def test__update__column_transformer_with_text(self):
        """Test reordering preserves TEXT modality correctly.

        Original: [num_0, cat_1, text_2, cat_3, num_4]
        After ColumnTransformer: [cat_1, cat_3, num_0, text_2, num_4]
        New categorical positions: [0, 1]
        """
        modalities = {
            FeatureModality.NUMERICAL: [0, 4],
            FeatureModality.CATEGORICAL: [1, 3],
            FeatureModality.TEXT: [2],
        }
        result = update_categorical_indices(modalities, [0, 1], n_features=5)
        assert result[FeatureModality.CATEGORICAL] == [0, 1]
        # Old NUM [0, 4] -> new positions [2, 4] (first and third non-cat slots)
        assert result[FeatureModality.NUMERICAL] == [2, 4]
        # Old TEXT [2] -> new position [3] (second non-cat slot)
        assert result[FeatureModality.TEXT] == [3]

    def test__update__single_categorical_moved(self):
        """Test single categorical column moved to front."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 2],
            FeatureModality.CATEGORICAL: [1],
        }
        # Categorical moved from index 1 to index 0
        result = update_categorical_indices(modalities, [0], n_features=3)
        assert result[FeatureModality.CATEGORICAL] == [0]
        # Old NUM [0, 2] -> new positions [1, 2]
        assert result[FeatureModality.NUMERICAL] == [1, 2]

    def test__update__all_categorical(self):
        """Test when all columns become categorical (expansion case)."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 1],
            FeatureModality.CATEGORICAL: [2],
        }
        result = update_categorical_indices(modalities, [0, 1, 2], n_features=3)
        assert result[FeatureModality.CATEGORICAL] == [0, 1, 2]
        # Numericals have no slots to map to
        assert result[FeatureModality.NUMERICAL] == []

    def test__update__no_categorical(self):
        """Test when categoricals are removed (all become non-categorical)."""
        modalities = {
            FeatureModality.NUMERICAL: [0],
            FeatureModality.CATEGORICAL: [1, 2],
        }
        result = update_categorical_indices(modalities, [], n_features=3)
        assert result[FeatureModality.CATEGORICAL] == []
        # Old categorical slots [1, 2] are now orphans, assigned to NUMERICAL
        assert result[FeatureModality.NUMERICAL] == [0, 1, 2]

    def test__update__more_features_expansion(self):
        """Test with more output features than input (e.g., one-hot encoding)."""
        modalities = {
            FeatureModality.NUMERICAL: [0],
            FeatureModality.CATEGORICAL: [1],
        }
        # After one-hot: 5 features, categorical expanded to [1, 2, 3]
        result = update_categorical_indices(modalities, [1, 2, 3], n_features=5)
        assert result[FeatureModality.CATEGORICAL] == [1, 2, 3]
        # Old NUM [0] -> new position [0], orphan [4] added to numerical
        assert result[FeatureModality.NUMERICAL] == [0, 4]

    def test__update__unsorted_input_same_positions(self):
        """Test that unsorted input is handled correctly (sorted internally)."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 2, 4],
            FeatureModality.CATEGORICAL: [1, 3],
        }
        # Same positions but provided unsorted: [3, 1] instead of [1, 3]
        result = update_categorical_indices(modalities, [3, 1], n_features=5)
        assert result[FeatureModality.CATEGORICAL] == [1, 3]
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4]

    def test__update__categoricals_moved_to_end(self):
        """Test categoricals moved to end (reverse of typical ColumnTransformer)."""
        modalities = {
            FeatureModality.NUMERICAL: [0, 2],
            FeatureModality.CATEGORICAL: [1, 3],
        }
        # Categoricals at end: positions [2, 3]
        result = update_categorical_indices(modalities, [2, 3], n_features=4)
        assert result[FeatureModality.CATEGORICAL] == [2, 3]
        # Old NUM [0, 2] -> new positions [0, 1]
        assert result[FeatureModality.NUMERICAL] == [0, 1]


class TestAppendNumericalFeatures:
    """Tests for append_numerical_features."""

    def test__append__single_feature(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test appending a single numerical feature."""
        result = append_numerical_features(
            basic_modalities, current_n_features=5, n_new_features=1
        )
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4, 5]
        assert result[FeatureModality.CATEGORICAL] == [1, 3]

    def test__append__multiple_features(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test appending multiple numerical features."""
        result = append_numerical_features(
            basic_modalities, current_n_features=5, n_new_features=3
        )
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4, 5, 6, 7]
        assert result[FeatureModality.CATEGORICAL] == [1, 3]

    def test__append__zero_features(
        self, basic_modalities: dict[FeatureModality, list[int]]
    ):
        """Test appending zero features doesn't change anything."""
        result = append_numerical_features(
            basic_modalities, current_n_features=5, n_new_features=0
        )
        assert result[FeatureModality.NUMERICAL] == [0, 2, 4]
        assert result[FeatureModality.CATEGORICAL] == [1, 3]

    def test__append__empty_numerical(self):
        """Test appending to empty numerical list."""
        modalities: dict[FeatureModality, list[int]] = {
            FeatureModality.NUMERICAL: [],
            FeatureModality.CATEGORICAL: [0, 1],
        }
        result = append_numerical_features(
            modalities, current_n_features=2, n_new_features=2
        )
        assert result[FeatureModality.NUMERICAL] == [2, 3]
        assert result[FeatureModality.CATEGORICAL] == [0, 1]
