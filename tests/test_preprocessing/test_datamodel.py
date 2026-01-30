"""Tests for the preprocessing datamodel module."""

from __future__ import annotations

import pytest

from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema


class TestFeatureSchema:
    """Tests for the FeatureSchema class."""

    def test__init__with_features(self) -> None:
        """Test FeatureSchema construction with features list."""
        features = [
            Feature(name="age", modality=FeatureModality.NUMERICAL),
            Feature(name="city", modality=FeatureModality.CATEGORICAL),
            Feature(name="income", modality=FeatureModality.NUMERICAL),
        ]
        schema = FeatureSchema(features=features)

        assert schema.num_columns == 3
        assert schema.feature_names == ["age", "city", "income"]
        assert schema.indices_for(FeatureModality.NUMERICAL) == [0, 2]
        assert schema.indices_for(FeatureModality.CATEGORICAL) == [1]

    def test__from_feature_modalities__basic(self) -> None:
        """Test from_feature_modalities factory method."""
        schema = FeatureSchema(
            features=[
                Feature(name=None, modality=FeatureModality.NUMERICAL),
                Feature(name=None, modality=FeatureModality.CATEGORICAL),
                Feature(name=None, modality=FeatureModality.NUMERICAL),
            ]
        )

        assert schema.num_columns == 3
        assert schema.feature_names == [None, None, None]
        assert schema.indices_for(FeatureModality.NUMERICAL) == [0, 2]
        assert schema.indices_for(FeatureModality.CATEGORICAL) == [1]

    def test__from_feature_modalities__with_custom_names(self) -> None:
        """Test from_feature_modalities with custom feature names."""
        schema = FeatureSchema(
            features=[
                Feature(name="height", modality=FeatureModality.NUMERICAL),
                Feature(name="weight", modality=FeatureModality.NUMERICAL),
            ]
        )

        assert schema.feature_names == ["height", "weight"]
        assert schema.num_columns == 2

    def test__feature_modalities__derived_property(self) -> None:
        """Test feature_modalities is correctly derived from features list."""
        features = [
            Feature(name="num1", modality=FeatureModality.NUMERICAL),
            Feature(name="cat1", modality=FeatureModality.CATEGORICAL),
            Feature(name="num2", modality=FeatureModality.NUMERICAL),
            Feature(name="txt1", modality=FeatureModality.TEXT),
        ]
        schema = FeatureSchema(features=features)

        assert schema.indices_for(FeatureModality.NUMERICAL) == [0, 2]
        assert schema.indices_for(FeatureModality.CATEGORICAL) == [1]
        assert schema.indices_for(FeatureModality.TEXT) == [3]

    def test__indices_for__missing_modality(self) -> None:
        """Test indices_for returns empty list for missing modality."""
        features = [Feature(name="a", modality=FeatureModality.NUMERICAL)]
        schema = FeatureSchema(features=features)

        assert schema.indices_for(FeatureModality.TEXT) == []


class TestFeatureSchemaAddColumns:
    """Tests for the add_columns method."""

    def test__add_columns__basic(self) -> None:
        """Test adding columns appends to features list."""
        features = [Feature(name="a", modality=FeatureModality.NUMERICAL)]
        schema = FeatureSchema(features=features)

        new_schema = schema.append_columns(FeatureModality.CATEGORICAL, num_new=2)

        assert new_schema.num_columns == 3
        assert new_schema.feature_names == ["a", "added_0", "added_1"]
        assert new_schema.indices_for(FeatureModality.NUMERICAL) == [0]
        assert new_schema.indices_for(FeatureModality.CATEGORICAL) == [1, 2]

    def test__add_columns__with_custom_names(self) -> None:
        """Test adding columns with custom names."""
        features = [Feature(name="a", modality=FeatureModality.NUMERICAL)]
        schema = FeatureSchema(features=features)

        new_schema = schema.append_columns(
            FeatureModality.NUMERICAL,
            num_new=2,
            names=["fingerprint_1", "fingerprint_2"],
        )

        assert new_schema.feature_names == ["a", "fingerprint_1", "fingerprint_2"]
        assert new_schema.indices_for(FeatureModality.NUMERICAL) == [0, 1, 2]

    def test__add_columns__name_count_mismatch_raises(self) -> None:
        """Test that mismatched name count raises error."""
        schema = FeatureSchema()

        with pytest.raises(ValueError, match="Expected 2 names"):
            schema.append_columns(
                FeatureModality.NUMERICAL, num_new=2, names=["only_one"]
            )

    def test__add_columns__immutability(self) -> None:
        """Test that add_columns returns new instance, doesn't modify original."""
        features = [Feature(name="a", modality=FeatureModality.NUMERICAL)]
        schema = FeatureSchema(features=features)

        new_schema = schema.append_columns(FeatureModality.CATEGORICAL, num_new=1)

        assert schema.num_columns == 1
        assert new_schema.num_columns == 2


class TestFeatureSchemaRemoveColumns:
    """Tests for the remove_columns method."""

    def test__remove_columns__basic(self) -> None:
        """Test removing columns filters out features at specified indices."""
        features = [
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.CATEGORICAL),
            Feature(name="c", modality=FeatureModality.NUMERICAL),
        ]
        schema = FeatureSchema(features=features)

        new_schema = schema.remove_columns([1])

        assert new_schema.num_columns == 2
        assert new_schema.feature_names == ["a", "c"]
        assert new_schema.indices_for(FeatureModality.NUMERICAL) == [0, 1]
        assert new_schema.indices_for(FeatureModality.CATEGORICAL) == []

    def test__remove_columns__immutability(self) -> None:
        """Test that remove_columns returns new instance, doesn't modify original."""
        features = [
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.NUMERICAL),
        ]
        schema = FeatureSchema(features=features)

        new_schema = schema.remove_columns([0])

        assert schema.num_columns == 2
        assert new_schema.num_columns == 1


class TestFeatureSchemaApplyPermutation:
    """Tests for the apply_permutation method."""

    def test__apply_permutation__basic_reorder(self) -> None:
        """Test applying a permutation reorders features."""
        features = [
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.CATEGORICAL),
            Feature(name="c", modality=FeatureModality.TEXT),
        ]
        schema = FeatureSchema(features=features)

        # Permutation: new position 0 <- old 2, new 1 <- old 0, new 2 <- old 1
        new_schema = schema.apply_permutation([2, 0, 1])

        assert new_schema.feature_names == ["c", "a", "b"]
        assert new_schema.indices_for(FeatureModality.TEXT) == [0]
        assert new_schema.indices_for(FeatureModality.NUMERICAL) == [1]
        assert new_schema.indices_for(FeatureModality.CATEGORICAL) == [2]

    def test__apply_permutation__immutability(self) -> None:
        """Test that apply_permutation returns new instance."""
        features = [
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.CATEGORICAL),
        ]
        schema = FeatureSchema(features=features)

        new_schema = schema.apply_permutation([1, 0])

        assert schema.feature_names == ["a", "b"]
        assert new_schema.feature_names == ["b", "a"]


class TestFeatureSchemaSliceForIndices:
    """Tests for the slice_for_indices method."""

    def test__slice_for_indices__basic(self) -> None:
        """Test slicing creates subset with remapped 0-based indices."""
        features = [
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.CATEGORICAL),
            Feature(name="c", modality=FeatureModality.NUMERICAL),
            Feature(name="d", modality=FeatureModality.TEXT),
        ]
        schema = FeatureSchema(features=features)

        # Select indices 1 and 3
        sliced = schema.slice_for_indices([1, 3])

        assert sliced.num_columns == 2
        assert sliced.feature_names == ["b", "d"]
        assert sliced.indices_for(FeatureModality.CATEGORICAL) == [0]
        assert sliced.indices_for(FeatureModality.TEXT) == [1]

    def test__slice_for_indices__preserves_names(self) -> None:
        """Test slicing preserves feature names."""
        features = [
            Feature(name="original_name", modality=FeatureModality.NUMERICAL),
            Feature(name="other", modality=FeatureModality.CATEGORICAL),
        ]
        schema = FeatureSchema(features=features)

        sliced = schema.slice_for_indices([0])

        assert sliced.feature_names == ["original_name"]

    def test__slice_for_indices__single_element(self) -> None:
        """Test slicing with a single index."""
        features = [
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.CATEGORICAL),
            Feature(name="c", modality=FeatureModality.TEXT),
        ]
        schema = FeatureSchema(features=features)

        sliced = schema.slice_for_indices([1])

        assert sliced.num_columns == 1
        assert sliced.feature_names == ["b"]
        assert sliced.indices_for(FeatureModality.CATEGORICAL) == [0]


class TestFeatureSchemaUpdateFromPreprocessingStepResult:
    """Tests for the update_from_preprocessing_step_result method."""

    def test__update_from_preprocessing_step_result__multiple_columns(self) -> None:
        """Test updating metadata when step processes multiple columns."""
        # Original: [cat, cat, num, num]
        features = [
            Feature(name="cat1", modality=FeatureModality.CATEGORICAL),
            Feature(name="cat2", modality=FeatureModality.CATEGORICAL),
            Feature(name="num1", modality=FeatureModality.NUMERICAL),
            Feature(name="num2", modality=FeatureModality.NUMERICAL),
        ]
        schema = FeatureSchema(features=features)

        # Step processed categorical columns [0, 1] and converted them to numerical
        step_result = FeatureSchema(
            features=[
                Feature(name="encoded_0", modality=FeatureModality.NUMERICAL),
                Feature(name="encoded_1", modality=FeatureModality.NUMERICAL),
            ]
        )

        updated = schema.update_from_preprocessing_step_result(
            original_indices=[0, 1], new_schema=step_result
        )

        # Original names preserved
        assert updated.feature_names == ["encoded_0", "encoded_1", "num1", "num2"]
        # All columns now numerical
        assert updated.indices_for(FeatureModality.NUMERICAL) == [0, 1, 2, 3]
        assert updated.indices_for(FeatureModality.CATEGORICAL) == []

    def test__update_from_preprocessing_step_result__preserves_unprocessed(
        self,
    ) -> None:
        """Test that unprocessed columns are preserved unchanged."""
        features = [
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.CATEGORICAL),
            Feature(name="c", modality=FeatureModality.TEXT),
        ]
        schema = FeatureSchema(features=features)

        # Only process column 1
        step_result = FeatureSchema(
            features=[Feature(name="x", modality=FeatureModality.NUMERICAL)]
        )

        updated = schema.update_from_preprocessing_step_result(
            original_indices=[1], new_schema=step_result
        )

        # Columns 0 and 2 unchanged
        assert updated.features[0].name == "a"
        assert updated.features[0].modality == FeatureModality.NUMERICAL
        assert updated.features[2].name == "c"
        assert updated.features[2].modality == FeatureModality.TEXT

    def test__update_from_preprocessing_step_result__immutability(self) -> None:
        """Test that update returns new instance, doesn't modify original."""
        features = [
            Feature(name="a", modality=FeatureModality.CATEGORICAL),
        ]
        schema = FeatureSchema(features=features)

        step_result = FeatureSchema(
            features=[Feature(name="x", modality=FeatureModality.NUMERICAL)]
        )

        updated = schema.update_from_preprocessing_step_result(
            original_indices=[0], new_schema=step_result
        )

        # Original unchanged
        assert schema.indices_for(FeatureModality.CATEGORICAL) == [0]
        # Updated has new modality
        assert updated.indices_for(FeatureModality.NUMERICAL) == [0]
