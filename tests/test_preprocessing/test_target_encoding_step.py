"""Tests for TargetEncodingStep."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing import PreprocessingPipeline
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.encode_categorical_features_step import (
    EncodeCategoricalFeaturesStep,
)
from tabpfn.preprocessing.steps.target_encoding_step import TargetEncodingStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_schema(modalities: list[FeatureModality]) -> FeatureSchema:
    return FeatureSchema(
        features=[Feature(name=None, modality=m) for m in modalities]
    )


def _make_mixed_data(
    n_samples: int = 100,
    n_num: int = 3,
    n_cat: int = 2,
    n_categories: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, FeatureSchema]:
    """Create mixed numerical + categorical data as a float array."""
    rng = np.random.default_rng(seed)
    X_num = rng.standard_normal((n_samples, n_num)).astype(np.float32)
    X_cat = rng.integers(0, n_categories, size=(n_samples, n_cat)).astype(np.float32)
    X = np.concatenate([X_num, X_cat], axis=1)
    modalities = [FeatureModality.NUMERICAL] * n_num + [
        FeatureModality.CATEGORICAL
    ] * n_cat
    return X, _make_schema(modalities)


def _make_binary_target(n_samples: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_samples).astype(np.float64)


def _make_multiclass_target(
    n_samples: int = 100, n_classes: int = 3, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n_samples).astype(np.float64)


def _make_regression_target(n_samples: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_samples)


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------
class TestBinaryClassification:
    def test_duplicate_true_shape(self) -> None:
        n_num, n_cat = 3, 2
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_binary_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        # Original columns + 1 TE column per categorical
        assert result.X.shape == (X.shape[0], n_num + n_cat + n_cat)

    def test_duplicate_false_shape(self) -> None:
        n_num, n_cat = 3, 2
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_binary_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=False, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        # Non-cat columns + 1 TE column per original categorical
        assert result.X.shape == (X.shape[0], n_num + n_cat)

    def test_schema_duplicate_true(self) -> None:
        n_num, n_cat = 3, 2
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_binary_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        # Original features preserved + new numerical features
        assert result.feature_schema.num_columns == n_num + n_cat + n_cat
        cat_indices = result.feature_schema.indices_for(FeatureModality.CATEGORICAL)
        assert len(cat_indices) == n_cat  # original categoricals still there

    def test_schema_duplicate_false(self) -> None:
        n_num, n_cat = 3, 2
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_binary_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=False, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        # All numerical (categoricals replaced)
        cat_indices = result.feature_schema.indices_for(FeatureModality.CATEGORICAL)
        assert len(cat_indices) == 0


# ---------------------------------------------------------------------------
# Multiclass classification — ordinal strategy (default)
# ---------------------------------------------------------------------------
class TestMulticlassOrdinal:
    def test_duplicate_true_shape(self) -> None:
        """Ordinal strategy: 1 TE column per categorical regardless of n_classes."""
        n_num, n_cat, n_classes = 3, 2, 4
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_multiclass_target(n_samples=X.shape[0], n_classes=n_classes)

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        # ordinal: 1 TE col per categorical, same as binary
        expected_cols = n_num + n_cat + n_cat
        assert result.X.shape == (X.shape[0], expected_cols)

    def test_duplicate_false_shape(self) -> None:
        n_num, n_cat, n_classes = 3, 2, 4
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_multiclass_target(n_samples=X.shape[0], n_classes=n_classes)

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=False, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        expected_cols = n_num + n_cat  # replace 1:1
        assert result.X.shape == (X.shape[0], expected_cols)

    def test_ordinal_is_default(self) -> None:
        """Verify ordinal is the default multiclass strategy."""
        step = TargetEncodingStep(task_type="classification", random_state=42)
        assert step.multiclass_strategy == "ordinal"

    def test_transform_consistent_shape(self) -> None:
        n_num, n_cat, n_classes = 2, 1, 3
        X_train, schema = _make_mixed_data(
            n_samples=80, n_num=n_num, n_cat=n_cat, seed=1
        )
        X_test, _ = _make_mixed_data(
            n_samples=20, n_num=n_num, n_cat=n_cat, seed=2
        )
        y_train = _make_multiclass_target(n_samples=80, n_classes=n_classes, seed=1)

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        train_result = step.fit_transform(X_train, schema, y=y_train)
        test_result = step.transform(X_test, is_test=True)

        expected_cols = n_num + n_cat + n_cat  # ordinal: 1 col per cat
        assert train_result.X.shape[1] == expected_cols
        assert test_result.X.shape[1] == expected_cols


# ---------------------------------------------------------------------------
# Multiclass classification — per_class strategy
# ---------------------------------------------------------------------------
class TestMulticlassPerClass:
    def test_duplicate_true_shape(self) -> None:
        """Per-class strategy: n_classes TE columns per categorical."""
        n_num, n_cat, n_classes = 3, 2, 4
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_multiclass_target(n_samples=X.shape[0], n_classes=n_classes)

        step = TargetEncodingStep(
            task_type="classification",
            multiclass_strategy="per_class",
            duplicate_features=True,
            random_state=42,
        )
        result = step.fit_transform(X, schema, y=y)

        expected_cols = n_num + n_cat + n_cat * n_classes
        assert result.X.shape == (X.shape[0], expected_cols)

    def test_duplicate_false_shape(self) -> None:
        n_num, n_cat, n_classes = 3, 2, 4
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_multiclass_target(n_samples=X.shape[0], n_classes=n_classes)

        step = TargetEncodingStep(
            task_type="classification",
            multiclass_strategy="per_class",
            duplicate_features=False,
            random_state=42,
        )
        result = step.fit_transform(X, schema, y=y)

        expected_cols = n_num + n_cat * n_classes
        assert result.X.shape == (X.shape[0], expected_cols)

    def test_transform_consistent_shape(self) -> None:
        n_num, n_cat, n_classes = 2, 1, 3
        X_train, schema = _make_mixed_data(
            n_samples=80, n_num=n_num, n_cat=n_cat, seed=1
        )
        X_test, _ = _make_mixed_data(
            n_samples=20, n_num=n_num, n_cat=n_cat, seed=2
        )
        y_train = _make_multiclass_target(n_samples=80, n_classes=n_classes, seed=1)

        step = TargetEncodingStep(
            task_type="classification",
            multiclass_strategy="per_class",
            duplicate_features=True,
            random_state=42,
        )
        train_result = step.fit_transform(X_train, schema, y=y_train)
        test_result = step.transform(X_test, is_test=True)

        expected_cols = n_num + n_cat + n_cat * n_classes
        assert train_result.X.shape[1] == expected_cols
        assert test_result.X.shape[1] == expected_cols


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
class TestRegression:
    def test_duplicate_true_shape(self) -> None:
        n_num, n_cat = 3, 2
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_regression_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="regression", duplicate_features=True, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        assert result.X.shape == (X.shape[0], n_num + n_cat + n_cat)

    def test_duplicate_false_shape(self) -> None:
        n_num, n_cat = 3, 2
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_regression_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="regression", duplicate_features=False, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        assert result.X.shape == (X.shape[0], n_num + n_cat)

    def test_te_values_are_continuous(self) -> None:
        """TE values for regression should be continuous (not integer category codes)."""
        n_num, n_cat = 3, 2
        X, schema = _make_mixed_data(n_num=n_num, n_cat=n_cat)
        y = _make_regression_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="regression", duplicate_features=True, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        # TE columns are the last n_cat columns
        te_cols = result.X[:, -n_cat:]
        # Should not be identical to integer category codes
        assert not np.array_equal(te_cols, X[:, n_num:])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_no_categorical_columns(self) -> None:
        """Step should be a no-op when there are no categoricals."""
        X = np.random.default_rng(42).standard_normal((50, 5)).astype(np.float32)
        schema = _make_schema([FeatureModality.NUMERICAL] * 5)
        y = _make_binary_target(n_samples=50)

        step = TargetEncodingStep(
            task_type="classification", random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        np.testing.assert_array_equal(result.X, X)
        assert result.feature_schema.num_columns == 5

    def test_y_none_raises(self) -> None:
        X, schema = _make_mixed_data()
        step = TargetEncodingStep(task_type="classification", random_state=42)

        with pytest.raises(ValueError, match="requires y"):
            step.fit_transform(X, schema, y=None)

    def test_nan_in_categorical(self) -> None:
        """NaN categories should be handled gracefully."""
        n_samples = 50
        X, schema = _make_mixed_data(n_samples=n_samples, n_num=2, n_cat=1)
        # Insert some NaN values
        X[0, 2] = np.nan
        X[10, 2] = np.nan
        y = _make_binary_target(n_samples=n_samples)

        step = TargetEncodingStep(
            task_type="classification", random_state=42
        )
        result = step.fit_transform(X, schema, y=y)

        # Should not have any NaN in TE columns
        assert not np.isnan(result.X[:, -1]).any()

    def test_all_same_category(self) -> None:
        """Single-category column should degenerate to global mean."""
        n_samples = 50
        X = np.zeros((n_samples, 2), dtype=np.float32)
        X[:, 0] = np.random.default_rng(42).standard_normal(n_samples)
        X[:, 1] = 0.0  # single category
        schema = _make_schema(
            [FeatureModality.NUMERICAL, FeatureModality.CATEGORICAL]
        )
        y = _make_binary_target(n_samples=n_samples)

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        result = step.fit_transform(X, schema, y=y)
        assert result.X.shape == (n_samples, 3)


# ---------------------------------------------------------------------------
# Test data (transform)
# ---------------------------------------------------------------------------
class TestTransform:
    def test_transform_consistent_shape(self) -> None:
        """Train and test should produce same column count."""
        X_train, schema = _make_mixed_data(n_samples=80, seed=1)
        X_test, _ = _make_mixed_data(n_samples=20, seed=2)
        y_train = _make_binary_target(n_samples=80, seed=1)

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        train_result = step.fit_transform(X_train, schema, y=y_train)
        test_result = step.transform(X_test, is_test=True)

        assert train_result.X.shape[1] == test_result.X.shape[1]
        assert test_result.X.shape[0] == 20

    def test_unseen_categories_get_global_prior(self) -> None:
        """Unseen categories in test should be encoded with global prior."""
        n_num, n_cat = 2, 1
        X_train, schema = _make_mixed_data(
            n_samples=80, n_num=n_num, n_cat=n_cat, n_categories=3, seed=1
        )
        y_train = _make_binary_target(n_samples=80, seed=1)

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        step.fit_transform(X_train, schema, y=y_train)

        # Create test data with an unseen category (99)
        X_test = np.zeros((5, n_num + n_cat), dtype=np.float32)
        X_test[:, n_num] = 99.0  # unseen category
        test_result = step.transform(X_test, is_test=True)

        # TE column should be the global prior for all rows
        te_col = test_result.X[:, -1]
        assert np.all(te_col == te_col[0])  # all same value

    def test_multiclass_ordinal_transform(self) -> None:
        """Multiclass with default ordinal strategy produces 1 col per cat."""
        n_num, n_cat, n_classes = 2, 1, 3
        X_train, schema = _make_mixed_data(
            n_samples=80, n_num=n_num, n_cat=n_cat, seed=1
        )
        X_test, _ = _make_mixed_data(
            n_samples=20, n_num=n_num, n_cat=n_cat, seed=2
        )
        y_train = _make_multiclass_target(n_samples=80, n_classes=n_classes, seed=1)

        step = TargetEncodingStep(
            task_type="classification", duplicate_features=True, random_state=42
        )
        train_result = step.fit_transform(X_train, schema, y=y_train)
        test_result = step.transform(X_test, is_test=True)

        expected_cols = n_num + n_cat + n_cat  # ordinal: 1 col per cat
        assert train_result.X.shape[1] == expected_cols
        assert test_result.X.shape[1] == expected_cols


# ---------------------------------------------------------------------------
# OOF property
# ---------------------------------------------------------------------------
class TestOOFProperty:
    def test_oof_differs_from_full_data_encoding(self) -> None:
        """OOF train encodings should differ from full-data test encodings."""
        X_train, schema = _make_mixed_data(
            n_samples=100, n_num=2, n_cat=1, n_categories=5, seed=42
        )
        y_train = _make_binary_target(n_samples=100, seed=42)

        step = TargetEncodingStep(
            task_type="classification",
            duplicate_features=True,
            n_folds=5,
            random_state=42,
        )
        train_result = step.fit_transform(X_train, schema, y=y_train)
        test_on_train = step.transform(X_train, is_test=True)

        # OOF encodings (train) vs full-data encodings (test on same data)
        oof_col = train_result.X[:, -1]
        full_col = test_on_train.X[:, -1]

        # They shouldn't be identical (OOF uses held-out folds)
        assert not np.allclose(oof_col, full_col)


# ---------------------------------------------------------------------------
# Smoothing effect
# ---------------------------------------------------------------------------
class TestSmoothing:
    def test_high_smoothing_approaches_global_prior(self) -> None:
        """With very high smoothing, all encodings should be close to global mean."""
        X, schema = _make_mixed_data(n_num=2, n_cat=1, n_categories=5)
        y = _make_binary_target(n_samples=X.shape[0])

        step = TargetEncodingStep(
            task_type="classification",
            smoothing=1e6,
            duplicate_features=True,
            random_state=42,
        )
        result = step.fit_transform(X, schema, y=y)

        te_col = result.X[:, -1]
        # All values should be nearly the same (close to global prior)
        assert np.std(te_col) < 0.01


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_produces_same_results(self) -> None:
        X, schema = _make_mixed_data()
        y = _make_binary_target(n_samples=X.shape[0])

        step1 = TargetEncodingStep(
            task_type="classification", random_state=42
        )
        result1 = step1.fit_transform(X.copy(), schema, y=y.copy())

        step2 = TargetEncodingStep(
            task_type="classification", random_state=42
        )
        result2 = step2.fit_transform(X.copy(), schema, y=y.copy())

        np.testing.assert_array_equal(result1.X, result2.X)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------
class TestPipelineIntegration:
    def test_target_encoding_then_categorical_encoding(self) -> None:
        """TargetEncoding + EncodeCategorical in a pipeline."""
        X, schema = _make_mixed_data(n_num=3, n_cat=2, n_categories=5)
        y = _make_binary_target(n_samples=X.shape[0])

        te_step = TargetEncodingStep(
            task_type="classification",
            duplicate_features=True,
            random_state=42,
        )
        cat_step = EncodeCategoricalFeaturesStep(
            categorical_transform_name="ordinal",
            random_state=42,
        )
        pipeline = PreprocessingPipeline(steps=[te_step, cat_step])
        result = pipeline.fit_transform(X, schema, y=y)

        # TE adds 2 numerical columns, ordinal encoder keeps same count
        assert result.X.shape[0] == X.shape[0]
        assert result.X.shape[1] == 3 + 2 + 2  # num + cat(ordinal) + te

    def test_pipeline_transform_consistency(self) -> None:
        """Pipeline fit_transform and transform should produce same column count."""
        X_train, schema = _make_mixed_data(n_samples=80, seed=1)
        X_test, _ = _make_mixed_data(n_samples=20, seed=2)
        y_train = _make_binary_target(n_samples=80, seed=1)

        te_step = TargetEncodingStep(
            task_type="classification",
            duplicate_features=True,
            random_state=42,
        )
        cat_step = EncodeCategoricalFeaturesStep(
            categorical_transform_name="ordinal",
            random_state=42,
        )
        pipeline = PreprocessingPipeline(steps=[te_step, cat_step])

        train_result = pipeline.fit_transform(X_train, schema, y=y_train)
        test_result = pipeline.transform(X_test)

        assert train_result.X.shape[1] == test_result.X.shape[1]

    def test_no_categorical_pipeline(self) -> None:
        """Pipeline with target encoding on all-numerical data."""
        X = np.random.default_rng(42).standard_normal((50, 5)).astype(np.float32)
        schema = _make_schema([FeatureModality.NUMERICAL] * 5)
        y = _make_binary_target(n_samples=50)

        te_step = TargetEncodingStep(
            task_type="classification", random_state=42
        )
        cat_step = EncodeCategoricalFeaturesStep(
            categorical_transform_name="ordinal",
            random_state=42,
        )
        pipeline = PreprocessingPipeline(steps=[te_step, cat_step])
        result = pipeline.fit_transform(X, schema, y=y)

        np.testing.assert_array_equal(result.X, X)
