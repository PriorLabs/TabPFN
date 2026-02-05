"""Tests for TabPFNLabelEncoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.label_encoder import TabPFNLabelEncoder


class TestTabPFNLabelEncoderBasic:
    """Tests for basic label encoding functionality."""

    def test__fit_transform__string_labels(self) -> None:
        """Test encoding string labels."""
        encoder = TabPFNLabelEncoder()
        y = np.array(["cat", "dog", "cat", "bird", "dog"])

        y_encoded, schema = encoder.fit_transform(y, max_num_classes=10)

        assert y_encoded.shape == (5,)
        assert schema.n_classes == 3
        np.testing.assert_array_equal(schema.classes, ["bird", "cat", "dog"])
        np.testing.assert_array_equal(schema.class_counts, [1, 2, 2])

    def test__fit_transform__non_contiguous_integers(self) -> None:
        """Test encoding non-contiguous integer labels (e.g., 0, 5, 10)."""
        encoder = TabPFNLabelEncoder()
        y = np.array([0, 5, 10, 0, 5, 10, 10])

        y_encoded, schema = encoder.fit_transform(y, max_num_classes=10)

        # Labels should be remapped to 0, 1, 2
        np.testing.assert_array_equal(y_encoded, [0, 1, 2, 0, 1, 2, 2])
        assert schema.n_classes == 3
        np.testing.assert_array_equal(schema.classes, [0, 5, 10])
        np.testing.assert_array_equal(schema.class_counts, [2, 2, 3])

    def test__fit_transform__negative_integers(self) -> None:
        """Test encoding negative integer labels."""
        encoder = TabPFNLabelEncoder()
        y = np.array([-1, 0, 1, -1, 0, 1])

        y_encoded, schema = encoder.fit_transform(y, max_num_classes=10)

        # Should be sorted and remapped: -1 -> 0, 0 -> 1, 1 -> 2
        np.testing.assert_array_equal(y_encoded, [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(schema.classes, [-1, 0, 1])


class TestTabPFNLabelEncoderUnconventionalInputs:
    """Tests for unconventional input types."""

    def test__fit_transform__boolean_labels(self) -> None:
        """Test encoding boolean labels."""
        encoder = TabPFNLabelEncoder()
        y = np.array([True, False, True, False, True])

        _, schema = encoder.fit_transform(y, max_num_classes=10)

        assert schema.n_classes == 2
        # Booleans sort as False < True
        np.testing.assert_array_equal(schema.class_counts, [2, 3])

    def test__fit_transform__python_list(self) -> None:
        """Test encoding from a Python list."""
        encoder = TabPFNLabelEncoder()
        y = [0, 1, 2, 0, 1]

        y_encoded, schema = encoder.fit_transform(y, max_num_classes=10)

        assert y_encoded.shape == (5,)
        assert schema.n_classes == 3

    def test__fit_transform__pandas_series(self) -> None:
        """Test encoding from a pandas Series."""
        encoder = TabPFNLabelEncoder()
        y = pd.Series(["a", "b", "c", "a", "b"], name="labels")

        y_encoded, schema = encoder.fit_transform(y, max_num_classes=10)

        np.testing.assert_array_equal(y_encoded, [0, 1, 2, 0, 1])
        assert schema.n_classes == 3
        np.testing.assert_array_equal(schema.classes, ["a", "b", "c"])

    def test__fit_transform__pandas_categorical(self) -> None:
        """Test encoding from a pandas Categorical."""
        encoder = TabPFNLabelEncoder()
        y = pd.Categorical(["low", "medium", "high", "low", "medium"])

        _, schema = encoder.fit_transform(y, max_num_classes=10)

        assert schema.n_classes == 3

    def test__fit_transform__unicode_strings(self) -> None:
        """Test encoding unicode string labels."""
        encoder = TabPFNLabelEncoder()
        y = np.array(["café", "naïve", "résumé", "café", "naïve"])

        _, schema = encoder.fit_transform(y, max_num_classes=10)

        assert schema.n_classes == 3
        assert "café" in schema.classes

    def test__fit_transform__empty_string_label(self) -> None:
        """Test encoding with empty string as one of the labels."""
        encoder = TabPFNLabelEncoder()
        y = np.array(["", "a", "b", "", "a"])

        _, schema = encoder.fit_transform(y, max_num_classes=10)

        assert schema.n_classes == 3
        assert "" in schema.classes


class TestTabPFNLabelEncoderValidation:
    """Tests for validation and error handling."""

    def test__fit_transform__exceeds_max_classes(self) -> None:
        """Test that exceeding max_num_classes raises an error."""
        encoder = TabPFNLabelEncoder()
        y = np.array([0, 1, 2, 3, 4, 5])  # 6 classes

        with pytest.raises(TabPFNValidationError, match="exceeds the maximum"):
            encoder.fit_transform(y, max_num_classes=3)
