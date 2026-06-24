#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the per-column ``mode`` helper in preprocessing/steps/utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing.steps.utils import mode


def test__mode__per_column():
    x = np.array([[1, 10], [1, 20], [2, 20]], dtype=float)
    np.testing.assert_array_equal(mode(x), [1, 20])


def test__mode__ignores_non_finite():
    x = np.array([[np.nan, 5], [1, np.inf], [1, -np.inf]], dtype=float)
    np.testing.assert_array_equal(mode(x), [1, 5])


def test__mode__all_non_finite_column_is_nan():
    x = np.array([[np.nan, 3], [np.inf, 3]], dtype=float)
    result = mode(x)
    assert np.isnan(result[0])
    assert result[1] == 3


def test__mode__1d_input():
    assert mode(np.array([7, 7, 9], dtype=float)) == 7
    assert np.isnan(mode(np.array([np.nan, np.inf], dtype=float)))


def test__mode__rejects_3d():
    with pytest.raises(ValueError, match="1D or 2D"):
        mode(np.zeros((2, 2, 2)))
