from __future__ import annotations

import numpy as np

from tabpfn.preprocessing.ensemble import _get_subsample_indices_for_estimators


def test__get_subsample_indices_for_estimators():
    """Test that different subsample_samples arguments work as expected."""
    kwargs = {
        "num_estimators": 3,
        "max_index": 5,
        "static_seed": 42,
    }

    subsample_samples = [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9]),
    ]
    expected_subsample_indices = [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9]),
        np.array([0, 1, 2, 3, 4]),
    ]
    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_samples,
        **kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index, expected_subsample_index in zip(
        subsample_indices, expected_subsample_indices
    ):
        assert subsample_index is not None
        assert (subsample_index == expected_subsample_index).all()

    subsample_samples = 0.5
    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_samples,
        **kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index in subsample_indices:
        assert subsample_index is not None
        assert len(subsample_index) == 3  # (max_index + 1) * 0.5

    subsample_samples = 2
    subsample_indices = _get_subsample_indices_for_estimators(
        subsample_samples=subsample_samples,
        **kwargs,
    )
    assert len(subsample_indices) == 3
    for subsample_index in subsample_indices:
        assert subsample_index is not None
        assert len(subsample_index) == 2
