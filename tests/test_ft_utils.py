from __future__ import annotations

import numpy as np
import torch

from tabpfn.utils import (
    pad_tensors,
    split_large_data,
)


def test_pad_tensors_2d_and_1d():
    """Test the pad_tensors function with 2D and 1D tensors."""
    # 2D tensors (features)
    tensors_2d = [torch.ones((2, 3)), torch.ones((3, 2)), torch.ones((1, 4))]
    padded = pad_tensors(tensors_2d, padding_val=-1, labels=False)
    assert all(
        t.shape == (3, 4) for t in padded
    ), f"Expected shape (3, 4), got {[t.shape for t in padded]}"
    assert padded[0][2, 3] == -1, "Padding value not set correctly for 2D case."

    # 1D tensors (labels)
    tensors_1d = [torch.arange(3), torch.arange(5), torch.arange(2)]
    padded_1d = pad_tensors(tensors_1d, padding_val=99, labels=True)
    assert all(
        t.shape == (5,) for t in padded_1d
    ), f"Expected shape (5,), got {[t.shape for t in padded_1d]}"
    assert padded_1d[0][3] == 99, "Padding value not set correctly for 1D case."


def test_split_large_data():
    """Test the split_large_data function with a large dataset."""
    total_size = 1000
    max_chunk = 100
    large_x = np.arange(total_size * 2).reshape((total_size, 2))
    large_y = np.arange(total_size)

    expected_num_chunks = ((total_size - 1) // max_chunk) + 1

    x_chunks, y_chunks = split_large_data(large_x, large_y, max_chunk)

    assert len(x_chunks) == expected_num_chunks, "Incorrect X chunk count"
    assert len(y_chunks) == expected_num_chunks, "Incorrect y chunk count"

    # Check that each chunk size is <= max_chunk
    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        assert len(x_chunk) <= max_chunk, "X chunk size exceeds max"
        assert len(y_chunk) <= max_chunk, "y chunk size exceeds max"
        assert len(x_chunk) == len(y_chunk), "X/y chunk length mismatch"

    # Check total size by summing chunk lengths
    assert sum(len(chunk) for chunk in x_chunks) == total_size, "X total size mismatch"
    assert sum(len(chunk) for chunk in y_chunks) == total_size, "y total size mismatch"

    # Check data integrity by reconstructing original arrays
    reconstructed_x = np.vstack(x_chunks)
    reconstructed_y = np.concatenate(y_chunks)

    np.testing.assert_array_equal(reconstructed_x, large_x, "Reconstructed X differs")
    np.testing.assert_array_equal(reconstructed_y, large_y, "Reconstructed y differs")

    # Test edge case: empty input
    x_empty, y_empty = split_large_data([], [], max_chunk)
    assert x_empty == [], "X should be empty list for empty input"
    assert y_empty == [], "y should be empty list for empty input"

    # Test edge case: max_data_size >= total_size
    x_single, y_single = split_large_data(large_x, large_y, total_size + 5)
    assert len(x_single) == 1, "Should be 1 X chunk if max_size is large"
    assert len(y_single) == 1, "Should be 1 y chunk if max_size is large"
    np.testing.assert_array_equal(x_single[0], large_x)
    np.testing.assert_array_equal(y_single[0], large_y)

    # Test edge case: max_data_size = 1
    x_max_one, y_max_one = split_large_data(large_x, large_y, 1)
    assert len(x_max_one) == total_size, "Should be total_size chunks if max_size=1"
    assert len(y_max_one) == total_size, "Should be total_size chunks if max_size=1"
    assert len(x_max_one[0]) == 1, "Each X chunk should have size 1"
    assert len(y_max_one[0]) == 1, "Each y chunk should have size 1"

    # Test edge case: max_data_size causes exact division
    total_size_exact = 90
    max_chunk_exact = 30
    large_x_exact = np.arange(total_size_exact * 2).reshape((total_size_exact, 2))
    large_y_exact = np.arange(total_size_exact)
    x_exact, y_exact = split_large_data(large_x_exact, large_y_exact, max_chunk_exact)
    assert len(x_exact) == total_size_exact // max_chunk_exact  # Should be 3 chunks
    assert len(y_exact) == total_size_exact // max_chunk_exact  # Should be 3 chunks
    assert len(x_exact[0]) == max_chunk_exact  # All chunks should be max size
    assert len(y_exact[0]) == max_chunk_exact
