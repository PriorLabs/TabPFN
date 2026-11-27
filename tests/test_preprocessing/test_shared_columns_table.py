#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable
from tabpfn.utils import as_tensor_to_read


def _base() -> np.ndarray:
    return np.arange(30, dtype=np.float32).reshape(6, 5)


def test__selecting_columns__matches_the_array() -> None:
    base = _base()
    table = SharedColumnsTable.of(base)[:, [3, 0, 4]][:, [True, False, True]]

    np.testing.assert_array_equal(np.asarray(table), base[:, [3, 4]])
    np.testing.assert_array_equal(table[2:5], base[2:5][:, [3, 4]])
    assert table.base is base
    assert table.shape == (6, 2)
    assert len(table) == 6


def test__selecting_columns__with_32_bit_sources__matches_the_array() -> None:
    # `np.arange` gives int32 on Windows with numpy 1.
    base = _base()
    table = SharedColumnsTable(
        base, np.arange(5, dtype=np.int32), np.empty((6, 0), np.float32)
    )[:, [3, 0, 4]]

    np.testing.assert_array_equal(np.asarray(table), base[:, [3, 0, 4]])


def test__with_columns__places_own_values() -> None:
    base = _base()
    values = np.full((6, 2), -1.0, dtype=np.float32)
    table = SharedColumnsTable.of(base).with_columns(values, slice(0, 2), [(2, 4)])

    np.testing.assert_array_equal(
        np.asarray(table), np.column_stack([values, base[:, 4]])
    )
    np.testing.assert_array_equal(
        table[1:4], np.column_stack([values, base[:, 4]])[1:4]
    )
    table.own_column(1)[:] = 7.0
    np.testing.assert_array_equal(np.asarray(table)[:, 1], np.full(6, 7.0))


def test__with_columns__once_and_in_the_base_dtype() -> None:
    table = SharedColumnsTable.of(_base())
    owning = table.with_columns(np.zeros((6, 1), np.float32), [0], [(1, 0)])

    with pytest.raises(ValueError, match="once"):
        owning.with_columns(np.zeros((6, 1), np.float32), [0], [(1, 0)])
    with pytest.raises(ValueError, match="dtype"):
        table.with_columns(np.zeros((6, 1)), [0], [(1, 0)])


def test__as_tensor_to_read__a_table__raises() -> None:
    with pytest.raises(TypeError, match="SharedColumnsTable"):
        as_tensor_to_read(SharedColumnsTable.of(_base()))


def test__is_base__only_for_every_column_in_order() -> None:
    table = SharedColumnsTable.of(_base())

    assert table.is_base()
    assert not table[:, [1, 0, 2, 3, 4]].is_base()
    assert not table[:, [0, 1, 2, 3]].is_base()


def test__own_column__of_a_shared_column__raises() -> None:
    with pytest.raises(ValueError, match="shared"):
        SharedColumnsTable.of(_base()).own_column(0)


def test__asarray__without_a_copy__raises_unless_the_table_is_its_base() -> None:
    base = _base()
    table = SharedColumnsTable.of(base)

    assert np.asarray(table) is base
    assert np.shares_memory(table[1:3], base)
    # Called directly: numpy 1's `np.asarray` takes no `copy`.
    with pytest.raises(ValueError, match="without a copy"):
        table[:, [1, 0]].__array__(copy=False)


def test__rows__a_slice_only() -> None:
    table = SharedColumnsTable.of(_base())[:, [1, 0]]

    with pytest.raises(IndexError, match="contiguous"):
        table[::2]
