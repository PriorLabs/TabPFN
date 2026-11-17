#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable


def _base() -> np.ndarray:
    return np.arange(30, dtype=np.float32).reshape(6, 5)


def test__selecting_columns__matches_the_array() -> None:
    base = _base()
    table = SharedColumnsTable.of(base)[:, [3, 0, 4]].compress(
        [True, False, True], axis=1
    )

    np.testing.assert_array_equal(np.asarray(table), base[:, [3, 4]])
    np.testing.assert_array_equal(table[2:5], base[2:5][:, [3, 4]])
    assert table.base is base
    assert table.shape == (6, 2)
    assert len(table) == 6


def test__with_columns__places_own_values_and_keeps_only_those_used() -> None:
    base = _base()
    values = np.full((6, 2), -1.0)
    table = SharedColumnsTable.of(base).with_columns(values, slice(0, 2), [(2, 4)])

    np.testing.assert_array_equal(
        np.asarray(table), np.column_stack([values, base[:, 4]])
    )
    assert table.own.dtype == base.dtype
    assert table.take([2]).own.shape == (6, 0)
    table.own_column(1)[:] = 7.0
    np.testing.assert_array_equal(np.asarray(table)[:, 1], np.full(6, 7.0))


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

    assert np.asarray(table, copy=False) is base
    assert np.shares_memory(table[1:3], base)
    with pytest.raises(ValueError, match="without a copy"):
        np.asarray(table[:, [1, 0]], copy=False)
