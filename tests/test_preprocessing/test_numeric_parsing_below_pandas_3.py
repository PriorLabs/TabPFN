#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the per-value numeric parse used below pandas 3.0.

RUF001 is off for this module: several cases are deliberately built from characters
that look like ASCII digits but are not, which is the property under test.

Delete this whole file when the declared pandas floor reaches 3.0. Everything here
covers `modality_detection._is_numeric_or_missing_for_old_pandas`, which only exists
because `pandas.to_numeric` segfaults below that version on a string whose
scientific-notation exponent lands in `[2**31, 2**32)`, e.g. `"8e2569614270"`
(https://github.com/pandas-dev/pandas/issues/63650). Once the floor is 3.0 the parse
goes away with it and these tests have nothing left to cover.
"""

# ruff: noqa: RUF001
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tabpfn.preprocessing import modality_detection
from tabpfn.preprocessing.modality_detection import (
    _is_numeric_or_missing_for_old_pandas,
)

#: Values every supported pandas agrees on, as (value, is_numeric). Both the per-value
#: parse and `pandas.to_numeric` itself are checked against these, so the pair of tests
#: together pins the two to each other rather than to a hand-written expectation.
_PANDAS_AGREES_ON: list[tuple[Any, bool]] = [
    # Plain numbers, in the spellings pandas accepts.
    ("0", True),
    ("123", True),
    ("000123", True),
    ("-7", True),
    ("+5", True),
    ("1.5", True),
    (".5", True),
    ("5.", True),
    ("-0.0", True),
    ("1e5", True),
    ("1E5", True),
    ("1e-5", True),
    ("1.0e+5", True),
    # Surrounding ASCII whitespace is ignored.
    (" 12 ", True),
    ("\t7\n", True),
    # A spelled-out infinity is a number; every spelling of NaN is not.
    ("inf", True),
    ("-inf", True),
    ("Infinity", True),
    ("nan", False),
    ("NaN", False),
    ("-nan", False),
    (" nan ", False),
    # Not numbers in any spelling.
    ("", False),
    (" ", False),
    ("abc", False),
    ("id_7", False),
    ("null", False),
    ("None", False),
    ("N/A", False),
    ("true", False),
    ("0x1A", False),
    ("1,000", False),
    ("1.2.3", False),
    ("--1", False),
    ("1j", False),
    ("e5", False),
    # The built-in `float` accepts these, so the parse has to reject them itself.
    ("1_000", False),
    ("1_0.5", False),
    ("٣", False),  # Arabic-Indic three
    ("１２", False),  # fullwidth one and two
    ("\xa0 5", False),  # non-breaking space
    # Missing reads as missing, not as text.
    (None, True),
    (np.nan, True),
    (pd.NA, True),
    (pd.NaT, True),
    # Already a number, so there is no spelling to check.
    (1, True),
    (2.5, True),
    (True, True),
    (np.int64(7), True),
    (np.float64(1.5), True),
    # Pandas reads `bytes`, so these are numbers ...
    (b"1", True),
    (b"1.5", True),
    (np.bytes_(b"1"), True),
    (b"x", False),
    # ... but it reads no other buffer, though the built-in `float` reads them all.
    (bytearray(b"1"), False),
    (memoryview(b"1"), False),
    # A cell holding a container. `pd.isna` answers element-wise for these, so the
    # missingness check has to tolerate them instead of raising.
    ([1, 2], False),
    ({"a": "b"}, False),
    ((1, 2), False),
    (np.array([1, 2]), False),
]

#: Values the parity test below cannot use. The first three crash `pandas.to_numeric`
#: outright below pandas 3.0, which is why this code exists at all; the last two are
#: the one disagreement between the pandas majors ("1e400" reads as NaN on 2 and as
#: infinity on 3), and the parse follows pandas 2, the only major it ever runs on.
_PANDAS_CANNOT_BE_ASKED: list[tuple[Any, bool]] = [
    ("8e2569614270", False),
    ("8e2569614270f3d8b9e7038efac9f116", False),
    ("1e2147483648", False),
    ("1e400", False),
    ("-1e400", False),
]


@pytest.mark.parametrize(
    ("value", "is_numeric"), _PANDAS_AGREES_ON + _PANDAS_CANNOT_BE_ASKED
)
def test__is_numeric_or_missing_for_old_pandas__reads_one_value(
    value: object, is_numeric: bool
) -> None:
    assert _is_numeric_or_missing_for_old_pandas(value) is is_numeric


@pytest.mark.parametrize(("value", "is_numeric"), _PANDAS_AGREES_ON)
def test__is_numeric_or_missing_for_old_pandas__expectations_are_pandas_own(
    value: object, is_numeric: bool
) -> None:
    """Pins the table above to `pandas.to_numeric`, not to what we think it does.

    The parse stands in for pandas, so pandas decides the right answer. Asserting the
    same table against both keeps a wrong expectation from being satisfied by a
    matching bug in the stand-in.
    """
    s = pd.Series([value], dtype=object)
    coerced = pd.to_numeric(s, errors="coerce")
    assert bool((coerced.notna() | s.isna()).all()) is is_numeric


@pytest.mark.parametrize("pandas_below_3", [True, False])
def test__is_numeric_pandas_series__both_branches_agree(
    pandas_below_3: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whichever branch runs, an ordinary column has to be classified the same way."""
    monkeypatch.setattr(modality_detection, "PANDAS_BELOW_3", pandas_below_3)

    assert modality_detection._is_numeric_pandas_series(pd.Series(["1", "2.5", None]))
    assert not modality_detection._is_numeric_pandas_series(pd.Series(["1", "abc"]))


def test__is_numeric_pandas_series__old_pandas_branch__never_calls_to_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The branch exists to keep a crashing value away from `pandas.to_numeric`.

    Forced on regardless of the installed pandas, so the guarantee is tested on every
    version rather than only where the crash reproduces. `pandas.to_numeric` is
    replaced with a raising stub because a real crash could not be caught.
    """
    monkeypatch.setattr(modality_detection, "PANDAS_BELOW_3", True)

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pandas.to_numeric was called with the crashing value")

    monkeypatch.setattr(pd, "to_numeric", _explode)

    values = [f"id_{i}" for i in range(50)]
    values[7] = "8e2569614270f3d8b9e7038efac9f116"

    assert not modality_detection._is_numeric_pandas_series(pd.Series(values))
