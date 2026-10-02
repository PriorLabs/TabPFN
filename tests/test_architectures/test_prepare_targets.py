"""Tests that `_prepare_targets` keeps the train targets intact at >= 2**16 rows.

On MPS with torch <= 2.14, `F.pad` with a non-zero fill value overwrites its input
once it reaches 2**16 rows, which turned every train target into NaN or another
row's value.
"""

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures import (
    tabpfn_v2,
    tabpfn_v2_5,
    tabpfn_v2_6,
    tabpfn_v3,
    tabpfn_v3_5,
)

_ARCHITECTURES = [tabpfn_v2, tabpfn_v2_5, tabpfn_v2_6, tabpfn_v3, tabpfn_v3_5]
_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize(
    "architecture", _ARCHITECTURES, ids=lambda m: m.__name__.rsplit(".", 1)[-1]
)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("num_train", [65535, 65536, 70000])
@pytest.mark.parametrize("num_test", [0, 8])
def test__prepare_targets__large_train_set__keeps_targets_and_nan_pads_test_rows(
    architecture, device: str, num_train: int, num_test: int
) -> None:
    y = torch.randn(num_train)
    out = architecture._prepare_targets(y.to(device), num_train + num_test, 1).cpu()
    assert out.shape == (num_train + num_test, 1, 1)
    torch.testing.assert_close(out[:num_train, 0, 0], y, rtol=0, atol=0)
    assert out[num_train:].isnan().all()
