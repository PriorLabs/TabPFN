#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.architectures.shared.workaround_mps_pad_bug."""

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.shared.workaround_mps_pad_bug import pad_rows_with_nan

_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape", [(5,), (5, 2), (5, 2, 1)])
@pytest.mark.parametrize("num_pad_rows", [0, 3])
def test__pad_rows_with_nan__any_shape__appends_nan_rows(
    device: str, shape: tuple[int, ...], num_pad_rows: int
) -> None:
    x = torch.randn(*shape)
    out = pad_rows_with_nan(x.to(device), shape[0] + num_pad_rows).cpu()
    assert out.shape == (shape[0] + num_pad_rows, *shape[1:])
    torch.testing.assert_close(out[: shape[0]], x, rtol=0, atol=0)
    assert out[shape[0] :].isnan().all()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("num_rows", [65535, 65536, 70000])
def test__pad_rows_with_nan__at_least_2_pow_16_rows__keeps_input_rows(
    device: str, num_rows: int
) -> None:
    x = torch.randn(num_rows, 1, 1)
    out = pad_rows_with_nan(x.to(device), num_rows + 8).cpu()
    torch.testing.assert_close(out[:num_rows], x, rtol=0, atol=0)
    assert out[num_rows:].isnan().all()
