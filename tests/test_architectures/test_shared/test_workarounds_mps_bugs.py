#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.architectures.shared.workarounds_mps_bugs."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tabpfn.architectures.shared import workarounds_mps_bugs
from tabpfn.architectures.shared.workarounds_mps_bugs import (
    MpsSafeLinear,
    maybe_replace_linears_on_mps,
    pad_rows_with_nan,
)

# Shape known to trigger the macOS-26 / M1 bug: >2D with a unary dim, small in_features.
_TRIGGER_SHAPE = (80, 1, 2)


def _make_model() -> nn.Module:
    """A module mixing a bias-enabled Linear, a bias=False Linear, and a nested one."""
    return nn.Sequential(
        nn.Linear(2, 8, bias=True),
        nn.GELU(),
        nn.Linear(8, 4, bias=False),
        nn.Sequential(nn.Linear(4, 3, bias=True)),
    )


def test__maybe_replace_linears_on_mps__mps_not_selected__does_not_test_for_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workarounds_mps_bugs,
        "_mps_linear_bias_is_buggy",
        lambda: pytest.fail("self-test must not run without an MPS device"),
    )
    model = _make_model()
    maybe_replace_linears_on_mps(model, [torch.device("cpu")])
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert all(type(m) is nn.Linear for m in linears)


def test__maybe_replace_linears_on_mps__bug_not_present__model_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workarounds_mps_bugs, "_mps_linear_bias_is_buggy", lambda: False
    )
    model = _make_model()
    maybe_replace_linears_on_mps(model, [torch.device("mps")])
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert all(type(m) is nn.Linear for m in linears)


def test__maybe_replace_linears_on_mps__retypes_only_bias_linears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workarounds_mps_bugs, "_mps_linear_bias_is_buggy", lambda: True)
    model = _make_model()
    maybe_replace_linears_on_mps(model, [torch.device("mps")])

    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    with_bias = [m for m in linears if m.bias is not None]
    without_bias = [m for m in linears if m.bias is None]

    assert with_bias, "expected at least one bias-enabled Linear"
    assert all(isinstance(m, MpsSafeLinear) for m in with_bias)
    assert all(type(m) is nn.Linear for m in without_bias)


def test__maybe_replace_linears_on_mps__force_replac__output_unchanged_when_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workarounds_mps_bugs, "_mps_linear_bias_is_buggy", lambda: True)
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(*_TRIGGER_SHAPE)
    before = model(x)

    maybe_replace_linears_on_mps(model, [torch.device("mps")])
    after = model(x)

    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test__MpsSafeLinear__on_cpu__matches_built_in_linear() -> None:
    torch.manual_seed(0)
    plain = nn.Linear(2, 192, bias=True)
    safe = MpsSafeLinear(2, 192, bias=True)
    safe.load_state_dict(plain.state_dict())
    x = torch.randn(*_TRIGGER_SHAPE)
    torch.testing.assert_close(safe(x), plain(x), rtol=0, atol=0)


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
