#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.architectures.shared.workaround_mps_linear_bug."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tabpfn.architectures.shared import workaround_mps_linear_bug
from tabpfn.architectures.shared.workaround_mps_linear_bug import (
    MpsSafeLinear,
    maybe_replace_linears_on_mps,
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


def test__maybe_replace_linears_on_mps__noop_when_not_buggy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workaround_mps_linear_bug, "_mps_linear_bias_is_buggy", lambda: False
    )
    model = _make_model()
    maybe_replace_linears_on_mps(model)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert all(type(m) is nn.Linear for m in linears)


def test__maybe_replace_linears_on_mps__retypes_only_linears_with_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workaround_mps_linear_bug, "_mps_linear_bias_is_buggy", lambda: True
    )
    model = _make_model()
    maybe_replace_linears_on_mps(model)

    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    with_bias = [m for m in linears if m.bias is not None]
    without_bias = [m for m in linears if m.bias is None]

    assert with_bias, "expected at least one bias-enabled Linear"
    assert all(isinstance(m, MpsSafeLinear) for m in with_bias)
    assert all(type(m) is nn.Linear for m in without_bias)


def test__maybe_replace_linears_on_mps__on_cpu__does_not_affect_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On CPU the swapped model must produce identical output (fall-through path)."""
    monkeypatch.setattr(
        workaround_mps_linear_bug, "_mps_linear_bias_is_buggy", lambda: True
    )
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(*_TRIGGER_SHAPE)
    before = model(x)

    maybe_replace_linears_on_mps(model)
    after = model(x)

    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test__MpsSafeLinear__on_cpu__matches_built_in_linear() -> None:
    torch.manual_seed(0)
    plain = nn.Linear(2, 192, bias=True)
    safe = MpsSafeLinear(2, 192, bias=True)
    safe.load_state_dict(plain.state_dict())
    x = torch.randn(*_TRIGGER_SHAPE)
    torch.testing.assert_close(safe(x), plain(x), rtol=0, atol=0)
