# Copyright (c) Prior Labs GmbH 2026.

"""Regression target projections must not require a runtime compiler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.test_architectures.test_tabpfn_v3 import _get_regression_model
from tests.test_architectures.test_tabpfn_v3_5 import _get_model


class _RejectBmm(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        assert func is not torch.ops.aten.bmm.default, "Target projection reached BMM"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("version", ["v3", "v3.5"])
@pytest.mark.parametrize("stage", ["col", "icl"])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_regression_target_projection(
    version: str,
    stage: Literal["col", "icl"],
    batch: int,
    contiguous: bool,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = (_get_regression_model() if version == "v3" else _get_model()).to(device)
    targets = torch.randn(1002, batch, device=device).T
    if contiguous:
        targets = targets.contiguous()
    targets.requires_grad_()
    project = getattr(model, f"_embed_{stage}_y")
    kwargs = {} if version == "v3" else {"task_type": "regression"}
    expected = torch.cat([project(row[None], **kwargs) for row in targets])
    expected_grad = torch.autograd.grad(expected.square().sum(), targets)[0]
    with _RejectBmm():
        actual = project(targets, **kwargs)
        actual_grad = torch.autograd.grad(actual.square().sum(), targets)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
