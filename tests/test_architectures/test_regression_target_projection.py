# Copyright (c) Prior Labs GmbH 2026.

"""Regression target projections must not require a runtime compiler."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.torch_version import TorchVersion

from tests.test_architectures.test_tabpfn_v3 import _get_regression_model
from tests.test_architectures.test_tabpfn_v3_5 import _get_model


@pytest.mark.parametrize("version", ["v3", "v3.5"])
def test_regression_target_projection(version: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (_get_regression_model() if version == "v3" else _get_model()).to(device)
    targets = torch.randn(1002, 8, device=device).T.requires_grad_()
    kwargs = {} if version == "v3" else {"task_type": "regression"}
    for stage in ("col", "icl"):
        project = getattr(model, f"_embed_{stage}_y")
        expected = torch.cat([project(row[None], **kwargs) for row in targets])
        expected_grad = torch.autograd.grad(expected.square().sum(), targets)[0]
        actual = project(targets, **kwargs)
        actual_grad = torch.autograd.grad(actual.square().sum(), targets)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.skipif(
    TorchVersion(torch.__version__) < TorchVersion("2.13"),
    reason="Eager Triton outer-product dispatch requires Torch 2.13+",
)
def test_regression_target_projection_without_compiler(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_architectures.test_regression_target_projection",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "TRITON_CACHE_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_without_compiler() -> None:
    from triton.runtime import build  # noqa: PLC0415

    def reject_compilation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Regression target encoding requested a C compiler")

    build._build = reject_compilation
    targets = torch.randn(1002, 8, device="cuda").T
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for version in ("v3", "v3.5"):
            model = (
                _get_regression_model() if version == "v3" else _get_model()
            ).cuda()
            kwargs = {} if version == "v3" else {"task_type": "regression"}
            for stage in ("col", "icl"):
                getattr(model, f"_embed_{stage}_y")(targets, **kwargs)
        torch.cuda.synchronize()


if __name__ == "__main__":
    _check_without_compiler()
