#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the ``enable_torch_compile`` performance option.

Compilation is only useful if the traced regions stay whole: a graph break
splits the region into separately-compiled fragments, silently giving back
most of what compiling bought. Breaks do not change results, so numerical
tests cannot see them — these tests read Dynamo's own graph-break log.

Dynamo is exercised without Inductor (``backend="eager"``): tracing, graph
breaks and guards all happen, but no C++ toolchain is needed and the test
stays fast.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import torch

import tabpfn
from tabpfn.architectures import tabpfn_v3
from tabpfn.architectures.interface import PerformanceOptions

pytestmark = pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile (Dynamo) is not supported on this platform/Python",
)

# A graph break is attributed to the innermost user frame of its traceback;
# a break deeper in torch merely passes through tabpfn's frames.
_USER_FRAME = re.compile(r'File "([^"]+)", line \d+, in (\w+)')

# torch's own SDPA-backend context manager is not traceable on every torch
# build. It is torch's to fix, so breaks it causes are not tabpfn's fault.
_TORCH_OWNED = ("sdpa_kernel",)

_TABPFN_ROOT = str(Path(tabpfn.__file__).parent)


def _tiny_model() -> tabpfn_v3.TabPFNV3:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=10,
        num_buckets=5,
        embed_dim=48,
        nlayers=1,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float32)
    return model


def _compiled_forward_graph_breaks(
    model: tabpfn_v3.TabPFNV3,
    x: torch.Tensor,
    y: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[torch.Tensor, list[str]]:
    """Run a compiled forward; return its output and Dynamo's break log."""
    # Dynamo only: this asserts tracing behaviour, and skipping Inductor
    # keeps the test toolchain-free and fast.
    eager_compile = torch.compile

    def compile_without_inductor(*args: object, **kwargs: object):  # noqa: ANN202
        kwargs["backend"] = "eager"
        return eager_compile(*args, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_without_inductor)

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Capture()
    torch._logging.set_logs(graph_breaks=True)
    logging.getLogger("torch._dynamo").addHandler(handler)
    try:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        with torch.no_grad():
            out = model(
                x, y, performance_options=PerformanceOptions(enable_torch_compile=True)
            )
    finally:
        logging.getLogger("torch._dynamo").removeHandler(handler)
        torch._logging.set_logs()

    return out, records


@torch.no_grad()
def test__enable_torch_compile__no_graph_break_in_tabpfn_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiled regions must not break on tabpfn's own code.

    Anything the model does per attention call that Dynamo cannot trace —
    an import, a registry lookup, a Python-level probe — fragments every
    compiled stage. The output stays correct, so only the break log shows
    it.
    """
    model = _tiny_model()
    torch.manual_seed(0)
    x = torch.randn(30, 2, 5, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [15, 2], dtype=torch.float32)

    out_eager = model(x, y)
    out_compiled, records = _compiled_forward_graph_breaks(model, x, y, monkeypatch)

    torch.testing.assert_close(out_compiled, out_eager, atol=1e-5, rtol=1e-5)
    # A silent fall-back to eager would make the assertions below vacuous.
    assert torch._dynamo.utils.counters["frames"]["ok"] > 0, "nothing was compiled"

    offenders = []
    for block in "\n".join(records).split("Graph break in user code")[1:]:
        frames = _USER_FRAME.findall(block)
        if not frames or any(owner in block for owner in _TORCH_OWNED):
            continue
        path, function = frames[-1]
        if path.startswith(_TABPFN_ROOT):
            offenders.append(f"{Path(path).name}::{function}")
    assert not offenders, f"graph breaks inside tabpfn code: {sorted(set(offenders))}"
