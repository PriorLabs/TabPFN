#  Copyright (c) Prior Labs GmbH 2026.
"""Repro: nn.Linear with bias=True gives wrong output on MPS on macOS 26.

The same shape with bias=False is correct, and the bias=True variant is
correct on CPU and on the macos-15 runner. Exits non-zero if any bias=True
variant diverges from CPU by more than TOL.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

TOL = 1e-3
# (in_features, out_features, bias, input_shape)
VARIANTS = [
    (2, 192, True, (80, 1, 2)),
    (2, 192, False, (80, 1, 2)),
    (4, 192, True, (80, 1, 4)),
    (2, 768, True, (80, 1, 2)),
]


def main() -> int:
    """Run the variant sweep, return exit code."""
    if not torch.backends.mps.is_available():
        return 0
    fails = 0
    for in_f, out_f, bias, shape in VARIANTS:
        torch.manual_seed(0)
        cpu = nn.Linear(in_f, out_f, bias=bias)
        mps = nn.Linear(in_f, out_f, bias=bias).to("mps")
        mps.load_state_dict(cpu.state_dict())
        x = torch.randn(*shape)
        rel = (
            (cpu(x) - mps(x.to("mps")).cpu()).abs().max() / cpu(x).abs().max()
        ).item()
        flag = "  <-- DIVERGED" if rel > TOL else ""
        print(  # noqa: T201
            f"Linear({in_f},{out_f}) bias={bias!s:5s} input={shape!s:14s}"
            f"  rel={rel:.3e}{flag}"
        )
        if rel > TOL:
            fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
