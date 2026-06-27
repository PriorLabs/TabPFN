#  Copyright (c) Prior Labs GmbH 2026.
"""Isolated repro: does nn.Linear(2, 192) produce wrong output on macos-26 MPS?

Bisecting TabPFN's forward pass localised the macos-26 MPS bug to
`target_embedder`, which is just an `nn.Linear(2, 192)` with bias=True
(src/tabpfn/architectures/tabpfn_v2.py:531). On macos-15 control, the same
module's CPU and MPS outputs were bit-identical (0.000e+00 diff); on macos-26
they diverged by ~0.96 absolute.

This script isolates that call: build the same Linear on CPU with a fixed seed,
copy its state_dict to an MPS instance so weights match exactly, then compare
outputs on the same input. Sweeps a few shape/bias variants for triangulation:

  - nn.Linear(2, 192, bias=True)    # exact target_embedder
  - nn.Linear(2, 192, bias=False)   # same shape minus bias
  - nn.Linear(6, 192, bias=False)   # feature_group_embedder (known-clean control)
  - nn.Linear(2, 192, bias=True) with input shape (80, 1, 2)    # exact call signature
  - nn.Linear(2, 192, bias=True) with input shape (80, 2)       # flat 2D variant
  - nn.Linear(2, 192, bias=True) with input shape (2,)          # 1D variant

If only the `nn.Linear(2, 192, bias=True)` variant diverges on macos-26, we have
a 10-line PyTorch reproducer with no TabPFN dependency.

Run:
  python scripts/repro_mps_linear.py

Exits 0 if all variants agree CPU vs MPS to within TOL, 1 if any diverge.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

TOL = 1e-3
# (label, in_features, out_features, bias, input_shape)
VARIANTS = [
    ("target_embedder exact", 2, 192, True, (80, 1, 2)),
    ("target_embedder no-bias", 2, 192, False, (80, 1, 2)),
    ("feature_group_embedder (clean ctrl)", 6, 192, False, (80, 14, 6)),
    ("target_embedder flat 2D input", 2, 192, True, (80, 2)),
    ("target_embedder 1D input", 2, 192, True, (2,)),
    ("Linear(2,192) bias=True large input", 2, 192, True, (1024, 2)),
    ("Linear(4,192) bias=True", 4, 192, True, (80, 1, 4)),
    ("Linear(2,768) bias=True", 2, 768, True, (80, 1, 2)),
]


def run_variant(
    in_f: int, out_f: int, *, bias: bool, input_shape: tuple[int, ...]
) -> tuple[float, float]:
    """Return (max_abs_diff, relative_diff) CPU vs MPS for one variant."""
    g = torch.Generator(device="cpu").manual_seed(0)
    # Build CPU Linear (deterministic weights).
    m_cpu = nn.Linear(in_f, out_f, bias=bias)
    # Re-seed and re-init so weight values are deterministic per variant.
    with torch.no_grad():
        m_cpu.weight.copy_(
            torch.randn(out_f, in_f, generator=g) / in_f**0.5,
        )
        if bias:
            m_cpu.bias.copy_(torch.randn(out_f, generator=g) * 0.1)
    # MPS Linear, weights copied from CPU exactly.
    m_mps = nn.Linear(in_f, out_f, bias=bias).to("mps")
    m_mps.load_state_dict(m_cpu.state_dict())

    x = torch.randn(*input_shape, generator=g)
    y_cpu = m_cpu(x)
    y_mps = m_mps(x.to("mps")).cpu()
    diff = (y_cpu - y_mps).abs().max().item()
    norm = y_cpu.abs().max().item() or 1.0
    return diff, diff / norm


def main() -> int:
    """Run the Linear variant sweep, return exit code."""
    if not torch.backends.mps.is_available():
        print("MPS not available — skipping")  # noqa: T201
        return 0

    print(  # noqa: T201
        f"{'variant':40s} {'shape':22s} {'bias':6s} {'max_abs':>11s} {'rel':>11s}"
    )
    fails = 0
    for label, in_f, out_f, bias, shape in VARIANTS:
        diff, rel = run_variant(in_f, out_f, bias=bias, input_shape=shape)
        flag = "  <-- DIVERGED" if rel > TOL else ""
        bias_s = "True" if bias else "False"
        print(  # noqa: T201
            f"{label:40s} {f'({in_f},{out_f})/{shape!s}':22s} {bias_s:6s} "
            f"{diff:11.3e} {rel:11.3e}{flag}"
        )
        if rel > TOL:
            fails += 1

    print(f"\nResult: {fails} divergent variant(s) of {len(VARIANTS)}")  # noqa: T201
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
