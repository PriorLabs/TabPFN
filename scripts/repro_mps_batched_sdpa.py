#  Copyright (c) Prior Labs GmbH 2026.
"""Reproduce the pytorch/pytorch#170837 MPS SDPA dispatch bug.

Per @jhavukainen on that issue, MPS's `scaled_dot_product_attention` dispatches
to an internal `sdpa_vector_attention` kernel under certain conditions, and
that kernel mishandles non-contiguous batched q/k/v: the first batch element
is correct, subsequent batch elements diverge. Forcing the inputs to be
contiguous before SDPA routes around the bug (the alternative MPSGraph path
is used). PR #170874 proposed the same dispatch-side fix in pytorch but went
stale and was never merged.

This script demonstrates the mechanism using a hand-rolled BERT-style fused
QKV projection (which is what produces the non-contiguous q/k/v in the wild).
Two outputs are compared against a CPU reference, per batch element, at a
range of shapes:

  permute:    q/k/v are non-contiguous views of the fused QKV tensor — the
              shape pytorch SDPA dispatches into the buggy kernel.
  contiguous: same logical computation but with `.contiguous()` calls between
              the permute and SDPA. Should always match CPU.

Run:
  python scripts/repro_mps_batched_sdpa.py

Exits 0 if MPS SDPA matches CPU everywhere, 1 if any case diverges by more
than `TOL`. The intent is to fail on a buggy environment (e.g. the GitHub
`macos-26-arm64` runner) and pass on a healthy one.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F  # noqa: N812

TOL = 1e-3
# Shapes derived from common transformer configurations; the bug is sensitive
# to shape so we sweep a few.
CONFIGS = [
    # (batch, seq_len, d_model, num_heads, label)
    (2, 7, 768, 12, "BERT-base-ish, seq=7, batch=2"),
    (2, 64, 768, 12, "BERT-base-ish, seq=64, batch=2"),
    (4, 128, 512, 8, "GPT-small-ish, seq=128, batch=4"),
    (8, 32, 256, 8, "TabPFN-ish, seq=32, batch=8"),
    (2, 7, 64, 4, "tiny, seq=7, batch=2"),
]


def attn_permute(
    x: torch.Tensor, qkv_weight: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Fused-QKV BERT pattern. q/k/v reach SDPA as non-contiguous views."""
    batch, seq, d_model = x.shape
    head_dim = d_model // num_heads
    qkv = x @ qkv_weight  # [B, L, 3*D]
    qkv = qkv.view(batch, seq, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each [B, H, L, head_dim], non-contiguous
    return F.scaled_dot_product_attention(q, k, v)


def attn_contiguous(
    x: torch.Tensor, qkv_weight: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Same computation; force q/k/v contiguous so SDPA picks the safe path."""
    batch, seq, d_model = x.shape
    head_dim = d_model // num_heads
    qkv = x @ qkv_weight
    qkv = qkv.view(batch, seq, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return F.scaled_dot_product_attention(q, k, v)


def run_case(
    device: str, batch: int, seq: int, d_model: int, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both attention paths on one device, return (permute, contiguous)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    # One canonical sequence, repeated across the batch — every batch element
    # is identical so any per-element divergence is purely an SDPA bug, not
    # genuine input-dependent variation.
    x_single = torch.randn(1, seq, d_model, generator=g)
    weight = torch.randn(d_model, 3 * d_model, generator=g) / d_model**0.5
    x = x_single.repeat(batch, 1, 1).to(device)
    weight = weight.to(device)
    return (
        attn_permute(x, weight, num_heads).cpu(),
        attn_contiguous(x, weight, num_heads).cpu(),
    )


def main() -> int:
    """Run the SDPA dispatch sweep, return exit code."""
    if not torch.backends.mps.is_available():
        print("MPS not available — skipping")  # noqa: T201
        return 0

    fails = 0
    for batch, seq, d_model, num_heads, label in CONFIGS:
        print(  # noqa: T201
            f"\n=== {label} (shape B={batch} L={seq} D={d_model} H={num_heads}) ==="
        )
        cpu_perm, cpu_cont = run_case("cpu", batch, seq, d_model, num_heads)
        mps_perm, mps_cont = run_case("mps", batch, seq, d_model, num_heads)

        # CPU permute and CPU contiguous must agree — sanity, never the bug.
        cpu_diff = (cpu_perm - cpu_cont).abs().max().item()
        print(  # noqa: T201
            f"  sanity: cpu permute vs cpu contiguous     max_abs_diff={cpu_diff:.3e}"
        )

        # Per batch element: does MPS match CPU?
        worst_perm, worst_cont = 0.0, 0.0
        for i in range(batch):
            d_perm = (cpu_perm[i] - mps_perm[i]).abs().max().item()
            d_cont = (cpu_cont[i] - mps_cont[i]).abs().max().item()
            worst_perm = max(worst_perm, d_perm)
            worst_cont = max(worst_cont, d_cont)
            flag_p = " <-- DIVERGED" if d_perm > TOL else ""
            flag_c = " <-- DIVERGED" if d_cont > TOL else ""
            print(  # noqa: T201
                f"  batch[{i}]: mps permute    vs cpu  "
                f"max_abs_diff={d_perm:.3e}{flag_p}"
            )
            print(  # noqa: T201
                f"  batch[{i}]: mps contiguous vs cpu  "
                f"max_abs_diff={d_cont:.3e}{flag_c}"
            )

        # Self-consistency: every row of the batch had identical input, so the
        # MPS output rows must be identical to each other. They aren't, on the
        # buggy dispatch path — the smoking gun for #170837.
        spread_perm = (mps_perm - mps_perm[0:1]).abs().max().item()
        spread_cont = (mps_cont - mps_cont[0:1]).abs().max().item()
        smoking = " <-- BATCH ROWS DIFFER (smoking gun for #170837)"
        flag_sp = smoking if spread_perm > TOL else ""
        flag_sc = " <-- BATCH ROWS DIFFER" if spread_cont > TOL else ""
        print(  # noqa: T201
            f"  self-cons: mps permute    batch[0] vs others  "
            f"max_spread={spread_perm:.3e}{flag_sp}"
        )
        print(  # noqa: T201
            f"  self-cons: mps contiguous batch[0] vs others  "
            f"max_spread={spread_cont:.3e}{flag_sc}"
        )

        if max(worst_perm, worst_cont, spread_perm, spread_cont) > TOL:
            fails += 1

    print(f"\nResult: {fails} failing config(s) of {len(CONFIGS)}")  # noqa: T201
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
