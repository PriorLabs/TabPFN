#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: ANN001, B023, D103, E501, FBT003, T201
"""Diagnostics: is the regression border remap deterministic on MPS?"""

from __future__ import annotations

import hashlib
import traceback

import numpy as np
import torch
from sklearn.datasets import make_regression

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.utils import (
    _apply_remap_weights,
    _cached_remap_weights,
    _grid_key,
    translate_probs_across_borders,
)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[
        :12
    ]


def distinct(fn, n: int = 30) -> str:
    outs = [fn().cpu() for _ in range(n)]
    worst = max(float((o.double() - outs[0].double()).abs().max()) for o in outs)
    return f"distinct={len({sha(o) for o in outs})}/{n} max_dev={worst:.3e}"


def main() -> None:
    print(f"torch {torch.__version__}", flush=True)
    X, y = make_regression(n_samples=40, n_features=5, random_state=42)
    torch.manual_seed(42)
    np.random.seed(42)  # noqa: NPY002
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3, device="mps", n_estimators=4, fit_mode="fit_with_cache"
    )
    model.fit(X, y)
    captured = list(model._iter_forward_executor(X, use_inference_mode=True))
    to = model.znorm_space_bardist_.borders
    for i, (borders_t, out) in enumerate(captured):
        frm = torch.as_tensor(borders_t, device=out.device)
        same = frm.shape == to.shape and torch.equal(frm.to(to.dtype), to)
        print(
            f"est[{i}] out {tuple(out.shape)} {out.dtype} {out.device} identity_grid={same}",
            flush=True,
        )
        print(
            "  translate (mps):",
            distinct(lambda: translate_probs_across_borders(out, frm=frm, to=to)),
            flush=True,
        )
        out_cpu, frm_cpu, to_cpu = out.cpu(), frm.cpu(), to.cpu()
        print(
            "  translate (cpu):",
            distinct(
                lambda: translate_probs_across_borders(out_cpu, frm=frm_cpu, to=to_cpu)
            ),
            flush=True,
        )
        if same:
            continue
        w = _cached_remap_weights(_grid_key(frm), _grid_key(to), out.dtype, out.device)
        ppd = w.pairs_per_destination
        print(
            f"  pairs={w.source.numel()} max_pairs_per_dest={int(ppd.max())} dest_with_>1_pair={int((ppd > 1).sum())}",
            flush=True,
        )
        probs = torch.softmax(out, -1)
        print(
            "  index_add_ only (mps):",
            distinct(
                lambda: torch.zeros(
                    out.shape[0], to.numel() - 1, device=out.device
                ).index_add_(
                    1, w.destination, probs.index_select(1, w.source).mul_(w.weight)
                )
            ),
            flush=True,
        )
        try:
            print(
                "  segment_reduce (mps):",
                distinct(
                    lambda: (
                        torch.segment_reduce(
                            probs.T.index_select(0, w.source).mul_(w.weight[:, None]),
                            "sum",
                            lengths=w.pairs_per_destination,
                            axis=0,
                        ).T
                    )
                ),
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  segment_reduce (mps) unavailable: {type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
        try:
            torch.use_deterministic_algorithms(True)
            print(
                "  index_add_ under use_deterministic_algorithms (mps):",
                distinct(lambda: _apply_remap_weights(out, w)),
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  deterministic mode error: {type(e).__name__}: {str(e)[:200]}",
                flush=True,
            )
        finally:
            torch.use_deterministic_algorithms(False)
        # Same arithmetic on CPU, compared with the MPS result.
        wc = _cached_remap_weights(
            _grid_key(frm), _grid_key(to), out.dtype, torch.device("cpu")
        )
        r_cpu = _apply_remap_weights(out_cpu, wc)
        r_mps = _apply_remap_weights(out, w).cpu()
        print(
            f"  mps vs cpu max_abs={float((r_cpu.double() - r_mps.double()).abs().max()):.3e}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
