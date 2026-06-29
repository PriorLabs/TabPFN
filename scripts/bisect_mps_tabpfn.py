#  Copyright (c) Prior Labs GmbH 2026.
"""Bisect a TabPFN forward pass to find the first MPS divergence vs CPU.

The macos-26 runner produces silently-wrong TabPFN predictions but a per-op
MPS-vs-CPU ladder is clean. This script narrows it down by running the same
fit+predict twice — once on CPU, once on MPS — with a forward_hook on every
nn.Module in the loaded architecture. For each module we capture BOTH its
input tensors and its output tensor, recorded in call order; afterwards CPU
and MPS lists are compared entry-by-entry. The first tensor whose relative
max-abs-diff exceeds TOL is the entry point to look at — and if it's an
input (kind="in[i]") the bug is upstream of that module, while if it's an
output (kind="out") the bug is inside that module's forward.

We force `inference_precision=torch.float32` and `n_estimators=1` to keep the
forward deterministic and one-shot; if the bug needs the ensemble pattern or
mixed precision, that itself is a clue (try TabPFN-v2 default config instead).

Run:
  python scripts/bisect_mps_tabpfn.py [v2|v2_5|v2_6|v3]   # defaults to v2

Exits 0 if all modules agree to within TOL on MPS vs CPU, 1 if any diverge.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import torch
from sklearn.datasets import make_classification

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

TOL = 1e-3
MAX_ROWS_PRINTED = 80
VERSION_MAP = {
    "v2": "V2",
    "v2_5": "V2_5",
    "v2_6": "V2_6",
    "v3": "V3",
}


def run_with_hooks(  # noqa: C901
    device: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    version: str,
) -> tuple[list[tuple[int, str, str, torch.Tensor]], np.ndarray]:
    """Fit+predict on device, return (captured activations, proba)."""
    model_version = getattr(ModelVersion, VERSION_MAP[version])
    clf = TabPFNClassifier.create_default_for_version(
        model_version,
        device=device,
        n_estimators=1,
        inference_precision=torch.float32,
        # Disable memory-saving chunking so CPU and MPS execute the same
        # whole-batch forward and we can pair hook outputs by index.
        memory_saving_mode=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_train, y_train)

    # Each entry: (call_idx, module_name, kind, tensor)
    # `kind` is "in[i]" for the i-th positional input, "out" for a single
    # output tensor, or "out[i]" for the i-th element of a tuple/list output.
    # Capturing both inputs and outputs lets us see whether a divergence at
    # module N is caused by N's own forward or by a wrong input handed to it.
    captured: list[tuple[int, str, str, torch.Tensor]] = []
    counter = [0]

    def stash(name: str, kind: str, t: torch.Tensor) -> None:
        captured.append((counter[0], name, kind, t.detach().cpu().float().clone()))
        counter[0] += 1

    def make_hook(name: str):  # noqa: ANN202
        def hook(_module: torch.nn.Module, inp, out) -> None:  # noqa: ANN001
            for i, t in enumerate(inp):
                if isinstance(t, torch.Tensor):
                    stash(name, f"in[{i}]", t)
            if isinstance(out, torch.Tensor):
                stash(name, "out", out)
            elif isinstance(out, (tuple, list)):
                for i, t in enumerate(out):
                    if isinstance(t, torch.Tensor):
                        stash(name, f"out[{i}]", t)

        return hook

    nn_model = clf.models_[0]
    handles = []
    for name, sub in nn_model.named_modules():
        if name:  # skip root
            handles.append(sub.register_forward_hook(make_hook(name)))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = clf.predict_proba(x_test)
    finally:
        for h in handles:
            h.remove()

    return captured, proba


def diff_summary(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_diff, relative_diff) between two equally-shaped tensors."""
    diff = (a - b).abs().max().item()
    norm = a.abs().max().item() or 1.0
    return diff, diff / norm


def main(argv: list[str]) -> int:
    """Run bisection, return exit code."""
    version = argv[1] if len(argv) > 1 else "v2"
    if version not in VERSION_MAP:
        print(f"unknown version {version!r}; choose from {list(VERSION_MAP)}")  # noqa: T201
        return 2

    if not torch.backends.mps.is_available():
        print("MPS not available — skipping")  # noqa: T201
        return 0

    # Small deterministic 3-class problem — close to the failing test.
    x, y = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=0,
    )
    x_train, x_test = x[:60], x[60:]
    y_train = y[:60]

    print(f"=== TabPFN-{version} CPU forward ===")  # noqa: T201
    cpu_act, cpu_proba = run_with_hooks("cpu", x_train, y_train, x_test, version)
    print(f"  captured {len(cpu_act)} tensors (inputs + outputs)")  # noqa: T201

    print(f"=== TabPFN-{version} MPS forward ===")  # noqa: T201
    mps_act, mps_proba = run_with_hooks("mps", x_train, y_train, x_test, version)
    print(f"  captured {len(mps_act)} tensors (inputs + outputs)")  # noqa: T201

    if len(cpu_act) != len(mps_act):
        print(  # noqa: T201
            f"  WARNING: CPU and MPS captured different counts ({len(cpu_act)} vs "
            f"{len(mps_act)}); pairing by index up to the shorter length."
        )

    print(  # noqa: T201
        f"\n=== per-module CPU vs MPS divergence (sorted by call order) ===\n"
        f"{'#':>4s} {'module':50s} {'kind':6s} {'shape':25s} "
        f"{'max_abs':>11s} {'rel':>11s}"
    )
    first_diverge: tuple[int, str, str] | None = None
    n_compared = min(len(cpu_act), len(mps_act))
    printed = 0
    for idx in range(n_compared):
        c_idx, c_name, c_kind, c_tensor = cpu_act[idx]
        _, m_name, m_kind, m_tensor = mps_act[idx]
        if c_name != m_name or c_kind != m_kind or c_tensor.shape != m_tensor.shape:
            print(  # noqa: T201
                f"  MISALIGN at idx {idx}: "
                f"cpu='{c_name}/{c_kind}' {tuple(c_tensor.shape)} "
                f"vs mps='{m_name}/{m_kind}' {tuple(m_tensor.shape)}"
            )
            break
        abs_diff, rel_diff = diff_summary(c_tensor, m_tensor)
        flag = ""
        if rel_diff > TOL:
            if first_diverge is None:
                flag = "  <-- FIRST DIVERGENCE"
                first_diverge = (idx, c_name, c_kind)
            else:
                flag = "  <-- diverged"
        # Always print divergences; cap clean rows to keep output readable.
        if rel_diff > TOL or printed < MAX_ROWS_PRINTED:
            print(  # noqa: T201
                f"{c_idx:>4d} {c_name:50s} {c_kind:6s} "
                f"{tuple(c_tensor.shape)!s:25s} "
                f"{abs_diff:11.3e} {rel_diff:11.3e}{flag}"
            )
            printed += 1

    proba_diff = float(np.abs(cpu_proba - mps_proba).max())
    print(  # noqa: T201
        f"\nfinal predict_proba max_abs_diff: {proba_diff:.3e}  "
        f"(cpu argmax={cpu_proba.argmax(axis=1).tolist()})  "
        f"(mps argmax={mps_proba.argmax(axis=1).tolist()})"
    )

    if first_diverge is not None:
        idx, name, kind = first_diverge
        explanation = (
            "INPUT already wrong → bug is UPSTREAM of this module"
            if kind.startswith("in[")
            else "input clean → bug is INSIDE this module's forward"
        )
        print(  # noqa: T201
            f"\nFIRST DIVERGENT TENSOR: call #{idx} '{name}/{kind}' — {explanation}"
        )
        return 1
    print("\nAll modules agree to within tolerance.")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
