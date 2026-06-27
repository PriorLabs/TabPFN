#  Copyright (c) Prior Labs GmbH 2026.
"""Bisect a TabPFN forward pass to find the first MPS divergence vs CPU.

The macos-26 runner produces silently-wrong TabPFN predictions but a per-op
MPS-vs-CPU ladder is clean. This script narrows it down by running the same
fit+predict twice — once on CPU, once on MPS — with a forward_hook on every
nn.Module in the loaded architecture. Each module's output tensor is moved
to CPU in float32 and recorded in call order; afterwards CPU and MPS lists
are compared module-by-module. The first module whose relative max-abs-diff
exceeds TOL is the entry point to look at.

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


def run_with_hooks(
    device: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    version: str,
) -> tuple[list[tuple[int, str, torch.Tensor]], np.ndarray]:
    """Fit+predict on device, return (captured activations, proba)."""
    model_version = getattr(ModelVersion, VERSION_MAP[version])
    clf = TabPFNClassifier.create_default_for_version(
        model_version,
        device=device,
        n_estimators=1,
        inference_precision=torch.float32,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_train, y_train)

    captured: list[tuple[int, str, torch.Tensor]] = []
    counter = [0]

    def make_hook(name: str):  # noqa: ANN202
        def hook(_module: torch.nn.Module, _inp, out) -> None:  # noqa: ANN001
            if isinstance(out, torch.Tensor):
                captured.append((counter[0], name, out.detach().cpu().float().clone()))
                counter[0] += 1
            elif isinstance(out, (tuple, list)):
                for i, t in enumerate(out):
                    if isinstance(t, torch.Tensor):
                        captured.append(
                            (
                                counter[0],
                                f"{name}[{i}]",
                                t.detach().cpu().float().clone(),
                            )
                        )
                        counter[0] += 1

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
    print(f"  captured {len(cpu_act)} module outputs")  # noqa: T201

    print(f"=== TabPFN-{version} MPS forward ===")  # noqa: T201
    mps_act, mps_proba = run_with_hooks("mps", x_train, y_train, x_test, version)
    print(f"  captured {len(mps_act)} module outputs")  # noqa: T201

    if len(cpu_act) != len(mps_act):
        print(  # noqa: T201
            f"  WARNING: CPU and MPS captured different counts ({len(cpu_act)} vs "
            f"{len(mps_act)}); pairing by index up to the shorter length."
        )

    print(  # noqa: T201
        f"\n=== per-module CPU vs MPS divergence (sorted by call order) ===\n"
        f"{'#':>4s} {'module':50s} {'shape':25s} {'max_abs':>11s} {'rel':>11s}"
    )
    first_diverge: tuple[int, str] | None = None
    n_compared = min(len(cpu_act), len(mps_act))
    printed = 0
    for idx in range(n_compared):
        c_idx, c_name, c_tensor = cpu_act[idx]
        _, m_name, m_tensor = mps_act[idx]
        if c_name != m_name or c_tensor.shape != m_tensor.shape:
            print(  # noqa: T201
                f"  MISALIGN at idx {idx}: cpu='{c_name}' {tuple(c_tensor.shape)} "
                f"vs mps='{m_name}' {tuple(m_tensor.shape)}"
            )
            break
        abs_diff, rel_diff = diff_summary(c_tensor, m_tensor)
        flag = ""
        if rel_diff > TOL:
            if first_diverge is None:
                flag = "  <-- FIRST DIVERGENCE"
                first_diverge = (idx, c_name)
            else:
                flag = "  <-- diverged"
        # Always print divergences; cap clean rows to keep output readable.
        if rel_diff > TOL or printed < MAX_ROWS_PRINTED:
            print(  # noqa: T201
                f"{c_idx:>4d} {c_name:50s} {tuple(c_tensor.shape)!s:25s} "
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
        print(  # noqa: T201
            f"\nFIRST DIVERGENT MODULE: call #{first_diverge[0]} '{first_diverge[1]}'"
        )
        return 1
    print("\nAll modules agree to within tolerance.")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
