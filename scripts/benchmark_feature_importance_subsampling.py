"""Benchmark feature-importance subsampling against naive methods.

Dataset
-------
Binary classification: ``n_informative`` truly predictive features embedded in
a fixed space of ``N_TOTAL_FEATURES`` columns (rest are pure noise).

We vary ``n_informative`` across [10, 50, 100] and compare three subsampling
strategies that each see only ``MAX_FEATURES_PER_EST`` features per estimator:

  feature_importance  ExtraTrees ranks features; top-50 always included,
                      remaining budget filled randomly.
  balanced            Round-robin random subsampling (TabPFN default).
  random              Independent random draw per estimator.

Each data point is the mean ± std accuracy from 3-fold stratified CV.

Usage
-----
    python scripts/benchmark_feature_importance_subsampling.py

Runtime
-------
~5-15 min on a single GPU (5k samples × 9 fits × 3 folds).
The plot is saved to scripts/feature_importance_subsampling_benchmark.png.
"""

from __future__ import annotations

import dataclasses
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless cluster nodes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from tabpfn import TabPFNClassifier

# ── experiment settings ────────────────────────────────────────────────────────
N_SAMPLES = 5_000
N_TOTAL_FEATURES = 300      # total columns; n_informative are signal, rest noise
MAX_FEATURES_PER_EST = 100  # subsampling budget per estimator
TOP_K_COUNT = 50            # always include 50 most important (feature_importance)
N_INFORMATIVE_LIST = [10, 50, 100]
N_ESTIMATORS = 4
N_CV_FOLDS = 3
RANDOM_SEED = 0
IMPORTANCE_N_FOLDS = 2      # ExtraTrees CV folds for importance estimation
# ──────────────────────────────────────────────────────────────────────────────


def _make_inference_config(method: str, base_transforms: list) -> dict:
    """Build the inference_config dict for a given subsampling method.

    Copies the model's default PREPROCESS_TRANSFORMS but lowers
    max_features_per_estimator to force subsampling.
    """
    modified_transforms = [
        dataclasses.replace(p, max_features_per_estimator=MAX_FEATURES_PER_EST)
        for p in base_transforms
    ]
    cfg: dict = {
        "FEATURE_SUBSAMPLING_METHOD": method,
        "PREPROCESS_TRANSFORMS": modified_transforms,
    }
    if method == "feature_importance":
        cfg["FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT"] = TOP_K_COUNT
        cfg["FEATURE_SUBSAMPLING_IMPORTANCE_N_FOLDS"] = IMPORTANCE_N_FOLDS
    return cfg


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    base_transforms: list,
) -> tuple[float, float]:
    """Return (mean, std) accuracy across stratified CV folds."""
    cfg = _make_inference_config(method, base_transforms)
    kf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for fold_i, (tr, te) in enumerate(kf.split(X, y)):
        clf = TabPFNClassifier(
            n_estimators=N_ESTIMATORS,
            inference_config=cfg,
            random_state=RANDOM_SEED + fold_i,
        )
        clf.fit(X[tr], y[tr])
        scores.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    print("Loading TabPFN model to get default inference config...")
    base_transforms = TabPFNClassifier().get_inference_config().PREPROCESS_TRANSFORMS
    print(
        f"Default preprocessors: {[p.name for p in base_transforms]}, "
        f"max_features overridden → {MAX_FEATURES_PER_EST}\n"
    )

    methods: dict[str, dict] = {
        "feature_importance": {
            "label": (
                f"feature_importance  "
                f"(top-{TOP_K_COUNT} fixed + {MAX_FEATURES_PER_EST - TOP_K_COUNT} random)"
            ),
            "color": "C0",
            "ls": "-",
        },
        "balanced": {
            "label": f"balanced  (round-robin random, budget={MAX_FEATURES_PER_EST})",
            "color": "C1",
            "ls": "--",
        },
        "random": {
            "label": f"random  (independent random, budget={MAX_FEATURES_PER_EST})",
            "color": "C2",
            "ls": ":",
        },
    }

    results: dict[str, dict[str, list]] = {
        m: {"mean": [], "std": []} for m in methods
    }

    for n_inf in N_INFORMATIVE_LIST:
        print(f"── n_informative={n_inf}  ({N_TOTAL_FEATURES - n_inf} noise features) ──")
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_TOTAL_FEATURES,
            n_informative=n_inf,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=2,
            random_state=RANDOM_SEED,
        )
        for method in methods:
            t0 = time.time()
            mean, std = evaluate(X, y, method, base_transforms)
            elapsed = time.time() - t0
            results[method]["mean"].append(mean)
            results[method]["std"].append(std)
            print(f"  {method:<22} acc={mean:.3f} ± {std:.3f}  ({elapsed:.0f}s)")
        print()

    # ── plot ──────────────────────────────────────────────────────────────────
    xs = np.asarray(N_INFORMATIVE_LIST)
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, meta in methods.items():
        means = np.asarray(results[method]["mean"])
        stds = np.asarray(results[method]["std"])
        ax.plot(
            xs, means,
            color=meta["color"], ls=meta["ls"],
            marker="o", lw=2.0, label=meta["label"],
        )
        ax.fill_between(
            xs, means - stds, means + stds,
            alpha=0.12, color=meta["color"],
        )

    ax.set_xlabel("Number of informative features  (rest are noise)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Feature subsampling: importance vs naive\n"
        f"{N_TOTAL_FEATURES} total features · {MAX_FEATURES_PER_EST} features/estimator · "
        f"{N_SAMPLES} samples · {N_ESTIMATORS} estimators",
        fontsize=11,
    )
    ax.set_xticks(xs)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = "scripts/feature_importance_subsampling_benchmark.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    main()
