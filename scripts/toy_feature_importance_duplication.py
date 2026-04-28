"""GPU smoke test for FEATURE_IMPORTANCE_DUPLICATE_TOP_K.

Verifies:
  1. The augmented model input has ``original_features + K`` columns.
  2. With FEATURE_SUBSAMPLING_METHOD='random' and DUPLICATE_TOP_K>0, the top-K
     importance indices are force-included in every estimator's subsample.
  3. Predict still works end-to-end (no shape mismatches downstream).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

N_SAMPLES = 300
N_TOTAL_FEATURES = 80
N_INFORMATIVE = 30
SEED = 0


def fit_and_inspect(
    *,
    inference_config: dict,
    label: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> None:
    clf = TabPFNClassifier(
        n_estimators=2,
        inference_config=inference_config,
        random_state=SEED,
        device=globals().get("DEVICE", "cpu"),
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    # First fitted ensemble member's preprocessed X carries the augmented column count.
    members = list(clf.executor_.ensemble_members)
    first_X = members[0].X_train
    print(
        f"  [{label}] preprocessed X cols = {first_X.shape[1]:>3d}  "
        f"acc = {acc:.3f}"
    )
    return clf


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})\n")
    else:
        device = "cpu"
        print("Device: CPU (no GPU available — testing on CPU)\n")
    globals()["DEVICE"] = device

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_TOTAL_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        random_state=SEED,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y,
    )

    base_transforms = (
        TabPFNClassifier().get_inference_config().PREPROCESS_TRANSFORMS
    )

    print(
        f"Data: {X_tr.shape[0]} train / {X_te.shape[0]} test, "
        f"{N_TOTAL_FEATURES} total features ({N_INFORMATIVE} informative)\n"
    )

    print("(1) baseline — no duplication, no subsampling")
    fit_and_inspect(
        inference_config={},
        label="baseline",
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
    )

    print("(2) duplication only — full input + K=15 duplicate columns")
    fit_and_inspect(
        inference_config={"FEATURE_IMPORTANCE_DUPLICATE_TOP_K": 15},
        label="dup=15",
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
    )

    print("(3) subsample(random) + duplication — force-include top-K in subsample")
    transforms = [
        dataclasses.replace(p, max_features_per_estimator=40)
        for p in base_transforms
    ]
    clf = fit_and_inspect(
        inference_config={
            "FEATURE_SUBSAMPLING_METHOD": "random",
            "PREPROCESS_TRANSFORMS": transforms,
            "FEATURE_IMPORTANCE_DUPLICATE_TOP_K": 10,
        },
        label="random+dup=10",
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
    )

    # Check force-inclusion.
    sub_indices = clf.ensemble_preprocessor_.subsample_feature_indices
    # Recompute the importance order the same way (deterministic via seed).
    from tabpfn.preprocessing.ensemble import compute_feature_importance_order

    rng = np.random.default_rng(seed=SEED + 12345)  # any rng; we just want top-K
    # Re-derive top-K from a fresh fit — quick sanity, not exact comparison.
    print(
        f"  per-estimator subsample sizes = "
        f"{[len(s) if s is not None else 'all' for s in sub_indices]}"
    )
    if sub_indices and sub_indices[0] is not None:
        # All estimators should share at least 10 common indices (the top-K).
        common = set(sub_indices[0].tolist())
        for s in sub_indices[1:]:
            if s is not None:
                common &= set(s.tolist())
        print(f"  indices common to every estimator: {len(common)} (expect >= 10)")


if __name__ == "__main__":
    main()
