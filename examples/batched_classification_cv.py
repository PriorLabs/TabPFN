#  Copyright (c) Prior Labs GmbH 2026.
"""Batched cross-validation with TabPFNClassifier.predict_proba_batched.

Cross-validation is a natural fit for batched inference: every fold is an
independent (train, test) pair, all folds share the same classes, and with a
row count divisible by the fold count they all share the same shape. That is
exactly what `predict_proba_batched` requires, so the whole CV run collapses
into one fused forward per estimator instead of one per fold.
"""

import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tabpfn import TabPFNClassifier

N_FOLDS = 5

X, y = load_breast_cancer(return_X_y=True)
# Trim to a multiple of N_FOLDS so every fold has an identical shape; ragged
# batches are rejected rather than padded, since padding would feed the model
# fake context rows.
X, y = X[: len(X) - len(X) % N_FOLDS], y[: len(y) - len(y) % N_FOLDS]

folds = list(
    StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(X, y)
)
X_trains = [X[train] for train, _ in folds]
y_trains = [y[train] for train, _ in folds]
X_tests = [X[test] for _, test in folds]
y_tests = [y[test] for _, test in folds]
print(f"{N_FOLDS} folds, {len(X_trains[0])} train rows and {len(X_tests[0])} test rows")

clf = TabPFNClassifier()
clf.fit(X_trains[0], y_trains[0])  # warm up: load the checkpoint outside the timings

start = time.perf_counter()
sequential = [
    clf.fit(X_tr, y_tr).predict_proba(X_te)
    for X_tr, y_tr, X_te in zip(X_trains, y_trains, X_tests, strict=True)
]
t_sequential = time.perf_counter() - start

start = time.perf_counter()
# One array of shape (n_folds, n_test, n_classes).
batched = clf.predict_proba_batched(X_trains, y_trains, X_tests)
t_batched = time.perf_counter() - start

auc = np.mean(
    [roc_auc_score(t, p[:, 1]) for t, p in zip(y_tests, batched, strict=True)]
)
drift = max(np.abs(a - b).max() for a, b in zip(sequential, batched, strict=True))
print(f"mean ROC AUC {auc:.4f} over folds")
print(f"max deviation from sequential {drift:.1e}")
print(f"sequential {t_sequential:.1f}s, batched {t_batched:.1f}s")
print(f"speedup {t_sequential / t_batched:.1f}x")
