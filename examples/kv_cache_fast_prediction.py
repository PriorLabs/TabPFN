#  Copyright (c) Prior Labs GmbH 2025.
"""TabPFN with KV cache vs. without on binary classification (synthetic).

`fit_mode="fit_with_cache"` builds a key–value (KV) cache *during* `fit`.
This front-loads the cost of computing the training-set representation so
`predict`/`predict_proba` run faster—especially when:
  • the training set is large, and/or
  • the test:train ratio is small (few predictions per many training points).

Trade-off: additional memory roughly O(N_samples * N_features) to hold the cache.
Implications:
  • Expect *slower* `fit` but *faster* `predict`/`predict_proba`.
  • Benefit grows with train size and repeated inference (CV folds, batch eval, etc.).
"""

import time
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    random_state=42,
    n_classes=2
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42, stratify=y
)

def bench(clf, name: str):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    # First inference already benefits if cache was built during fit.
    t1 = time.perf_counter()
    preds = clf.predict(X_test)
    t_pred = time.perf_counter() - t1

    print(f"[{name}] fit: {t_fit:.4f}s | predict_proba: {t_pred:.4f}s "
          f"| Acc: {accuracy_score(y_test, preds):.3f} ")

# Baseline: no cache
clf_no_cache = TabPFNClassifier()  # default mode (training part recomputed at predict time)
bench(clf_no_cache, "no_cache")

# With KV cache: cache is built during `fit`, so first predict is faster
clf_kv = TabPFNClassifier(fit_mode="fit_with_cache")
bench(clf_kv, "kv_cache")
