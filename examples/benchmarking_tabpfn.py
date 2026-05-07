#  Copyright (c) Prior Labs GmbH 2026.
"""Example of benchmarking TabPFN against XGBoost on the "German Credit Data" dataset from OpenML.

This script compares TabPFN (default) against two XGBoost configurations on a binary
classification task with 1000 samples, reporting ROC-AUC, fit time, and predict time.

You can find more information on benchmarking TabPFN in our documentation:
https://docs.priorlabs.ai/benchmarking

Note: XGBoost is not a dependency of TabPFN. Install it separately if needed:
    pip install xgboost
"""

import time

import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# 1. Data — German-Credit-Data-Creditability (OpenML did=46562, 1000 samples, binary classification)
X, y = fetch_openml(data_id=46562, as_frame=True, return_X_y=True)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = []

# 3. TabPFN (default)
tabpfn = TabPFNClassifier(device="auto")
t0 = time.perf_counter()
tabpfn.fit(X_train, y_train)
fit_tabpfn = time.perf_counter() - t0
t0 = time.perf_counter()
proba_tabpfn = tabpfn.predict_proba(X_test)[:, 1]
predict_tabpfn = time.perf_counter() - t0
results.append(
    {
        "Model": "TabPFN v2.6 (default)",
        "ROC-AUC": f"{roc_auc_score(y_test, proba_tabpfn):.4f}",
        "Fit time": f"{fit_tabpfn:.2f}s",
        "Predict time": f"{predict_tabpfn:.2f}s",
        "Notes": "none",
    }
)

# 4. XGBoost sensible defaults
xgb_params = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "seed": 42,
}
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
t0 = time.perf_counter()
booster = xgb.train(xgb_params, dtrain, num_boost_round=100)
fit_xgb = time.perf_counter() - t0
t0 = time.perf_counter()
proba_xgb = booster.predict(dtest)
predict_xgb = time.perf_counter() - t0
results.append(
    {
        "Model": "XGBoost (sensible defaults)",
        "ROC-AUC": f"{roc_auc_score(y_test, proba_xgb):.4f}",
        "Fit time": f"{fit_xgb:.2f}s",
        "Predict time": f"{predict_xgb:.2f}s",
        "Notes": "n_estimators=100",
    }
)

# 5. XGBoost CV-tuned n_estimators (5-fold CV with early stopping)
cv_result = xgb.cv(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=20,
    seed=42,
)
best_rounds = len(cv_result)
t0 = time.perf_counter()
booster_tuned = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)
fit_xgb_tuned = time.perf_counter() - t0
t0 = time.perf_counter()
proba_xgb_tuned = booster_tuned.predict(dtest)
predict_xgb_tuned = time.perf_counter() - t0
results.append(
    {
        "Model": "XGBoost (CV-tuned n_estimators)",
        "ROC-AUC": f"{roc_auc_score(y_test, proba_xgb_tuned):.4f}",
        "Fit time": f"{fit_xgb_tuned:.2f}s",
        "Predict time": f"{predict_xgb_tuned:.2f}s",
        "Notes": f"5-fold CV with early stopping, best_rounds={best_rounds}",
    }
)

print(pd.DataFrame(results).to_string(index=False))
