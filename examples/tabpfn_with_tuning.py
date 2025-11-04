#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for binary classification with an eval_metric and tuning.

This example demonstrates how to calibrate and tune the predictions
of a TabPFNClassifier with an eval_metric and tuning_config.
"""

from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from tabpfn import TabPFNClassifier

MINORITY_FRAC = 0.04

# Generate an imbalanced dataset
X, y = make_classification(
    n_samples=3_000,
    n_features=4,
    n_classes=2,
    n_informative=4,
    n_redundant=0,
    weights=[float(1.0 - MINORITY_FRAC), float(MINORITY_FRAC)],
    random_state=42,
)

print(f"Generated dataset with imbalance ratio: {len(y[y == 1]) / len(y[y == 0]):.3f}")

stratified_splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.33,
    random_state=42,
)
train_index, test_index = next(stratified_splitter.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Initialize a classifier with tuning and fit
clf_no_tuning = TabPFNClassifier(eval_metric="f1")
clf_no_tuning.fit(X_train, y_train)

# Predict F1 score without tuning
predictions = clf_no_tuning.predict(X_test)
print(f"F1 Score without tuning: {f1_score(y_test, predictions):.3f}")

# Initialize a classifier with tuning and fit
clf_with_tuning = TabPFNClassifier(
    eval_metric="f1",
    tuning_config={"tune_decision_thresholds": True},
)
# This will tune the temperature and decision thresholds
clf_with_tuning.fit(X_train, y_train)

# Predict F1 score with tuning
predictions = clf_with_tuning.predict(X_test)
print(f"F1 Score with tuning: {f1_score(y_test, predictions):.3f}")
