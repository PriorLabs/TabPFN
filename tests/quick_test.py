"""TabPFN Example Usage.

Toy script to check that the TabPFN is working.
Uses breast cancer (classification) and diabetes (regression) datasets.
"""

from __future__ import annotations

import logging

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    tabpfn = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, n_estimators=3
    )
    tabpfn.fit(X_train[:99], y_train[:99])
    print("predicting")  # noqa: T201
    print(tabpfn.predict(X_test))  # noqa: T201
    print("predicting_proba")  # noqa: T201
    print(tabpfn.predict_proba(X_test))  # noqa: T201

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    tabpfn = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_5, n_estimators=3
    )
    tabpfn.fit(X_train[:99], y_train[:99])
    print("predicting reg")  # noqa: T201
    print(tabpfn.predict(X_test, output_type="mean"))  # noqa: T201

    print("predicting full")  # noqa: T201
    print(  # noqa: T201
        tabpfn.predict(
            X_test[:30],
            output_type="full",
            quantiles=[0.1, 0.5, 0.9],
        )
    )
