"""Test to verify that TabPFNClassifier handles missing values gracefully."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier


def test_classifier_handles_na_values() -> None:
    """Ensure that TabPFNClassifier can train and predict
    with NA values in input data.
    """
    data = {
        "feature1": ["a", "b", pd.NA, "d"],
        "feature2": [1, 2, 3, 4],
        "target": [0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    X = df[["feature1", "feature2"]]
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = TabPFNClassifier(device="cpu")

    # Train and predict
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Assert predictions length matches
    assert len(predictions) == len(y_test)

    # Predictions should be valid labels only
    assert set(predictions).issubset(set(y.unique()))
