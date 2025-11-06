"""Test TabPFNClassifier and TabPFNRegressor on small sample datasets."""

import numpy as np
from sklearn.datasets import (
    load_iris,
    make_classification,
    make_regression,
    load_breast_cancer,
    make_moons,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier, TabPFNRegressor


def test_binary_classification():
    """Test TabPFNClassifier on binary classification tasks."""
    print("\n" + "=" * 60)
    print("TESTING BINARY CLASSIFICATION")
    print("=" * 60)

    # Test 1: Breast Cancer dataset
    print("\n1. Breast Cancer Dataset:")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = TabPFNClassifier(
        n_estimators=4,
        random_state=42,
        model_path="tabpfn-v2-classifier.ckpt"
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")

    # Test 2: Synthetic dataset with make_moons
    print("\n2. Make Moons Dataset (Non-linear):")
    X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = TabPFNClassifier(n_estimators=4, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")


def test_multiclass_classification():
    """Test TabPFNClassifier on multiclass classification tasks."""
    print("\n" + "=" * 60)
    print("TESTING MULTICLASS CLASSIFICATION")
    print("=" * 60)

    # Test 1: Iris dataset
    print("\n1. Iris Dataset (3 classes):")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = TabPFNClassifier(n_estimators=4, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    print(f"   Accuracy: {accuracy:.4f}")

    # Test 2: Synthetic multiclass dataset
    print("\n2. Synthetic Multiclass Dataset (5 classes):")
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = TabPFNClassifier(n_estimators=4, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    print(f"   Accuracy: {accuracy:.4f}")


def test_regression():
    """Test TabPFNRegressor on regression tasks."""
    print("\n" + "=" * 60)
    print("TESTING REGRESSION")
    print("=" * 60)

    # Test 1: Simple linear regression
    print("\n1. Simple Linear Regression:")
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, noise=10.0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    regressor = TabPFNRegressor(n_estimators=4, random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   R² Score: {r2:.4f}")

    # Test 2: Non-linear regression
    print("\n2. Non-linear Regression:")
    # Create a non-linear dataset
    n_samples = 200
    np.random.seed(42)
    X = np.random.rand(n_samples, 5) * 10
    y = (
        X[:, 0] ** 2
        + 2 * X[:, 1]
        + np.sin(X[:, 2]) * 10
        + X[:, 3] * X[:, 4]
        + np.random.randn(n_samples) * 5
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    regressor = TabPFNRegressor(n_estimators=4, random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   R² Score: {r2:.4f}")


def test_with_categorical_features():
    """Test TabPFN with categorical features."""
    print("\n" + "=" * 60)
    print("TESTING WITH CATEGORICAL FEATURES")
    print("=" * 60)

    # Create a dataset with mixed numeric and categorical features
    n_samples = 300
    np.random.seed(42)

    # Numeric features
    X_numeric = np.random.randn(n_samples, 3)

    # Categorical features (will be treated as ordinal)
    X_cat1 = np.random.choice([0, 1, 2], size=n_samples)  # 3 categories
    X_cat2 = np.random.choice([0, 1, 2, 3], size=n_samples)  # 4 categories

    # Combine features
    X = np.column_stack([X_numeric, X_cat1, X_cat2])

    # Create target based on both numeric and categorical features
    y = (
        X[:, 0] * 2
        + X[:, 1] ** 2
        + X[:, 3] * 3  # categorical feature
        + X[:, 4] * 2  # categorical feature
        + np.random.randn(n_samples) * 0.5
    )

    # Make it a classification problem
    y_class = (y > np.median(y)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.3, random_state=42
    )

    # Specify categorical indices
    categorical_indices = [3, 4]  # Last two columns are categorical

    clf = TabPFNClassifier(
        n_estimators=4,
        categorical_features_indices=categorical_indices,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Categorical features: {categorical_indices}")
    print(f"   Accuracy: {accuracy:.4f}")


def test_small_sample_performance():
    """Test TabPFN on very small datasets where it should excel."""
    print("\n" + "=" * 60)
    print("TESTING SMALL SAMPLE PERFORMANCE")
    print("=" * 60)

    # Test with only 20 training samples
    print("\n1. Binary Classification with 20 training samples:")
    X, y = make_classification(
        n_samples=50,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.6,
        random_state=42,  # Only 20 training samples
    )

    clf = TabPFNClassifier(n_estimators=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   Accuracy: {accuracy:.4f}")

    # Test regression with small sample
    print("\n2. Regression with 30 training samples:")
    X, y = make_regression(
        n_samples=50, n_features=5, n_informative=3, noise=5.0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,  # Only 30 training samples
    )

    regressor = TabPFNRegressor(n_estimators=8, random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"   Train set size: {X_train.shape}")
    print(f"   Test set size: {X_test.shape}")
    print(f"   R² Score: {r2:.4f}")


if __name__ == "__main__":
    import os
    os.environ["TABPFN_TELEMETRY_SOURCE"] = "test"

    for _ in range(10):
        test_binary_classification()
        test_multiclass_classification()
        test_regression()
        test_with_categorical_features()
        test_small_sample_performance()