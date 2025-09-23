import pandas as pd
import pytest
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

def test_classifier_handles_na_values():
    data = {
        'feature1': ['a', 'b', pd.NA, 'd'],
        'feature2': [1, 2, 3, 4],
        'target': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2']]
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = TabPFNClassifier(device='cpu')

    # The classifier should handle NA values internally
    try:
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # Assert predictions length matches y_test
        assert len(predictions) == len(y_test)
    except Exception as e:
        pytest.fail(f"TabPFNClassifier failed to handle NA values: {e}")
