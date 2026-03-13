"""
TabPFN Client Example Usage
--------------------------
Toy script to check that the TabPFN client is working.
Use the breast cancer dataset for classification and the diabetes dataset for regression,
and try various prediction types.
"""

import logging
from unittest.mock import patch

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# from tabpfn_client import UserDataClient
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    use_server = True
    # use_server = False

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    tabpfn = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, n_estimators=3
    )
    # print("checking estimator", check_estimator(tabpfn))
    tabpfn.fit(X_train[:99], y_train[:99])
    print("predicting")
    print(tabpfn.predict(X_test))
    print("predicting_proba")
    print(tabpfn.predict_proba(X_test))

    # can be slow if you have a lot of data
    # print(UserDataClient.get_data_summary())

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    tabpfn = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_5, n_estimators=3
    )
    # print("checking estimator", check_estimator(tabpfn))
    tabpfn.fit(X_train[:99], y_train[:99])
    print("predicting reg")
    print(tabpfn.predict(X_test, output_type="mean"))

    # print(UserDataClient.get_data_summary())
    # test predict_full
    print("predicting ")
    print(
        tabpfn.predict(X_test[:30], output_type="full", quantiles=[0.1, 0.5, 0.9])
    )
