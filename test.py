from __future__ import annotations

import numpy as np
import sklearn.datasets

from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: F401

model = TabPFNClassifier(n_estimators=2, fit_mode="fit_with_cache")
print(model)  # noqa: T201

X, y = sklearn.datasets.load_iris(return_X_y=True)
X = X.to_numpy()
print(X.shape)  # noqa: T201

model.fit(X, y)
predictions = model.predict(X)
proba = model.predict_proba(X)
proba_preds = np.argmax(proba, axis=1)

proba_two = model.predict_proba(X)

np.testing.assert_equal(proba, proba_two)

# print(model.predict(X))
# print(y)

"""
model = TabPFNRegressor(
    n_estimators=2,
    fit_mode="fit_preprocessors",
    download=True,
)
print(model)  # noqa: T201

X, y = sklearn.datasets.make_regression(n_samples=50, n_features=10)
print(X.shape)  # noqa: T201

model.fit(X, y)
print(model.predict(X))  # noqa: T201
print(y)  # noqa: T201
"""
