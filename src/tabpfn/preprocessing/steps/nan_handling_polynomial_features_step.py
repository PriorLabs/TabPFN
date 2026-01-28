"""Nan Handling Polynomial Features Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
)

from tabpfn.preprocessing.pipeline_interfaces import (
    PreprocessingStep,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import append_numerical_features
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureModality


class NanHandlingPolynomialFeaturesStep(PreprocessingStep):
    """Nan Handling Polynomial Features Step."""

    def __init__(
        self,
        *,
        max_features: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()

        self.max_poly_features = max_features
        self.random_state = random_state

        self.poly_factor_1_idx: np.ndarray | None = None
        self.poly_factor_2_idx: np.ndarray | None = None
        self.n_polynomials_: int = 0

        self.standardizer = StandardScaler(with_mean=False)

    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_modalities: dict[FeatureModality, list[int]],
    ) -> dict[FeatureModality, list[int]]:
        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"
        _, rng = infer_random_state(self.random_state)

        if X.shape[0] == 0 or X.shape[1] == 0:
            self.n_polynomials_ = 0
            return feature_modalities

        # How many polynomials can we create?
        n_polynomials = (X.shape[1] * (X.shape[1] - 1)) // 2 + X.shape[1]
        n_polynomials = (
            min(self.max_poly_features, n_polynomials)
            if self.max_poly_features
            else n_polynomials
        )
        self.n_polynomials_ = n_polynomials

        X = self.standardizer.fit_transform(X)

        # Randomly select the indices of the factors
        self.poly_factor_1_idx = rng.choice(
            np.arange(0, X.shape[1]),
            size=n_polynomials,
            replace=True,
        )
        self.poly_factor_2_idx = np.ones_like(self.poly_factor_1_idx) * -1
        for i in range(len(self.poly_factor_1_idx)):
            while self.poly_factor_2_idx[i] == -1:
                poly_factor_1_ = self.poly_factor_1_idx[i]
                # indices of the factors that have already been used
                used_indices = self.poly_factor_2_idx[
                    self.poly_factor_1_idx == poly_factor_1_
                ]
                # remaining indices, only factors with higher index can be selected
                # to avoid duplicates
                indices_ = set(range(poly_factor_1_, X.shape[1])) - set(
                    used_indices.tolist(),
                )
                if len(indices_) == 0:
                    self.poly_factor_1_idx[i] = rng.choice(np.arange(0, X.shape[1]))
                    continue
                self.poly_factor_2_idx[i] = rng.choice(list(indices_))

        # Polynomial features are appended as new numerical columns
        return append_numerical_features(
            feature_modalities, X.shape[1], n_new_features=n_polynomials
        )

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"

        if X.shape[0] == 0 or X.shape[1] == 0:
            return X

        X = self.standardizer.transform(X)  # type: ignore

        poly_features_xs = X[:, self.poly_factor_1_idx] * X[:, self.poly_factor_2_idx]

        return np.hstack((X, poly_features_xs))
