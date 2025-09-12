"""Feature Preprocessing Transformer Step."""

from __future__ import annotations

from abc import abstractmethod
from collections import UserList
from typing import TYPE_CHECKING, NamedTuple
from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np
    import torch


class TransformResult(NamedTuple):
    """Result of a feature preprocessing step."""

    X: np.ndarray | torch.Tensor
    categorical_features: list[int]


# TODO(eddiebergman): I'm sure there's a way to handle this when using dataframes.
class FeaturePreprocessingTransformerStep:
    """Base class for feature preprocessing steps.

    It's main abstraction is really just to provide categorical indices along the
    pipeline.
    """

    categorical_features_after_transform_: list[int]

    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> TransformResult:
        """Fits the preprocessor and transforms the data."""
        self.fit(X, categorical_features)
        # TODO(eddiebergman): If we could get rid of this... anywho, needed for
        # the AddFingerPrint
        result = self._transform(X, is_test=False)
        return TransformResult(result, self.categorical_features_after_transform_)

    @abstractmethod
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        """Underlying method of the preprocessor to implement by subclassses.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.

        Returns:
            list of indices of categorical features after the transform.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, categorical_features: list[int]) -> Self:
        """Fits the preprocessor.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.
        """
        self.categorical_features_after_transform_ = self._fit(X, categorical_features)
        assert self.categorical_features_after_transform_ is not None, (
            "_fit should have returned a list of the indexes of the categorical"
            "features after the transform."
        )
        return self

    @abstractmethod
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        """Underlying method of the preprocessor to implement by subclassses.

        Args:
            X: 2d array of shape (n_samples, n_features)
            is_test: Should be removed, used for the `AddFingerPrint` step.

        Returns:
            2d np.ndarray of shape (n_samples, new n_features)
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> TransformResult:
        """Transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        # TODO: Get rid of this, it's always test in `transform`
        result = self._transform(X, is_test=True)
        return TransformResult(result, self.categorical_features_after_transform_)


class SequentialFeatureTransformer(UserList):
    """A transformer that applies a sequence of feature preprocessing steps.
    This is very related to sklearn's Pipeline, but it is designed to work with
    categorical_features lists that are always passed on.

    Currently this class is only used once, thus this could also be made
    less general if needed.
    """

    def __init__(self, steps: list[FeaturePreprocessingTransformerStep]):
        super().__init__(steps)
        self.steps = steps
        self.categorical_features_: list[int] | None = None

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        categorical_features: list[int],
    ) -> TransformResult:
        """Fit and transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical features.
        """
        for step in self.steps:
            X, categorical_features = step.fit_transform(X, categorical_features)
            assert isinstance(categorical_features, list), (
                f"The {step=} must return list of categorical features,"
                f" but {type(step)} returned {categorical_features}"
            )

        self.categorical_features_ = categorical_features
        return TransformResult(X, categorical_features)

    def fit(
        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
    ) -> Self:
        """Fit all the steps in the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.
        """
        assert (
            len(self) > 0
        ), "The SequentialFeatureTransformer must have at least one step."
        self.fit_transform(X, categorical_features)
        return self

    def transform(self, X: np.ndarray) -> TransformResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        assert (
            len(self) > 0
        ), "The SequentialFeatureTransformer must have at least one step."
        assert self.categorical_features_ is not None, (
            "The SequentialFeatureTransformer must be fit before it"
            " can be used to transform."
        )
        categorical_features = []
        for step in self:
            X, categorical_features = step.transform(X)

        assert categorical_features == self.categorical_features_, (
            f"Expected categorical features {self.categorical_features_},"
            f"but got {categorical_features}"
        )
        return TransformResult(X, categorical_features)


__all__ = [
    "FeaturePreprocessingTransformerStep",
    "SequentialFeatureTransformer",
    "TransformResult",
]
