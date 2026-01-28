"""Interfaces for creating preprocessing pipelines."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple
from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np
    import torch

    from tabpfn.preprocessing.datamodel import FeatureModality


class PreprocessingStepResult(NamedTuple):
    """Result of a feature preprocessing step."""

    X: np.ndarray | torch.Tensor
    feature_modalities: dict[FeatureModality, list[int]]


class PreprocessingStep:
    """Base class for feature preprocessing steps.

    It's main abstraction is really just to provide feature modalities along the
    pipeline.
    """

    feature_modalities_after_transform_: dict[FeatureModality, list[int]]

    def fit_transform(
        self,
        X: np.ndarray,
        feature_modalities: dict[FeatureModality, list[int]],
    ) -> PreprocessingStepResult:
        """Fits the preprocessor and transforms the data."""
        self.fit(X, feature_modalities)
        # TODO(eddiebergman): If we could get rid of this... anywho, needed for
        # the AddFingerPrint
        result = self._transform(X, is_test=False)
        return PreprocessingStepResult(result, self.feature_modalities_after_transform_)

    @abstractmethod
    def _fit(
        self, X: np.ndarray, feature_modalities: dict[FeatureModality, list[int]]
    ) -> dict[FeatureModality, list[int]]:
        """Underlying method of the preprocessor to implement by subclassses.

        Args:
            X: 2d array of shape (n_samples, n_features)
            feature_modalities: dictionary of feature modalities.

        Returns:
            dictionary of feature modalities after the transform.
        """
        raise NotImplementedError

    def fit(
        self, X: np.ndarray, feature_modalities: dict[FeatureModality, list[int]]
    ) -> Self:
        """Fits the preprocessor.

        Args:
            X: 2d array of shape (n_samples, n_features)
            feature_modalities: dictionary of feature modalities.
        """
        self.feature_modalities_after_transform_ = self._fit(X, feature_modalities)
        assert self.feature_modalities_after_transform_ is not None, (
            "_fit should have returned a dictionary of feature modalities after the transform."  # noqa: E501
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

    def transform(self, X: np.ndarray) -> PreprocessingStepResult:
        """Transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        # TODO: Get rid of this, it's always test in `transform`
        result = self._transform(X, is_test=True)
        return PreprocessingStepResult(result, self.feature_modalities_after_transform_)


class PreprocessingPipeline:
    """A transformer that applies a sequence of feature preprocessing steps.

    This is very related to sklearn's Pipeline, but it is designed to work with
    feature_modalities dictionaries that are always passed on.

    Currently this class is only used once, thus this could also be made
    less general if needed.
    """

    def __init__(self, steps: list[PreprocessingStep]) -> None:
        super().__init__()
        self.steps = steps
        self.feature_modalities_: dict[FeatureModality, list[int]] | None = None

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        feature_modalities: dict[FeatureModality, list[int]],
    ) -> PreprocessingStepResult:
        """Fit and transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            feature_modalities: dictionary of feature modalities.
        """
        for step in self.steps:
            X, feature_modalities = step.fit_transform(X, feature_modalities)
            assert isinstance(feature_modalities, dict), (
                f"The {step=} must return dictionary of feature modalities,"
                f" but {type(step)} returned {feature_modalities}"
            )

        self.feature_modalities_ = feature_modalities
        return PreprocessingStepResult(X, feature_modalities)

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        feature_modalities: dict[FeatureModality, list[int]],
    ) -> Self:
        """Fit all the steps in the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            feature_modalities: dictionary of feature modalities.
        """
        assert len(self) > 0, (
            "The SequentialFeatureTransformer must have at least one step."
        )
        self.fit_transform(X, feature_modalities)
        return self

    def transform(self, X: np.ndarray) -> PreprocessingStepResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        assert len(self.steps) > 0, (
            "The SequentialFeatureTransformer must have at least one step."
        )
        assert self.feature_modalities_ is not None, (
            "The SequentialFeatureTransformer must be fit before it"
            " can be used to transform."
        )
        feature_modalities = {}
        for step in self.steps:
            X, feature_modalities = step.transform(X)

        assert feature_modalities == self.feature_modalities_, (
            f"Expected feature modalities {self.feature_modalities_},"
            f"but got {feature_modalities}"
        )
        return PreprocessingStepResult(X, feature_modalities)
