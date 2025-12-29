"""Feature Preprocessing Transformer Step."""

from __future__ import annotations

from abc import abstractmethod
from collections import UserList
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple
from typing_extensions import Self, override

if TYPE_CHECKING:
    import torch

    from tabpfn.classifier import XType, YType


import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    check_is_fitted,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tabpfn.constants import DEFAULT_NUMPY_PREPROCESSING_DTYPE


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
        assert len(self) > 0, (
            "The SequentialFeatureTransformer must have at least one step."
        )
        self.fit_transform(X, categorical_features)
        return self

    def transform(self, X: np.ndarray) -> TransformResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        assert len(self) > 0, (
            "The SequentialFeatureTransformer must have at least one step."
        )
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


class OrderPreservingColumnTransformer(ColumnTransformer):
    """An ColumnTransformer that preserves the column order after transformation."""

    def __init__(
        self,
        transformers: Sequence[
            tuple[
                str,
                BaseEstimator,
                str
                | int
                | slice
                | Iterable[str | int]
                | Callable[[Any], Iterable[str | int]],
            ]
        ],
        **kwargs: Any,
    ):
        """Implementation base on https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html.

        Parameters
        ----------
        transformers : sequence of (name, transformer, columns) tuples
            List of (name, transformer, columns) tuples specifying the transformers.
        **kwargs : additional keyword arguments
            Passed to sklearn.compose.ColumnTransformer.
        """
        super().__init__(transformers=transformers, **kwargs)

        # Check if there is a single transformer, of subtype OneToOneFeatureMixin
        assert all(
            isinstance(t, OneToOneFeatureMixin)
            for name, t, _ in transformers
            if name != "remainder"
        ), (
            "OrderPreservingColumnTransformer currently only supports transformers "
            "that are instances of OneToOneFeatureMixin."
        )

        assert len([t for name, _, t in transformers if name != "remainder"]) <= 1, (
            "OrderPreservingColumnTransformer only supports up to one transformer."
        )

    @override
    def transform(self, X: XType, **kwargs: dict[str, Any]) -> XType:
        original_columns = (
            X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
        )
        X_t = super().transform(X, **kwargs)
        return self._preserve_order(X=X_t, original_columns=original_columns)

    @override
    def fit_transform(
        self, X: XType, y: YType = None, **kwargs: dict[str, Any]
    ) -> XType:
        original_columns = (
            X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
        )
        X_t = super().fit_transform(X, y, **kwargs)
        return self._preserve_order(X=X_t, original_columns=original_columns)

    def _preserve_order(
        self, X: XType, original_columns: list | range | pd.Index
    ) -> XType:
        check_is_fitted(self)
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D (shape={X.shape})"
        for name, _, col_subset in reversed(self.transformers_):
            if (
                len(col_subset) > 0
                and len(col_subset) < X.shape[-1]
                and name != "remainder"
            ):
                col_subset_list = list(col_subset)
                # Map original columns to indices in the transformed array
                transformed_columns = col_subset_list + [
                    c for c in original_columns if c not in col_subset_list
                ]
                indices = [transformed_columns.index(c) for c in original_columns]
                # restore the column order from before the transfomer has been applied
                X = X.iloc[:, indices] if isinstance(X, pd.DataFrame) else X[:, indices]
        return X


def get_ordinal_encoder(
    *,
    numpy_dtype: np.floating = DEFAULT_NUMPY_PREPROCESSING_DTYPE,  # type: ignore
) -> ColumnTransformer:
    """Create a ColumnTransformer that ordinally encodes string/category columns."""
    oe = OrdinalEncoder(
        # TODO: Could utilize the categorical dtype values directly instead of "auto"
        categories="auto",
        dtype=numpy_dtype,  # type: ignore
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,  # Missing stays missing
    )

    # Documentation of sklearn, deferring to pandas is misleading here. It's done
    # using a regex on the type of the column, and using `object`, `"object"` and
    # `np.object` will not pick up strings.
    to_convert = ["category", "string"]

    # When using a ColumnTransformer with inner transformers applied to only a subset of
    # columns, the original column order of the data is not preserved. Because we do not
    # update the categorical indices after encoding, these indices may no longer align
    # with the true categorical columns.

    # Subsequent components rely on these categorical indices. For instance:
    # - QuantileTransformer should only be applied to numerical features.
    # - EncodeCategoricalFeaturesStep should be applied to all categorical features.

    # Despite the column shuffling introduced by the vanilla ColumnTransformer, we
    # observed better overall performance when using it. Therefore, we keep it.

    return ColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


__all__ = [
    "FeaturePreprocessingTransformerStep",
    "SequentialFeatureTransformer",
    "TransformResult",
]
