"""Order-preserving column transformer for tabular data preprocessing.

This module provides an OrderPreservingColumnTransformer that extends scikit-learn's
ColumnTransformer to ensure the original column order is preserved after transformation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    check_is_fitted,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

if TYPE_CHECKING:
    from tabpfn.classifier import XType, YType


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

    @override
    def transform(self, X: XType, **kwargs: dict[str, Any]) -> XType:
        return self._preserve_order(X, fit=False, **kwargs)

    @override
    def fit_transform(
        self, X: XType, y: YType = None, **kwargs: dict[str, Any]
    ) -> XType:
        return self._preserve_order(X, y, fit=True, **kwargs)

    def _preserve_order(
        self, X: XType, y: YType = None, *, fit: bool = False, **kwargs: dict[str, Any]
    ) -> XType:
        X_t = (
            super().fit_transform(X, y, **kwargs)
            if fit
            else super().transform(X, **kwargs)
        )
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D (shape={X.shape})"
        original_columns = (
            X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
        )
        # Safety check for duplicate column names in data
        # Duplicate column names destroy the mapping and thus are not supported.
        if len(set(original_columns)) != len(original_columns):
            raise ValueError("Duplicate column names detected in data.")

        return self._restore_column_order_if_necessary(X_t, original_columns)

    def _restore_column_order_if_necessary(
        self,
        X: XType,
        original_columns: pd.RangeIndex | range | list | pd.Index,
    ) -> XType:
        check_is_fitted(self)


            
        # We check if there exists an OrdinalEncoder in the list of encoders,
        # as this is an encoder, which changes the feature order
        categorical_cols = self._get_transformer_columns(OrdinalEncoder)

        if len(categorical_cols) > 0 and len(categorical_cols) < X.shape[-1]:
            # In case that the data a mixture of categorical & numerical features,
            # we need to restore the column order
            # OrdinalEncoder reorders:
            # - categorical columns (in the order they appear in the dataframe)
            # - numerical columns (in the order they appear in the dataframe)

            # map original columns to indices in the transformed array
            transformed_columns = categorical_cols + [
                c for c in original_columns if c not in categorical_cols
            ]
            indices = [transformed_columns.index(c) for c in original_columns]
            X = X.iloc[:, indices] if isinstance(X, pd.DataFrame) else X[:, indices]

        return X

    def _get_transformer_columns(
        self,
        encoder_cls: type[OneToOneFeatureMixin | BaseEstimator],
    ) -> list[Any]:
        """Collect all columns handled by transformers of a given class type.

        Parameters:
        ----------
        encoder_cls : class
            Transformer class (e.g., OrdinalEncoder).

        Returns:
        -------
        columns : list or None
            Combined list of columns processed by transformers of this type.
        """
        cols: list[Any] = []

        # self.transformers_ is only present once we called .fit() or .fit_transform()
        for _, transformer, columns in getattr(self, "transformers_", []):
            if isinstance(transformer, encoder_cls):
                if isinstance(columns, (list, tuple, pd.Index, range)):
                    cols.extend(columns)
                else:
                    cols.append(columns)

        return cols
