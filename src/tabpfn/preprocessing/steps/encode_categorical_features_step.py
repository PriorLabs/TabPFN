"""Encode Categorical Features Step."""

from __future__ import annotations

import warnings
from typing_extensions import override

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from tabpfn.preprocessing.steps.preprocessing_helpers import (
    FeaturePreprocessingTransformerStep,
    TransformResult,
)
from tabpfn.utils import infer_random_state


def _get_least_common_category_count(x_column: np.ndarray) -> int:
    if len(x_column) == 0:
        return 0
    counts = np.unique(x_column, return_counts=True)[1]
    return int(counts.min())


class EncodeCategoricalFeaturesStep(FeaturePreprocessingTransformerStep):
    """Encode Categorical Features Step."""

    def __init__(
        self,
        categorical_transform_name: str = "ordinal",
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.categorical_transform_name = categorical_transform_name
        self.random_state = random_state

        self.categorical_transformer_ = None

    def _get_transformer(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[ColumnTransformer | None, list[int]]:
        if self.categorical_transform_name.startswith("ordinal"):
            name = self.categorical_transform_name[len("ordinal") :]
            # Create a column transformer
            if name.startswith("_common_categories"):
                name = name[len("_common_categories") :]
                categorical_features = [
                    i
                    for i, col in enumerate(X.T)
                    if i in categorical_features
                    and _get_least_common_category_count(col) >= 10
                ]
            elif name.startswith("_very_common_categories"):
                name = name[len("_very_common_categories") :]
                categorical_features = [
                    i
                    for i, col in enumerate(X.T)
                    if i in categorical_features
                    and _get_least_common_category_count(col) >= 10
                    and len(np.unique(col)) < (len(X) // 10)  # type: ignore
                ]

            assert name in ("_shuffled", ""), (
                "unknown categorical transform name, should be 'ordinal'"
                f" or 'ordinal_shuffled' it was {self.categorical_transform_name}"
            )

            ct = ColumnTransformer(
                [
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),  # 'sparse' has been deprecated
                        categorical_features,
                    ),
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            return ct, categorical_features

        if self.categorical_transform_name == "onehot":
            # Create a column transformer
            ct = ColumnTransformer(
                [
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            drop="if_binary",
                            sparse_output=False,
                            handle_unknown="ignore",
                        ),
                        categorical_features,
                    ),
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            return ct, categorical_features

        if self.categorical_transform_name in ("numeric", "none"):
            return None, categorical_features
        raise ValueError(
            f"Unknown categorical transform {self.categorical_transform_name}",
        )

    @override
    def _fit(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> list[int]:
        ct, categorical_features = self._get_transformer(X, categorical_features)
        if ct is None:
            self.categorical_transformer_ = None
            return categorical_features

        _, rng = infer_random_state(self.random_state)

        if self.categorical_transform_name.startswith("ordinal"):
            ct.fit(X)
            categorical_features = list(range(len(categorical_features)))

            self.random_mappings_ = {}
            if self.categorical_transform_name.endswith("_shuffled"):
                for col_ix in categorical_features:
                    col_cats = len(
                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
                    )
                    perm = rng.permutation(col_cats)
                    self.random_mappings_[col_ix] = perm

        elif self.categorical_transform_name == "onehot":
            Xt = ct.fit_transform(X)
            if Xt.size >= 1_000_000:
                ct = None
            else:
                categorical_features = list(range(Xt.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
        else:
            raise ValueError(
                f"Unknown categorical transform {self.categorical_transform_name}",
            )

        self.categorical_transformer_ = ct
        return categorical_features

    def _fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        ct, categorical_features = self._get_transformer(X, categorical_features)
        if ct is None:
            self.categorical_transformer_ = None
            return X, categorical_features

        _, rng = infer_random_state(self.random_state)

        if self.categorical_transform_name.startswith("ordinal"):
            Xt = ct.fit_transform(X)
            categorical_features = list(range(len(categorical_features)))

            self.random_mappings_ = {}
            if self.categorical_transform_name.endswith("_shuffled"):
                for col_ix in categorical_features:
                    col_cats = len(
                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
                    )
                    perm = rng.permutation(col_cats)
                    self.random_mappings_[col_ix] = perm

                    Xcol: np.ndarray = Xt[:, col_ix]  # type: ignore
                    not_nan_mask = ~np.isnan(Xcol)
                    Xcol[not_nan_mask] = perm[Xcol[not_nan_mask].astype(int)].astype(
                        Xcol.dtype,
                    )

        elif self.categorical_transform_name == "onehot":
            Xt = ct.fit_transform(X)
            if Xt.size >= 1_000_000:
                ct = None
                Xt = X
            else:
                categorical_features = list(range(Xt.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
        else:
            raise ValueError(
                f"Unknown categorical transform {self.categorical_transform_name}",
            )

        self.categorical_transformer_ = ct
        return Xt, categorical_features  # type: ignore

    @override
    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> TransformResult:
        Xt, cat_ix = self._fit_transform(X, categorical_features)
        self.categorical_features_after_transform_ = cat_ix
        return TransformResult(Xt, cat_ix)

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        if self.categorical_transformer_ is None:
            return X

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Found unknown categories in col.*"
            )  # These warnings are expected when transforming test data
            transformed = self.categorical_transformer_.transform(X)
        if self.categorical_transform_name.endswith("_shuffled"):
            for col, mapping in self.random_mappings_.items():
                not_nan_mask = ~np.isnan(transformed[:, col])  # type: ignore
                transformed[:, col][not_nan_mask] = mapping[
                    transformed[:, col][not_nan_mask].astype(int)
                ].astype(transformed[:, col].dtype)
        return transformed  # type: ignore


__all__ = [
    "EncodeCategoricalFeaturesStep",
]
