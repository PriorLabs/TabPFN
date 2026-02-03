"""Adds SVD features to the data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from typing_extensions import override

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_interface import PreprocessingStep
from tabpfn.preprocessing.steps.utils import make_standard_scaler_safe
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    import numpy as np


class AddSVDFeaturesStep(PreprocessingStep):
    """Adds SVD features to the data."""

    def __init__(
        self,
        global_transformer_name: Literal[
            "svd", "svd_quarter_components"
        ] = "svd_quarter_components",
        random_state: int | np.random.Generator | None = None,
    ):
        """Initializes the AddSVDFeaturesStep."""
        super().__init__()
        self.global_transformer_name = global_transformer_name
        self.random_state = random_state
        self.is_no_op: bool = False

    def num_added_features(self, n_samples: int, n_features: int) -> int:
        """Return the number of added features."""
        if n_features < 2:
            return n_features

        transformer = get_svd_features_transformer(
            self.global_transformer_name,
            n_samples,
            n_features,
        )
        return next(
            s[1].n_components
            for s in transformer.steps
            if isinstance(s[1], TruncatedSVD)
        )

    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        n_samples, n_features = X.shape
        if n_features < 2:
            self.is_no_op = True
            return feature_schema

        static_seed, _ = infer_random_state(self.random_state)
        transformer = get_svd_features_transformer(
            self.global_transformer_name,
            n_samples,
            n_features,
            random_state=static_seed,
        )
        transformer.fit(X)

        self.transformer_ = transformer
        self.feature_schema_updated_ = feature_schema

        return feature_schema

    @override
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        if self.is_no_op:
            return X, None, None

        assert self.feature_schema_updated_ is not None
        assert self.transformer_ is not None

        return X, self.transformer_.transform(X), FeatureModality.NUMERICAL


def get_svd_features_transformer(
    global_transformer_name: Literal["svd", "svd_quarter_components"],
    n_samples: int,
    n_features: int,
    random_state: int | None = None,
) -> Pipeline:
    """Returns a transformer to add SVD features to the data."""
    if global_transformer_name == "svd":
        divisor = 2
    elif global_transformer_name == "svd_quarter_components":
        divisor = 4
    else:
        raise ValueError(f"Invalid global transformer name: {global_transformer_name}.")

    n_components = max(1, min(n_samples // 10 + 1, n_features // divisor))
    return Pipeline(
        steps=[
            (
                "save_standard",
                make_standard_scaler_safe(
                    ("standard", StandardScaler(with_mean=False)),
                ),
            ),
            (
                "svd",
                TruncatedSVD(
                    algorithm="arpack",
                    n_components=n_components,
                    random_state=random_state,
                ),
            ),
        ],
    )
