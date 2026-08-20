#  Copyright (c) Prior Labs GmbH 2026.

"""Adds SVD features to the data."""

from __future__ import annotations

import math
from typing import Literal
from typing_extensions import override

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_interface import PreprocessingStep
from tabpfn.preprocessing.steps.utils import make_scaler_safe
from tabpfn.utils import infer_random_state


def _pin_layout(X: np.ndarray) -> np.ndarray:
    """Return ``X`` in Fortran order, the layout this step's solver is calibrated on.

    ``TruncatedSVD(algorithm="arpack")`` is an iterative Lanczos solver, and on a
    near-degenerate spectrum -- singular values within a fraction of a percent of each
    other, which wide tables routinely produce -- the basis it converges to depends on
    the order the underlying BLAS accumulates in, and so on whether the array it is
    handed is C- or Fortran-contiguous. The values are unaffected and so is the
    subspace; which basis of it comes back is not. Pinning the layout here is what
    keeps the components stable against a layout change in any earlier step.

    A no-op when the input is already Fortran-contiguous.
    """
    return np.asfortranarray(X)


def get_svd_n_components(
    global_transformer_name: Literal["svd", "svd_quarter_components"],
    n_samples: int,
    n_features: int,
) -> int:
    """Compute the number of SVD components matching the TabPFN convention.

    Used by both the sklearn and torch SVD feature steps.
    """
    if global_transformer_name == "svd":
        divisor = 2
    elif global_transformer_name == "svd_quarter_components":
        divisor = 4
    else:
        raise ValueError(f"Invalid global transformer name: {global_transformer_name}.")
    return max(1, min(n_samples // 10 + 1, n_features // divisor))


def get_svd_component_pool_size(n_samples: int, n_features: int) -> int:
    """Largest number of components the solver can extract from this shape.

    ``TruncatedSVD(algorithm="arpack")`` wraps ``scipy.sparse.linalg.svds``,
    which requires strictly fewer components than the smaller matrix dimension.
    This is the size of the pool the extra random components are drawn from.
    """
    return max(1, min(n_samples, n_features) - 1)


def get_svd_n_extra_random_components(
    n_top_components: int,
    pool_size: int,
    extra_random_fraction: float,
) -> int:
    """How many components to draw at random from below the top-k.

    Rounds up, so any positive fraction adds at least one component, and never
    asks for more than the pool holds below the top-k.
    """
    if extra_random_fraction <= 0:
        return 0
    wanted = math.ceil(extra_random_fraction * n_top_components)
    return max(0, min(wanted, pool_size - n_top_components))


def select_svd_component_indices(
    n_top_components: int,
    n_extra_random: int,
    pool_size: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Pick which components of the full pool to keep.

    Always the top ``n_top_components`` by singular value, plus
    ``n_extra_random`` drawn uniformly without replacement from the rest of the
    spectrum. Each ensemble member seeds this differently, so the members see
    different low-variance directions of the same data.

    Shared by the sklearn and torch SVD steps so both select the same indices
    for the same seed.
    """
    top = np.arange(n_top_components)
    if n_extra_random <= 0:
        return top
    static_seed, _ = infer_random_state(random_state)
    rng = np.random.default_rng(static_seed)
    extra = rng.choice(
        np.arange(n_top_components, pool_size),
        size=n_extra_random,
        replace=False,
    )
    return np.concatenate([top, np.sort(extra)])


class AddSVDFeaturesStep(PreprocessingStep):
    """Append low-rank SVD projection features to the input.

    This keeps the original `X` and adds additional numerical features given by a
    truncated SVD of (scaled) `X`, i.e. a compressed/global view of the feature
    space. This can be used for numerical columns or other modalities that are encoded
    as numericals (e.g. categoricals that use target encoding or one-hot encoding).
    """

    def __init__(
        self,
        global_transformer_name: Literal[
            "svd", "svd_quarter_components"
        ] = "svd_quarter_components",
        random_state: int | np.random.Generator | None = None,
        extra_random_component_fraction: float = 0.0,
    ):
        """Initializes the AddSVDFeaturesStep.

        Args:
            global_transformer_name: Which component-count convention to use.
            random_state: Seeds the solver and, when
                ``extra_random_component_fraction`` is positive, the draw of the
                extra components.
            extra_random_component_fraction: When positive, the full spectrum is
                decomposed and this fraction of the top-k count is appended as
                components drawn at random from below the top-k (0.5 adds half
                as many again). Zero keeps the top-k only, which is both the
                default and the cheaper path.
        """
        super().__init__()
        self.global_transformer_name = global_transformer_name
        self.random_state = random_state
        self.extra_random_component_fraction = extra_random_component_fraction
        self.is_no_op: bool = False
        self.component_indices_: np.ndarray | None = None

    @override
    def added_feature_prefix(self) -> str:
        return "svd"

    def _component_counts(
        self, n_samples: int, n_features: int
    ) -> tuple[int, int, int]:
        """Return (n_top, n_extra_random, pool_size) for this input shape."""
        n_top = get_svd_n_components(
            self.global_transformer_name,
            n_samples=n_samples,
            n_features=n_features,
        )
        pool_size = get_svd_component_pool_size(n_samples, n_features)
        n_extra = get_svd_n_extra_random_components(
            n_top, pool_size, self.extra_random_component_fraction
        )
        return n_top, n_extra, pool_size

    @override
    def num_added_features(self, n_samples: int, feature_schema: FeatureSchema) -> int:
        """Return the number of added features."""
        n_features = feature_schema.num_columns
        if n_features < 2:
            return 0

        n_top, n_extra, _ = self._component_counts(n_samples, n_features)
        return n_top + n_extra

    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        self.is_no_op = False
        self.component_indices_ = None
        n_samples, n_features = X.shape
        if n_features < 2:
            self.is_no_op = True
            return feature_schema

        n_top, n_extra, pool_size = self._component_counts(n_samples, n_features)

        static_seed, _ = infer_random_state(self.random_state)
        transformer = get_svd_features_transformer(
            self.global_transformer_name,
            n_samples,
            n_features,
            random_state=static_seed,
            # The extra components are drawn from anywhere below the top-k, so
            # the whole spectrum has to be decomposed to have a pool to draw
            # from. Without them only the top-k are computed, as before.
            n_components=pool_size if n_extra > 0 else None,
        )
        transformer.fit(_pin_layout(X))

        if n_extra > 0:
            self.component_indices_ = select_svd_component_indices(
                n_top, n_extra, pool_size, static_seed
            )
            _keep_svd_components(transformer, self.component_indices_)

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

        # Only the solver's input is pinned; `X` itself is handed back as it came.
        return X, self.transformer_.transform(_pin_layout(X)), FeatureModality.NUMERICAL


def _keep_svd_components(transformer: Pipeline, indices: np.ndarray) -> None:
    """Restrict a fitted SVD to ``indices``, so transform only projects onto those.

    ``TruncatedSVD.transform`` is ``X @ components_.T`` and nothing else, so
    dropping the rows we did not select is what makes the step emit the chosen
    components -- and only those -- without a second projection. The parallel
    per-component attributes are trimmed alongside to keep the fitted estimator
    self-consistent.
    """
    svd = transformer.steps[1][1]
    assert isinstance(svd, TruncatedSVD)
    svd.components_ = svd.components_[indices]
    for attribute in (
        "singular_values_",
        "explained_variance_",
        "explained_variance_ratio_",
    ):
        values = getattr(svd, attribute, None)
        if values is not None:
            setattr(svd, attribute, values[indices])


def get_svd_features_transformer(
    global_transformer_name: Literal["svd", "svd_quarter_components"],
    n_samples: int,
    n_features: int,
    random_state: int | None = None,
    n_components: int | None = None,
) -> Pipeline:
    """Returns a transformer to add SVD features to the data.

    ``n_components`` overrides the count implied by ``global_transformer_name``,
    which is how the extra-random-component path decomposes the full spectrum.
    """
    if n_components is None:
        n_components = get_svd_n_components(
            global_transformer_name, n_samples, n_features
        )
    return Pipeline(
        steps=[
            (
                "save_standard",
                make_scaler_safe("standard", StandardScaler(with_mean=False)),
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
