"""Reshape the feature distributions using different transformations."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, TypeVar
from typing_extensions import override

import numpy as np
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from tabpfn.preprocessing.steps.adaptive_quantile_transformer import (
    AdaptiveQuantileTransformer,
)
from tabpfn.preprocessing.steps.kdi_transformer import (
    KDITransformerWithNaN,
    get_all_kdi_transformers,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import (
    FeaturePreprocessingTransformerStep,
    TransformResult,
)
from tabpfn.preprocessing.steps.safe_power_transformer import SafePowerTransformer
from tabpfn.preprocessing.steps.squashing_scaler_transformer import SquashingScaler
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

T = TypeVar("T")


def _identity(x: T) -> T:
    return x


def _inf_to_nan_func(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _exp_minus_1(x: np.ndarray) -> np.ndarray:
    return np.exp(x) - 1  # type: ignore


inf_to_nan_transformer = FunctionTransformer(
    func=_inf_to_nan_func,
    inverse_func=_identity,
    check_inverse=False,
)
nan_impute_transformer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean",
    # keep empty features for inverse to function
    keep_empty_features=True,
)
nan_impute_transformer.inverse_transform = (
    _identity  # do not inverse np.nan values.  # type: ignore
)

_make_finite_transformer = [
    ("inf_to_nan", inf_to_nan_transformer),
    ("nan_impute", nan_impute_transformer),
]


def _make_standard_scaler_safe(
    _name_scaler_tuple: tuple[str, TransformerMixin],
    *,
    no_name: bool = False,
) -> Pipeline:
    # Make sure that all data that enters and leaves a scaler is finite.
    # This is needed in edge cases where, for example, a division by zero
    # occurs while scaling or when the input contains not number values.
    return Pipeline(
        steps=[
            *[(n + "_pre ", deepcopy(t)) for n, t in _make_finite_transformer],
            ("placeholder", _name_scaler_tuple) if no_name else _name_scaler_tuple,
            *[(n + "_post", deepcopy(t)) for n, t in _make_finite_transformer],
        ],
    )


def _make_box_cox_safe(input_transformer: TransformerMixin | Pipeline) -> Pipeline:
    """Make box cox save.

    The Box-Cox transformation can only be applied to strictly positive data.
    With first MinMax scaling, we achieve this without loss of function.
    Additionally, for test data, we also need clipping.
    """
    return Pipeline(
        steps=[
            ("mm", MinMaxScaler(feature_range=(0.1, 1), clip=True)),
            ("box_cox", input_transformer),
        ],
    )


def _add_safe_standard_to_safe_power_without_standard(
    input_transformer: TransformerMixin,
) -> Pipeline:
    """In edge cases PowerTransformer can create inf values and similar. Then, the post
    standard scale crashes. This fixes this issue.
    """
    return Pipeline(
        steps=[
            ("input_transformer", input_transformer),
            ("standard", _make_standard_scaler_safe(("standard", StandardScaler()))),
        ],
    )


def _skew(x: np.ndarray) -> float:
    """skewness: 3 * (mean - median) / std."""
    return float(3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0))


class ReshapeFeatureDistributionsStep(FeaturePreprocessingTransformerStep):
    """Reshape the feature distributions using different transformations."""

    APPEND_TO_ORIGINAL_THRESHOLD = 500

    @staticmethod
    def get_column_types(X: np.ndarray) -> list[str]:
        """Returns a list of column types for the given data, that indicate how
        the data should be preprocessed.
        """
        # TODO(eddiebergman): Bad to keep calling skew again and again here...
        column_types = []
        for col in range(X.shape[1]):
            if np.unique(X[:, col]).size < 10:
                column_types.append(f"ordinal_{col}")
            elif (
                _skew(X[:, col]) > 1.1
                and np.min(X[:, col]) >= 0
                and np.max(X[:, col]) <= 1
            ):
                column_types.append(f"skewed_pos_1_0_{col}")
            elif _skew(X[:, col]) > 1.1 and np.min(X[:, col]) > 0:
                column_types.append(f"skewed_pos_{col}")
            elif _skew(X[:, col]) > 1.1:
                column_types.append(f"skewed_{col}")
            elif shapiro(X[0:3000, col]).statistic > 0.95:
                column_types.append(f"normal_{col}")
            else:
                column_types.append(f"other_{col}")
        return column_types

    def __init__(
        self,
        *,
        transform_name: str = "safepower",
        apply_to_categorical: bool = False,
        append_to_original: bool | Literal["auto"] = False,
        max_features_per_estimator: int = 500,
        global_transformer_name: str | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()

        if max_features_per_estimator <= 0:
            raise ValueError("max_features_per_estimator must be a positive integer.")

        self.transform_name = transform_name
        self.apply_to_categorical = apply_to_categorical
        self.append_to_original = append_to_original
        self.random_state = random_state
        self.max_features_per_estimator = max_features_per_estimator
        self.global_transformer_name = global_transformer_name
        self.transformer_: Pipeline | ColumnTransformer | None = None

    def _set_transformer_and_cat_ix(  # noqa: PLR0912
        self,
        n_samples: int,
        n_features: int,
        categorical_features: list[int],
    ) -> tuple[Pipeline | ColumnTransformer, list[int]]:
        if "adaptive" in self.transform_name:
            raise NotImplementedError("Adaptive preprocessing raw removed.")

        static_seed, rng = infer_random_state(self.random_state)

        all_preprocessors = get_all_reshape_feature_distribution_preprocessors(
            n_samples,
            random_state=static_seed,
        )
        if n_features > self.max_features_per_estimator:
            subsample_features = self.max_features_per_estimator
            self.subsampled_features_ = rng.choice(
                list(range(n_features)),
                subsample_features,
                replace=False,
            )
            categorical_features = [
                new_idx
                for new_idx, idx in enumerate(self.subsampled_features_)
                if idx in categorical_features
            ]
            n_features = subsample_features
        else:
            self.subsampled_features_ = np.arange(n_features)

        if (
            self.global_transformer_name is not None
            and self.global_transformer_name != "None"
            and not (
                self.global_transformer_name in ["svd", "svd_quarter_components"]
                and n_features < 2
            )
        ):
            global_transformer_ = get_all_global_transformers(
                n_samples,
                n_features,
                random_state=static_seed,
            )[self.global_transformer_name]
        else:
            global_transformer_ = None

        all_feats_ix = list(range(n_features))
        transformers = []

        numerical_ix = [i for i in range(n_features) if i not in categorical_features]

        append_decision = (
            n_features < self.APPEND_TO_ORIGINAL_THRESHOLD
            and n_features < (self.max_features_per_estimator / 2)
        )
        self.append_to_original = (
            append_decision
            if self.append_to_original == "auto"
            else self.append_to_original
        )

        # -------- Append to original ------
        # If we append to original, all the categorical indices are kept in place
        # as the first transform is a passthrough on the whole X as it is above
        if self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        elif self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            # Includes the categoricals passed through
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        # -------- Don't append to original ------
        # We only have categorical indices if we don't transform them
        # The first transformer will be a passthrough on the categorical indices
        # Making them the first
        elif not self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            cat_ix = []  # We have none left, they've been transformed

        elif not self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            transformers.append(("cats", "passthrough", categorical_features))
            cat_ix = list(range(len(categorical_features)))  # They are at start

        else:
            raise ValueError(
                f"Unrecognized combination of {self.apply_to_categorical=}"
                f" and {self.append_to_original=}",
            )

        # NOTE: No need to keep track of categoricals here, already done above
        if self.transform_name != "per_feature":
            _transformer = all_preprocessors[self.transform_name]
            transformers.append(("feat_transform", _transformer, trans_ixs))
        else:
            preprocessors = list(all_preprocessors.values())
            transformers.extend(
                [
                    (f"transformer_{i}", rng.choice(preprocessors), [i])  # type: ignore
                    for i in trans_ixs
                ],
            )

        transformer = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.0,  # No sparse
        )

        # Apply a global transformer which accepts the entire dataset instead of
        # one column
        # NOTE: We assume global_transformer does not destroy the semantic meaning of
        # categorical_features_.
        if global_transformer_:
            transformer = Pipeline(
                [
                    ("preprocess", transformer),
                    ("global_transformer", global_transformer_),
                ],
            )

        self.transformer_ = transformer

        return transformer, cat_ix

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        transformer.fit(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return cat_ix

    @override
    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> TransformResult:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        Xt = transformer.fit_transform(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return TransformResult(Xt, cat_ix)  # type: ignore

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.transformer_ is not None, "You must call fit first"
        return self.transformer_.transform(X[:, self.subsampled_features_])  # type: ignore


def get_all_global_transformers(
    num_examples: int,
    num_features: int,
    random_state: int | None = None,
) -> dict[str, FeatureUnion | Pipeline]:
    """Returns a dictionary of global transformers to transform the data."""
    return {
        "scaler": _make_standard_scaler_safe(("standard", StandardScaler())),
        "svd": FeatureUnion(
            [
                # default FunctionTransformer yields the identity function
                ("passthrough", FunctionTransformer()),
                (
                    "svd",
                    Pipeline(
                        steps=[
                            (
                                "save_standard",
                                _make_standard_scaler_safe(
                                    ("standard", StandardScaler(with_mean=False)),
                                ),
                            ),
                            (
                                "svd",
                                TruncatedSVD(
                                    algorithm="arpack",
                                    n_components=max(
                                        1,
                                        min(
                                            num_examples // 10 + 1,
                                            num_features // 2,
                                        ),
                                    ),
                                    random_state=random_state,
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        ),
        "svd_quarter_components": FeatureUnion(
            [
                ("passthrough", FunctionTransformer(func=_identity)),
                (
                    "svd",
                    Pipeline(
                        steps=[
                            (
                                "save_standard",
                                _make_standard_scaler_safe(
                                    ("standard", StandardScaler(with_mean=False)),
                                ),
                            ),
                            (
                                "svd",
                                TruncatedSVD(
                                    algorithm="arpack",
                                    n_components=max(
                                        1,
                                        min(
                                            num_examples // 10 + 1,
                                            num_features // 4,
                                        ),
                                    ),
                                    random_state=random_state,
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        ),
    }


def get_adaptive_preprocessors(
    num_examples: int = 100,
    random_state: int | None = None,
) -> dict[str, ColumnTransformer]:
    """Returns a dictionary of adaptive column transformers that can be used to
    preprocess the data. Adaptive column transformers are used to preprocess the
    data based on the column type, they receive a pandas dataframe with column
    names, that indicate the column type. Column types are not datatypes,
    but rather a string that indicates how the data should be preprocessed.

    Args:
        num_examples: The number of examples in the dataset.
        random_state: The random state to use for the transformers.
    """
    return {
        "adaptive": ColumnTransformer(
            [
                (
                    "skewed_pos_1_0",
                    FunctionTransformer(
                        func=np.exp,
                        inverse_func=np.log,
                        check_inverse=False,
                    ),
                    make_column_selector("skewed_pos_1_0*"),
                ),
                (
                    "skewed_pos",
                    _make_box_cox_safe(
                        _add_safe_standard_to_safe_power_without_standard(
                            SafePowerTransformer(
                                standardize=False,
                                method="box-cox",
                            ),
                        ),
                    ),
                    make_column_selector("skewed_pos*"),
                ),
                (
                    "skewed",
                    _add_safe_standard_to_safe_power_without_standard(
                        SafePowerTransformer(
                            standardize=False,
                            method="yeo-johnson",
                        ),
                    ),
                    make_column_selector("skewed*"),
                ),
                (
                    "other",
                    AdaptiveQuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=max(num_examples // 10, 2),
                        random_state=random_state,
                    ),
                    # "other" or "ordinal"
                    make_column_selector("other*"),
                ),
                (
                    "ordinal",
                    # default FunctionTransformer yields the identity function
                    FunctionTransformer(),
                    # "other" or "ordinal"
                    make_column_selector("ordinal*"),
                ),
                (
                    "normal",
                    # default FunctionTransformer yields the identity function
                    FunctionTransformer(),
                    make_column_selector("normal*"),
                ),
            ],
            remainder="passthrough",
        ),
    }


def get_all_reshape_feature_distribution_preprocessors(
    num_examples: int,
    random_state: int | None = None,
) -> dict[str, TransformerMixin | Pipeline]:
    """Returns a dictionary of preprocessing to preprocess the data."""
    all_preprocessors = {
        "power": _add_safe_standard_to_safe_power_without_standard(
            PowerTransformer(standardize=False),
        ),
        "safepower": _add_safe_standard_to_safe_power_without_standard(
            SafePowerTransformer(standardize=False),
        ),
        "power_box": _make_box_cox_safe(
            _add_safe_standard_to_safe_power_without_standard(
                PowerTransformer(standardize=False, method="box-cox"),
            ),
        ),
        "safepower_box": _make_box_cox_safe(
            _add_safe_standard_to_safe_power_without_standard(
                SafePowerTransformer(standardize=False, method="box-cox"),
            ),
        ),
        "log": FunctionTransformer(
            func=np.log,
            inverse_func=np.exp,
            check_inverse=False,
        ),
        "1_plus_log": FunctionTransformer(
            func=np.log1p,
            inverse_func=_exp_minus_1,
            check_inverse=False,
        ),
        "exp": FunctionTransformer(
            func=np.exp,
            inverse_func=np.log,
            check_inverse=False,
        ),
        "quantile_uni_coarse": AdaptiveQuantileTransformer(
            output_distribution="uniform",
            n_quantiles=max(num_examples // 10, 2),
            random_state=random_state,
        ),
        "quantile_norm_coarse": AdaptiveQuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(num_examples // 10, 2),
            random_state=random_state,
        ),
        "quantile_uni": AdaptiveQuantileTransformer(
            output_distribution="uniform",
            n_quantiles=max(num_examples // 5, 2),
            random_state=random_state,
        ),
        "quantile_norm": AdaptiveQuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(num_examples // 5, 2),
            random_state=random_state,
        ),
        "quantile_uni_fine": AdaptiveQuantileTransformer(
            output_distribution="uniform",
            n_quantiles=num_examples,
            random_state=random_state,
        ),
        "quantile_norm_fine": AdaptiveQuantileTransformer(
            output_distribution="normal",
            n_quantiles=num_examples,
            random_state=random_state,
        ),
        "squashing_scaler_default": SquashingScaler(),
        "squashing_scaler_max10": SquashingScaler(max_absolute_value=10.0),
        "robust": RobustScaler(unit_variance=True),
        # default FunctionTransformer yields the identity function
        "none": FunctionTransformer(),
        **get_all_kdi_transformers(),
    }

    with contextlib.suppress(Exception):
        all_preprocessors["norm_and_kdi"] = FeatureUnion(
            [
                (
                    "norm",
                    AdaptiveQuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=max(num_examples // 10, 2),
                        random_state=random_state,
                    ),
                ),
                (
                    "kdi",
                    KDITransformerWithNaN(alpha=1.0, output_distribution="uniform"),
                ),
            ],
        )

    all_preprocessors.update(
        get_adaptive_preprocessors(
            num_examples,
            random_state=random_state,
        ),
    )

    return all_preprocessors


__all__ = [
    "ReshapeFeatureDistributionsStep",
    "get_all_reshape_feature_distribution_preprocessors",
]
