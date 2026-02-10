"""Reshape the feature distributions using different transformations."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

import numpy as np
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
)

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
)
from tabpfn.preprocessing.steps.adaptive_quantile_transformer import (
    AdaptiveQuantileTransformer,
)
from tabpfn.preprocessing.steps.kdi_transformer import (
    KDITransformerWithNaN,
    get_all_kdi_transformers,
)
from tabpfn.preprocessing.steps.safe_power_transformer import SafePowerTransformer
from tabpfn.preprocessing.steps.squashing_scaler_transformer import SquashingScaler
from tabpfn.preprocessing.steps.utils import wrap_with_safe_standard_scaler
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


def _exp_minus_1(x: np.ndarray) -> np.ndarray:
    return np.exp(x) - 1  # type: ignore


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


def _skew(x: np.ndarray) -> float:
    """skewness: 3 * (mean - median) / std."""
    return float(3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0))


class ReshapeFeatureDistributionsStep(PreprocessingStep):
    """Reshape feature distributions using various transformations.

    This step should receive ALL columns (not modality-sliced) because it:
    1. Handles feature subsampling when too many features exist
    2. Applies different logic based on `apply_to_categorical` flag
    3. Can append transformed features to originals (`append_to_original`)

    # TODO(ben): Add separate PreprocessingStep's for all of the above
    # so that we can register this with modalities

    When using with PreprocessingPipeline, register as a bare step (no modalities):
        pipeline = PreprocessingPipeline(steps=[ReshapeFeatureDistributionsStep()])

    Configuration options:
        - transform_name: The transformation to apply (e.g., "squashing_scaler_default",
            "quantile_uni_coarse")
        - apply_to_categorical: Whether to transform categorical columns too
        - append_to_original: If True, keep original and append transformed as new
            columns
        - max_features_per_estimator: Subsample features if above this threshold
        - global_transformer_name: Optional global transform like "svd" that adds
            features

    Output column ordering:
        - With append_to_original=True: [original_cols, transformed_cols, (svd_cols)]
        - With append_to_original=False, apply_to_categorical=False:
            [categorical_passthrough, numerical_transformed, (svd_cols)]
        - With append_to_original=False, apply_to_categorical=True:
            [all_transformed, (svd_cols)]
    """

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
        self.transformer_: Pipeline | ColumnTransformer | None = None

    def _create_transformers_and_new_schema(
        self,
        n_samples: int,
        n_features: int,
        feature_schema: FeatureSchema,
    ) -> tuple[Pipeline | ColumnTransformer, FeatureSchema]:
        if "adaptive" in self.transform_name:
            raise NotImplementedError("Adaptive preprocessing raw removed.")

        static_seed, rng = infer_random_state(self.random_state)
        categorical_features = feature_schema.indices_for(FeatureModality.CATEGORICAL)

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
            # Update modalities to reflect subsampled features
            # Create a new schema with only the kept indices
            kept_indices = list(self.subsampled_features_)
            feature_schema = feature_schema.slice_for_indices(kept_indices)
            categorical_features = feature_schema.indices_for(
                FeatureModality.CATEGORICAL
            )
            n_features = subsample_features
        else:
            self.subsampled_features_ = np.arange(n_features)

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

        self.transformer_ = transformer

        # Compute output feature count for modality update
        # Include: base features + appended transformed (if append_to_original)
        n_output_features = (
            n_features + len(trans_ixs) if self.append_to_original else n_features
        )

        # Build the new metadata with updated categorical indices
        # Non-categorical indices become numerical
        new_schema = FeatureSchema.from_only_categorical_indices(
            categorical_indices=sorted(cat_ix),
            num_columns=n_output_features,
        )

        return transformer, new_schema

    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        n_samples, n_features = X.shape
        transformer, output_schema = self._create_transformers_and_new_schema(
            n_samples,
            n_features,
            feature_schema,
        )
        transformer.fit(X[:, self.subsampled_features_])
        self.transformer_ = transformer
        return output_schema

    @override
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        assert self.transformer_ is not None, "You must call fit first"
        return self.transformer_.transform(X[:, self.subsampled_features_]), None, None  # type: ignore


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
                        wrap_with_safe_standard_scaler(
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
                    wrap_with_safe_standard_scaler(
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
        "power": wrap_with_safe_standard_scaler(
            PowerTransformer(standardize=False),
        ),
        "safepower": wrap_with_safe_standard_scaler(
            SafePowerTransformer(standardize=False),
        ),
        "power_box": _make_box_cox_safe(
            wrap_with_safe_standard_scaler(
                PowerTransformer(standardize=False, method="box-cox"),
            ),
        ),
        "safepower_box": _make_box_cox_safe(
            wrap_with_safe_standard_scaler(
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
