#  Copyright (c) Prior Labs GmbH 2026.

"""Feature names must be unique within any feature schema.

Uniqueness is guaranteed by construction (see the naming helpers in
``datamodel`` and the per-step naming): input features are named from the
DataFrame columns (or positional ``f{i}`` names), and features added by
preprocessing steps are named from the transform plus an index. There is no
runtime assertion of this invariant in production code; instead these tests
verify it holds end-to-end across the preprocessing pipeline presets.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from tabpfn.preprocessing.configs import EnsembleConfig, PreprocessorConfig
from tabpfn.preprocessing.datamodel import (
    Feature,
    FeatureModality,
    FeatureSchema,
    build_input_feature_names,
    make_names_unique,
)
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline

RANDOM_STATE = 42


def assert_unique_feature_names(schema: FeatureSchema) -> None:
    """Assert all feature names in the schema are non-None and unique."""
    names = schema.feature_names
    assert all(n is not None for n in names), (
        f"schema has unnamed (None) features: {names}"
    )
    assert len(names) == len(set(names)), (
        f"schema has duplicate feature names: "
        f"{[n for n in set(names) if names.count(n) > 1]}"
    )


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #


class TestMakeNamesUnique:
    def test__no_collisions__unchanged(self) -> None:
        assert make_names_unique(["a", "b", "c"]) == ["a", "b", "c"]

    def test__duplicates__suffixed(self) -> None:
        assert make_names_unique(["a", "a", "a"]) == ["a", "a_1", "a_2"]

    def test__collides_with_existing(self) -> None:
        assert make_names_unique(["a", "b"], existing=["a"]) == ["a_1", "b"]

    def test__result_is_always_unique(self) -> None:
        # Resolution is order-dependent (only already-emitted names are known),
        # but the result is always collision-free.
        out = make_names_unique(["a", "a", "a_1"])
        assert len(out) == len(set(out))
        assert out[0] == "a"


class TestBuildInputFeatureNames:
    def test__dataframe_names_prefixed(self) -> None:
        assert build_input_feature_names(["age", "city"], 2) == [
            "input_age",
            "input_city",
        ]

    def test__duplicate_dataframe_names_disambiguated(self) -> None:
        assert build_input_feature_names(["x", "x", "x"], 3) == [
            "input_x",
            "input_x_1",
            "input_x_2",
        ]

    def test__array_input_positional(self) -> None:
        assert build_input_feature_names(None, 3) == ["f0", "f1", "f2"]


class TestFromOnlyCategoricalIndicesNames:
    def test__default_positional_names(self) -> None:
        schema = FeatureSchema.from_only_categorical_indices([1], num_columns=3)
        assert schema.feature_names == ["f0", "f1", "f2"]

    def test__explicit_names_preserved(self) -> None:
        schema = FeatureSchema.from_only_categorical_indices(
            [1], num_columns=3, names=["a", "b", "c"]
        )
        assert schema.feature_names == ["a", "b", "c"]
        assert schema.indices_for(FeatureModality.CATEGORICAL) == [1]

    def test__wrong_name_count_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 3 names"):
            FeatureSchema.from_only_categorical_indices(
                [1], num_columns=3, names=["a", "b"]
            )


class TestAppendColumnsDedup:
    def test__prefix_generates_names(self) -> None:
        schema = FeatureSchema(
            features=[Feature(name="input_a", modality=FeatureModality.NUMERICAL)]
        )
        new = schema.append_columns(
            FeatureModality.NUMERICAL, num_new=2, name_prefix="svd"
        )
        assert new.feature_names == ["input_a", "svd_0", "svd_1"]

    def test__appended_names_deduped_against_existing(self) -> None:
        schema = FeatureSchema(
            features=[Feature(name="svd_0", modality=FeatureModality.NUMERICAL)]
        )
        new = schema.append_columns(
            FeatureModality.NUMERICAL, num_new=1, name_prefix="svd"
        )
        assert new.feature_names == ["svd_0", "svd_0_1"]


# --------------------------------------------------------------------------- #
# End-to-end pipeline uniqueness
# --------------------------------------------------------------------------- #


def _mixed_dataframe(
    rng: np.random.Generator,
    *,
    n_samples: int = 60,
    n_numerical: int = 3,
    n_categorical: int = 2,
    duplicate_column_names: bool = False,
) -> pd.DataFrame:
    """A DataFrame with numerical + low-cardinality categorical columns."""
    data: dict[str, np.ndarray] = {}
    for i in range(n_numerical):
        col = rng.standard_normal(n_samples) * 10
        col[rng.random(n_samples) < 0.1] = np.nan
        data[f"num{i}"] = col
    for i in range(n_categorical):
        data[f"cat{i}"] = rng.integers(0, 4, size=n_samples).astype(float)

    df = pd.DataFrame(data)
    if duplicate_column_names:
        # pandas permits duplicate column labels; the input naming must
        # disambiguate them.
        df.columns = ["dup"] * df.shape[1]
    return df


def _schema_from_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, FeatureSchema]:
    X = df.to_numpy(dtype=np.float64)
    schema = detect_feature_modalities(
        X=X,
        feature_names=list(df.columns),
        min_samples_for_inference=1,
        max_unique_for_category=6,
        min_unique_for_numerical=5,
    )
    return X, schema


_PREPROCESSOR_CONFIGS: dict[str, PreprocessorConfig] = {
    "none_numeric": PreprocessorConfig(name="none", categorical_name="numeric"),
    "none_ordinal": PreprocessorConfig(name="none", categorical_name="ordinal"),
    "none_onehot": PreprocessorConfig(name="none", categorical_name="onehot"),
    "none_ordinal_shuffled": PreprocessorConfig(
        name="none", categorical_name="ordinal_shuffled"
    ),
    "quantile_uni_coarse": PreprocessorConfig(
        name="quantile_uni_coarse", categorical_name="numeric"
    ),
    "squashing_scaler": PreprocessorConfig(
        name="squashing_scaler_default",
        categorical_name="ordinal_very_common_categories_shuffled",
    ),
    "robust_onehot": PreprocessorConfig(name="robust", categorical_name="onehot"),
    "quantile_append_original": PreprocessorConfig(
        name="quantile_uni_coarse", categorical_name="numeric", append_original=True
    ),
    "none_with_svd": PreprocessorConfig(
        name="none", categorical_name="numeric", global_transformer_name="svd"
    ),
    "squashing_with_svd_quarter": PreprocessorConfig(
        name="squashing_scaler_default",
        categorical_name="ordinal_very_common_categories_shuffled",
        global_transformer_name="svd_quarter_components",
    ),
}


def _ensemble_config(
    preprocess_config: PreprocessorConfig,
    *,
    add_fingerprint_feature: bool = False,
    polynomial_features: Literal["no", "all"] | int = "no",
    feature_shift_count: int = 0,
    feature_shift_decoder: Literal["shuffle", "rotate"] | None = None,
) -> EnsembleConfig:
    return EnsembleConfig(
        preprocess_config=preprocess_config,
        add_fingerprint_feature=add_fingerprint_feature,
        polynomial_features=polynomial_features,
        feature_shift_count=feature_shift_count,
        feature_shift_decoder=feature_shift_decoder,
        outlier_removal_std=None,
        _model_index=0,
        passthrough_inf=False,
    )


@pytest.mark.parametrize("config_name", list(_PREPROCESSOR_CONFIGS))
@pytest.mark.parametrize("add_fingerprint", [False, True])
@pytest.mark.parametrize("polynomial_features", ["no", "all"])
def test__pipeline_output_has_unique_feature_names(
    config_name: str,
    add_fingerprint: bool,
    polynomial_features: Literal["no", "all"],
) -> None:
    """The schema after a full preprocessing pipeline has unique names."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = _mixed_dataframe(rng)
    X, schema = _schema_from_dataframe(df)
    assert_unique_feature_names(schema)  # input naming itself is unique

    config = _ensemble_config(
        _PREPROCESSOR_CONFIGS[config_name],
        add_fingerprint_feature=add_fingerprint,
        polynomial_features=polynomial_features,
    )
    pipeline = create_preprocessing_pipeline(config, random_state=RANDOM_STATE)
    result = pipeline.fit_transform(X, schema)

    assert_unique_feature_names(result.feature_schema)
    assert result.feature_schema.num_columns == result.X.shape[1]


def test__duplicate_input_columns_are_disambiguated() -> None:
    """Duplicate DataFrame column labels still yield unique input names."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = _mixed_dataframe(rng, duplicate_column_names=True)
    _, schema = _schema_from_dataframe(df)
    assert_unique_feature_names(schema)
    assert schema.feature_names[0] == "input_dup"
