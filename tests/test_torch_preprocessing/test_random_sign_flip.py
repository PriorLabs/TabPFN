#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig
from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.torch.factory import create_gpu_preprocessing_pipeline
from tabpfn.preprocessing.torch.pipeline_interface import TorchPreprocessingPipeline
from tabpfn.preprocessing.torch.steps import TorchRandomSignFlipStep


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [42, None])
def test_signs_are_reused_across_rows_and_predictions(
    dtype: torch.dtype, seed: int | None
) -> None:
    step = TorchRandomSignFlipStep(random_state=seed)
    x = torch.arange(1, 129, dtype=dtype).reshape(4, 2, 16)
    x[1, 0, 0] = torch.nan
    result = step.fit_transform(x, list(range(16)), num_train_rows=2)
    signs = result.x[:1] / x[:1]
    if seed is not None:
        assert set(signs.unique().tolist()) == {-1.0, 1.0}
        assert not torch.equal(signs[:, 0], signs[:, 1])
    torch.testing.assert_close(result.x, x * signs, equal_nan=True)

    for cache in (None, result.fitted_cache):
        prediction = step.fit_transform(
            x[2:], list(range(16)), num_train_rows=1, fitted_cache=cache
        )
        torch.testing.assert_close(prediction.x, (x * signs)[2:])
    assert result.x.dtype == dtype
    assert result.x.device == x.device
    assert x[0, 0, 0] == 1


@pytest.mark.parametrize("n_categorical", [2, 16])
def test_pipeline_preserves_categoricals_and_cached_signs(n_categorical: int) -> None:
    schema = FeatureSchema.from_only_categorical_indices(list(range(n_categorical)), 16)
    pipeline = TorchPreprocessingPipeline(
        [(TorchRandomSignFlipStep(42), {FeatureModality.NUMERICAL})],
        keep_fitted_cache=True,
    )
    x = torch.ones(4, 16)
    fitted = pipeline(x, schema, num_train_rows=2)
    prediction = pipeline(x[:1], schema, use_fitted_cache=True)
    torch.testing.assert_close(fitted.x[:1], prediction.x)
    torch.testing.assert_close(fitted.x[:, :n_categorical], x[:, :n_categorical])
    assert fitted.feature_schema == schema
    if n_categorical < 16:
        assert (fitted.x[:, n_categorical:] == -1).any()


def test_sign_flip_preserves_gradients() -> None:
    x = torch.ones(3, 1, 16, requires_grad=True)
    result = TorchRandomSignFlipStep(42).fit_transform(
        x, list(range(16)), num_train_rows=2
    )
    result.x.sum().backward()
    torch.testing.assert_close(x.grad, result.x)


def _make_config(*, enabled: bool) -> ClassifierEnsembleConfig:
    return ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig(
            "none", random_sign_flip_numericals=enabled
        ),
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_count=0,
        feature_shift_decoder=None,
        outlier_removal_std=None,
        _model_index=0,
        passthrough_inf=False,
        class_permutation=None,
    )


def test_sign_flip_is_opt_in() -> None:
    config = _make_config(enabled=False)
    assert not PreprocessorConfig("none").random_sign_flip_numericals
    assert create_gpu_preprocessing_pipeline(config) is None
    enabled = replace(
        config,
        preprocess_config=replace(
            config.preprocess_config, random_sign_flip_numericals=True
        ),
    )
    assert create_gpu_preprocessing_pipeline(enabled) is not None
    assert str(config.preprocess_config) != str(enabled.preprocess_config)


@pytest.mark.parametrize("enable_gpu_preprocessing", [False, True])
def test_ensemble_members_have_reproducible_independent_signs(
    enable_gpu_preprocessing: bool,
) -> None:
    X = np.random.default_rng(0).uniform(1, 2, (10, 32))
    schema = FeatureSchema.from_only_categorical_indices([], X.shape[1])

    def run_ensemble() -> list[torch.Tensor]:
        preprocessor = TabPFNEnsemblePreprocessor(
            configs=[_make_config(enabled=True) for _ in range(2)],
            n_samples=len(X),
            feature_schema=schema,
            random_state=42,
            n_preprocessing_jobs=1,
            enable_gpu_preprocessing=enable_gpu_preprocessing,
        )
        outputs = []
        for member in preprocessor.fit_transform_ensemble_members_iterator(
            X, np.arange(len(X)) % 2, parallel_mode="in-order"
        ):
            assert member.gpu_preprocessor is not None
            outputs.append(
                member.gpu_preprocessor(
                    torch.as_tensor(member.X_train), member.feature_schema
                ).x
            )
        return outputs

    first, second = run_ensemble(), run_ensemble()
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a.abs(), torch.from_numpy(X))
    assert not torch.equal(first[0], first[1])
