#  Copyright (c) Prior Labs GmbH 2026.
"""Reducing stored member outputs reproduces the online predictions exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures import tabpfn_v2_5
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.base import ModelSpecs
from tabpfn.classifier import (
    finalize_classifier_probas,
    reduce_classifier_member_logits,
)
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
from tabpfn.regressor import logits_to_output, reduce_regressor_member_outputs

N_TRAIN, N_TEST, N_FEATURES = 60, 9, 4


def _specs(*, regression: bool) -> ModelSpecs:
    config = tabpfn_v2_5.TabPFNV2p5Config(
        emsize=8,
        features_per_group=1,
        max_num_classes=0 if regression else 10,
        nhead=2,
        nlayers=2,
        num_buckets=100,
    )
    torch.manual_seed(0)
    return ModelSpecs(
        model=tabpfn_v2_5.get_architecture(config=config),
        architecture_config=config,
        inference_config=InferenceConfig.get_default(
            task_type="regression" if regression else "multiclass",
            model_version=ModelVersion.V2_5,
        ),
        norm_criterion=(
            FullSupportBarDistribution(torch.linspace(-3, 3, config.num_buckets + 1))
            if regression
            else None
        ),
    )


def _data(*, regression: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_TRAIN + N_TEST, N_FEATURES))
    if regression:
        # Positive and skewed, so the power transform changes the borders.
        y = np.exp(X[:N_TRAIN, 0] + 0.3 * rng.standard_normal(N_TRAIN))
    else:
        y = (X[:N_TRAIN, 0] > 0).astype(int) + (X[:N_TRAIN, 1] > 0.5)
    return X[:N_TRAIN], y, X[N_TRAIN:]


@pytest.mark.parametrize("average_before_softmax", [False, True])
def test__classifier__predict_proba__equals_reduced_raw_logits(
    average_before_softmax: bool,
) -> None:
    X_train, y_train, X_test = _data(regression=False)
    clf = TabPFNClassifier(
        n_estimators=3,
        device="cpu",
        random_state=0,
        inference_precision=torch.float32,
        softmax_temperature=0.8,
        average_before_softmax=average_before_softmax,
        model_path=_specs(regression=False),
    ).fit(X_train, y_train)

    raw_logits = clf.predict_raw_logits(X_test)
    assert raw_logits.shape == (3, N_TEST, 3)
    probas = reduce_classifier_member_logits(
        torch.from_numpy(raw_logits),
        softmax_temperature=clf.softmax_temperature_,
        average_before_softmax=average_before_softmax,
        downsample_correction_weights=None,
        class_counts=None,
    )
    probas = finalize_classifier_probas(
        probas.numpy(),
        classification_thresholds=None,
        use_sklearn_16_decimal_precision=(
            clf.inference_config_.USE_SKLEARN_16_DECIMAL_PRECISION
        ),
    )

    np.testing.assert_array_equal(probas, clf.predict_proba(X_test))


@pytest.mark.parametrize("target_sign_flip", [False, True])
@pytest.mark.parametrize("average_before_softmax", [False, True])
def test__regressor__predict__equals_reduced_member_outputs(
    target_sign_flip: bool,
    average_before_softmax: bool,
) -> None:
    X_train, y_train, X_test = _data(regression=True)
    reg = TabPFNRegressor(
        n_estimators=4,
        device="cpu",
        random_state=0,
        inference_precision=torch.float32,
        softmax_temperature=0.8,
        average_before_softmax=average_before_softmax,
        inference_config={
            "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, "safepower"),
            "REGRESSION_TARGET_SIGN_FLIP": target_sign_flip,
        },
        model_path=_specs(regression=True),
    ).fit(X_train, y_train)
    flips = {config.flip_target for config in reg.ensemble_configs_}
    assert flips == ({False, True} if target_sign_flip else {False})

    members = reg.predict_member_outputs(X_test)
    # Rebuild the decoding distribution from its borders, as an offline scorer must.
    borders = reg.znorm_space_bardist_.borders.clone()
    logits = reduce_regressor_member_outputs(
        members,
        znorm_borders=borders,
        softmax_temperature=reg.softmax_temperature_,
        average_before_softmax=average_before_softmax,
        ensemble_softmax_temperature=1.0,
        downsample_log_weights=None,
    )
    mean = logits_to_output(
        output_type="mean",
        logits=logits,
        znorm_space_bardist=FullSupportBarDistribution(borders),
        y_mean=reg.y_train_mean_,
        y_std=reg.y_train_std_,
        quantiles=[],
    )

    online = reg.predict(X_test, output_type="full")
    assert torch.equal(logits, online["logits"])
    np.testing.assert_array_equal(mean, online["mean"])
