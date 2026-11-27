#  Copyright (c) Prior Labs GmbH 2026.

"""In-memory models retain regression metadata without task-specific wrappers."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tabpfn import ModelSpecs, TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures import tabpfn_v2, tabpfn_v2_5, tabpfn_v3, tabpfn_v3_5
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.base import initialize_tabpfn_model
from tabpfn.constants import ModelVersion
from tabpfn.finetuning.train_util import clone_model_for_evaluation
from tabpfn.inference_config import InferenceConfig


@pytest.fixture
def multitask_specs() -> ModelSpecs:
    config = tabpfn_v3_5.TabPFNV3p5Config(
        max_num_classes=3,
        num_buckets=10,
        embed_dim=16,
        nlayers=1,
        icl_num_heads=2,
        icl_num_kv_heads_test=1,
        dist_embed_num_heads=2,
        dist_embed_num_blocks=1,
        dist_embed_num_inducing_points=4,
        feat_agg_num_heads=2,
        feat_agg_num_blocks=1,
        feat_agg_num_cls_tokens=1,
    )
    return ModelSpecs(
        tabpfn_v3_5.get_architecture(config).eval(),
        config,
        InferenceConfig.get_default("multiclass", ModelVersion.V2_5),
    )


@pytest.fixture(params=[ModelVersion.V2, ModelVersion.V2_5])
def legacy_specs(request: pytest.FixtureRequest) -> ModelSpecs:
    architecture = tabpfn_v2 if request.param == ModelVersion.V2 else tabpfn_v2_5
    config_type = (
        tabpfn_v2.TabPFNV2Config
        if request.param == ModelVersion.V2
        else tabpfn_v2_5.TabPFNV2p5Config
    )
    config = config_type(
        emsize=8,
        features_per_group=1,
        max_num_classes=0,
        nhead=2,
        nlayers=1,
        num_buckets=10,
    )
    return ModelSpecs(
        architecture.get_architecture(config).eval(),
        config,
        InferenceConfig.get_default("regression", request.param),
        FullSupportBarDistribution(torch.linspace(-3, 3, 11)),
    )


@pytest.mark.parametrize("as_list", [False, True])
def test__model_specs__multitask__matches_checkpoint(
    multitask_specs: ModelSpecs, tmp_path: Path, *, as_list: bool
) -> None:
    """Both estimators use one bundle and match normal loading, without disk I/O."""
    specs = multitask_specs
    path = tmp_path / "multitask.ckpt"
    torch.save(
        {
            "state_dict": specs.model.state_dict(),
            "architecture_name": "tabpfn_v3_5",
            "config": asdict(specs.architecture_config),
            "inference_config": asdict(specs.inference_config),
        },
        path,
    )
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2))
    for cls, y, predict in (
        (TabPFNClassifier, np.arange(20) % 3, "predict_proba"),
        (TabPFNRegressor, rng.normal(size=20), "predict"),
    ):
        kwargs = {"device": "cpu", "n_estimators": 1, "random_state": 0}
        reference = cls(model_path=path, **kwargs).fit(x, y)
        expected = getattr(reference, predict)(x[:3])
        with patch(
            "tabpfn.base.load_model_criterion_config",
            side_effect=AssertionError("In-memory specs must not load checkpoints"),
        ):
            estimator = cls(model_path=[specs] if as_list else specs, **kwargs).fit(
                x, y
            )
            np.testing.assert_allclose(
                getattr(estimator, predict)(x[:3]), expected, rtol=1e-6, atol=1e-6
            )
        assert estimator.models_[0] is specs.model


def test__model_specs__derived_borders__are_independent(
    multitask_specs: ModelSpecs,
) -> None:
    _, _, distribution, _ = initialize_tabpfn_model(multitask_specs, "regressor")
    torch.testing.assert_close(
        distribution.borders, multitask_specs.model.regression_borders
    )
    original = multitask_specs.model.regression_borders.clone()
    distribution.borders.add_(1)
    torch.testing.assert_close(multitask_specs.model.regression_borders, original)


def test__model_specs__explicit_distribution__takes_precedence(
    multitask_specs: ModelSpecs,
) -> None:
    explicit = FullSupportBarDistribution(torch.linspace(-2, 2, 11))
    specs = replace(multitask_specs, norm_criterion=explicit)
    assert initialize_tabpfn_model(specs, "regressor")[2] is explicit
    assert initialize_tabpfn_model(specs, "classifier")[2] is None


def test__model_specs__missing_borders__requires_explicit_distribution(
    legacy_specs: ModelSpecs,
) -> None:
    specs = replace(legacy_specs, norm_criterion=None)
    with pytest.raises(ValueError, match="requires norm_criterion"):
        initialize_tabpfn_model(specs, "regressor")
    assert (
        initialize_tabpfn_model(legacy_specs, "regressor")[2]
        is legacy_specs.norm_criterion
    )


def test__model_specs__ensemble__validates_borders(multitask_specs: ModelSpecs) -> None:
    different = replace(
        multitask_specs,
        norm_criterion=FullSupportBarDistribution(torch.linspace(-2, 2, 11)),
    )
    with pytest.raises(ValueError, match="same borders"):
        initialize_tabpfn_model([multitask_specs, different], "regressor")
    # Classification does not interpret regression distributions.
    assert (
        initialize_tabpfn_model([multitask_specs, different], "classifier")[2] is None
    )
    assert (
        initialize_tabpfn_model([multitask_specs, multitask_specs], "regressor")[2]
        is not None
    )


def test__model_specs__legacy_finetuning_copy__preserves_predictions(
    legacy_specs: ModelSpecs,
) -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(20, 2)), rng.normal(size=20)
    kwargs = {"device": "cpu", "n_estimators": 1, "random_state": 0}
    original = TabPFNRegressor(model_path=legacy_specs, **kwargs).fit(x, y)
    copied = clone_model_for_evaluation(original, kwargs, TabPFNRegressor)
    assert isinstance(copied.model_path, ModelSpecs)
    assert copied.model_path.model is not original.models_[0]
    assert not hasattr(copied.model_path.model, "regression_borders")
    assert copied.model_path.norm_criterion is not original.znorm_space_bardist_
    copied.fit(x, y)
    np.testing.assert_allclose(copied.predict(x[:3]), original.predict(x[:3]))


def test__model_specs__half_precision_model__keeps_large_targets_finite(
    multitask_specs: ModelSpecs,
) -> None:
    # Inference can cast the shared network in place. Reusing its specs must not
    # leave target-space arithmetic in float16, whose maximum is only 65504.
    multitask_specs.model.half()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2))
    y = 100_000 + rng.normal(size=20) * 100
    regressor = TabPFNRegressor(
        model_path=multitask_specs,
        device="cpu",
        inference_precision=torch.float32,
        n_estimators=1,
        random_state=0,
    ).fit(x, y)
    assert regressor.znorm_space_bardist_.borders.dtype == torch.float32
    assert torch.isfinite(regressor.raw_space_bardist_.borders).all()
    assert np.isfinite(regressor.predict(x[:3])).all()


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("explicit_criterion", [False, True])
def test__model_specs__classification_only_model__rejects_regression(
    *, as_list: bool, explicit_criterion: bool
) -> None:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=3,
        num_buckets=10,
        embed_dim=48,
        nlayers=1,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    specs = ModelSpecs(
        model=tabpfn_v3.get_architecture(config),
        architecture_config=config,
        inference_config=InferenceConfig.get_default("multiclass", ModelVersion.V2_5),
        norm_criterion=(
            FullSupportBarDistribution(torch.linspace(-3, 3, 11))
            if explicit_criterion
            else None
        ),
    )
    assert specs.model.regression_borders is not None
    with pytest.raises(ValueError, match="classification-only"):
        initialize_tabpfn_model([specs] if as_list else specs, "regressor")
    assert initialize_tabpfn_model(specs, "classifier")[0] == [specs.model]
