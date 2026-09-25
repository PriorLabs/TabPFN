#  Copyright (c) Prior Labs GmbH 2026.
"""The inference engines run equal-shape ensemble members as one batched forward."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from numpy.random import default_rng

from tabpfn.architectures import tabpfn_v3
from tabpfn.inference import (
    InferenceEngineCachePreprocessing,
    InferenceEngineExplicitKVCache,
    InferenceEngineOnDemand,
    _batch_member_inputs,
    _constant,
    _member_groups,
    _members_per_forward,
    _yield_in_member_order,
)
from tabpfn.preprocessing import (
    PreprocessorConfig,
    generate_classification_ensemble_configs,
)
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.torch import FeatureSchema
from tabpfn.settings import settings

N_TRAIN, N_TEST, N_FEATURES, N_CLASSES, N_MEMBERS = 40, 7, 4, 3, 5
# The features plus the fingerprint feature the test preprocessor appends.
N_PREPARED_COLUMNS = N_FEATURES + 1
CPU = torch.device("cpu")
ENGINES = ["on_demand", "cache_preprocessing", "explicit_kv_cache"]


def _model() -> tabpfn_v3.TabPFNV3:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=10,
        num_buckets=5,
        embed_dim=32,
        nlayers=2,
        icl_num_heads=4,
        dist_embed_num_heads=4,
        dist_embed_num_blocks=1,
        feat_agg_num_heads=4,
        feat_agg_num_blocks=1,
        feat_agg_num_cls_tokens=2,
        dist_embed_num_inducing_points=8,
    )
    torch.manual_seed(0)
    return tabpfn_v3.get_architecture(config).to(torch.float32).eval()


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = default_rng(0)
    X_train = rng.standard_normal((N_TRAIN, N_FEATURES))
    y_train = rng.integers(0, N_CLASSES, size=N_TRAIN)
    return X_train, y_train, rng.standard_normal((N_TEST, N_FEATURES))


def _preprocessor(n_members: int = N_MEMBERS) -> TabPFNEnsemblePreprocessor:
    """Members that differ in feature order and class permutation only, as v3.5's."""
    configs = generate_classification_ensemble_configs(
        num_estimators=n_members,
        add_fingerprint_feature=True,
        polynomial_features="no",
        feature_shift_decoder="shuffle",
        preprocessor_configs=[
            PreprocessorConfig(
                "none", categorical_name="numeric", max_features_per_estimator=500
            )
        ],
        class_shift_method="shuffle",
        n_classes=N_CLASSES,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )
    return TabPFNEnsemblePreprocessor(
        configs=configs,
        n_samples=N_TRAIN,
        feature_schema=FeatureSchema.from_only_categorical_indices([], N_FEATURES),
        random_state=default_rng(0),
        n_preprocessing_jobs=1,
    )


def _engine(kind: str, model: tabpfn_v3.TabPFNV3) -> object:
    X_train, y_train, _ = _data()
    common = {
        "ensemble_preprocessor": _preprocessor(),
        "models": [model],
        "devices": [CPU],
        "dtype_byte_size": 4,
        "force_inference_dtype": None,
        "save_peak_mem": False,
    }
    if kind == "on_demand":
        return InferenceEngineOnDemand(X_train, y_train, **common)
    if kind == "cache_preprocessing":
        return InferenceEngineCachePreprocessing(
            X_train, y_train, inference_mode=True, **common
        )
    return InferenceEngineExplicitKVCache(
        X_train, y_train, autocast=False, task_type="multiclass", **common
    )


def _predict(engine: object) -> list[tuple[torch.Tensor, object]]:
    _, _, X_test = _data()
    return list(engine.iter_outputs(X_test, autocast=False, task_type="multiclass"))  # type: ignore[attr-defined]


@torch.no_grad()
@pytest.mark.parametrize("kind", ENGINES)
def test__iter_outputs__batched_matches_sequential_in_member_order(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    batched = _predict(_engine(kind, model))

    monkeypatch.setattr(type(model), "batches_ensemble_members", False)
    sequential = _predict(_engine(kind, model))

    assert len(batched) == len(sequential) == N_MEMBERS
    for (out_b, config_b), (out_s, config_s) in zip(batched, sequential, strict=True):
        assert repr(config_b) == repr(config_s)
        assert out_b.shape == out_s.shape == (N_TEST, 10)
        torch.testing.assert_close(out_b, out_s, atol=1e-5, rtol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("kind", ENGINES)
def test__iter_outputs__row_budget_bounds_the_rows_per_forward(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget of two members' rows splits the work; the outputs are unchanged.

    The engines that run the full forward at predict split the members into
    forwards of at most two. The KV-cache engine builds the cache in forwards of
    at most two members, but the cache still holds every member, and predict
    splits the test rows instead.
    """
    model = _model()
    reference = _predict(_engine(kind, model))
    cached = kind == "explicit_kv_cache"
    rows = N_TEST if cached else N_TRAIN + N_TEST
    monkeypatch.setattr(
        settings.tabpfn, "max_batched_member_cells", 2 * rows * N_PREPARED_COLUMNS
    )
    calls: list[tuple[int, int]] = []
    original = type(model).forward

    def counting(
        self: object, x: torch.Tensor, *args: object, **kwargs: object
    ) -> object:
        calls.append((x.shape[1], x.shape[0]))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(type(model), "forward", counting)
    engine = _engine(kind, model)
    build_calls, calls = calls, []
    outputs = _predict(engine)

    if cached:
        assert engine.cache_groups == [list(range(N_MEMBERS))]  # type: ignore[attr-defined]
        assert all(batch <= 2 * N_TEST // N_TRAIN + 1 for batch, _ in build_calls)
        assert sum(batch for batch, _ in build_calls) == N_MEMBERS
        (batch,) = {batch for batch, _ in calls}
        assert batch == N_MEMBERS
        assert all(rows <= 2 * N_TEST // batch for _, rows in calls)
        assert sum(rows for _, rows in calls) == N_TEST
    else:
        assert not build_calls
        assert all(batch <= 2 for batch, _ in calls)
        assert sum(batch for batch, _ in calls) == N_MEMBERS
    for (out, _), (expected, _) in zip(outputs, reference, strict=True):
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test__explicit_kv_cache__one_cache_holds_all_members() -> None:
    engine = _engine("explicit_kv_cache", _model())
    assert isinstance(engine, InferenceEngineExplicitKVCache)
    assert engine.ensemble_members[0].X_train.shape[1] == N_PREPARED_COLUMNS
    assert engine.cache_groups == [list(range(N_MEMBERS))]
    (cache,) = engine.kv_caches
    assert cache.train_shape == (N_MEMBERS, N_TRAIN)
    for entry in cache.kv.values():
        assert entry.key.shape[0] == N_MEMBERS


@torch.no_grad()
def test__explicit_kv_cache__built_one_member_at_a_time_matches_one_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row budget bounds the build; the concatenated cache predicts the same."""
    model = _model()
    expected = _predict(_engine("explicit_kv_cache", model))

    monkeypatch.setattr(settings.tabpfn, "max_batched_member_rows", N_TRAIN)
    engine = _engine("explicit_kv_cache", model)
    assert isinstance(engine, InferenceEngineExplicitKVCache)
    assert engine.cache_groups == [list(range(N_MEMBERS))]
    (cache,) = engine.kv_caches
    assert cache.train_shape == (N_MEMBERS, N_TRAIN)
    for (out, _), (out_expected, _) in zip(_predict(engine), expected, strict=True):
        torch.testing.assert_close(out, out_expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test__explicit_kv_cache__pickled_copy_predicts_the_same() -> None:
    engine = _engine("explicit_kv_cache", _model())
    assert isinstance(engine, InferenceEngineExplicitKVCache)
    expected = _predict(engine)

    copy = pickle.loads(pickle.dumps(engine._create_copy_for_pickling()))  # noqa: S301
    copy._set_models([engine.model_caches[0].get(CPU)])
    copy.to([CPU], None, 4)
    for (out, _), (out_expected, _) in zip(_predict(copy), expected, strict=True):
        torch.testing.assert_close(out, out_expected, atol=1e-5, rtol=1e-5)


def test__member_groups__splits_by_key_and_size() -> None:
    keys = ["a", "b", "a", "a", "b", "a"]
    assert _member_groups(keys, lambda key, _n: 2 if key == "a" else 10) == [
        [0, 2],
        [3, 5],
        [1, 4],
    ]
    assert _member_groups(keys, _constant(0)) == [[0], [2], [3], [5], [1], [4]]


def test__member_groups__spreads_over_devices_before_batching() -> None:
    """Every device gets a group; only the members left per device share one."""
    keys = ["a"] * 8
    assert _member_groups(keys, _constant(100), num_devices=2) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert _member_groups(keys, _constant(100), num_devices=3) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
    ]
    assert _member_groups(keys, _constant(100), num_devices=16) == [
        [i] for i in range(8)
    ]
    assert _member_groups(keys, _constant(3), num_devices=2) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
    ]


def test__members_per_forward__follows_the_tighter_of_both_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.tabpfn, "max_batched_member_rows", 10_000)
    monkeypatch.setattr(settings.tabpfn, "max_batched_member_cells", 10_000)
    assert _members_per_forward(rows_per_member=100, columns_per_member=10) == 10
    assert _members_per_forward(rows_per_member=100, columns_per_member=20) == 5
    assert _members_per_forward(rows_per_member=2000, columns_per_member=10) == 1
    monkeypatch.setattr(settings.tabpfn, "max_batched_member_cells", 10**9)
    assert _members_per_forward(rows_per_member=100, columns_per_member=20) == 100
    assert _members_per_forward(rows_per_member=4000, columns_per_member=1) == 2
    monkeypatch.setattr(settings.tabpfn, "max_batched_member_rows", 0)
    assert _members_per_forward(rows_per_member=1, columns_per_member=1) == 1


def test__batch_member_inputs__lays_members_along_the_batch_dimension() -> None:
    xs = [torch.full((6, 1, 3), float(i)) for i in range(4)]
    ys = [torch.full((4,), float(i)) for i in range(4)]
    X, y = _batch_member_inputs(xs, ys)
    assert X.shape == (6, 4, 3)
    assert y.shape == (4, 4)
    assert torch.equal(X[:, 2], xs[2][:, 0])
    assert torch.equal(y[:, 2], ys[2])
    _, y_2d = _batch_member_inputs(xs, [y.unsqueeze(1) for y in ys])
    assert torch.equal(y_2d, y)
    X1, y1 = _batch_member_inputs(xs[:1], ys[:1])
    assert X1 is xs[0]
    assert y1 is ys[0]


def test__yield_in_member_order__reorders_and_streams() -> None:
    groups = [[2, 3], [0], [1]]
    outputs = iter(
        [torch.tensor([[2.0, 3.0]]), torch.tensor([[0.0]]), torch.tensor([[1.0]])]
    )
    result = list(_yield_in_member_order(groups, outputs, total=4))
    assert [float(r) for r in result] == [0.0, 1.0, 2.0, 3.0]
