#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the opt-in built-model cache in ``tabpfn.model_loading``."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tabpfn import model_loading


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    # Only stat() is read (Checkpoint.identity); the build itself is patched.
    p = tmp_path / "model.ckpt"
    p.write_bytes(b"not a real checkpoint")
    return p


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    model_loading.clear_built_model_cache()
    yield
    model_loading.clear_built_model_cache()


def _patch_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace the real build with a counter returning a fresh sentinel tuple."""
    calls = {"n": 0}

    def fake_build(*_args: object, **_kwargs: object) -> tuple:
        calls["n"] += 1
        return (object(), None, object(), object())

    monkeypatch.setattr(model_loading, "_build_model", fake_build)
    return calls


def test_cache_hit_reuses_built_model(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    first = model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    second = model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )

    assert first is second  # same built model handed back
    assert calls["n"] == 1  # built once, not twice


def test_one_build_serves_both_fit_modes(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    """Every architecture ignores the flag, so the builds are identical."""
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    for cache_trainset_representation in (False, True, False, True):
        model_loading.load_model(
            path=ckpt,
            estimator_type="classifier",
            cache_trainset_representation=cache_trainset_representation,
        )
    assert calls["n"] == 1


def test_cache_disabled_by_default(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    """Opt-in: nothing changes for a caller that has not set the env var."""
    calls = _patch_build(monkeypatch)
    monkeypatch.delenv("TABPFN_MODEL_CACHE_SIZE", raising=False)

    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    assert calls["n"] == 2


def test_size_two_holds_a_classifier_and_a_regressor(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both tasks fit in two entries, so neither evicts the other."""
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "2")

    for estimator_type in ("classifier", "regressor", "classifier", "regressor"):
        model_loading.load_model(
            path=ckpt,
            estimator_type=estimator_type,
            cache_trainset_representation=False,
        )
    assert calls["n"] == 2


def test_cache_can_be_disabled(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "0")

    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    assert calls["n"] == 2


def test_invalid_size_leaves_the_cache_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "not-a-number")
    assert model_loading._get_built_model_cache_size() == 0


def test_models_for_different_devices_are_not_shared(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """Sharing across devices would hand out a model another estimator moved."""
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    cpu = model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("cpu")],
    )
    meta = model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("meta")],
    )
    cpu_again = model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("cpu")],
    )

    assert cpu is not meta
    assert cpu is cpu_again
    assert calls["n"] == 2


def test_models_for_different_dtypes_are_not_shared(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """The cast is destructive, so an fp16 model can never serve fp32."""
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    def load(dtype: torch.dtype | None) -> tuple:
        return model_loading.load_model(
            path=ckpt,
            estimator_type="classifier",
            cache_trainset_representation=False,
            devices=[torch.device("cpu")],
            force_inference_dtype=dtype,
        )

    full = load(None)
    half = load(torch.float16)
    assert full is not half
    assert full is load(None)
    assert half is load(torch.float16)
    assert calls["n"] == 2


def test_unspecified_devices_are_kept_apart_from_device_specific_entries(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("cpu")],
    )
    assert calls["n"] == 2


def test_clear_built_model_cache_forces_a_rebuild(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    def load() -> tuple:
        return model_loading.load_model(
            path=ckpt,
            estimator_type="classifier",
            cache_trainset_representation=False,
            devices=[torch.device("cpu")],
        )

    first = load()
    model_loading.clear_built_model_cache()
    second = load()

    assert first is not second
    assert calls["n"] == 2


def test_lru_eviction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "1")
    a = tmp_path / "a.ckpt"
    a.write_bytes(b"a")
    b = tmp_path / "b.ckpt"
    b.write_bytes(b"b")

    model_loading.load_model(
        path=a, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=b, estimator_type="classifier", cache_trainset_representation=False
    )  # evicts a
    model_loading.load_model(
        path=a, estimator_type="classifier", cache_trainset_representation=False
    )  # rebuilds a
    assert calls["n"] == 3


def test_load_model_signature_is_tracked_by_the_cache():
    """Tripwire: cache correctness depends on `load_model`'s exact inputs.

    A parameter that affects the build, or that describes something the caller
    applies to the model it is handed, belongs in the key — otherwise a hit
    returns a model that does not match the request.
    """
    params = set(inspect.signature(model_loading.load_model).parameters)
    assert params == {
        "path",
        "estimator_type",
        "cache_trainset_representation",
        "devices",
        "force_inference_dtype",
    }, (
        f"load_model parameters changed to {sorted(params)}; check whether "
        "`_BUILT_MODEL_CACHE`'s key needs to cover the new one."
    )
