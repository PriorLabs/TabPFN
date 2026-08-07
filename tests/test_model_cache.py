#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the opt-in built-model cache in ``tabpfn.model_loading``."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest

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

    first = model_loading.load_model(path=ckpt, cache_trainset_representation=False)
    second = model_loading.load_model(path=ckpt, cache_trainset_representation=False)

    assert first is second  # same built model handed back
    assert calls["n"] == 1  # built once, not twice


def test_mutating_build_is_never_cached(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    # cache_trainset_representation=True mutates the model during fit, so it is
    # rebuilt every time rather than served from the shared cache.
    model_loading.load_model(path=ckpt, cache_trainset_representation=True)
    model_loading.load_model(path=ckpt, cache_trainset_representation=True)
    assert calls["n"] == 2


def test_cache_disabled_by_default(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.delenv("TABPFN_MODEL_CACHE_SIZE", raising=False)  # default 0 = off

    model_loading.load_model(path=ckpt, cache_trainset_representation=False)
    model_loading.load_model(path=ckpt, cache_trainset_representation=False)
    assert calls["n"] == 2  # no caching; prior behaviour preserved


def test_lru_eviction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "1")
    a = tmp_path / "a.ckpt"
    a.write_bytes(b"a")
    b = tmp_path / "b.ckpt"
    b.write_bytes(b"b")

    model_loading.load_model(path=a, cache_trainset_representation=False)
    model_loading.load_model(path=b, cache_trainset_representation=False)  # evicts a
    model_loading.load_model(path=a, cache_trainset_representation=False)  # rebuilds a
    assert calls["n"] == 3


def test_load_model_signature_is_tracked_by_the_cache():
    """Tripwire: the cache correctness depends on `load_model`'s exact inputs.

    It keys on (path, file-identity) and caches only when
    ``cache_trainset_representation`` is False. So a *new build-affecting*
    parameter must be added to the cache key (else a hit returns a stale model),
    and a *new mutation flag* must extend the gate (else a mutated model gets
    shared). If this assertion fails, revisit `_BUILT_MODEL_CACHE` / `load_model`
    before updating the expected set.
    """
    params = set(inspect.signature(model_loading.load_model).parameters)
    assert params == {"path", "cache_trainset_representation"}, (
        f"load_model parameters changed to {sorted(params)}; the built-model "
        "cache key and/or its cache_trainset_representation gate must be updated."
    )
