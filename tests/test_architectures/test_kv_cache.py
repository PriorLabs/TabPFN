#  Copyright (c) Prior Labs GmbH 2026.
"""Tests for KV cache quantization, including the fp8 storage format.

Everything here is hardware-independent: fp8 KV storage is plain tensor casts
(``quantized = clamp(t / scale)`` cast to ``float8_e4m3fn``, ``float =
quantized * scale`` back), so the whole suite runs on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier
from tabpfn.architectures.kv_cache import (
    FP8_KV_DTYPE,
    KVCacheEntry,
    _dequantize_tensor,
    _quantize_tensor,
)
from tabpfn.architectures.tabpfn_v3 import TabPFNV3Config, get_cache_size
from tabpfn.inference import resolve_kv_cache_precision


def _kv_tensor(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    # (B, N, num_kv_heads, head_dim), with per-head magnitude variation so
    # per-head scales actually differ.
    t = torch.randn(2, 64, 4, 8)
    return t * torch.tensor([0.1, 1.0, 5.0, 50.0]).view(1, 1, 4, 1)


def test_fp8_roundtrip_error_band() -> None:
    t = _kv_tensor()
    quantized, scale = _quantize_tensor(t, FP8_KV_DTYPE)
    assert quantized.dtype == FP8_KV_DTYPE
    restored = _dequantize_tensor(quantized, scale, torch.float32)
    # e4m3 has a 3-bit mantissa: a few percent relative error.
    assert (restored - t).norm() / t.norm() < 0.05


def test_fp8_uses_the_same_scalar_scale_contract_as_int8() -> None:
    t = _kv_tensor()
    _, scale = _quantize_tensor(t, FP8_KV_DTYPE)
    assert scale.dim() == 0
    # Unlike int8's uniform grid, fp8's relative precision is unaffected by
    # the per-head amplitude spread under a single scale.
    for head in range(4):
        head_slice = t[:, :, head]
        quantized, s = _quantize_tensor(t, FP8_KV_DTYPE)
        restored = _dequantize_tensor(quantized, s, torch.float32)[:, :, head]
        assert (restored - head_slice).norm() / head_slice.norm() < 0.05


def test_int8_behaviour_unchanged() -> None:
    t = _kv_tensor()
    quantized, scale = _quantize_tensor(t, torch.int8)
    assert quantized.dtype == torch.int8
    assert scale.dim() == 0  # per-tensor scalar, as before
    restored = _dequantize_tensor(quantized, scale, torch.float32)
    # int8's uniform grid is dominated by the loudest head here, so the
    # tolerance is loose; the point is the code path is untouched.
    assert (restored - t).norm() / t.norm() < 0.2


def test_fp8_never_produces_nan_and_respects_absmax() -> None:
    # e4m3fn has no inf: values above the max would become NaN without the
    # clamp. Include an extreme outlier and an all-zero tensor.
    t = _kv_tensor()
    t[0, 0, 0, 0] = 1e30
    quantized, scale = _quantize_tensor(t, FP8_KV_DTYPE)
    restored = _dequantize_tensor(quantized, scale, torch.float32)
    assert not torch.isnan(restored).any()
    assert restored.abs().max() <= t.abs().max() * 1.01

    zeros = torch.zeros(1, 4, 2, 8)
    quantized, scale = _quantize_tensor(zeros, FP8_KV_DTYPE)
    assert torch.all(_dequantize_tensor(quantized, scale, torch.float32) == 0)


def test_unsupported_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported quantization dtype"):
        _quantize_tensor(_kv_tensor(), torch.int16)


def test_entry_quantize_to_fp8_and_back() -> None:
    k, v = _kv_tensor(1), _kv_tensor(2)
    entry = KVCacheEntry(key=k, value=v).quantize(FP8_KV_DTYPE)
    assert entry.key.dtype == FP8_KV_DTYPE
    assert entry.value.dtype == FP8_KV_DTYPE
    restored = entry.dequantize(torch.float32)
    assert (restored.key - k).norm() / k.norm() < 0.05
    assert (restored.value - v).norm() / v.norm() < 0.05


class _FakeArchitecture:
    def __init__(self, supported: tuple[str, ...]):
        self._supported = supported

    def get_supported_kv_cache_precisions(self) -> tuple[str, ...]:
        return self._supported


def test_resolve_accepts_fp8_and_default_stays_int8() -> None:
    arch = _FakeArchitecture(("auto", "int8", "fp8"))
    assert resolve_kv_cache_precision("fp8", architecture=arch) == "fp8"
    # The default is unchanged by this feature: unset still resolves to int8.
    assert resolve_kv_cache_precision(None, architecture=arch) == "int8"


def test_resolve_falls_back_to_auto_when_unsupported() -> None:
    arch = _FakeArchitecture(("auto",))
    with pytest.warns(UserWarning, match="not supported"):
        assert resolve_kv_cache_precision("fp8", architecture=arch) == "auto"


def test_get_cache_size_accepts_fp8() -> None:
    config = TabPFNV3Config()
    kwargs = {
        "n_train": 1000,
        "n_features": 20,
        "model_config": config,
        "base_dtype": torch.float32,
    }
    # int8 and fp8 are both one byte per element plus scales.
    assert get_cache_size(kv_cache_precision="fp8", **kwargs) == get_cache_size(
        kv_cache_precision="int8", **kwargs
    )
    with pytest.raises(ValueError, match="Invalid kv_cache_precision"):
        get_cache_size(kv_cache_precision="int4", **kwargs)  # type: ignore[arg-type]


def test_classifier_fp8_cache_end_to_end_on_cpu() -> None:
    """The whole feature is tensor casts only, so it must work on plain CPU."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=60) > 0).astype(int)
    clf = TabPFNClassifier(
        fit_mode="fit_with_cache",
        kv_cache_precision="fp8",
        n_estimators=1,
        device="cpu",
        random_state=0,
    )
    clf.fit(X[:40], y[:40])
    dtypes = {
        entry.key.dtype
        for cache in clf.executor_.kv_caches
        for entry in cache.kv.values()
    }
    assert dtypes == {FP8_KV_DTYPE}
    proba = clf.predict_proba(X[40:])
    assert np.all(np.isfinite(proba))
    assert proba.shape == (20, 2)
