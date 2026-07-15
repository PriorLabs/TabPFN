#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the v3 single-file model."""

from __future__ import annotations

import dataclasses
import sys

import pytest
import torch

from tabpfn import TabPFNClassifier
from tabpfn.architectures import tabpfn_v3
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.kv_cache import (
    KVCacheEntry,
    QuantizedKVCacheEntry,
)
from tabpfn.architectures.tabpfn_v3 import TabPFNV3Cache, get_cache_size
from tabpfn.utils import get_autocast_context


def _get_model() -> tabpfn_v3.TabPFNV3:
    """Construct v2.5 and base such that they have the same outputs."""
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=10,
        num_buckets=5,
        embed_dim=48,
        nlayers=1,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float32)
    return model


@torch.no_grad()
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled() -> None:
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float32)

    output_without_memory_saving = arch(x, y, only_return_standard_out=False)
    output_with_memory_saving = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(save_peak_memory_factor=4),
    )

    msg = "Output keys do not match between implementations"
    assert output_with_memory_saving.keys() == output_without_memory_saving.keys(), msg
    for key in output_with_memory_saving:
        assert torch.allclose(
            output_with_memory_saving[key],
            output_without_memory_saving[key],
            atol=1e-6,
        ), f"Outputs for {key} do not match between implementations."


@torch.no_grad()
def test__forward_pass_equal_with_checkpointing_enabled_and_disabled() -> None:
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float32)

    output_without_recomputation = arch(x, y, only_return_standard_out=False)
    output_with_recomputation = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(force_recompute_layer=True),
    )

    msg = "Output keys do not match between implementations"
    assert output_with_recomputation.keys() == output_without_recomputation.keys(), msg
    for key in output_with_recomputation:
        assert torch.allclose(
            output_with_recomputation[key],
            output_without_recomputation[key],
            atol=1e-6,
        ), f"Outputs for {key} do not match between implementations."


@torch.no_grad()
def test__batch_size_one__padding_still_works() -> None:
    arch = _get_model()

    x = torch.randn(100, 1, 1, dtype=torch.float32) * 0.1
    x[10, 0] = float("nan")
    x[11, 0] = float("inf")
    y = torch.randint(0, 10, [97, 1], dtype=torch.float32)
    output = arch(x, y)

    assert output.shape == (3, 1, 10)


@torch.no_grad()
def test__forward__no_test_set_works_batch_size_one() -> None:
    arch = _get_model()

    x = torch.randn(1, 1, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [1, 1], dtype=torch.float32)

    out = arch(x, y, only_return_standard_out=False)
    assert out["standard"].shape == (0, 1, 10)


@torch.no_grad()
def test__mem_eff_forward_matches_standard_forward() -> None:
    """Memory-efficient inference path must be numerically identical to standard."""
    arch = _get_model()

    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float32)

    # Standard path: disable memory-efficient inference via forward argument.
    output_standard = arch(x, y, only_return_standard_out=False)

    # Memory-efficient path: small fixed chunk sizes to force chunking
    # even on this tiny dataset.
    arch.inference_row_chunk_size = 50
    arch.inference_col_chunk_size = 10
    output_mem_eff = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(use_chunkwise_inference=True),
    )

    assert isinstance(output_standard, dict)
    assert isinstance(output_mem_eff, dict)
    assert output_mem_eff.keys() == output_standard.keys(), (
        "Output keys do not match between standard and memory-efficient paths."
    )
    for key in output_mem_eff:
        assert torch.allclose(output_mem_eff[key], output_standard[key], atol=1e-5), (
            f"Outputs for '{key}' differ between standard and memory-efficient "
            "forward passes."
        )


@torch.no_grad()
def test__chunked_inference_recovers_from_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recoverable OOM during chunked inference must not crash the forward.

    The column-chunk handler reacts to an OOM by freeing memory, halving the
    chunk and retrying. This used to call ``torch.mps.empty_cache()``
    unconditionally, which raises ``Cannot execute emptyCache() without MPS
    backend`` on any non-MPS device (CUDA GPUs, CPU-only Linux), turning a
    recoverable OOM into a hard crash on the CI runners. The recovered output
    must also match the standard forward pass.
    """
    arch = _get_model()
    # Chunkwise inference (and hence the OOM recovery path) only runs in eval mode.
    arch.eval()

    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float32)

    expected = arch(x, y, only_return_standard_out=False)

    # Force chunking so the inducing-hidden (column) recovery path is exercised.
    arch.inference_row_chunk_size = 50
    arch.inference_col_chunk_size = 10

    # Raise a single OOM the first time a column chunk is processed, so the
    # handler must free memory, halve the column chunk and retry.
    # Patch on the class (not the instance) so that, should torch.compile be
    # enabled, the bound method still exposes ``__func__`` for ``_compiled``.
    original_process_col_chunk = tabpfn_v3.TabPFNV3._process_col_chunk
    calls = {"n": 0}

    def _process_col_chunk_oom_once(
        self: tabpfn_v3.TabPFNV3, *args: object, **kwargs: object
    ) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return original_process_col_chunk(self, *args, **kwargs)

    monkeypatch.setattr(
        tabpfn_v3.TabPFNV3, "_process_col_chunk", _process_col_chunk_oom_once
    )

    recovered = arch(
        x,
        y,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(use_chunkwise_inference=True),
    )

    assert calls["n"] > 1, "the simulated OOM never triggered a retry"
    assert recovered.keys() == expected.keys()
    for key in recovered:
        assert torch.allclose(recovered[key], expected[key], atol=1e-5), (
            f"Output '{key}' after OOM recovery differs from the standard forward pass."
        )


def _get_regression_model() -> tabpfn_v3.TabPFNV3:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=-1,
        num_buckets=100,
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
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float32)
    return model


@torch.no_grad()
@pytest.mark.parametrize("use_chunkwise", [False, True])
def test__kv_cache__matches_standard_forward(use_chunkwise: bool) -> None:
    """KV-cache inference must produce identical output to standard forward."""
    arch = _get_regression_model()

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5, dtype=torch.float32) * 0.1
    y = torch.randn(10, dtype=torch.float32)

    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    # Standard forward (no cache)
    out_standard = arch(x, y, performance_options=perf)

    # Build cache
    out_store, cache = arch(x, y, performance_options=perf, return_kv_cache=True)

    assert isinstance(cache, TabPFNV3Cache)
    assert not cache.is_empty()
    assert cache.train_embeddings is not None
    assert len(cache.kv) == 2  # nlayers=2

    # Store-mode output matches standard
    assert torch.allclose(out_standard, out_store, atol=1e-6), (
        "return_kv_cache=True output differs from standard."
    )

    # Use cache for inference
    out_cached = arch(x, y, performance_options=perf, kv_cache=cache)

    assert torch.allclose(out_standard, out_cached, atol=1e-6), (
        "kv_cache inference output differs from standard."
    )


@torch.no_grad()
def test__kv_cache__multiclass_matches_standard() -> None:
    """KV-cache inference for multiclass produces identical output."""
    arch = _get_model()

    torch.manual_seed(42)
    x = torch.randn(20, 1, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, (10,), dtype=torch.float32)

    perf = PerformanceOptions(use_chunkwise_inference=False)

    out_standard = arch(x, y, performance_options=perf)
    out_store, cache = arch(x, y, performance_options=perf, return_kv_cache=True)
    out_cached = arch(x, y, performance_options=perf, kv_cache=cache)

    assert torch.allclose(out_standard, out_store, atol=1e-6)
    assert torch.allclose(out_standard, out_cached, atol=1e-6)


@torch.no_grad()
def test__kv_cache__row_chunked_matches_unchunked() -> None:
    """Cached forward with a small inference_row_chunk_size must match unchunked.

    Exercises the chunked branch of ``_forward_with_cache`` (R_test >
    row_chunk_size), which the existing cache tests don't hit because
    ``inference_row_chunk_size="auto"`` short-circuits to a single chunk
    on small R_test.
    """
    arch = _get_regression_model()

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5, dtype=torch.float32) * 0.1
    y = torch.randn(10, dtype=torch.float32)

    perf = PerformanceOptions(use_chunkwise_inference=False)

    # Reference: default "auto" → single-chunk on 10 test rows
    out_standard = arch(x, y, performance_options=perf)
    _, cache = arch(x, y, performance_options=perf, return_kv_cache=True)

    # Force multi-chunk test-row processing: 10 test rows / 3 per chunk = 4 chunks
    arch.inference_row_chunk_size = 3
    out_cached_chunked = arch(x, y, performance_options=perf, kv_cache=cache)

    assert torch.allclose(out_standard, out_cached_chunked, atol=1e-6), (
        "Row-chunked cached forward differs from unchunked."
    )


@torch.no_grad()
def test__kv_cache__gqa_matches_standard() -> None:
    """KV-cache inference with GQA (num_kv_heads_test) produces identical output."""
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=-1,
        num_buckets=100,
        embed_dim=32,
        nlayers=2,
        icl_num_heads=4,
        icl_num_kv_heads=4,
        icl_num_kv_heads_test=2,
        dist_embed_num_heads=4,
        dist_embed_num_blocks=1,
        feat_agg_num_heads=4,
        feat_agg_num_blocks=1,
        feat_agg_num_cls_tokens=2,
        dist_embed_num_inducing_points=8,
    )
    arch = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    arch.to(torch.float32)

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5, dtype=torch.float32) * 0.1
    y = torch.randn(10, dtype=torch.float32)

    perf = PerformanceOptions(use_chunkwise_inference=False)

    out_standard = arch(x, y, performance_options=perf)
    _, cache = arch(x, y, performance_options=perf, return_kv_cache=True)
    out_cached = arch(x, y, performance_options=perf, kv_cache=cache)

    assert torch.allclose(out_standard, out_cached, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("use_chunkwise", [False, True])
@pytest.mark.parametrize(
    "autocast_dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                sys.platform == "win32" and not torch.cuda.is_available(),
                reason=(
                    "bf16 CPU kernels crash with STATUS_ILLEGAL_INSTRUCTION "
                    "(0xc000001d) on Windows CI runners"
                ),
            ),
        ),
    ],
)
def test__kv_cache__works_under_autocast(
    use_chunkwise: bool, autocast_dtype: torch.dtype
) -> None:
    """KV cache inference works correctly under torch.autocast (fp16/bf16)."""
    arch = _get_regression_model().float()  # model in fp32

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5) * 0.1
    y = torch.randn(10)

    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    # Build cache WITHOUT autocast (fp32 cache)
    _, cache = arch(x, y, performance_options=perf, return_kv_cache=True)
    assert cache is not None

    # Standard forward WITHOUT autocast (reference)
    out_standard = arch(x, y, performance_options=perf)

    # Use cache WITH autocast — this is the scenario that triggered the
    # fp32-cache-under-fp16-input dtype mismatch.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        out_cached_autocast = arch(x, y, performance_options=perf, kv_cache=cache)

    # Also test standard forward under autocast for reference
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        out_standard_autocast = arch(x, y, performance_options=perf)

    # Autocast introduces precision differences; use a loose tolerance
    assert torch.allclose(
        out_standard.float(), out_cached_autocast.float(), atol=1e-2
    ), (
        f"Autocast ({autocast_dtype}) KV-cache output too far from standard "
        f"(max diff: {(out_standard.float() - out_cached_autocast.float()).abs().max().item():.2e})"  # noqa: E501
    )
    assert torch.allclose(
        out_standard_autocast.float(), out_cached_autocast.float(), atol=1e-2
    ), (
        f"Autocast ({autocast_dtype}) KV-cache output differs from autocast standard "
        f"(max diff: {(out_standard_autocast.float() - out_cached_autocast.float()).abs().max().item():.2e})"  # noqa: E501
    )


@torch.no_grad()
def test__kv_cache_entry__quantize_dequantize_roundtrip() -> None:
    """Quantize/dequantize roundtrip preserves values within int8 tolerance."""
    torch.manual_seed(0)
    entry = KVCacheEntry(key=torch.randn(2, 32, 4, 16), value=torch.randn(2, 32, 4, 16))

    q = entry.quantize()
    assert isinstance(q, QuantizedKVCacheEntry)
    assert q.key.dtype == torch.int8
    assert q.value.dtype == torch.int8

    d = q.dequantize(torch.float32)
    assert d.key.dtype == torch.float32
    # Per-tensor int8 error: max ~absmax/127
    assert (entry.key - d.key).abs().max() < entry.key.abs().amax() / 64
    assert (entry.value - d.value).abs().max() < entry.value.abs().amax() / 64


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test__kv_cache_entry__quantize_all_zero_is_finite(dtype: torch.dtype) -> None:
    """All-zero inputs must round-trip without NaN/Inf across float dtypes.

    Regression test for the scale-floor: a fixed 1e-12 underflows to 0 in
    float16 and reintroduces division by zero on line 44.
    """
    entry = KVCacheEntry(
        key=torch.zeros(1, 4, 1, 2, dtype=dtype),
        value=torch.zeros(1, 4, 1, 2, dtype=dtype),
    )
    q = entry.quantize()
    d = q.dequantize(dtype)
    assert torch.all(torch.isfinite(d.key))
    assert torch.all(torch.isfinite(d.value))
    assert d.key.abs().max().item() == 0
    assert d.value.abs().max().item() == 0


def test__kv_cache_entry__quantize_unsupported_dtype_raises() -> None:
    """Quantization with an unregistered integer dtype should raise."""
    entry = KVCacheEntry(
        key=torch.randn(1, 1, 1, 1),
        value=torch.randn(1, 1, 1, 1),
    )
    with pytest.raises(ValueError, match="Unsupported quantization dtype"):
        entry.quantize(dtype=torch.int16)


def test__kv_cache__quantize_passthrough_on_already_quantized() -> None:
    """quantize() must not re-quantize existing QuantizedKVCacheEntry values."""
    torch.manual_seed(0)
    entry = KVCacheEntry(
        key=torch.randn(1, 4, 1, 2),
        value=torch.randn(1, 4, 1, 2),
    )
    cache = TabPFNV3Cache(kv={0: entry})
    q1 = cache.quantize()
    q2 = q1.quantize()
    e1 = q1.kv[0]
    e2 = q2.kv[0]
    assert isinstance(e2, QuantizedKVCacheEntry)
    # Identity in storage — passthrough returns the same entry object.
    assert e1 is e2


@torch.no_grad()
@pytest.mark.parametrize("use_chunkwise", [False, True])
def test__quantized_kv_cache__close_to_standard_forward(use_chunkwise: bool) -> None:
    """Int8-quantized KV cache produces output close to standard forward.

    Decomposes error so a regression in the cache path itself (which should
    match standard at near machine precision) can't hide behind the loose
    int8 tolerance used for the quantization step.
    """
    arch = _get_regression_model()

    torch.manual_seed(42)
    x = torch.randn(20, 1, 5, dtype=torch.float32) * 0.1
    y = torch.randn(10, dtype=torch.float32)

    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    out_standard = arch(x, y, performance_options=perf)
    _, cache = arch(x, y, performance_options=perf, return_kv_cache=True)
    out_cached = arch(x, y, performance_options=perf, kv_cache=cache)

    q_cache = cache.quantize()
    # Verify quantization happened
    for entry in q_cache.kv.values():
        assert isinstance(entry, QuantizedKVCacheEntry)
        assert entry.key.dtype == torch.int8
    # Train embeddings stay in native dtype
    assert q_cache.train_embeddings.dtype == torch.float32

    out_quantized = arch(x, y, performance_options=perf, kv_cache=q_cache)

    # Cache path itself should match standard at near machine precision.
    assert torch.allclose(out_standard, out_cached, atol=1e-5), (
        f"Non-quantized cached output diverges from standard "
        f"(max diff: {(out_standard - out_cached).abs().max().item():.2e})"
    )
    # Quantization adds small additional error on top of the cached forward.
    assert torch.allclose(out_cached, out_quantized, atol=1e-2), (
        f"Quantized cache output too far from non-quantized cached "
        f"(max diff: {(out_cached - out_quantized).abs().max().item():.2e})"
    )


@torch.no_grad()
def test__quantized_kv_cache__multiclass__close_to_standard_forward() -> None:
    """Int8-quantized KV cache produces output close to standard forward for mclass."""
    arch = _get_model()

    torch.manual_seed(42)
    x = torch.randn(100, 2, 20, dtype=torch.float32) * 0.1
    y = torch.randint(0, 10, [97, 2], dtype=torch.float32)

    out_standard = arch(x, y)
    _, cache = arch(x, y, return_kv_cache=True)
    out_cached = arch(x, y, kv_cache=cache)

    q_cache = cache.quantize()
    # Verify quantization happened
    for entry in q_cache.kv.values():
        assert isinstance(entry, QuantizedKVCacheEntry)
        assert entry.key.dtype == torch.int8
    # Train embeddings stay in full precision
    assert q_cache.train_embeddings is not None
    assert q_cache.train_embeddings.dtype == torch.float32

    out_quantized = arch(x, y, kv_cache=q_cache)

    assert torch.allclose(out_standard, out_cached, atol=1e-5), (
        f"Non-quantized cached multiclass output diverges from standard "
        f"(max diff: {(out_standard - out_cached).abs().max().item():.2e})"
    )
    assert torch.allclose(out_cached, out_quantized, atol=1e-2), (
        f"Quantized multiclass cache output too far from non-quantized cached "
        f"(max diff: {(out_cached - out_quantized).abs().max().item():.2e})"
    )


def _sum_cache_tensors(obj: object) -> int:
    """Recursively sum ``numel * element_size`` over every tensor in a cache.

    Walks dataclasses / dicts / lists so a newly-added cached tensor field is
    automatically included -- the completeness guard for ``calculate_cache_size``.
    """
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_sum_cache_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_sum_cache_tensors(v) for v in obj)
    if dataclasses.is_dataclass(obj):
        return sum(
            _sum_cache_tensors(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        )
    return 0


def _build_cache(
    arch: tabpfn_v3.TabPFNV3,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    quantize_kv_cache: bool,
) -> TabPFNV3Cache:
    """Build a cache the way the inference engine does: forward to populate it,
    then apply the ``kv_cache_dtype="int8"`` step (``cache.quantize()``).
    """
    _, cache = arch(x, y, return_kv_cache=True)
    return cache.quantize() if quantize_kv_cache else cache


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("quantize_kv_cache", [True, False])
def test__calculate_cache_size__matches_whole_classifier_cache(
    quantize_kv_cache: bool,
    dtype: torch.dtype,
) -> None:
    """get_cache_size equals the exact byte size of every tensor in a real
    classifier cache -- KV (+ int8 scales), decoder activations, inducing states,
    and scaler stats.

    Parametrized over ``dtype`` to cover the engine's forced-precision path
    (``inference_precision`` set to a ``torch.dtype``): the engine casts the model
    (``set_dtype`` -> ``model.type``) and inputs to that dtype and runs the forward
    with autocast disabled, so every non-KV term lands at that one dtype -- exactly
    what get_cache_size models. (The autocast default is a separate, mixed-dtype
    path this exact-equality check does not cover.)
    """
    arch = _get_model()
    arch.type(dtype)  # mirror the engine's set_dtype for forced precision.
    cfg = arch.config
    n_train, n_features = 10, 5

    torch.manual_seed(0)
    x = (torch.randn(20, 1, n_features, dtype=torch.float32) * 0.1).to(dtype)
    y = torch.randint(0, 10, (n_train,), dtype=torch.float32).to(dtype)
    cache = _build_cache(arch, x, y, quantize_kv_cache=quantize_kv_cache)

    total = get_cache_size(
        n_train=n_train,
        n_features=n_features,
        model_config=cfg,
        base_dtype=dtype,
        quantize_kv_cache=quantize_kv_cache,
    )
    # Exact: accounts for every tensor the real cache holds.
    assert total == _sum_cache_tensors(cache)


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Autocast inference is only enabled on CUDA (disabled on CPU/MPS), so "
    "the mixed-precision cache it produces can only be built on a GPU.",
)
@pytest.mark.parametrize("quantize_kv_cache", [True, False])
def test__calculate_cache_size__matches_whole_classifier_cache_autocast(
    quantize_kv_cache: bool,
) -> None:
    """get_cache_size matches the exact byte size of a real classifier cache built
    on the GPU autocast path (the default for ``inference_precision='auto'`` on a
    GPU).

    Unlike the forced-precision path, autocast keeps fp32 model weights and casts
    ops at runtime -- matmul-lineage tensors (KV, and ``train_embeddings`` via its
    explicit cast to the KV dtype) become the 2-byte compute dtype, while
    reduction/norm-lineage tensors (``inducing_hidden``, ``scaler_cache``) stay
    fp32. ``get_cache_size(dtype="autocast")`` must size each term at its real
    precision and still match to the byte.
    """
    device = torch.device("cuda")
    # fp32 weights: autocast casts individual ops at runtime, it does not force a
    # model dtype (mirrors force_inference_dtype=None on the autocast path).
    arch = _get_model().to(device)
    cfg = arch.config
    n_train, n_features = 10, 5

    torch.manual_seed(0)
    x = torch.randn(20, 1, n_features, dtype=torch.float32, device=device) * 0.1
    y = torch.randint(0, 10, (n_train,), dtype=torch.float32, device=device)
    with get_autocast_context(device, enabled=True):
        _, cache = arch(x, y, return_kv_cache=True)
    if quantize_kv_cache:
        cache = cache.quantize()

    total = get_cache_size(
        n_train=n_train,
        n_features=n_features,
        model_config=cfg,
        # "autocast": KV/train_embeddings sized at fp16, inducing/scaler at fp32.
        base_dtype="autocast",
        quantize_kv_cache=quantize_kv_cache,
    )
    assert total == _sum_cache_tensors(cache)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test__calculate_cache_size__matches_whole_regression_cache(
    dtype: torch.dtype,
) -> None:
    """get_cache_size equals the exact byte size of every tensor in a real
    regression cache, across the engine's forced-precision dtypes.

    Regression currently *stores* train_embeddings even though it never uses
    them, so counting the full physical cache is correct today. If a future
    change stops caching those for regression, this exact-equality assert breaks
    -- signalling that get_cache_size should then drop that term for regression.
    """
    arch = _get_regression_model()
    arch.type(dtype)  # mirror the engine's set_dtype for forced precision.
    cfg = arch.config
    n_train, n_features = 10, 5

    torch.manual_seed(0)
    x = torch.randn(20, 1, n_features, dtype=torch.float32).to(dtype)
    y = torch.randn(n_train, dtype=torch.float32).to(dtype)
    cache = _build_cache(arch, x, y, quantize_kv_cache=True)

    total = get_cache_size(
        n_train=n_train, n_features=n_features, model_config=cfg, base_dtype=dtype
    )
    assert total == _sum_cache_tensors(cache)


def test__calculate_cache_size__mqa_smaller_than_mha() -> None:
    """Fewer cached KV heads (MQA on the test partition) shrinks the KV term."""
    common = {
        "max_num_classes": -1,
        "num_buckets": 5,
        "embed_dim": 48,
        "nlayers": 1,
        "icl_num_heads": 4,
        "dist_embed_num_heads": 4,
        "feat_agg_num_heads": 4,
    }
    mha = tabpfn_v3.TabPFNV3Config(**common)  # H_kv = icl_num_heads = 4
    mqa = tabpfn_v3.TabPFNV3Config(**common, icl_num_kv_heads_test=1)  # H_kv = 1
    n_train = 50
    # quantize_kv_cache defaults to True, so the KV cache is int8 (1 byte)
    # regardless of base_dtype; base_dtype only sizes the (cancelling) non-KV terms.
    kw = {"n_train": n_train, "n_features": 5, "base_dtype": torch.float32}
    est_mha = get_cache_size(model_config=mha, **kw)
    est_mqa = get_cache_size(model_config=mqa, **kw)

    # mha and mqa differ ONLY in the KV term (H_kv 4 vs 1); every other term
    # (activations, inducing, scaler) is identical, so it cancels in the diff.
    icl_emsize = mha.embed_dim * mha.feat_agg_num_cls_tokens
    head_dim = icl_emsize // mha.icl_num_heads
    kv_per_head = mha.nlayers * 2 * n_train * head_dim  # int8 KV -> 1 byte/element
    assert est_mqa < est_mha
    assert est_mha - est_mqa == (4 - 1) * kv_per_head


@pytest.mark.slow
def test__calculate_cache_size__tabpfn3_classifier_1000_rows() -> None:
    """Pin calculate_cache_size for the real Prior-Labs/tabpfn_3 classifier at
    1,000 train rows (1 estimator, engine defaults: int8 KV, fp16 rest).
    """
    clf = TabPFNClassifier()
    # Loads the checkpoint (config + weights) without needing fit data.
    clf._initialize_model_variables()
    config = clf.model_.config

    common = {
        "n_train": 1000,
        "n_features": 1,
        "model_config": config,
        "base_dtype": torch.float16,
        "quantize_kv_cache": True,
    }
    # get_cache_size always sums the full cache. Fixed terms (n_features = 1):
    #   KV int8:            nlayers*2*H_kv*head_dim * N = 24*2*1*64 * 1000 = 3,072,000
    #   + int8 KV scales:   nlayers*2 * 2 bytes         = 24*2*2           =        96
    #   + scaler stats:     2 * n_features(1) * 2 bytes                    =         4
    #   + fp16 activations: icl_emsize * N * 2 = 512 * 1000 * 2            = 1,024,000
    # The inducing term depends on the shipped dist-embedder config, so derive it
    # from the config instead of hardcoding it.
    n_features = 1
    inducing = (
        config.dist_embed_num_blocks
        * n_features
        * config.dist_embed_num_inducing_points
        * config.embed_dim
    ) * torch.float16.itemsize

    total = get_cache_size(**common)

    # Numbers need manual update if we bump the default architecture.
    assert total == 3_072_000 + 96 + 4 + 1_024_000 + inducing
