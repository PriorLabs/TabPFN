#  Copyright (c) Prior Labs GmbH 2026.
"""Verify that the v2 implementation computes exactly the same outputs as the base."""

from __future__ import annotations

import sys
from typing import cast

import pytest
import torch

from tabpfn import model_loading
from tabpfn.architectures import base, tabpfn_v2_sf
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.shared.column_embeddings import load_column_embeddings
from tabpfn.architectures.tabpfn_v2_sf import TabPFNV2Cache


def _create_identical_small_v2_and_base(
    max_num_classes: int = 10,
) -> tuple[tabpfn_v2_sf.TabPFNV2, PerFeatureTransformer]:
    """Construct the v2 and base architectures such that they have the same outputs."""
    configv2 = tabpfn_v2_sf.TabPFNV2Config(
        max_num_classes=max_num_classes,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        features_per_group=2,
        # Use 420 (not the production default 42) so the column embeddings are
        # generated purely from the random generator rather than loaded from disk,
        # matching the base architecture configured below.
        seed=420,
    )
    config_base = base.ModelConfig(
        max_num_classes=max_num_classes,
        num_buckets=5,
        emsize=192,
        nlayers=1,
        nhead=6,
        nhid_factor=4,
        features_per_group=2,
        remove_duplicate_features=False,
        nan_handling_enabled=True,
        # Needs to be 420 to match the single-file implementation
        # and avoid special handling of seed=42 in the multi-file
        # implementation.
        seed=420,
    )

    # Get the architectures
    arch_v2 = tabpfn_v2_sf.get_architecture(
        configv2, cache_trainset_representation=False
    )
    arch_base = base.get_architecture(config_base, cache_trainset_representation=False)
    # Overwrite zero-initialized outputs to make sure we catch differences in
    # attention outputs.
    for param in arch_base.parameters():
        if param.abs().sum() < 1e-6:
            param.data += torch.randn_like(param) * 1e-1

    # load_state_dict translates the base-architecture key names automatically.
    arch_v2.load_state_dict(arch_base.state_dict(), strict=True)

    arch_v2.to(torch.float64)
    arch_base.to(torch.float64)

    return arch_v2, arch_base


@torch.no_grad()
def test__load_state_dict__from_base_checkpoint_strict() -> None:
    """A base-architecture state dict can be loaded strictly (all keys consumed)."""
    arch_v2, arch_base = _create_identical_small_v2_and_base()
    # Re-loading the (already translated) base state dict must succeed with strict=True,
    # i.e. the translated keys exactly cover the single-file architecture's parameters.
    result = arch_v2.load_state_dict(arch_base.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


@torch.no_grad()
def test__load_state_dict__from_regression_base_checkpoint_strict() -> None:
    """A regression base state dict loads strictly despite the shifted y-encoder index.

    Regression checkpoints omit the multiclass target step, so the y-encoder's linear
    projection lives at ``y_encoder.1.layer`` instead of ``y_encoder.2.layer``. The key
    translation must locate it by name rather than assuming the classification index.
    """
    arch_v2, arch_base = _create_identical_small_v2_and_base(max_num_classes=0)
    base_state = arch_base.state_dict()
    assert "y_encoder.1.layer.weight" in base_state
    assert "y_encoder.2.layer.weight" not in base_state

    result = arch_v2.load_state_dict(base_state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


@torch.no_grad()
def test__load_state_dict__native_keys_still_work() -> None:
    """A round-trip of the single-file architecture's own state dict still works."""
    arch_v2, _ = _create_identical_small_v2_and_base()
    arch_v2.load_state_dict(arch_v2.state_dict(), strict=True)


class TestTabPFNv2NewVsOldImplementation:
    """Test that the v2 implementation computes exactly the same outputs as the base."""

    @torch.no_grad()
    @pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
    def test__forward__v2_and_base_have_same_output(self) -> None:
        loaded_models, _, loaded_configs, _ = model_loading.load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download_if_not_exists=True,
        )
        arch_base = cast(PerFeatureTransformer, loaded_models[0])
        config_base = loaded_configs[0]
        # Force seed 42 on both implementations, which makes them load the
        # pre-generated column embeddings from disk for the first 2000 columns. This
        # exercises the single-file implementation's disk-loading path end-to-end.
        arch_base.random_embedding_seed = 42

        arch_v2 = tabpfn_v2_sf.get_architecture(
            tabpfn_v2_sf.TabPFNV2Config(
                max_num_classes=config_base.max_num_classes,
                num_buckets=config_base.num_buckets,
                seed=42,
            ),
            cache_trainset_representation=False,
        )
        # load_state_dict translates the base-architecture key names automatically.
        arch_v2.load_state_dict(arch_base.state_dict(), strict=True)

        arch_v2.to(torch.float64)
        arch_base.to(torch.float64)

        # Create dummy input data
        x = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
        y = torch.randint(0, 10, [97, 2], dtype=torch.float64)
        # Forward pass through both architectures
        output_v2 = arch_v2(x, y, only_return_standard_out=False)
        output_base = arch_base(x, y, only_return_standard_out=False)

        assert output_v2.keys(), "No output returned."
        msg = "Output keys do not match between implementations"
        assert output_v2.keys() == output_base.keys(), msg
        for key in output_v2:
            msg = f"Outputs for {key} do not match between implementations."
            assert torch.allclose(output_v2[key], output_base[key]), msg

    @pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
    def test__backward__v2_and_base_x_inputs_have_same_gradients(self) -> None:
        # Doing a forward and backward pass is more expensive, so use a smaller model.
        arch_v2, arch_base = _create_identical_small_v2_and_base()

        # Use float64 (like the forward comparison tests): the functional preprocessing
        # is mathematically equivalent to the base encoder steps but not bit-identical,
        # so float32 gradients differ at rounding level (~1e-7) while float64 matches.
        x_v2 = torch.randn(100, 2, 20, dtype=torch.float64) * 0.1
        x_base = x_v2.clone().detach()
        y_v2 = torch.randint(0, 10, [97, 2], dtype=torch.float64)
        y_base = y_v2.clone().detach()

        x_v2.requires_grad = True
        x_base.requires_grad = True

        # Forward pass and backward pass through both architectures
        arch_v2(x_v2, y_v2, only_return_standard_out=True).sum().backward()
        arch_base(x_base, y_base, only_return_standard_out=True).sum().backward()

        msg = "Gradients for input x do not match between implementations."
        assert torch.allclose(x_v2.grad, x_base.grad), msg


def _handler_inputs() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(x, y)`` pairs that each exercise one preprocessing handler.

    Every case must produce identical outputs in the single-file and base
    architectures, validating the functional reimplementation of each encoder step
    (NaN/Inf handling, constant-feature removal, feature-group normalization and the
    multiclass target densification).
    """
    torch.manual_seed(0)
    num_train, num_test, num_features, batch = 40, 13, 8, 2
    num_rows = num_train + num_test

    def make_x() -> torch.Tensor:
        return torch.randn(num_rows, batch, num_features, dtype=torch.float64) * 0.5

    def make_y() -> torch.Tensor:
        return torch.randint(0, 10, (num_train, batch), dtype=torch.float64)

    cases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # Baseline: clean, well-behaved data (no handler is actually triggered).
    cases["plain"] = (make_x(), make_y())

    # NaNs scattered through the features, including a fully-NaN feature column.
    x = make_x()
    x[5, 0, 2] = float("nan")
    x[20, 1, 4] = float("nan")
    x[:, 0, 6] = float("nan")
    cases["feature_nans"] = (x, make_y())

    # +Inf / -Inf in the features (distinct indicator values from NaN).
    x = make_x()
    x[3, 0, 1] = float("inf")
    x[7, 1, 5] = float("-inf")
    x[11, 0, 5] = float("inf")
    cases["feature_infs"] = (x, make_y())

    # Constant feature columns -> exercises remove-empty + feature-group normalization.
    x = make_x()
    x[:, :, 0] = 3.14
    x[:, :, 3] = 0.0
    cases["constant_features"] = (x, make_y())

    # Duplicated feature columns and duplicated rows (kept as-is; no dedup step).
    x = make_x()
    x[:, :, 1] = x[:, :, 2]
    x[10] = x[3]
    x[25] = x[3]
    cases["duplicate_features_and_rows"] = (x, make_y())

    # NaNs in the training targets -> exercises the target NaN handler.
    y = make_y()
    y[2, 0] = float("nan")
    y[10, 1] = float("nan")
    cases["target_nans"] = (make_x(), y)

    # Non-contiguous / sparse classes -> exercises the multiclass densification.
    y = torch.randint(0, 10, (num_train, batch), dtype=torch.float64)
    y[y == 5] = 7
    y[y == 2] = 9
    cases["sparse_classes"] = (make_x(), y)

    # Everything at once.
    x = make_x()
    x[4, 0, 2] = float("nan")
    x[8, 1, 5] = float("inf")
    x[:, :, 0] = 1.0
    y = make_y()
    y[1, 0] = float("nan")
    cases["combined"] = (x, y)

    return cases


@pytest.mark.parametrize("case", list(_handler_inputs()))
@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward__handlers_match_base(case: str) -> None:
    """Each preprocessing handler produces the same output as the base architecture."""
    arch_v2, arch_base = _create_identical_small_v2_and_base()
    x, y = _handler_inputs()[case]

    output_v2 = arch_v2(x, y, only_return_standard_out=False)
    output_base = arch_base(x, y, only_return_standard_out=False)

    assert output_v2.keys() == output_base.keys()
    for key in output_v2:
        assert torch.allclose(output_v2[key], output_base[key]), (
            f"Outputs for {key} do not match between implementations for case {case!r}."
        )


@pytest.mark.parametrize("case", ["feature_nans", "feature_infs", "constant_features"])
@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__kv_cache__matches_standard_forward_with_handlers(case: str) -> None:
    """The cached feature statistics reproduce the standard forward on tricky inputs."""
    arch = _build_small_arch(seed=420)
    arch.to(torch.float64)
    x_full, y_train = _handler_inputs()[case]
    num_train = y_train.shape[0]
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train)

    out_store, cache = arch(x_full, y_train, return_kv_cache=True)
    assert cache.feature_cache is not None
    assert torch.allclose(out_standard, out_store, atol=1e-10)

    out_cached = arch(x_test, y_train, kv_cache=cache, x_is_test_only=True)
    assert torch.allclose(out_standard, out_cached, atol=1e-10), (
        f"kv_cache inference differs from the standard forward for case {case!r}."
    )


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled() -> None:
    arch, _ = _create_identical_small_v2_and_base()

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
            output_with_memory_saving[key], output_without_memory_saving[key]
        ), f"Outputs for {key} do not match between implementations."


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__forward_pass_equal_with_checkpointing_enabled_and_disabled() -> None:
    arch, _ = _create_identical_small_v2_and_base()

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
            output_with_recomputation[key], output_without_recomputation[key]
        ), f"Outputs for {key} do not match between implementations."


def _build_small_arch(seed: int, emsize: int = 192) -> tabpfn_v2_sf.TabPFNV2:
    return tabpfn_v2_sf.get_architecture(
        tabpfn_v2_sf.TabPFNV2Config(
            max_num_classes=10,
            num_buckets=5,
            emsize=emsize,
            nlayers=1,
            nhead=6,
            features_per_group=2,
            seed=seed,
        ),
        cache_trainset_representation=False,
    )


@torch.no_grad()
def test__column_embeddings__seed_42_loaded_from_disk() -> None:
    """With the production seed (42) and 48-d subspace, embeddings come from disk."""
    arch = _build_small_arch(seed=42)
    disk_embeddings = load_column_embeddings()

    num_cols = 7
    x = torch.zeros(2, 3, num_cols, 192)
    out = arch.add_column_embeddings(x.clone())

    # The first `num_cols` rows of the disk embeddings, projected, should have been
    # added to every (batch, row) position.
    expected_subspace = disk_embeddings[:num_cols].to(dtype=x.dtype)
    expected = arch.feature_positional_embedding_embeddings(expected_subspace)
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


@torch.no_grad()
def test__column_embeddings__non_production_seed_does_not_use_disk() -> None:
    """A non-42 seed (or non-48 subspace) ignores the disk embeddings."""
    disk_embeddings = load_column_embeddings()

    num_cols = 7
    x = torch.zeros(2, 3, num_cols, 192)

    arch_420 = _build_small_arch(seed=420)
    out_420 = arch_420.add_column_embeddings(x.clone())
    expected_disk = arch_420.feature_positional_embedding_embeddings(
        disk_embeddings[:num_cols].to(dtype=x.dtype)
    )
    # The random (seed 420) embeddings must differ from the disk-based ones.
    assert not torch.allclose(out_420[0, 0], expected_disk, atol=1e-6)

    # A smaller emsize (subspace != 48) also bypasses the disk embeddings even for
    # seed 42, falling back to the random generator.
    arch_small = _build_small_arch(seed=42, emsize=96)
    x_small = torch.zeros(2, 3, num_cols, 96)
    generator = torch.Generator().manual_seed(42)
    expected_random = torch.randn((num_cols, 96 // 4), generator=generator)
    expected_random = arch_small.feature_positional_embedding_embeddings(
        expected_random
    )
    out_small = arch_small.add_column_embeddings(x_small.clone())
    assert torch.allclose(out_small[0, 0], expected_random, atol=1e-6)


def _make_kv_cache_data() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(x_full, y_train, num_train)`` for the KV-cache tests."""
    torch.manual_seed(0)
    num_train, num_test, num_features = 30, 7, 5
    x_full = torch.randn(num_train + num_test, 1, num_features, dtype=torch.float32)
    x_full = x_full * 0.5
    y_train = torch.randint(0, 10, (num_train, 1), dtype=torch.float32)
    return x_full, y_train, num_train


@torch.no_grad()
def test__kv_cache__matches_standard_forward() -> None:
    """KV-cache inference must match the standard (full train+test) forward."""
    arch = _build_small_arch(seed=420)
    arch.to(torch.float32)
    x_full, y_train, num_train = _make_kv_cache_data()
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train)

    # Build the cache; the store-mode output must match the standard forward.
    out_store, cache = arch(x_full, y_train, return_kv_cache=True)
    assert isinstance(cache, TabPFNV2Cache)
    assert not cache.is_empty()
    assert len(cache.kv) == 1  # nlayers=1
    assert cache.train_shape == (1, num_train)
    assert cache.feature_cache is not None
    assert torch.allclose(out_standard, out_store, atol=1e-10), (
        "return_kv_cache=True output differs from the standard forward."
    )

    # Use the cache on test-only data.
    out_cached = arch(x_test, y_train, kv_cache=cache, x_is_test_only=True)
    assert out_cached.shape == out_standard.shape
    assert torch.allclose(out_standard, out_cached, atol=1e-10), (
        "kv_cache inference output differs from the standard forward."
    )


@torch.no_grad()
def test__kv_cache__non_standard_out_matches() -> None:
    """The cache path also returns matching embeddings dicts."""
    arch = _build_small_arch(seed=420)
    arch.to(torch.float32)
    x_full, y_train, num_train = _make_kv_cache_data()
    x_test = x_full[num_train:]

    out_standard = arch(x_full, y_train, only_return_standard_out=False)
    _, cache = arch(x_full, y_train, return_kv_cache=True)
    out_cached = arch(
        x_test,
        y_train,
        only_return_standard_out=False,
        kv_cache=cache,
        x_is_test_only=True,
    )
    for key in ("standard", "test_embeddings"):
        assert torch.allclose(out_standard[key], out_cached[key], atol=1e-10), (
            f"cache path output for {key} differs from the standard forward."
        )


@torch.no_grad()
def test__kv_cache__x_is_test_only_requires_populated_cache() -> None:
    arch = _build_small_arch(seed=420)
    arch.to(torch.float32)
    x_full, y_train, _ = _make_kv_cache_data()
    with pytest.raises(ValueError, match="requires a populated kv_cache"):
        arch(x_full, y_train, x_is_test_only=True)
