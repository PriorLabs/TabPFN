#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the v3.5 single-file model.

v3.5 is a multitask model: `task_type` is a per-`forward()` argument, so one
instance, and one checkpoint, handles both multiclass and regression. It ranks
cells against `cell_ecdf_num_buckets` bucket edges per column rather than
against every train row, so the inference cache stops growing with the table.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import sys
from typing import Literal

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier
from tabpfn.architectures import tabpfn_v3_5
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.kv_cache import (
    FP8_KV_DTYPE,
    QUANTIZED_KV_DTYPE,
    KVCacheEntry,
    QuantizedKVCacheEntry,
)
from tabpfn.architectures.tabpfn_v3_5 import (
    TabPFNV3p5,
    TabPFNV3p5Cache,
    TabPFNV3p5Config,
    get_cache_size,
)
from tabpfn.constants import ModelVersion, TaskType
from tabpfn.utils import get_autocast_context

MAX_NUM_CLASSES = 5
NUM_TRAIN, NUM_TEST, BATCH, NUM_FEATURES = 20, 4, 2, 5
TASK_TYPES: list[TaskType] = ["multiclass", "regression"]

# Shrunk to keep the tests fast; every stage of the model is still exercised.
_SMALL_CONFIG: dict[str, object] = {
    "max_num_classes": MAX_NUM_CLASSES,
    "num_buckets": 32,
    "embed_dim": 32,
    "nlayers": 2,
    "icl_num_heads": 4,
    "icl_num_kv_heads_test": 1,
    "dist_embed_num_heads": 4,
    "dist_embed_num_blocks": 1,
    "feat_agg_num_heads": 4,
    "feat_agg_num_blocks": 1,
    "feat_agg_num_cls_tokens": 2,
    "dist_embed_num_inducing_points": 8,
    # Small enough that the chunked inference path splits the test inputs.
    "inference_row_chunk_size": 8,
    "inference_col_chunk_size": 2,
}


def _config(config_overrides: dict[str, object]) -> TabPFNV3p5Config:
    config, _unused = tabpfn_v3_5.parse_config({**_SMALL_CONFIG, **config_overrides})
    return config


def _get_model(**config_overrides: object) -> TabPFNV3p5:
    """A small v3.5 model in eval mode, with no all-zero parameters."""
    config = _config(config_overrides)
    arch = tabpfn_v3_5.get_architecture(config, cache_trainset_representation=False)
    # Several modules zero-init their residual out-projections; a fully-zero
    # projection masks its sublayer and would hide a bug in it.
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in arch.parameters():
            if param.numel() > 0 and bool((param == 0).all()):
                param.normal_(std=0.02, generator=gen)
    arch.to(torch.float32)
    return arch.eval()


def _inputs(
    task_type: TaskType,
    *,
    n_train_classes: int = MAX_NUM_CLASSES,
    batch: int = BATCH,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(NUM_TRAIN + NUM_TEST, batch, NUM_FEATURES) * 0.1
    if task_type == "regression":
        return x, torch.randn(NUM_TRAIN, batch)
    y = torch.arange(NUM_TRAIN).unsqueeze(1).repeat(1, batch) % n_train_classes
    return x, y.float()


def _assert_outputs_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    atol: float,
) -> None:
    assert actual.keys() == expected.keys(), "Output keys do not match"
    for key, value in expected.items():
        assert torch.allclose(actual[key], value, atol=atol), (
            f"Outputs for '{key}' do not match."
        )


# ---------------------------------------------------------------------------
# Config and module
# ---------------------------------------------------------------------------


def test__config__defaults__match_the_v3_5_checkpoint() -> None:
    """The defaults rebuild the released checkpoint's architecture.

    The two head sizes stay unset: they come from the checkpoint, and the base
    `ArchitectureConfig` leaves them at -1.
    """
    config = TabPFNV3p5Config()
    assert dataclasses.asdict(config) == {
        "name": "TabPFN-v3.5",
        "max_num_classes": -1,
        "num_buckets": -1,
        "embed_dim": 128,
        "dist_embed_num_blocks": 3,
        "dist_embed_num_heads": 8,
        "dist_embed_num_inducing_points": 128,
        "feature_group_size": 3,
        "feat_agg_num_blocks": 3,
        "feat_agg_num_heads": 8,
        "feat_agg_num_cls_tokens": 8,
        "feat_agg_rope_base": 100_000,
        "nlayers": 24,
        "icl_num_heads": 16,
        "icl_num_kv_heads": None,
        "icl_num_kv_heads_test": 1,
        "decoder_head_dim": 64,
        "decoder_num_heads": 6,
        "decoder_use_softmax_scaling": True,
        "ff_factor": 2,
        "softmax_scaling_mlp_hidden_dim": 64,
        "fourier_encoding_num_frequencies": 32,
        "cell_ecdf_num_frequencies": 4,
        "cell_ecdf_num_buckets": 8192,
        "cell_embed_row_chunk_size": 2048,
        "inference_row_chunk_size": 2048,
        "inference_col_chunk_size": 4,
    }


def test__module_imports__only_tabpfn_and_third_party() -> None:
    """The architecture must stay self-contained within the tabpfn package."""
    tree = ast.parse(inspect.getsource(tabpfn_v3_5))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots == {
        "__future__",
        "collections",
        "contextlib",
        "dataclasses",
        "functools",
        "logging",
        "math",
        "numpy",
        "pydantic",
        "tabpfn",
        "torch",
        "typing",
        "typing_extensions",
    }


def test__parse_config__training_only_keys__reported_as_unused() -> None:
    """Training checkpoints carry loss and muP keys; v3.5 must ignore them."""
    training_only = {"enable_mup": True, "weight_regression_ce_loss": 1.0}
    _config, unused = tabpfn_v3_5.parse_config({**_SMALL_CONFIG, **training_only})
    assert set(unused) == set(training_only)


def test__get_supported_kv_cache_precisions__advertises_the_quantized_dtypes() -> None:
    """The engine resolves to "auto" unless the architecture lists its dtypes."""
    assert _get_model().get_supported_kv_cache_precisions() == ("auto", "int8", "fp8")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__forward__output_shapes(task_type: TaskType) -> None:
    arch = _get_model()
    x, y = _inputs(task_type)
    out = arch(x, y, task_type=task_type)
    width = (
        MAX_NUM_CLASSES if task_type == "multiclass" else _SMALL_CONFIG["num_buckets"]
    )
    assert out.shape == (NUM_TEST, BATCH, width)


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__forward__only_return_standard_out_false__returns_embeddings(
    task_type: TaskType,
) -> None:
    """v3.5 carries no losses, so the dict output holds only the three tensors."""
    arch = _get_model()
    x, y = _inputs(task_type)
    output = arch(x, y, task_type=task_type, only_return_standard_out=False)
    assert set(output) == {"standard", "train_embeddings", "test_embeddings"}


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__forward_pass_equal_with_save_peak_memory_enabled_and_disabled(
    task_type: TaskType,
) -> None:
    arch = _get_model()
    x, y = _inputs(task_type)

    without = arch(x, y, task_type=task_type, only_return_standard_out=False)
    with_saving = arch(
        x,
        y,
        task_type=task_type,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(save_peak_memory_factor=4),
    )
    _assert_outputs_equal(with_saving, without, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__forward_pass_equal_with_checkpointing_enabled_and_disabled(
    task_type: TaskType,
) -> None:
    arch = _get_model()
    x, y = _inputs(task_type)

    without = arch(x, y, task_type=task_type, only_return_standard_out=False)
    with_recompute = arch(
        x,
        y,
        task_type=task_type,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(force_recompute_layer=True),
    )
    _assert_outputs_equal(with_recompute, without, atol=1e-6)


@torch.no_grad()
def test__batch_size_one__nan_and_inf_in_features__still_works() -> None:
    arch = _get_model()
    x = torch.randn(100, 1, 1, dtype=torch.float32) * 0.1
    x[10, 0] = float("nan")
    x[11, 0] = float("inf")
    y = torch.randint(0, MAX_NUM_CLASSES, [97, 1], dtype=torch.float32)

    output = arch(x, y, task_type="multiclass")

    assert output.shape == (3, 1, MAX_NUM_CLASSES)
    assert torch.isfinite(output).all()


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__forward__no_test_set_works_batch_size_one(task_type: TaskType) -> None:
    arch = _get_model()
    x = torch.randn(1, 1, NUM_FEATURES, dtype=torch.float32) * 0.1
    y = torch.randint(0, MAX_NUM_CLASSES, [1, 1], dtype=torch.float32)

    out = arch(x, y, task_type=task_type, only_return_standard_out=False)

    assert out["standard"].shape[:2] == (0, 1)


@torch.no_grad()
@pytest.mark.parametrize(
    "invalid_target",
    [-1.0, -0.5, MAX_NUM_CLASSES - 0.5, MAX_NUM_CLASSES, -np.inf, np.inf],
)
def test__forward__multiclass_target_out_of_range__raises(
    invalid_target: float,
) -> None:
    arch = _get_model()
    x, y = _inputs("multiclass")
    y[0, 0] = invalid_target
    with pytest.raises(ValueError, match="Target is out of range"):
        arch(x, y, task_type="multiclass")


@torch.no_grad()
def test__forward__nan_and_highest_class_in_second_batch__cached_matches_uncached() -> (
    None
):
    arch = _get_model()
    x, y = _inputs("multiclass", n_train_classes=2)
    y[0, 0] = np.nan
    y[0, 1] = MAX_NUM_CLASSES - 1
    expected = arch(x, y, task_type="multiclass")

    _, cache = arch(x[:NUM_TRAIN], y, task_type="multiclass", return_kv_cache=True)
    actual = arch(
        x[NUM_TRAIN:],
        y,
        task_type="multiclass",
        kv_cache=cache,
        x_is_test_only=True,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__chunked_inference_matches_standard_forward(task_type: TaskType) -> None:
    """The row/column chunking must not change the prediction."""
    arch = _get_model()
    x, y = _inputs(task_type)
    options = arch.get_default_performance_options()
    assert options.use_chunkwise_inference

    standard = arch(
        x,
        y,
        task_type=task_type,
        only_return_standard_out=False,
        performance_options=PerformanceOptions(use_chunkwise_inference=False),
    )
    chunked = arch(
        x,
        y,
        task_type=task_type,
        only_return_standard_out=False,
        performance_options=options,
    )
    _assert_outputs_equal(chunked, standard, atol=1e-5)


@torch.no_grad()
def test__chunked_inference_recovers_from_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recoverable OOM during chunked inference must not crash the forward.

    The column-chunk handler reacts to an OOM by freeing memory, halving the
    chunk and retrying. The recovered output must match the standard forward.
    """
    arch = _get_model()
    x, y = _inputs("multiclass")
    expected = arch(x, y, task_type="multiclass", only_return_standard_out=False)

    # Raise a single OOM the first time a column chunk is processed, so the handler
    # must free memory, halve the column chunk and retry. Patched on the class so
    # the bound method still exposes `__func__` for `_compiled`.
    original_process_col_chunk = TabPFNV3p5._process_col_chunk
    calls = {"n": 0}

    def _process_col_chunk_oom_once(
        self: TabPFNV3p5, *args: object, **kwargs: object
    ) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return original_process_col_chunk(self, *args, **kwargs)

    monkeypatch.setattr(TabPFNV3p5, "_process_col_chunk", _process_col_chunk_oom_once)

    recovered = arch(
        x,
        y,
        task_type="multiclass",
        only_return_standard_out=False,
        performance_options=PerformanceOptions(use_chunkwise_inference=True),
    )

    assert calls["n"] > 1, "the simulated OOM never triggered a retry"
    _assert_outputs_equal(recovered, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Many-class decoder
# ---------------------------------------------------------------------------


@torch.no_grad()
def test__forward_many_class_head__fewer_classes_than_max__pads_absent_columns() -> (
    None
):
    """Classes missing from the train targets still get a column, at the zero logit.

    The decoder narrows its one-hot to the classes present and pads the output
    back, so the absent columns must carry the logit of a zero attention output.
    """
    arch = _get_model()
    n_train_classes = 2
    x, y = _inputs("multiclass", n_train_classes=n_train_classes)

    out = arch(x, y, task_type="multiclass")

    assert out.shape[-1] == MAX_NUM_CLASSES
    zero_logit = float(np.log(1e-5 + 3e-5))
    assert torch.allclose(
        out[..., n_train_classes:],
        torch.full_like(out[..., n_train_classes:], zero_logit),
    )


@torch.no_grad()
@pytest.mark.skipif(sys.platform == "win32", reason="float64 tests fail on Windows")
def test__many_class_decoder__unused_classes__matches_full_width_one_hot() -> None:
    """Narrowing the one-hot to the present classes must not change the output.

    `head_dim` below the class count makes the full-width reference span three
    folded attention passes where the narrowed path needs one, so a mismatch in
    the trim, the chunking or the padding surfaces here.
    """
    num_classes, input_size, num_heads, head_dim = 10, 12, 3, 4
    batch, num_train, num_test = 2, 17, 3

    torch.manual_seed(42)
    decoder = tabpfn_v3_5.ManyClassDecoder(
        max_num_classes=num_classes,
        input_size=input_size,
        head_dim=head_dim,
        num_heads=num_heads,
    ).to(torch.float64)
    train_emb = torch.randn(batch, num_train, input_size, dtype=torch.float64)
    test_emb = torch.randn(batch, num_test, input_size, dtype=torch.float64)
    # Only classes 0..2 occur, so the decoder trims 10 columns down to 3.
    targets = (torch.arange(num_train) % 3).repeat(batch, 1).to(torch.float64)

    train_keys = decoder.project_keys(train_emb)
    actual = decoder(train_keys, test_emb, targets, num_present_classes=3)

    q_BMHD = decoder.q_projection(test_emb).view(batch, num_test, num_heads, head_dim)
    one_hot_BNHT = (
        torch.nn.functional.one_hot(targets.long(), num_classes=num_classes)
        .to(torch.float64)
        .unsqueeze(2)
        .expand(-1, -1, num_heads, -1)
        .contiguous()
    )
    reference_BMT = tabpfn_v3_5._chunked_class_attention(
        q_BMHD.contiguous(), train_keys, one_hot_BNHT
    ).mean(2)
    expected = torch.log(torch.clamp(reference_BMT.transpose(0, 1), min=1e-5) + 3e-5)

    assert actual.shape == (num_test, batch, num_classes)
    assert torch.allclose(actual, expected, atol=1e-12), (
        f"max abs diff: {(actual - expected).abs().max()}"
    )


# ---------------------------------------------------------------------------
# In-context ECDF
# ---------------------------------------------------------------------------


def _exact_midranks(x_BRiC: torch.Tensor, num_train: int) -> torch.Tensor:
    """Midrank ECDF against every train row, the definition v3.5 approximates."""
    sorted_BCN = x_BRiC[:, :num_train].transpose(1, 2).contiguous().sort(dim=-1).values
    values_BCRi = x_BRiC.transpose(1, 2).contiguous()
    left = torch.searchsorted(sorted_BCN, values_BCRi, side="left")
    right = torch.searchsorted(sorted_BCN, values_BCRi, side="right")
    return (0.5 * (left + right).float() / num_train).transpose(1, 2)


def _bucketed_midranks(
    x_BRiC: torch.Tensor, num_train: int, num_buckets: int
) -> torch.Tensor:
    context = tabpfn_v3_5._build_ecdf_context(x_BRiC, num_train, num_buckets)
    assert context.dtype == tabpfn_v3_5.ECDF_CONTEXT_DTYPE
    batch, _rows, columns = x_BRiC.shape
    assert context.shape == (3, batch, columns, min(num_buckets, num_train))
    return tabpfn_v3_5._in_context_ecdf(x_BRiC, context)


@pytest.mark.parametrize(
    ("kind", "num_buckets", "column"),
    [
        # Fewer train rows than buckets: the buckets are the rows.
        ("all-rows-fit", 10_000, torch.linspace(-3.0, 3.0, 4000)),
        # More rows than buckets, but few enough distinct values to keep them all.
        ("low-cardinality", 100, torch.randint(0, 7, (4000,)).float()),
        # A value seen once still gets its own bucket edge.
        ("one-rare-value", 100, torch.cat([torch.zeros(3999), torch.ones(1)])),
        # A constant column: every row lands on the single edge.
        ("constant", 100, torch.full((4000,), 4.2)),
    ],
)
def test__in_context_ecdf__buckets_cover_every_value__matches_exact_midranks(
    kind: str, num_buckets: int, column: torch.Tensor
) -> None:
    """Where no distinct value is dropped, bucketing must change nothing at all."""
    del kind
    x_BRiC = column.reshape(1, -1, 1)
    ranks = _bucketed_midranks(x_BRiC, x_BRiC.shape[1], num_buckets)
    assert torch.equal(ranks, _exact_midranks(x_BRiC, x_BRiC.shape[1]))


def test__build_ecdf_context__dense_values_in_a_wide_range__keep_their_rank() -> None:
    """Edges must follow the rows, not the distinct values, once they run out.

    Half the rows sit on six values holding six of ~10 000 distinct indices, so
    spacing the edges over distinct values skips every one of them and ranks half
    the column inside one bucket.
    """
    torch.manual_seed(0)
    num_rows, num_buckets = 20_000, 512
    dense = torch.arange(6.0).repeat_interleave(num_rows // 12)
    tail = torch.rand(num_rows - dense.numel()) * 1000 + 10
    # The six dense values ride along as test rows so they get ranked too.
    x_BRiC = torch.cat([dense, tail, torch.arange(6.0)]).reshape(1, -1, 1)

    context = tabpfn_v3_5._build_ecdf_context(x_BRiC, num_rows, num_buckets)
    ranks = tabpfn_v3_5._in_context_ecdf(x_BRiC, context)
    expected = _exact_midranks(x_BRiC, num_rows)
    assert (ranks[:, num_rows:] - expected[:, num_rows:]).abs().max() < 1 / num_buckets


@pytest.mark.parametrize(("num_buckets", "tolerance"), [(1000, 1e-3), (100, 1e-2)])
def test__in_context_ecdf__more_distinct_values_than_buckets__error_below_one_bucket(
    num_buckets: int, tolerance: float
) -> None:
    """Dropped values cost at most a bucket's worth of rank, on any column scale."""
    torch.manual_seed(0)
    num_rows = 20_000
    scales = [1.0, 1e3, 1e5, 1e8, 1e-4, 1e30, 1e-30]
    x_BRiC = torch.stack([torch.randn(num_rows) * s for s in scales], dim=-1).unsqueeze(
        0
    )
    ranks = _bucketed_midranks(x_BRiC, num_rows, num_buckets)
    assert torch.isfinite(ranks).all()
    assert (ranks - _exact_midranks(x_BRiC, num_rows)).abs().max() < tolerance


def test__build_ecdf_context__column_chunking__builds_the_same_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The context is built per column, so the chunk width must not matter."""
    torch.manual_seed(0)
    x_BRiC = torch.randn(2, 200, 7)
    one_pass = tabpfn_v3_5._build_ecdf_context(x_BRiC, 150, 32)
    # 200 rows over a 300-cell budget gives one column per pass, not a divisor of 7.
    monkeypatch.setattr(tabpfn_v3_5, "_ECDF_CELL_BUDGET", 300)
    chunked = tabpfn_v3_5._build_ecdf_context(x_BRiC, 150, 32)
    assert chunked.shape == one_pass.shape
    assert torch.equal(chunked, one_pass)


@pytest.mark.parametrize(
    ("kind", "num_train", "num_buckets"),
    [
        ("every value an edge", 40, 8192),
        ("interpolating", 300, 8),
        ("duplicate edges", 300, 64),
    ],
)
def test__in_context_ecdf__cells_under_optimisation__keep_a_finite_gradient(
    kind: str, num_train: int, num_buckets: int
) -> None:
    """Prompt tuning optimises the cells, so the ranks must stay differentiable."""
    torch.manual_seed(0)
    values = (
        torch.randint(0, 4, (1, 400, 3)).float()
        if kind == "duplicate edges"
        else torch.randn(1, max(num_train, 400), 3)
    )
    values.requires_grad_()
    context = tabpfn_v3_5._build_ecdf_context(values, num_train, num_buckets)
    tabpfn_v3_5._in_context_ecdf(values, context).sum().backward()
    assert torch.isfinite(values.grad).all()


def test__in_context_ecdf__row_chunking__does_not_move_a_single_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranking is per cell, so the chunk boundaries must not show up anywhere."""
    torch.manual_seed(0)
    x_BRiC = torch.randn(2, 500, 3)
    context = tabpfn_v3_5._build_ecdf_context(x_BRiC, 400, 64)
    one_pass = tabpfn_v3_5._in_context_ecdf(x_BRiC, context)
    # Small enough to split the 500 rows many times, and not a divisor of them.
    monkeypatch.setattr(tabpfn_v3_5, "_ECDF_CELL_BUDGET", 21)
    assert torch.equal(tabpfn_v3_5._in_context_ecdf(x_BRiC, context), one_pass)


def test__in_context_ecdf__value_inside_a_bucket__is_interpolated_not_snapped() -> None:
    """Within a bucket the rank rises with the value instead of stepping."""
    x_BRiC = torch.arange(1000.0).reshape(1, -1, 1)
    ranks = _bucketed_midranks(x_BRiC, 1000, 10).flatten()
    assert (ranks[1:] > ranks[:-1]).all()
    # On a uniform column linear interpolation recovers the exact ranks.
    assert (ranks - _exact_midranks(x_BRiC, 1000).flatten()).abs().max() < 1e-3


@torch.no_grad()
@pytest.mark.parametrize("cell_ecdf_num_buckets", [8192, 8])
def test__preprocess_raw__cached_context__ranks_test_rows_identically(
    cell_ecdf_num_buckets: int,
) -> None:
    """A cached run must rank test rows against exactly the buckets it stored.

    8 buckets over 20 train rows is lossy, which is what makes this fail if either
    path builds its own context.
    """
    arch = _get_model(cell_ecdf_num_buckets=cell_ecdf_num_buckets)
    x, _ = _inputs("multiclass")
    _, _, ecdf_full, scaler_stats = arch._preprocess_raw(x, num_train=NUM_TRAIN)
    _, _, ecdf_from_cache, _ = arch._preprocess_raw(
        x[NUM_TRAIN:], num_train=0, scaler_cache=scaler_stats
    )
    assert torch.equal(ecdf_full[:, NUM_TRAIN:], ecdf_from_cache)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
@pytest.mark.parametrize("use_chunkwise", [False, True])
# 16 buckets over 20 train rows makes the cached ranks lossy, so the cached and
# uncached paths agree only if both rank against the same buckets.
@pytest.mark.parametrize("cell_ecdf_num_buckets", [10_000, 16])
def test__kv_cache__matches_standard_forward(
    task_type: TaskType, use_chunkwise: bool, cell_ecdf_num_buckets: int
) -> None:
    """Reusing the cache on test-only rows must reproduce the full forward.

    Not bit-for-bit: the cached call feeds the attention kernels a test-only
    sequence instead of train+test, and the kernel's reduction order follows the
    sequence length. The tolerance is a float-noise bound.
    """
    arch = _get_model(cell_ecdf_num_buckets=cell_ecdf_num_buckets)
    x, y = _inputs(task_type)
    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    out_standard = arch(x, y, task_type=task_type, performance_options=perf)
    out_store, cache = arch(
        x, y, task_type=task_type, performance_options=perf, return_kv_cache=True
    )

    assert isinstance(cache, TabPFNV3p5Cache)
    assert not cache.is_empty()
    assert len(cache.kv) == _SMALL_CONFIG["nlayers"]
    assert cache.train_shape == (BATCH, NUM_TRAIN)
    torch.testing.assert_close(out_store, out_standard, rtol=0, atol=1e-6)

    # Test-only rows against the cache, and the full tensor against the cache.
    out_test_only = arch(
        x[NUM_TRAIN:],
        y,
        task_type=task_type,
        performance_options=perf,
        kv_cache=cache,
        x_is_test_only=True,
    )
    out_full = arch(x, y, task_type=task_type, performance_options=perf, kv_cache=cache)
    torch.testing.assert_close(out_test_only, out_standard, rtol=0, atol=1e-5)
    torch.testing.assert_close(out_full, out_standard, rtol=0, atol=1e-5)


@torch.no_grad()
def test__kv_cache__x_is_test_only_without_cache__raises() -> None:
    arch = _get_model()
    x, y = _inputs("multiclass")
    with pytest.raises(ValueError, match="x_is_test_only=True requires kv_cache"):
        arch(x[NUM_TRAIN:], y, task_type="multiclass", x_is_test_only=True)


@torch.no_grad()
def test__kv_cache__row_chunked_matches_unchunked() -> None:
    """Cached forward with a small inference_row_chunk_size must match unchunked."""
    arch = _get_model()
    x, y = _inputs("regression")
    perf = PerformanceOptions(use_chunkwise_inference=False)

    out_standard = arch(x, y, task_type="regression", performance_options=perf)
    _, cache = arch(
        x, y, task_type="regression", performance_options=perf, return_kv_cache=True
    )

    # Force multi-chunk test-row processing: 4 test rows / 3 per chunk = 2 chunks.
    arch.inference_row_chunk_size = 3
    out_cached_chunked = arch(
        x, y, task_type="regression", performance_options=perf, kv_cache=cache
    )
    torch.testing.assert_close(out_cached_chunked, out_standard, rtol=0, atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize(
    "config_overrides",
    [
        {"icl_num_kv_heads": 2, "icl_num_kv_heads_test": 1},
        {"icl_num_kv_heads": 4, "icl_num_kv_heads_test": 2},
        {"icl_num_kv_heads_test": None},
    ],
)
def test__kv_cache__gqa_variants_match_standard(
    config_overrides: dict[str, object],
) -> None:
    """KV-cache inference with GQA / MQA head layouts reproduces the forward."""
    arch = _get_model(**config_overrides)
    x, y = _inputs("regression")

    out_standard = arch(x, y, task_type="regression")
    _, cache = arch(x, y, task_type="regression", return_kv_cache=True)
    out_cached = arch(
        x[NUM_TRAIN:], y, task_type="regression", kv_cache=cache, x_is_test_only=True
    )
    torch.testing.assert_close(out_cached, out_standard, rtol=0, atol=1e-5)


@torch.no_grad()
def test__kv_cache__regression_caches_no_decoder_keys() -> None:
    """Regression has no many-class decoder, so its cache omits that term."""
    arch = _get_model()
    x, y = _inputs("regression")
    _, cache = arch(x, y, task_type="regression", return_kv_cache=True)
    assert cache.decoder_keys is None

    x_cls, y_cls = _inputs("multiclass")
    _, cls_cache = arch(x_cls, y_cls, task_type="multiclass", return_kv_cache=True)
    decoder = arch.heads.many_class_decoder
    assert cls_cache.decoder_keys.shape == (
        BATCH,
        NUM_TRAIN,
        decoder.num_heads,
        decoder.head_dim,
    )


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__kv_cache__cached_path__omits_train_embeddings(task_type: TaskType) -> None:
    """The cached path cannot report train embeddings; only the keys survive."""
    arch = _get_model()
    x, y = _inputs(task_type)
    _, cache = arch(x, y, task_type=task_type, return_kv_cache=True)
    output = arch(
        x[NUM_TRAIN:],
        y,
        task_type=task_type,
        kv_cache=cache,
        x_is_test_only=True,
        only_return_standard_out=False,
    )
    assert set(output) == {"standard", "test_embeddings"}


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
    """An fp32 cache is usable under an fp16/bf16 autocast forward."""
    arch = _get_model()
    x, y = _inputs("regression")
    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    out_standard = arch(x, y, task_type="regression", performance_options=perf)
    _, cache = arch(
        x, y, task_type="regression", performance_options=perf, return_kv_cache=True
    )

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        out_cached_autocast = arch(
            x, y, task_type="regression", performance_options=perf, kv_cache=cache
        )
        out_standard_autocast = arch(
            x, y, task_type="regression", performance_options=perf
        )

    # Autocast introduces precision differences; use a loose tolerance. bf16
    # carries three significant digits, so it needs the looser one.
    atol = 2e-2 if autocast_dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        out_cached_autocast.float(), out_standard.float(), rtol=0, atol=atol
    )
    torch.testing.assert_close(
        out_cached_autocast.float(), out_standard_autocast.float(), rtol=0, atol=atol
    )


# ---------------------------------------------------------------------------
# KV cache quantization
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("dtype", [QUANTIZED_KV_DTYPE, FP8_KV_DTYPE])
def test__quantize__kv_entries__only_the_kv_is_converted(dtype: torch.dtype) -> None:
    arch = _get_model()
    x, y = _inputs("multiclass")
    _, cache = arch(x, y, task_type="multiclass", return_kv_cache=True)
    quantized = cache.quantize(dtype)

    assert all(isinstance(e, QuantizedKVCacheEntry) for e in quantized.kv.values())
    assert all(e.key.dtype == dtype for e in quantized.kv.values())
    # The ECDF context is already narrow; the KV quantization leaves it alone.
    assert cache.ecdf_context.dtype == tabpfn_v3_5.ECDF_CONTEXT_DTYPE
    assert quantized.ecdf_context is cache.ecdf_context
    assert set(cache.scaler_cache) == {"mean", "std"}
    # Everything outside the KV cache is passed through untouched.
    assert quantized.decoder_keys is cache.decoder_keys
    assert quantized.scaler_cache is cache.scaler_cache
    assert quantized.inducing_hidden is cache.inducing_hidden
    assert quantized.train_shape == cache.train_shape


def test__kv_cache__quantize_passthrough_on_already_quantized() -> None:
    """quantize() must not re-quantize existing QuantizedKVCacheEntry values."""
    torch.manual_seed(0)
    entry = KVCacheEntry(key=torch.randn(1, 4, 1, 2), value=torch.randn(1, 4, 1, 2))
    cache = TabPFNV3p5Cache(kv={0: entry})
    q1 = cache.quantize()
    q2 = q1.quantize()
    assert isinstance(q2.kv[0], QuantizedKVCacheEntry)
    assert q1.kv[0] is q2.kv[0]


@torch.no_grad()
@pytest.mark.parametrize("task_type", ["multiclass", "regression"])
def test__forward__kv_cache__missing_cells_in_train_and_test__matches_uncached(
    task_type: TaskType,
) -> None:
    """Filled test cells must tie with the filled train cells they rank against."""
    arch = _get_model()
    x, y = _inputs(task_type)
    gen = torch.Generator().manual_seed(1)
    x = x.masked_fill(torch.rand(x.shape, generator=gen) < 0.4, float("nan"))
    full = arch(x, y, task_type=task_type)
    _, cache = arch(x, y, task_type=task_type, return_kv_cache=True)
    from_cache = arch(
        x[NUM_TRAIN:], y, task_type=task_type, kv_cache=cache, x_is_test_only=True
    )
    torch.testing.assert_close(full, from_cache, rtol=0, atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("cache_dtype", [QUANTIZED_KV_DTYPE, FP8_KV_DTYPE])
def test__kv_cache__layerwise_quantization_matches_post_forward(
    cache_dtype: torch.dtype,
) -> None:
    """Quantizing during construction produces the same cache as afterward."""
    arch = _get_model()
    x, y = _inputs("multiclass")

    _, full_precision = arch(x, y, task_type="multiclass", return_kv_cache=True)
    _, layerwise = arch(
        x,
        y,
        task_type="multiclass",
        return_kv_cache=True,
        performance_options=PerformanceOptions(kv_cache_dtype=cache_dtype),
    )
    post_forward = full_precision.quantize(cache_dtype)

    assert layerwise.decoder_keys.dtype == full_precision.decoder_keys.dtype
    for layer_idx in post_forward.kv:
        expected = post_forward.kv[layer_idx]
        actual = layerwise.kv[layer_idx]
        assert isinstance(expected, QuantizedKVCacheEntry)
        assert isinstance(actual, QuantizedKVCacheEntry)
        # torch.equal lacks CPU float8 support in the lowest supported PyTorch.
        # Comparing after an exact float32 widening works for int8 and float8.
        assert torch.equal(actual.key.float(), expected.key.float())
        assert torch.equal(actual.value.float(), expected.value.float())
        # The scales are an absmax over a fresh forward pass, and BLAS on some
        # platforms (macOS arm64) is not bitwise reproducible across runs.
        torch.testing.assert_close(
            actual.key_scale, expected.key_scale, rtol=1e-6, atol=0
        )
        torch.testing.assert_close(
            actual.value_scale, expected.value_scale, rtol=1e-6, atol=0
        )


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test__forward__quantized_kv_cache__equals_the_dequantized_cache(
    task_type: TaskType,
) -> None:
    """Quantizing must add nothing but the dequantize on the way back in."""
    arch = _get_model()
    x, y = _inputs(task_type)
    _, cache = arch(x, y, task_type=task_type, return_kv_cache=True)
    quantized = cache.quantize()
    dequantized = dataclasses.replace(
        quantized,
        kv={i: e.dequantize(torch.float32) for i, e in quantized.kv.items()},
    )
    predict = functools.partial(
        arch, x[NUM_TRAIN:], y, task_type=task_type, x_is_test_only=True
    )
    assert torch.equal(predict(kv_cache=quantized), predict(kv_cache=dequantized))


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
@pytest.mark.parametrize("use_chunkwise", [False, True])
def test__quantized_kv_cache__close_to_standard_forward(
    task_type: TaskType, use_chunkwise: bool
) -> None:
    """Int8-quantized KV cache produces output close to the standard forward.

    Decomposes error so a regression in the cache path itself (which should match
    standard at near machine precision) can't hide behind the loose int8 tolerance.
    """
    arch = _get_model()
    x, y = _inputs(task_type)
    perf = PerformanceOptions(use_chunkwise_inference=use_chunkwise)

    out_standard = arch(x, y, task_type=task_type, performance_options=perf)
    _, cache = arch(
        x, y, task_type=task_type, performance_options=perf, return_kv_cache=True
    )
    out_cached = arch(
        x, y, task_type=task_type, performance_options=perf, kv_cache=cache
    )
    out_quantized = arch(
        x, y, task_type=task_type, performance_options=perf, kv_cache=cache.quantize()
    )

    torch.testing.assert_close(out_cached, out_standard, rtol=0, atol=1e-5)
    torch.testing.assert_close(out_quantized, out_cached, rtol=0, atol=1e-2)


# ---------------------------------------------------------------------------
# Cache size
# ---------------------------------------------------------------------------


def _sum_cache_tensors(obj: object) -> int:
    """Recursively sum ``numel * element_size`` over every tensor in a cache.

    Walks dataclasses / dicts / lists so a newly-added cached tensor field is
    automatically included -- the completeness guard for ``get_cache_size``.
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


def _quantize(
    cache: TabPFNV3p5Cache, kv_cache_precision: Literal["auto", "int8", "fp8"]
) -> TabPFNV3p5Cache:
    """Apply the quantization step the inference engine applies."""
    if kv_cache_precision == "int8":
        return cache.quantize()
    if kv_cache_precision == "fp8":
        return cache.quantize(FP8_KV_DTYPE)
    return cache


@torch.no_grad()
@pytest.mark.parametrize("task_type", TASK_TYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("kv_cache_precision", ["int8", "fp8", "auto"])
@pytest.mark.parametrize("config_overrides", [{}, {"icl_num_kv_heads_test": None}])
def test__get_cache_size__matches_whole_cache(
    task_type: TaskType,
    kv_cache_precision: Literal["auto", "int8", "fp8"],
    dtype: torch.dtype,
    config_overrides: dict[str, object],
) -> None:
    """get_cache_size equals the exact byte size of every tensor in a real cache.

    Parametrized over ``dtype`` to cover the engine's forced-precision path: the
    engine casts the model and inputs to that dtype and runs the forward with
    autocast disabled, so every non-KV term lands at that one dtype.
    """
    arch = _get_model(**config_overrides)
    arch.type(dtype)  # mirror the engine's set_dtype for forced precision.
    # get_cache_size describes one estimator, so batch size 1.
    x, y = _inputs(task_type, batch=1)
    x, y = x.to(dtype), y.to(dtype)
    _, cache = arch(x, y, task_type=task_type, return_kv_cache=True)
    cache = _quantize(cache, kv_cache_precision)

    total = get_cache_size(
        n_train=NUM_TRAIN,
        n_features=NUM_FEATURES,
        model_config=_config(config_overrides),
        task_type=task_type,
        base_dtype=dtype,
        kv_cache_precision=kv_cache_precision,
    )
    assert total == _sum_cache_tensors(cache)


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Autocast inference is only enabled on CUDA (disabled on CPU/MPS), so "
    "the mixed-precision cache it produces can only be built on a GPU.",
)
@pytest.mark.parametrize("task_type", TASK_TYPES)
@pytest.mark.parametrize("kv_cache_precision", ["int8", "fp8", "auto"])
def test__get_cache_size__matches_whole_cache_autocast(
    task_type: TaskType, kv_cache_precision: Literal["auto", "int8", "fp8"]
) -> None:
    """get_cache_size matches a real cache built on the GPU autocast path.

    Autocast keeps fp32 model weights and casts ops at runtime, so the cache mixes
    dtypes; ``get_cache_size(base_dtype="autocast")`` must size each term at its
    real precision and still match to the byte.
    """
    device = torch.device("cuda")
    arch = _get_model().to(device)
    x, y = _inputs(task_type, batch=1)
    x, y = x.to(device), y.to(device)
    with get_autocast_context(device, enabled=True):
        _, cache = arch(x, y, task_type=task_type, return_kv_cache=True)
    cache = _quantize(cache, kv_cache_precision)

    total = get_cache_size(
        n_train=NUM_TRAIN,
        n_features=NUM_FEATURES,
        model_config=_config({}),
        task_type=task_type,
        base_dtype="autocast",
        kv_cache_precision=kv_cache_precision,
    )
    assert total == _sum_cache_tensors(cache)


def test__get_cache_size__mqa_smaller_than_mha() -> None:
    """Fewer cached KV heads (MQA on the test partition) shrinks the KV term."""
    common = {**_SMALL_CONFIG, "icl_num_kv_heads_test": None}
    mha, _ = tabpfn_v3_5.parse_config(common)  # H_kv = icl_num_heads = 4
    mqa, _ = tabpfn_v3_5.parse_config({**common, "icl_num_kv_heads_test": 1})
    n_train = 50
    # kv_cache_precision defaults to "int8", so the KV cache is int8 (1 byte)
    # regardless of base_dtype; base_dtype only sizes the (cancelling) non-KV terms.
    kw = {
        "n_train": n_train,
        "n_features": 5,
        "task_type": "multiclass",
        "base_dtype": torch.float32,
    }
    est_mha = get_cache_size(model_config=mha, **kw)
    est_mqa = get_cache_size(model_config=mqa, **kw)

    # mha and mqa differ ONLY in the KV term (H_kv 4 vs 1); every other term
    # (activations, inducing, scaler, ecdf) is identical, so it cancels in the diff.
    icl_emsize = mha.embed_dim * mha.feat_agg_num_cls_tokens
    head_dim = icl_emsize // mha.icl_num_heads
    kv_per_head = mha.nlayers * 2 * n_train * head_dim  # int8 KV -> 1 byte/element
    assert est_mqa < est_mha
    assert est_mha - est_mqa == (4 - 1) * kv_per_head


def test__get_cache_size__regression_omits_the_decoder_keys() -> None:
    config, _ = tabpfn_v3_5.parse_config(_SMALL_CONFIG)
    kw = {
        "n_train": 50,
        "n_features": 5,
        "model_config": config,
        "base_dtype": torch.float32,
    }
    multiclass = get_cache_size(task_type="multiclass", **kw)
    regression = get_cache_size(task_type="regression", **kw)
    decoder_keys = 50 * config.decoder_num_heads * config.decoder_head_dim * 4
    assert multiclass - regression == decoder_keys


def test__get_cache_size__ecdf_term_stops_growing_at_the_bucket_count() -> None:
    """Past `cell_ecdf_num_buckets` train rows, only the KV cache keeps growing."""
    config, _ = tabpfn_v3_5.parse_config({**_SMALL_CONFIG, "cell_ecdf_num_buckets": 16})
    kw = {
        "n_features": 5,
        "model_config": config,
        "task_type": "regression",
        "base_dtype": torch.float32,
        "kv_cache_precision": "auto",
    }
    head_dim = config.embed_dim * config.feat_agg_num_cls_tokens // config.icl_num_heads
    kv_per_row = config.nlayers * 2 * config.icl_num_kv_heads_test * head_dim * 4
    below = get_cache_size(n_train=16, **kw)
    above = get_cache_size(n_train=32, **kw)
    assert above - below == 16 * kv_per_row


def test__get_cache_size__invalid_precision__raises() -> None:
    config, _ = tabpfn_v3_5.parse_config(_SMALL_CONFIG)
    with pytest.raises(ValueError, match="Invalid kv_cache_precision"):
        get_cache_size(
            n_train=10,
            n_features=5,
            model_config=config,
            task_type="multiclass",
            base_dtype=torch.float32,
            kv_cache_precision="int4",  # type: ignore[arg-type]
        )


@pytest.mark.slow
def test__get_cache_size__tabpfn3_5_classifier_1000_rows() -> None:
    """Pin get_cache_size for the real TabPFN-v3.5 checkpoint at 1,000 train rows
    (1 estimator, engine defaults: int8 KV, fp16 rest).
    """
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)
    # Loads the checkpoint (config + weights) without needing fit data.
    clf._initialize_model_variables()
    config = clf.configs_[0]
    assert isinstance(config, TabPFNV3p5Config)

    n_train, n_features = 1000, 1
    total = get_cache_size(
        n_train=n_train,
        n_features=n_features,
        model_config=config,
        task_type="multiclass",
        base_dtype=torch.float16,
        kv_cache_precision="int8",
    )

    # Fixed terms for the shipped config (nlayers 24, H_kv 1, head_dim 64, 6x64
    # decoder), all at fp16 apart from the int8 KV:
    #   KV int8:             24 * 2 * 1 * 64 * 1000       = 3,072,000
    #   + int8 KV scales:    24 * 2 * 2 bytes             =        96
    #   + scaler stats:      2 * n_features * 2 bytes     =         4
    #   + fp16 decoder keys: 6 * 64 * 1000 * 2            =   768,000
    # The inducing and ECDF terms depend on the shipped embedder config, so they
    # are derived from it instead of hardcoded.
    inducing = (
        config.dist_embed_num_blocks
        * n_features
        * config.dist_embed_num_inducing_points
        * config.embed_dim
    ) * torch.float16.itemsize
    ecdf = (
        3
        * n_features
        * min(config.cell_ecdf_num_buckets, n_train)
        * tabpfn_v3_5.ECDF_CONTEXT_DTYPE.itemsize
    )
    # Numbers need manual update if we bump the default architecture.
    assert total == 3_072_000 + 96 + 4 + 768_000 + inducing + ecdf
