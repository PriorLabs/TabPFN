#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for shared-pool stage-0-2 embedding and per-context gathering.

The design these cover: with the stage-0-2 statistics held fixed at values fitted
on a training pool, a row's embedding is a function of that row alone. Pool rows
can then be embedded once and each context assembled by gathering, instead of
re-running stages 0-2 for every context.
"""

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures import tabpfn_v3
from tabpfn.architectures.interface import PerformanceOptions


def _get_model(*, nlayers: int = 1) -> tabpfn_v3.TabPFNV3:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=10,
        num_buckets=5,
        embed_dim=48,
        nlayers=nlayers,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float32)
    model.eval()
    return model


@torch.no_grad()
def test__gathered_pool_embeddings_match_direct_embedding() -> None:
    """A gathered pool row equals that row embedded on its own.

    This is the property the whole shared-pool design rests on: given the pool's
    statistics, stage 0-2 is row-independent.
    """
    arch = _get_model()
    n_pool, n_features = 200, 12
    x_pool = torch.randn(n_pool, 1, n_features) * 0.1
    y_pool = torch.randint(0, 10, [n_pool, 1], dtype=torch.float32)

    pool_emb, pool_stats = arch.embed_pool(x_pool, y_pool)
    assert pool_emb.shape[0] == 1
    assert pool_emb.shape[1] == n_pool

    context = torch.tensor([7, 3, 199, 0, 42, 42])
    gathered = pool_emb[:, context]
    directly = arch.embed_rows(x_pool[context], pool_stats)

    torch.testing.assert_close(gathered, directly, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test__query_embedding_is_independent_of_its_context() -> None:
    """The same query row embeds identically regardless of the context it joins."""
    arch = _get_model()
    x_pool = torch.randn(150, 1, 8) * 0.1
    y_pool = torch.randint(0, 10, [150, 1], dtype=torch.float32)
    _, pool_stats = arch.embed_pool(x_pool, y_pool)

    query = torch.randn(1, 1, 8) * 0.1
    alone = arch.embed_rows(query, pool_stats)
    with_others = arch.embed_rows(torch.cat([query, x_pool[:5]]), pool_stats)

    torch.testing.assert_close(alone, with_others[:, :1], rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test__predict_from_gathered_matches_predict_from_direct_embedding() -> None:
    """End to end: assembling a context by gather gives the same logits."""
    arch = _get_model(nlayers=2)
    n_pool, n_features, k = 120, 10, 16
    x_pool = torch.randn(n_pool, 1, n_features) * 0.1
    y_pool = torch.randint(0, 10, [n_pool, 1], dtype=torch.float32)
    pool_emb, pool_stats = arch.embed_pool(x_pool, y_pool)

    context = torch.randperm(n_pool)[:k]
    query = torch.randn(1, 1, n_features) * 0.1
    y_context = y_pool[context]

    query_emb = arch.embed_rows(query, pool_stats)
    gathered = torch.cat([pool_emb[:, context], query_emb], dim=1)
    from_gather = arch(None, y_context, precomputed_stage012=gathered)

    rows = torch.cat([x_pool[context], query])
    direct = arch(
        None, y_context, precomputed_stage012=arch.embed_rows(rows, pool_stats)
    )

    torch.testing.assert_close(from_gather, direct, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test__contexts_batch_independently() -> None:
    """Scoring B contexts in one batch equals scoring each on its own."""
    arch = _get_model(nlayers=2)
    n_pool, n_features, k, batch = 120, 10, 16, 4
    x_pool = torch.randn(n_pool, 1, n_features) * 0.1
    y_pool = torch.randint(0, 10, [n_pool, 1], dtype=torch.float32)
    pool_emb, pool_stats = arch.embed_pool(x_pool, y_pool)

    contexts = torch.stack([torch.randperm(n_pool)[:k] for _ in range(batch)])
    queries = torch.randn(batch, 1, n_features) * 0.1

    # Batched: one batch element per context.
    ctx_emb = pool_emb[0][contexts]  # (batch, k, Cl, E)
    q_emb = torch.stack(
        [arch.embed_rows(queries[i : i + 1], pool_stats)[0] for i in range(batch)]
    )
    batched_input = torch.cat([ctx_emb, q_emb], dim=1)
    y_batched = y_pool[contexts].squeeze(-1).transpose(0, 1)

    batched = arch(None, y_batched, precomputed_stage012=batched_input)

    # Output is sequence-first, (M, B, n_classes), so batch elements live on dim 1.
    assert batched.shape[:2] == (1, batch)
    for i in range(batch):
        single = arch(
            None,
            y_pool[contexts[i]],
            precomputed_stage012=batched_input[i : i + 1],
        )
        torch.testing.assert_close(batched[:, i : i + 1], single, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test__result_is_invariant_to_which_contexts_share_a_batch() -> None:
    """A context scores the same alone as beside a context with a wider class range.

    The many-class decoder sizes its one-hot targets from the batch's labels, so
    this is the guard that batching (and therefore chunk size) cannot move a
    result. Classes above the batch maximum contribute an all-zero column, which
    is why the width does not matter.
    """
    arch = _get_model(nlayers=2)
    n_pool, n_features, k = 120, 10, 16
    x_pool = torch.randn(n_pool, 1, n_features) * 0.1
    y_pool = torch.randint(0, 9, [n_pool, 1], dtype=torch.float32)
    y_pool[0] = 9.0  # the only row of the widest class
    pool_emb, pool_stats = arch.embed_pool(x_pool, y_pool)
    query = torch.randn(1, 1, n_features) * 0.1
    query_emb = arch.embed_rows(query, pool_stats)

    narrow = torch.arange(1, k + 1)  # excludes the class-9 row
    wide = torch.arange(0, k)  # includes it
    emb_narrow = torch.cat([pool_emb[:, narrow], query_emb], dim=1)
    emb_wide = torch.cat([pool_emb[:, wide], query_emb], dim=1)

    alone = arch(None, y_pool[narrow], precomputed_stage012=emb_narrow)
    together = arch(
        None,
        torch.cat([y_pool[narrow], y_pool[wide]], dim=1),
        precomputed_stage012=torch.cat([emb_narrow, emb_wide], dim=0),
    )
    torch.testing.assert_close(alone, together[:, :1], rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test__forward_does_not_mutate_supplied_embeddings() -> None:
    """Stage 3 adds the target embedding in place; a supplied buffer must survive.

    Without this the same gathered tensor scored twice would pick up the target
    embedding twice, and any reused pool buffer would be silently corrupted.
    """
    arch = _get_model(nlayers=2)
    x_pool = torch.randn(60, 1, 8) * 0.1
    y_pool = torch.randint(0, 10, [60, 1], dtype=torch.float32)
    pool_emb, pool_stats = arch.embed_pool(x_pool, y_pool)

    context = torch.arange(12)
    emb = torch.cat(
        [pool_emb[:, context], arch.embed_rows(torch.randn(1, 1, 8) * 0.1, pool_stats)],
        dim=1,
    )
    before = emb.clone()
    first = arch(None, y_pool[context], precomputed_stage012=emb)
    torch.testing.assert_close(emb, before, rtol=0, atol=0)

    second = arch(None, y_pool[context], precomputed_stage012=emb)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@torch.no_grad()
def test__precomputed_stage012_rejects_kv_cache_request() -> None:
    arch = _get_model()
    emb = torch.randn(1, 5, arch.config.feat_agg_num_cls_tokens, arch.config.embed_dim)
    y = torch.randint(0, 10, [4, 1], dtype=torch.float32)
    with pytest.raises(ValueError, match="return_kv_cache is not supported"):
        arch(None, y, precomputed_stage012=emb, return_kv_cache=True)


@torch.no_grad()
def test__forward_requires_x_when_no_precomputed_embeddings() -> None:
    arch = _get_model()
    y = torch.randint(0, 10, [4, 1], dtype=torch.float32)
    with pytest.raises(ValueError, match="x may only be None"):
        arch(None, y)


@torch.no_grad()
def test__pool_embedding_is_the_same_chunked_or_not() -> None:
    """A pool larger than the row chunk takes a different code path internally.

    Chunked stages 0-2 precompute the inducing states over all train rows and then
    walk the rows in chunks; the unchunked path computes them inline. A pool big
    enough to trigger chunking must still embed to the same values, or the pool
    pass would depend on ``inference_row_chunk_size``.
    """
    arch = _get_model()
    n_pool = 400
    x_pool = torch.randn(n_pool, 1, 6) * 0.1
    y_pool = torch.randint(0, 10, [n_pool, 1], dtype=torch.float32)

    arch.inference_row_chunk_size = 64
    chunked, chunked_stats = arch.embed_pool(x_pool, y_pool)

    unchunked, unchunked_stats = arch.embed_pool(
        x_pool,
        y_pool,
        performance_options=PerformanceOptions(use_chunkwise_inference=False),
    )

    torch.testing.assert_close(chunked, unchunked, rtol=1e-4, atol=1e-5)
    pairs = zip(
        chunked_stats.inducing_hidden, unchunked_stats.inducing_hidden, strict=True
    )
    for a, b in pairs:
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def _get_regression_model() -> tabpfn_v3.TabPFNV3:
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=0,
        num_buckets=5,
        embed_dim=48,
        nlayers=2,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(torch.float32)
    model.eval()
    return model


@torch.no_grad()
def test__pool_gathering_works_for_regression_too() -> None:
    """Stages 0-2 are task-agnostic, so the regressor needs no separate mechanism.

    The target is embedded into the pool rows during stages 0-2, so this only holds
    while the target scaling is a pool-level constant. Standardising y per context
    would make a pool row's embedding depend on the context it lands in, which is
    the same thing that rules out context-specific inducing states.
    """
    arch = _get_regression_model()
    assert arch.task_type == "regression"
    n_pool, n_features, k = 120, 10, 16
    x_pool = torch.randn(n_pool, 1, n_features) * 0.1
    y_pool = torch.randn(n_pool, 1)  # continuous, standardised on the pool upstream

    pool_emb, pool_stats = arch.embed_pool(x_pool, y_pool)
    context = torch.randperm(n_pool)[:k]
    query = torch.randn(1, 1, n_features) * 0.1

    gathered = torch.cat([pool_emb[:, context], arch.embed_rows(query, pool_stats)], 1)
    direct = arch.embed_rows(torch.cat([x_pool[context], query]), pool_stats)

    torch.testing.assert_close(
        pool_emb[:, context], direct[:, :k], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        arch(None, y_pool[context], precomputed_stage012=gathered),
        arch(None, y_pool[context], precomputed_stage012=direct),
        rtol=1e-4,
        atol=1e-5,
    )
