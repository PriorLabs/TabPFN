#  Copyright (c) Prior Labs GmbH 2026.

"""Per-test-row contexts drawn from one shared, already-preprocessed pool.

Every test row attends over its own ``k`` rows of a single training pool, given as
indices into that pool. Because the pool is fitted once, stages 0-2 of the v3
architecture become a function of each row alone, so the pool is embedded once and
each context is assembled by gathering row embeddings rather than by re-running the
pre-ICL stages per context.

Shared by the classifier and the regressor: everything here stops at the model's
raw output, and each estimator decodes that in its own way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from tabpfn.architectures.tabpfn_v3 import PoolStats, TabPFNV3
from tabpfn.inference import _maybe_run_gpu_preprocessing
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.utils import get_autocast_context

if TYPE_CHECKING:
    from tabpfn.constants import XType
    from tabpfn.preprocessing.ensemble import TabPFNEnsembleMember

#: Contexts scored per fused forward. Measured throughput is flat above ~128, so
#: larger chunks trade a lot of memory for a few percent; the spare memory is worth
#: more, since the pool embedding is the big resident tensor.
DEFAULT_CONTEXT_CHUNK_SIZE = 128


class SharedPoolError(ValueError):
    """Raised when an estimator or input cannot support shared-pool inference."""


def validate_context_indices(
    context_indices: np.ndarray | torch.Tensor,
    *,
    n_pool: int,
    n_test: int,
    n_estimators: int,
) -> torch.Tensor:
    """Normalise the index set to ``(n_estimators, n_test, k)`` and check it.

    Two shapes are accepted. ``(n_test, k)`` gives every ensemble member the same
    context for a test row. ``(n_estimators, n_test, k)`` gives each member its own,
    which is how ensemble diversity is expressed here: the classic per-estimator row
    subsample is a special case where every test row of a member shares one context.

    Args:
        context_indices: Pool indices, of either accepted shape.
        n_pool: Number of rows in the pool.
        n_test: Number of test rows being scored.
        n_estimators: Number of ensemble members.

    Returns:
        A ``(n_estimators, n_test, k)`` long tensor. The 2-D form is broadcast, so it
        costs no extra memory.

    Raises:
        SharedPoolError: On a wrong rank, a mismatched test-row or estimator count,
            an empty context, or an index outside the pool.
    """
    idx = torch.as_tensor(np.asarray(context_indices)).long()
    if idx.ndim == 2:
        # A view, not a copy: every member reads the same underlying indices.
        idx = idx.unsqueeze(0).expand(n_estimators, -1, -1)
    elif idx.ndim == 3:
        if idx.shape[0] != n_estimators:
            raise SharedPoolError(
                f"context_indices has {idx.shape[0]} estimator slots but the model "
                f"has {n_estimators} ensemble members. Pass (n_test, k) to share one "
                "context across members, or (n_estimators, n_test, k) to vary it."
            )
    else:
        raise SharedPoolError(
            f"context_indices must be (n_test, k) or (n_estimators, n_test, k), got "
            f"shape {tuple(idx.shape)}. Every test row needs the same number of "
            "context rows: the fused forward stacks them on the batch dimension and "
            "ragged contexts cannot stack."
        )

    if idx.shape[1] != n_test:
        raise SharedPoolError(
            f"context_indices covers {idx.shape[1]} test rows but X_test has {n_test}."
        )
    if idx.shape[2] == 0:
        raise SharedPoolError("Contexts must hold at least one pool row.")
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n_pool):
        raise SharedPoolError(
            f"context_indices must index the pool, i.e. lie in [0, {n_pool}); got "
            f"[{int(idx.min())}, {int(idx.max())}]."
        )
    return idx


def require_full_pool_members(estimator: object, members: Sequence) -> int:
    """Check every member holds the whole pool, and return its row count.

    Per-estimator row subsampling is incompatible with caller-supplied indices: each
    member would hold a *different* subset, so the same index would name a different
    row in each, and the row the caller meant would never be read. Nothing about that
    is detectable downstream, so it has to be refused here.

    Contexts that vary by estimator are still available, and are strictly more
    expressive: pass ``(n_estimators, n_test, k)`` indices instead.

    Args:
        estimator: The fitted estimator.
        members: Its ensemble members.

    Returns:
        The number of pool rows every member holds.

    Raises:
        SharedPoolError: If subsampling is configured or the members disagree.
    """
    config = getattr(estimator, "inference_config_", None)
    if getattr(config, "SUBSAMPLE_SAMPLES", None) is not None:
        raise SharedPoolError(
            "SUBSAMPLE_SAMPLES is not supported with per-row contexts: it gives each "
            "estimator a different subset of the training rows, so a pool index would "
            "mean a different row in every member. Drop it and pass "
            "(n_estimators, n_test, k) context_indices instead, which expresses the "
            "same per-estimator variation and more."
        )

    sizes = {int(np.asarray(m.X_train).shape[0]) for m in members}
    if len(sizes) != 1:
        raise SharedPoolError(
            f"Ensemble members hold different numbers of training rows ({sorted(sizes)}"
            "), so there is no single pool for the indices to address."
        )
    n_pool = sizes.pop()

    expected = getattr(estimator, "n_train_samples_", None)
    if expected is not None and n_pool != expected:
        raise SharedPoolError(
            f"Ensemble members hold {n_pool} rows but {expected} were fitted, so the "
            "members are not carrying the whole pool and caller-supplied indices "
            "cannot address it."
        )
    return n_pool


def require_shared_pool_support(estimator: object, model: object) -> TabPFNV3:
    """Reject estimator configurations shared-pool inference cannot honour.

    Args:
        estimator: The fitted estimator being scored.
        model: The architecture the estimator loaded.

    Returns:
        The model, narrowed to :class:`TabPFNV3`.

    Raises:
        SharedPoolError: If the architecture or the estimator's configuration is
            incompatible.
    """
    if not isinstance(model, TabPFNV3):
        raise SharedPoolError(
            "Shared-pool inference needs a TabPFN-v3 model: it relies on v3's split "
            "between the row-independent stages 0-2 and the ICL stage, which the "
            f"earlier architectures interleave. Got {type(model).__name__}."
        )
    if getattr(estimator, "tuning_config", None):
        raise SharedPoolError(
            "tuning_config is not supported: its calibrated state is fitted per "
            "dataset and there is no single dataset here."
        )
    if getattr(estimator, "balance_probabilities", False):
        raise SharedPoolError(
            "balance_probabilities is not supported with per-row contexts: the class "
            "balance differs per context, so there is no one correction to apply."
        )
    if getattr(estimator, "inference_precision", None) is torch.float64:
        raise SharedPoolError(
            "inference_precision=torch.float64 is not supported by the fused forward."
        )
    return model


def model_for_member(
    executor: object,
    member: TabPFNEnsembleMember,
    device: torch.device,
) -> TabPFNV3:
    """Resolve the architecture instance this member's config points at."""
    index = member.config._model_index
    model = executor.model_caches[index].get(device)  # type: ignore[attr-defined]
    if not isinstance(model, TabPFNV3):
        raise SharedPoolError(
            "Shared-pool inference needs a TabPFN-v3 model; got "
            f"{type(model).__name__}."
        )
    return model


def _member_pool_and_query(
    member: TabPFNEnsembleMember,
    X_test_clean: XType,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Preprocess this member's pool and query rows together.

    The GPU pipeline is fitted on the pool rows only (``num_train_rows=n_pool``),
    which is what makes the resulting pool rows reusable across every context.
    """
    x_pool = torch.as_tensor(np.asarray(member.X_train), dtype=torch.float32)
    x_query = torch.as_tensor(
        np.asarray(member.transform_X_test(X_test_clean)), dtype=torch.float32
    )
    n_pool = x_pool.shape[0]
    full, schema = _maybe_run_gpu_preprocessing(
        torch.cat([x_pool, x_query], dim=0).to(device),
        member.gpu_preprocessor,
        member.feature_schema,
        num_train_rows=n_pool,
    )
    cat_ix = schema.indices_for(FeatureModality.CATEGORICAL) or []
    return full[:n_pool], full[n_pool:], cat_ix


def score_member(
    model: TabPFNV3,
    member: TabPFNEnsembleMember,
    X_test_clean: XType,
    context_indices: torch.Tensor,
    *,
    device: torch.device,
    autocast: bool,
    chunk_size: int = DEFAULT_CONTEXT_CHUNK_SIZE,
    only_return_standard_out: bool = True,
) -> torch.Tensor:
    """Score every context for one ensemble member.

    Embeds the member's pool once, then walks the contexts in chunks, gathering row
    embeddings and running only the ICL stage.

    Args:
        model: The v3 architecture.
        member: The ensemble member, carrying its own preprocessed view of the pool.
        X_test_clean: Test rows, already validated and dtype-fixed by the caller.
        context_indices: ``(n_test, k)`` pool indices, one row per test row.
        device: Device to run on.
        autocast: Whether to run under autocast.
        chunk_size: Contexts per fused forward.
        only_return_standard_out: Passed through to the model.

    Returns:
        Model output for all contexts, concatenated on the batch dimension.
    """
    pool_rows, query_rows, _cat_ix = _member_pool_and_query(
        member, X_test_clean, device
    )
    y_pool = torch.as_tensor(np.asarray(member.y_train), dtype=torch.float32).to(device)

    n_test = context_indices.shape[0]
    idx = context_indices.to(device)

    with get_autocast_context(device, enabled=autocast), torch.inference_mode():
        # One pool element on the batch dimension: the statistics are shared by
        # every context, which is precisely what makes the gather valid.
        pool_emb, pool_stats = model.embed_pool(
            pool_rows.unsqueeze(1), y_pool.reshape(-1, 1)
        )
        query_emb = model.embed_rows(query_rows.unsqueeze(1), pool_stats)

        pool_by_row = pool_emb[0]
        outputs = []
        for start in range(0, n_test, chunk_size):
            end = min(start + chunk_size, n_test)
            chunk = idx[start:end]
            context_emb = pool_by_row[chunk]
            queries = query_emb[0, start:end].unsqueeze(1)
            stacked = torch.cat([context_emb, queries], dim=1)
            y_context = y_pool[chunk].transpose(0, 1)
            outputs.append(
                model(
                    None,
                    y_context,
                    precomputed_stage012=stacked,
                    only_return_standard_out=only_return_standard_out,
                )
            )
    # Outputs are sequence-first (M, B, ...) with M=1, so contexts concatenate on
    # dim 1 and the query axis is squeezed by the caller.
    return torch.cat(outputs, dim=1)


def pool_stats_for_member(
    model: TabPFNV3,
    member: TabPFNEnsembleMember,
    X_test_clean: XType,
    device: torch.device,
    *,
    autocast: bool,
) -> tuple[torch.Tensor, PoolStats, torch.Tensor]:
    """Embed one member's pool and query rows without scoring anything.

    Exposed for callers that want to reuse the pool embedding across several
    different context selections, which is free: a query row's embedding does not
    depend on the context it will later be scored against.
    """
    pool_rows, query_rows, _ = _member_pool_and_query(member, X_test_clean, device)
    y_pool = torch.as_tensor(np.asarray(member.y_train), dtype=torch.float32).to(device)
    with get_autocast_context(device, enabled=autocast), torch.inference_mode():
        pool_emb, pool_stats = model.embed_pool(
            pool_rows.unsqueeze(1), y_pool.reshape(-1, 1)
        )
        query_emb = model.embed_rows(query_rows.unsqueeze(1), pool_stats)
    return pool_emb, pool_stats, query_emb
