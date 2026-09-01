# ruff: noqa: T201
#  Copyright (c) Prior Labs GmbH 2026.

"""Runtime benchmark for shared-pool per-context inference on TabPFN v3.

Compares, at the architecture level, two ways of scoring ``B`` test rows that each
attend over their own ``k``-row context drawn from one shared training pool:

``batched_recompute``
    What ``predict_proba_batched`` does model-side: contexts are stacked on the
    model's batch dimension and scored with one fused forward per chunk, but each
    context is built from raw rows, so stages 0-2 re-run for every context. A pool
    row is re-embedded once per context it appears in. This is a batched baseline,
    not a sequential one -- the fused forward of #1045 is already in it, and the
    only thing the pooled rung changes is where stages 0-2 happen.

``pooled``
    Same fused forward, but the pool is embedded once and each context is assembled
    by gathering row embeddings, so only the ICL stage runs per context. Reported
    with a breakdown so the ICL time -- the irreducible part, and the ceiling any
    implementation can reach -- is visible.

Runtime only: predictions are checked for shape and finiteness, not compared. The
comparison excludes the per-triple CPU preprocessing that ``predict_proba_batched``
also pays, so the speed-ups here are a conservative lower bound on the end-to-end
figure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from tabpfn.architectures import tabpfn_v3


@dataclass
class Config:
    """Benchmark configuration for one measured point."""

    n_pool: int
    batch: int
    k: int
    n_features: int
    nlayers: int
    embed_dim: int
    chunk: int
    device: str
    dtype: str
    repeats: int
    warmups: int
    k_global: int

    @property
    def redundancy(self) -> float:
        """Average number of contexts each pool row appears in."""
        return self.batch * self.k / self.n_pool


def build_model(cfg: Config) -> tabpfn_v3.TabPFNV3:
    """Build a v3 model with randomly initialised weights.

    Runtime depends on shapes and layer count, not on the weight values, so an
    untrained model measures the same thing a checkpoint would while keeping the
    benchmark self-contained.
    """
    heads = 8
    config = tabpfn_v3.TabPFNV3Config(
        max_num_classes=10,
        num_buckets=5,
        embed_dim=cfg.embed_dim,
        nlayers=cfg.nlayers,
        icl_num_heads=heads,
        dist_embed_num_heads=heads,
        feat_agg_num_heads=heads,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    model.to(device=cfg.device, dtype=getattr(torch, cfg.dtype))
    model.eval()
    return model


def make_data(
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool rows, pool labels, per-row context indices, and query rows."""
    dtype = getattr(torch, cfg.dtype)
    gen = torch.Generator(device="cpu").manual_seed(0)
    x_pool = (torch.randn(cfg.n_pool, 1, cfg.n_features, generator=gen) * 0.1).to(
        device=cfg.device, dtype=dtype
    )
    y_pool = torch.randint(
        0, 10, [cfg.n_pool, 1], generator=gen, dtype=torch.float32
    ).to(device=cfg.device, dtype=dtype)
    # Uniform random contexts. Runtime does not depend on which rows are picked, so
    # no selection machinery is needed here (accuracy work would need it).
    contexts = torch.stack(
        [torch.randperm(cfg.n_pool, generator=gen)[: cfg.k] for _ in range(cfg.batch)]
    ).to(cfg.device)
    queries = (torch.randn(cfg.batch, 1, cfg.n_features, generator=gen) * 0.1).to(
        device=cfg.device, dtype=dtype
    )
    return x_pool, y_pool, contexts, queries


def sync(device: str) -> None:
    """Block until queued device work has finished, so timings are real."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def run_batched_recompute(
    arch: tabpfn_v3.TabPFNV3,
    cfg: Config,
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    contexts: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Batched forward per chunk, re-running stages 0-2 for every context."""
    outputs = []
    for start in range(0, cfg.batch, cfg.chunk):
        end = min(start + cfg.chunk, cfg.batch)
        idx = contexts[start:end]
        # (k, chunk, C) context rows plus the query row appended as the test row.
        ctx_rows = x_pool[idx.reshape(-1), 0].reshape(end - start, cfg.k, -1)
        ctx_rows = ctx_rows.transpose(0, 1)
        q_rows = queries[start:end].transpose(0, 1)
        x = torch.cat([ctx_rows, q_rows], dim=0)
        y = y_pool[idx.reshape(-1), 0].reshape(end - start, cfg.k).transpose(0, 1)
        outputs.append(arch(x, y))
    return torch.cat(outputs, dim=1), {}


@torch.no_grad()
def run_pooled(
    arch: tabpfn_v3.TabPFNV3,
    cfg: Config,
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    contexts: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Embed the pool once, then gather each context and run the ICL stage."""
    parts: dict[str, float] = {}

    t0 = time.perf_counter()
    pool_emb, pool_stats = arch.embed_pool(x_pool, y_pool)
    sync(cfg.device)
    parts["pool_pass"] = time.perf_counter() - t0

    # queries is already (n_rows, B=1, C): every query is a row of the single
    # batch element the pool statistics were fitted for.
    t0 = time.perf_counter()
    query_emb = arch.embed_rows(queries, pool_stats)
    sync(cfg.device)
    parts["query_embed"] = time.perf_counter() - t0

    pool_rows = pool_emb[0]
    gather_s = 0.0
    icl_s = 0.0
    outputs = []
    for start in range(0, cfg.batch, cfg.chunk):
        end = min(start + cfg.chunk, cfg.batch)
        t0 = time.perf_counter()
        ctx_emb = pool_rows[contexts[start:end]]
        q_emb = query_emb[0, start:end].unsqueeze(1)
        stacked = torch.cat([ctx_emb, q_emb], dim=1)
        y = y_pool[contexts[start:end].reshape(-1), 0]
        y = y.reshape(end - start, cfg.k).transpose(0, 1)
        sync(cfg.device)
        gather_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        outputs.append(arch(None, y, precomputed_stage012=stacked))
        sync(cfg.device)
        icl_s += time.perf_counter() - t0

    parts["gather"] = gather_s
    parts["icl"] = icl_s
    return torch.cat(outputs, dim=1), parts


@torch.no_grad()
def run_global(
    arch: tabpfn_v3.TabPFNV3,
    cfg: Config,
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One shared context for every test row, as TabPFN-Rel runs today.

    The context is built once into a KV cache and every test row attends over it,
    so the context self-attention is paid once rather than per test row. This is
    the baseline the per-row rungs have to justify themselves against: it does
    less attention work when ``k_global`` is small, and more when it is large,
    since it scales as ``k_global*(k_global + B)`` against ``B*k**2``.
    """
    parts: dict[str, float] = {}
    n_ctx = min(cfg.k_global, x_pool.shape[0])
    x_ctx = x_pool[:n_ctx]
    y_ctx = (y_pool[:n_ctx, 0] if y_pool.dim() == 2 else y_pool[:n_ctx]).reshape(
        n_ctx, 1
    )

    t0 = time.perf_counter()
    _, cache = arch(x_ctx, y_ctx, return_kv_cache=True)
    sync(cfg.device)
    parts["cache_build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs = []
    for start in range(0, cfg.batch, cfg.chunk):
        end = min(start + cfg.chunk, cfg.batch)
        outputs.append(
            arch(queries[start:end], y_ctx, kv_cache=cache, x_is_test_only=True)
        )
    sync(cfg.device)
    parts["test_forward"] = time.perf_counter() - t0
    return torch.cat(outputs, dim=0), parts


def time_rung(
    fn: Callable[[], tuple[torch.Tensor, dict[str, float]]],
    cfg: Config,
) -> tuple[dict[str, object], torch.Tensor]:
    """Warm up, then time ``fn`` and report the median with its breakdown."""
    for _ in range(cfg.warmups):
        fn()
    sync(cfg.device)
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    times = []
    parts_runs: list[dict[str, float]] = []
    out = None
    for _ in range(cfg.repeats):
        sync(cfg.device)
        t0 = time.perf_counter()
        out, parts = fn()
        sync(cfg.device)
        times.append(time.perf_counter() - t0)
        parts_runs.append(parts)

    assert out is not None
    assert torch.isfinite(out).all(), "rung produced non-finite output"
    peak = (
        torch.cuda.max_memory_allocated() / 2**20
        if cfg.device.startswith("cuda")
        else float("nan")
    )
    median_parts = {
        key: statistics.median(p[key] for p in parts_runs) for key in parts_runs[0]
    }
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "peak_mib": peak,
        "out_shape": tuple(out.shape),
        "parts_s": median_parts,
    }, out


def main() -> int:
    """Run both rungs at one configuration and report the comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pool", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--n-features", type=int, default=100)
    ap.add_argument("--nlayers", type=int, default=24)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument(
        "--k-global",
        type=int,
        default=0,
        help=(
            "also time a single shared context of this many rows, KV-cached and "
            "reused by every test row (how TabPFN-Rel runs today). 0 disables it."
        ),
    )
    ap.add_argument(
        "--rungs",
        default="batched_recompute,pooled",
        help=(
            "comma-separated rungs to time. Selecting a subset matters when one "
            "rung cannot fit a configuration the others can: the rungs share a "
            "process, so an OOM in one loses the measurements for all of them."
        ),
    )
    ap.add_argument("--json", action="store_true", help="emit one JSON line")
    ap.add_argument(
        "--check-agreement",
        action="store_true",
        help=(
            "also report how far the two rungs' probabilities differ. A smoke "
            "check for gross bugs, not an accuracy measurement: the rungs fit "
            "their statistics on different rows by design."
        ),
    )
    args = ap.parse_args()

    cfg = Config(
        n_pool=args.n_pool,
        batch=args.batch,
        k=args.k,
        n_features=args.n_features,
        nlayers=args.nlayers,
        embed_dim=args.embed_dim,
        chunk=args.chunk,
        device=args.device,
        dtype=args.dtype,
        repeats=args.repeats,
        warmups=args.warmups,
        k_global=args.k_global,
    )
    arch = build_model(cfg)
    x_pool, y_pool, contexts, queries = make_data(cfg)

    rungs: list[tuple[str, Callable[..., tuple[torch.Tensor, dict[str, float]]]]] = [
        ("batched_recompute", run_batched_recompute),
        ("pooled", run_pooled),
    ]
    wanted = [r.strip() for r in args.rungs.split(",") if r.strip()]
    rungs = [r for r in rungs if r[0] in wanted]
    if cfg.k_global > 0:
        rungs.append(
            (
                "global",
                lambda a, c, xp, yp, _contexts, q: run_global(a, c, xp, yp, q),
            )
        )

    results = {}
    outputs = {}
    for name, fn in rungs:
        results[name], outputs[name] = time_rung(
            lambda fn=fn: fn(arch, cfg, x_pool, y_pool, contexts, queries), cfg
        )

    speedup = (
        results["batched_recompute"]["median_s"] / results["pooled"]["median_s"]
        if "batched_recompute" in results and "pooled" in results
        else None
    )
    icl_share = (
        results["pooled"]["parts_s"]["icl"] / results["pooled"]["median_s"]
        if "pooled" in results and results["pooled"]["parts_s"]
        else float("nan")
    )
    agreement = None
    if args.check_agreement and len(outputs) >= 2:
        # Not an accuracy benchmark: the two rungs fit their statistics on
        # different row sets by design, so they are not expected to match. This
        # only catches a gross bug, where the pooled path is wrong rather than
        # merely different.
        p_mat = torch.softmax(outputs["batched_recompute"].float(), dim=-1)
        p_pool = torch.softmax(outputs["pooled"].float(), dim=-1)
        agreement = {
            "mean_abs_prob_delta": float((p_mat - p_pool).abs().mean()),
            "max_abs_prob_delta": float((p_mat - p_pool).abs().max()),
            "top1_agreement": float(
                (p_mat.argmax(-1) == p_pool.argmax(-1)).float().mean()
            ),
        }

    if "global" in results:
        summary_global = results["pooled"]["median_s"] / results["global"]["median_s"]
    else:
        summary_global = None

    summary = {
        "config": asdict(cfg),
        "agreement": agreement,
        "pooled_over_global": summary_global,
        "redundancy": cfg.redundancy,
        "results": results,
        "speedup_pooled_vs_batched_recompute": speedup,
        "icl_share_of_pooled": icl_share,
    }

    if args.json:
        print(json.dumps(summary))
        return 0

    print(
        f"n_pool={cfg.n_pool} B={cfg.batch} k={cfg.k} d={cfg.n_features} "
        f"layers={cfg.nlayers} chunk={cfg.chunk} {cfg.device}/{cfg.dtype} "
        f"redundancy={cfg.redundancy:.1f}x"
    )
    for name, res in results.items():
        parts = res["parts_s"]
        detail = (
            "  " + "  ".join(f"{k}={v:.3f}s" for k, v in parts.items()) if parts else ""
        )
        print(
            f"  {name:18s} {res['median_s']:8.3f}s  peak={res['peak_mib']:8.1f}MiB"
            f"{detail}"
        )
    if speedup is not None:
        pct = icl_share * 100
        print(f"  speedup      {speedup:8.2f}x   (ICL is {pct:.0f}% of pooled)")
    if summary_global is not None:
        print(f"  pooled/global {summary_global:8.2f}x  (>1 means slower than today)")
    if agreement is not None:
        print(
            f"  agreement    top1={agreement['top1_agreement'] * 100:.1f}%  "
            f"mean|dp|={agreement['mean_abs_prob_delta']:.4f}  "
            f"max|dp|={agreement['max_abs_prob_delta']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
