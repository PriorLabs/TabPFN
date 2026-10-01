#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: B905,C901,D103,E501,PLC0415,T201
"""Diagnostics: where do two identically fitted MPS regressors diverge?

Prints, for a fit_with_cache regressor and its save/load round-trip, the
aggregated logits, per-estimator outputs, KV caches, weights and decoded means,
and bisects any predict-to-predict difference down to the first module whose
output changes.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import platform
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.datasets import make_regression

import tabpfn
import tabpfn.architectures.shared.scaled_dot_product_attention as sdpa_module
import tabpfn.regressor as regressor_module
from tabpfn import TabPFNRegressor
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.constants import ModelVersion

ROW = 31
DEVICE = "mps"
TMP = Path(tempfile.mkdtemp())


def log(*args: Any) -> None:
    print(*args, flush=True)


def sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous()
    if a.dtype == torch.bfloat16:
        a = a.view(torch.int16)
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()[:12]


def diff(a: torch.Tensor, b: torch.Tensor) -> str:
    a = a.detach().cpu()
    b = b.detach().cpu()
    if a.shape != b.shape:
        return f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"DTYPE {a.dtype} vs {b.dtype}"
    if a.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        neq = a != b
        return f"int n_diff={int(neq.sum())}/{a.numel()}"
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        a, b = a.float(), b.float()
    af, bf = a.double(), b.double()
    neq = ~((af == bf) | (af.isnan() & bf.isnan()))
    n = int(neq.sum())
    if n == 0:
        return "IDENTICAL"
    d = (af - bf).abs()[neq]
    return f"n_diff={n}/{a.numel()} max_abs={float(d.max()):.3e}"


def tensors_of(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if depth > 8 or obj is None:
        return out
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(tensors_of(v, f"{prefix}[{k}]", depth + 1))
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            out.update(tensors_of(v, f"{prefix}[{i}]", depth + 1))
    elif hasattr(obj, "__dict__") and not isinstance(obj, type | torch.nn.Module):
        for k, v in vars(obj).items():
            out.update(tensors_of(v, f"{prefix}.{k}", depth + 1))
    return out


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

_captured: dict[str, Any] = {}
_orig_l2o = regressor_module._logits_to_output


def _spy_l2o(**kwargs: Any) -> Any:
    if kwargs["output_type"] == "mean":
        _captured["logits"] = kwargs["logits"].detach().clone()
    return _orig_l2o(**kwargs)


regressor_module._logits_to_output = _spy_l2o

_backend_counts: collections.Counter = collections.Counter()
_backend_lock = threading.Lock()
_orig_find = sdpa_module.find_attention_backend


def _spy_find(q, k, v, **kwargs):  # noqa: ANN001, ANN202
    backend = _orig_find(q, k, v, **kwargs)
    qkv = kwargs.get("quantized_kv")
    kdtype = "quantized" if qkv is not None else (k.dtype if k is not None else None)
    key = (
        getattr(backend, "name", None),
        str(q.dtype),
        str(kdtype),
        tuple(q.shape),
        q.is_contiguous(),
    )
    with _backend_lock:
        _backend_counts[key] += 1
    return backend


sdpa_module.find_attention_backend = _spy_find


def predict_with_capture(model: TabPFNRegressor, X: np.ndarray) -> dict[str, Any]:
    per_est: list[tuple[np.ndarray, torch.Tensor]] = []
    orig_iter = model._iter_forward_executor

    def wrapped(*a: Any, **k: Any):  # noqa: ANN202
        for borders_t, out in orig_iter(*a, **k):
            per_est.append((np.asarray(borders_t).copy(), out.detach().cpu().clone()))
            yield borders_t, out

    model._iter_forward_executor = wrapped  # type: ignore[method-assign]
    try:
        pred = model.predict(X)
    finally:
        del model._iter_forward_executor
    logits = _captured.pop("logits")
    return {"pred": pred, "logits": logits, "per_est": per_est}


def decode_means(model: TabPFNRegressor, logits: torch.Tensor) -> dict[str, np.ndarray]:
    zb = model.znorm_space_bardist_
    out = {}
    with torch.inference_mode():
        out["z_f32_dev"] = zb.mean(logits).cpu().numpy()
        zcpu = copy.deepcopy(zb).to("cpu")
        out["z_f32_cpu"] = zcpu.mean(logits.cpu()).numpy()
        out["z_f64_cpu"] = (
            copy.deepcopy(zcpu).double().mean(logits.cpu().double()).numpy()
        )
        old_raw = FullSupportBarDistribution(
            zb.borders.detach() * model.y_train_std_ + model.y_train_mean_
        ).float()
        out["oldraw_f32_dev"] = old_raw.mean(logits).cpu().numpy()
        out["oldraw_f32_cpu"] = (
            copy.deepcopy(old_raw).to("cpu").mean(logits.cpu()).numpy()
        )
    return out


def describe_model(tag: str, model: TabPFNRegressor) -> None:
    ex = model.executor_
    log(f"[{tag}] executor={type(ex).__name__} devices={model.devices_}")
    log(
        f"[{tag}] use_autocast_={getattr(model, 'use_autocast_', None)} "
        f"forced_inference_dtype_={getattr(model, 'forced_inference_dtype_', None)} "
        f"engine.force_inference_dtype={getattr(ex, 'force_inference_dtype', None)} "
        f"keep_cache_on_device={getattr(ex, 'keep_cache_on_device', None)} "
        f"kv_cache_precision={getattr(ex, 'kv_cache_precision', None)} "
        f"cache_groups={getattr(ex, 'cache_groups', None)}"
    )
    log(
        f"[{tag}] y_mean={model.y_train_mean_!r} y_std={model.y_train_std_!r} "
        f"avg_before_softmax={model.average_before_softmax} "
        f"temp={getattr(model, 'ensemble_softmax_temperature_', None)}"
    )
    for name in ("znorm_space_bardist_", "raw_space_bardist_"):
        b = getattr(model, name).borders
        log(f"[{tag}] {name}.borders dtype={b.dtype} device={b.device} sha={sha(b)}")
    for i, mc in enumerate(ex.model_caches):
        m = mc.get_any()
        p = next(m.parameters())
        classes = collections.Counter(type(mod).__name__ for mod in m.modules())
        linear_kinds = {k: v for k, v in classes.items() if "Linear" in k}
        log(
            f"[{tag}] model[{i}] {type(m).__name__} param dtype={p.dtype} "
            f"device={p.device} id={id(m)} linear_kinds={linear_kinds}"
        )
    for i, cache in enumerate(ex.kv_caches):
        ts = tensors_of(cache, f"kv[{i}]")
        dtypes = collections.Counter(str(t.dtype) for t in ts.values())
        devs = collections.Counter(str(t.device) for t in ts.values())
        noncontig = sum(not t.is_contiguous() for t in ts.values())
        offs = sum(t.storage_offset() != 0 for t in ts.values())
        log(
            f"[{tag}] kv[{i}] {type(cache).__name__} n_tensors={len(ts)} "
            f"dtypes={dict(dtypes)} devices={dict(devs)} noncontig={noncontig} "
            f"nonzero_offset={offs}"
        )


def compare_models(tag: str, a: TabPFNRegressor, b: TabPFNRegressor) -> None:
    for i, (ca, cb) in enumerate(zip(a.executor_.kv_caches, b.executor_.kv_caches)):
        ta, tb = tensors_of(ca, f"kv[{i}]"), tensors_of(cb, f"kv[{i}]")
        bad = []
        for k in ta:
            if k not in tb:
                bad.append(f"{k}: missing")
                continue
            d = diff(ta[k], tb[k])
            if d != "IDENTICAL":
                bad.append(f"{k}: {d}")
            elif ta[k].stride() != tb[k].stride():
                bad.append(
                    f"{k}: same values, strides {ta[k].stride()} vs {tb[k].stride()}"
                )
        log(f"[{tag}] kv[{i}] tensors={len(ta)} differing={len(bad)}")
        for line in bad[:10]:
            log(f"    {line}")
    sa = a.executor_.model_caches[0].get_any().state_dict()
    sb = b.executor_.model_caches[0].get_any().state_dict()
    bad = [k for k in sa if diff(sa[k], sb[k]) != "IDENTICAL"]
    log(f"[{tag}] weights: {len(sa)} tensors, differing={len(bad)} {bad[:5]}")
    za = a.znorm_space_bardist_.borders
    zb = b.znorm_space_bardist_.borders
    log(f"[{tag}] znorm borders: {diff(za, zb)}")
    for k in ("y_train_mean_", "y_train_std_"):
        log(f"[{tag}] {k}: {getattr(a, k)!r} vs {getattr(b, k)!r}")
    ma, mb = a.executor_.ensemble_members, b.executor_.ensemble_members
    for i, (ea, eb) in enumerate(zip(ma, mb)):
        ya, yb = np.asarray(ea.y_train), np.asarray(eb.y_train)
        log(
            f"[{tag}] member[{i}] y_train equal={np.array_equal(ya, yb)} "
            f"dtype={ya.dtype}/{yb.dtype}"
        )


def compare_runs(tag: str, ra: dict, rb: dict, ma: TabPFNRegressor) -> None:
    log(f"[{tag}] pred row{ROW}: {ra['pred'][ROW]!r} vs {rb['pred'][ROW]!r}")
    pd = np.abs(ra["pred"] - rb["pred"])
    log(f"[{tag}] pred max_abs={pd.max():.3e} rows_diff={np.flatnonzero(pd).tolist()}")
    log(
        f"[{tag}] logits {ra['logits'].dtype} {ra['logits'].device} "
        f"sha {sha(ra['logits'])} vs {sha(rb['logits'])}: "
        f"{diff(ra['logits'], rb['logits'])}"
    )
    la, lb = ra["logits"].cpu().double(), rb["logits"].cpu().double()
    rows = torch.nonzero((la != lb).any(-1)).flatten().tolist()
    log(f"[{tag}] logits rows differing: {rows}")
    if ROW in rows:
        log(f"[{tag}] logits row{ROW}: {diff(la[ROW], lb[ROW])}")
    for i, ((ba, oa), (bb, ob)) in enumerate(zip(ra["per_est"], rb["per_est"])):
        r = torch.nonzero(
            (oa.double() != ob.double()).reshape(oa.shape[0], -1).any(-1)
        ).flatten()
        log(
            f"[{tag}] est[{i}] out {tuple(oa.shape)} {oa.dtype}: {diff(oa, ob)} "
            f"rows={r.tolist()[:12]} borders_equal={np.array_equal(ba, bb)}"
        )
    da = decode_means(ma, ra["logits"].to(ma.devices_[0]))
    db = decode_means(ma, rb["logits"].to(ma.devices_[0]))
    for k in da:
        neq = np.flatnonzero(da[k] != db[k]).tolist()
        log(
            f"[{tag}] decode {k}: row{ROW} {da[k][ROW]!r} vs {db[k][ROW]!r} "
            f"rows_diff={neq}"
        )


def make_model(**kw: Any) -> TabPFNRegressor:
    params = {"device": DEVICE, "n_estimators": 4, "fit_mode": "fit_with_cache"}
    params.update(kw)
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3, **params)


def roundtrip(model: TabPFNRegressor, name: str) -> TabPFNRegressor:
    path = TMP / f"{name}.tabpfn_fit"
    model.save_fit_state(path)
    return TabPFNRegressor.load_from_fit_state(path, device=DEVICE)


def section(title: str) -> None:
    log("\n" + "=" * 78 + f"\n== {title}\n" + "=" * 78)


def seed_all() -> None:
    torch.manual_seed(42)
    np.random.seed(42)  # noqa: NPY002


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def exp_main(
    X: np.ndarray, y: np.ndarray
) -> tuple[TabPFNRegressor, TabPFNRegressor, dict]:
    section("E1: test sequence, original vs loaded, repeated predicts")
    seed_all()
    orig = make_model()
    orig.fit(X, y)
    describe_model("orig", orig)
    loaded = roundtrip(orig, "e1")
    describe_model("loaded", loaded)
    compare_models("orig~loaded (before predict)", orig, loaded)

    runs = {}
    runs["orig#1"] = predict_with_capture(orig, X)
    runs["loaded#1"] = predict_with_capture(loaded, X)
    for i in range(2, 5):
        runs[f"orig#{i}"] = predict_with_capture(orig, X)
        runs[f"loaded#{i}"] = predict_with_capture(loaded, X)
    for k, r in runs.items():
        log(f"  {k:10s} row{ROW}={r['pred'][ROW]!r} logits_sha={sha(r['logits'])}")
    compare_runs("orig#1~loaded#1", runs["orig#1"], runs["loaded#1"], orig)
    compare_runs("orig#1~orig#2", runs["orig#1"], runs["orig#2"], orig)
    compare_runs("loaded#1~loaded#2", runs["loaded#1"], runs["loaded#2"], orig)
    compare_models("orig~loaded (after predict)", orig, loaded)

    section("E1b: decode determinism on fixed logits (MPS, 20x)")
    logits = runs["orig#1"]["logits"].to(orig.devices_[0])
    vals = [orig.znorm_space_bardist_.mean(logits).cpu().numpy() for _ in range(20)]
    log(f"  distinct row{ROW} values: {sorted({float(v[ROW]) for v in vals})}")
    log(f"  all identical: {all(np.array_equal(vals[0], v) for v in vals)}")
    big = torch.empty(logits.numel() + 7, device=logits.device, dtype=logits.dtype)
    shifted = big[7:].view_as(logits)
    shifted.copy_(logits)
    v_shift = orig.znorm_space_bardist_.mean(shifted).cpu().numpy()
    log(f"  offset-7 copy identical: {np.array_equal(vals[0], v_shift)}")

    section("E2: fit determinism (three fresh fits, same seed)")
    fits = []
    for i in range(3):
        seed_all()
        m = make_model()
        m.fit(X, y)
        fits.append((m, predict_with_capture(m, X)))
        log(
            f"  fit#{i} row{ROW}={fits[-1][1]['pred'][ROW]!r} sha={sha(fits[-1][1]['logits'])}"
        )
    compare_models("fit#0~fit#1", fits[0][0], fits[1][0])
    compare_runs("fit#0~fit#1", fits[0][1], fits[1][1], fits[0][0])
    compare_models("fit#0~fit#2", fits[0][0], fits[2][0])

    section("E2b: same KV cache, swap caches between orig and loaded")
    try:
        loaded2 = roundtrip(orig, "e2b")
        loaded2.executor_.kv_caches = [c.to(DEVICE) for c in orig.executor_.kv_caches]
        r = predict_with_capture(loaded2, X)
        log(
            f"  loaded-with-orig-cache row{ROW}={r['pred'][ROW]!r} sha={sha(r['logits'])}"
        )
        compare_runs("orig#1~loaded_with_orig_cache", runs["orig#1"], r, orig)
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    del fits
    return orig, loaded, runs


def exp_bisect(
    model_a: TabPFNRegressor, model_b: TabPFNRegressor, X: np.ndarray, tag: str
) -> None:
    """Record every module's output during predict, compare in call order."""
    section(f"E3: module-level bisection {tag}")

    def record(model: TabPFNRegressor) -> list[tuple[str, torch.Tensor]]:
        rec: list[tuple[str, torch.Tensor]] = []
        handles = []
        m = model.executor_.model_caches[0].get_any()
        for name, mod in m.named_modules():

            def hook(_mod, _inp, out, name=name):  # noqa: ANN001, ANN202
                ts = tensors_of(out, "")
                for k, t in ts.items():
                    if t.is_floating_point():
                        rec.append((f"{name}{k}", t.detach().cpu().clone()))

            handles.append(mod.register_forward_hook(hook))
        try:
            model.predict(X)
        finally:
            for h in handles:
                h.remove()
        _captured.pop("logits", None)
        return rec

    ra, rb = record(model_a), record(model_b)
    log(f"  recorded {len(ra)} vs {len(rb)} module outputs")
    shown = 0
    first = None
    for i, ((na, ta), (nb, tb)) in enumerate(zip(ra, rb)):
        d = diff(ta, tb) if na == nb else f"NAME {na} vs {nb}"
        if d != "IDENTICAL":
            if first is None:
                first = i
            if shown < 25:
                log(f"  #{i} {na} {tuple(ta.shape)} {ta.dtype}: {d}")
                shown += 1
    if first is None:
        log("  all module outputs identical")
    else:
        log(f"  first differing call index: {first} of {len(ra)}")
        lo = max(0, first - 5)
        for i in range(lo, first):
            log(
                f"  (identical) #{i} {ra[i][0]} {tuple(ra[i][1].shape)} {ra[i][1].dtype}"
            )


def exp_attention_determinism() -> None:
    section("E4: torch-mps SDPA determinism on fixed inputs (shapes from the log)")
    from tabpfn.architectures.shared.torch_mps_backend import torch_mps_sdpa

    g = torch.Generator().manual_seed(0)
    cases = [
        ("gqa 40x40 h8/1 d64", 40, 40, 8, 1, 64),
        ("40x40 h8/8 d64", 40, 40, 8, 8, 64),
        ("15x15 h8/8 d16", 15, 15, 8, 8, 16),
        ("40x128 h8/8 d16", 40, 128, 8, 8, 16),
        ("128x40 h8/8 d16", 128, 40, 8, 8, 16),
    ]
    for dtype in (torch.float16, torch.float32):
        for name, sq, sk, hq, hk, d in cases:
            B = 5
            q = torch.randn(B, hq, sq, d, generator=g).to(DEVICE, dtype)
            k = torch.randn(B, hk, sk, d, generator=g).to(DEVICE, dtype)
            v = torch.randn(B, hk, sk, d, generator=g).to(DEVICE, dtype)
            outs = [
                torch_mps_sdpa(q, k, v, enable_gqa=hq != hk).cpu() for _ in range(30)
            ]
            distinct = len({sha(o) for o in outs})
            worst = max(
                float((o.double() - outs[0].double()).abs().max()) for o in outs
            )
            log(
                f"  {dtype} {name}: distinct outputs over 30 runs={distinct} max_dev={worst:.3e}"
            )


def exp_matmul_determinism() -> None:
    section("E5: MPS softmax@values (decode) determinism, 40x5000")
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 5000, generator=g).to(DEVICE)
    vals = torch.randn(5000, generator=g).to(DEVICE)
    outs = [(torch.softmax(logits, -1) @ vals).cpu() for _ in range(50)]
    log(f"  distinct outputs over 50 runs: {len({sha(o) for o in outs})}")


def exp_seeds(n_seeds: int) -> None:
    section(f"E6: {n_seeds} datasets; orig vs loaded: logits / new decode / old decode")
    counts = collections.Counter()
    for seed in range(n_seeds):
        try:
            Xs, ys = make_regression(n_samples=40, n_features=5, random_state=seed)
            seed_all()
            o = make_model()
            o.fit(Xs, ys)
            lo = roundtrip(o, f"s{seed}")
            ro = predict_with_capture(o, Xs)
            rl = predict_with_capture(lo, Xs)
            ro2 = predict_with_capture(o, Xs)
            dev = o.devices_[0]
            dn_o = decode_means(o, ro["logits"].to(dev))
            dn_l = decode_means(o, rl["logits"].to(dev))
            logit_rows = (
                torch.nonzero((ro["logits"].cpu() != rl["logits"].cpu()).any(-1))
                .flatten()
                .tolist()
            )
            same_model_rows = (
                torch.nonzero((ro["logits"].cpu() != ro2["logits"].cpu()).any(-1))
                .flatten()
                .tolist()
            )
            new_rows = np.flatnonzero(ro["pred"] != rl["pred"]).tolist()
            old_rows = np.flatnonzero(
                dn_o["oldraw_f32_dev"] != dn_l["oldraw_f32_dev"]
            ).tolist()
            fail_new = not np.allclose(ro["pred"], rl["pred"], rtol=0, atol=1.5e-6)
            fail_old = not np.allclose(
                dn_o["oldraw_f32_dev"], dn_l["oldraw_f32_dev"], rtol=0, atol=1.5e-6
            )
            counts["logits_differ"] += bool(logit_rows)
            counts["same_model_logits_differ"] += bool(same_model_rows)
            counts["new_decode_fails_test"] += fail_new
            counts["old_decode_fails_test"] += fail_old
            log(
                f"  seed={seed} logit_rows={logit_rows} same_model_rows={same_model_rows} "
                f"new_pred_rows={new_rows} old_pred_rows={old_rows} "
                f"fail_new={fail_new} fail_old={fail_old}"
            )
            del o, lo
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    log(f"  summary over {n_seeds} seeds: {dict(counts)}")


def exp_precision_variants(X: np.ndarray, y: np.ndarray) -> None:
    section("E7: variants (inference_precision, kv_cache_precision)")
    variants = {
        "fp32": {"inference_precision": torch.float32},
        "kv_auto": {"kv_cache_precision": "auto"},
        "fp32+kv_auto": {
            "inference_precision": torch.float32,
            "kv_cache_precision": "auto",
        },
    }
    for name, kw in variants.items():
        try:
            seed_all()
            o = make_model(**kw)
            o.fit(X, y)
            lo = roundtrip(o, f"v_{name}")
            rs = [predict_with_capture(m, X) for m in (o, lo, o, lo)]
            shas = [sha(r["logits"]) for r in rs]
            log(
                f"  {name}: autocast={o.use_autocast_} forced={o.forced_inference_dtype_} "
                f"logits sha o1,l1,o2,l2={shas} row{ROW}={[r['pred'][ROW] for r in rs]}"
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def main() -> None:
    log(f"python {sys.version}")
    log(f"platform {platform.platform()} {platform.machine()}")
    log(
        f"torch {torch.__version__} tabpfn {tabpfn.__version__ if hasattr(tabpfn, '__version__') else '?'}"
    )
    log(
        f"mps available={torch.backends.mps.is_available()} built={torch.backends.mps.is_built()}"
    )
    try:
        import mlx.core as mx

        log(f"mlx {mx.__version__}")
    except Exception as e:  # noqa: BLE001
        log(f"mlx unavailable: {e}")
    from tabpfn.architectures.shared.attention_backends import (
        registered_attention_backends,
    )

    log(f"attention backends: {[b.name for b in registered_attention_backends()]}")

    X, y = make_regression(n_samples=40, n_features=5, random_state=42)
    exp_attention_determinism()
    exp_matmul_determinism()
    try:
        orig, loaded, runs = exp_main(X, y)
        a_val, b_val = runs["orig#1"]["pred"][ROW], runs["orig#2"]["pred"][ROW]
        exp_bisect(orig, orig, X, "orig vs orig (same model, two predicts)")
        exp_bisect(orig, loaded, X, "orig vs loaded")
        log(f"  (E1 values for context: orig#1={a_val!r} orig#2={b_val!r})")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    exp_precision_variants(X, y)
    exp_seeds(12)

    section("attention backend usage (name, q dtype, k dtype, q shape, q contiguous)")
    for k, v in sorted(_backend_counts.items(), key=lambda kv: -kv[1])[:40]:
        log(f"  {v:5d} {k}")


if __name__ == "__main__":
    main()
