#  Copyright (c) Prior Labs GmbH 2026.
"""Quick numeric v3.5-fast API check against an explicit source checkout.

Runs eager or aggregation compilation via PerformanceOptions.enable_torch_compile.
ICL, preprocessing, precision and the new checkout's batching policy are retained.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--repo", type=Path, default=ROOT.parents[1])
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--mode", choices=["eager", "compiled"], required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--reference", type=Path)
p.add_argument("--train", type=int, default=50000)
p.add_argument("--test", type=int, default=1024)
p.add_argument("--features", type=int, default=200)
p.add_argument("--estimators", type=int, default=4)
args = p.parse_args()
sys.path[:0] = [str(args.repo / "src"), str(ROOT)]
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.cache / "inductor")
os.environ["TRITON_CACHE_DIR"] = str(args.cache / "triton")
os.environ["MPLCONFIGDIR"] = str(args.out.parent / "matplotlib")

import numpy as np
import torch
from sklearn.datasets import make_classification

import tabpfn
from tabpfn import TabPFNClassifier
from tabpfn.architectures import tabpfn_v3_5 as architecture

torch.set_num_threads(8)
torch.manual_seed(0)
torch._dynamo.config.cache_size_limit = 64
assert torch.cuda.is_available(), "This check requires a GPU allocation"
assert Path(tabpfn.__file__).resolve().is_relative_to(args.repo.resolve())
model_path = args.checkpoint.resolve()
assert model_path.exists(), model_path

result = {
    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    "revision": subprocess.check_output(  # noqa: S603 -- fixed git command
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "compute_capability": list(torch.cuda.get_device_capability()),
    "inference_precision": "auto (estimator default)",
    "tabpfn_source": tabpfn.__file__,
    "checkpoint": str(model_path),
    "icl_compiled": False,
    "compilation_control": "PerformanceOptions.enable_torch_compile",
    "tuning": "default",
    "cache": str(args.cache),
}
with model_path.open("rb") as f:
    result["checkpoint_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
print("START", json.dumps(result), flush=True)
x, y = make_classification(
    n_samples=args.train + args.test,
    n_features=args.features,
    n_informative=min(30, args.features - 2),
    n_classes=3,
    random_state=0,
)
x = x.astype(np.float32)
clf = TabPFNClassifier(
    model_path=model_path,
    device="cuda",
    n_estimators=args.estimators,
    auto_scale_n_estimators=False,
    categorical_features_indices=[],
    random_state=0,
)
t = time.perf_counter()
clf.fit(x[: args.train], y[: args.train])
result["fit_s"] = time.perf_counter() - t
result["estimators"] = clf.n_estimators_
assert result["estimators"] == args.estimators
args.out.parent.mkdir(parents=True, exist_ok=True)


def enable_compilation(
    model: architecture.TabPFNV3p5, inputs: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    # The sklearn API does not expose PerformanceOptions directly. This benchmark
    # adapter only sets the existing per-forward flag; region selection lives in
    # the architecture and no model methods are replaced.
    options = (
        kwargs.get("performance_options") or model.get_default_performance_options()
    )
    kwargs["performance_options"] = dataclasses.replace(
        options, enable_torch_compile=True
    )
    return inputs, kwargs


if args.mode == "compiled":
    for model in clf.models_:
        model.register_forward_pre_hook(enable_compilation, with_kwargs=True)


def save() -> None:
    args.out.write_text(json.dumps(result, indent=2) + "\n")


result["times_s"] = []
result["peak_allocated_gb_per_call"] = []
first = None
for i in range(3):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = clf.predict_proba(x[args.train :])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if first is None:
        first = output.copy()
    result["times_s"].append(elapsed)
    result["peak_allocated_gb_per_call"].append(torch.cuda.max_memory_allocated() / 1e9)
    print("ITER", i, elapsed, flush=True)
    save()

result["warm_median_s"] = statistics.median(result["times_s"][1:])
result["all_finite"] = bool(np.isfinite(output).all())
result["repeat_max_abs"] = float(np.abs(first - output).max())
result["probability_sum_max_error"] = float(np.abs(output.sum(-1) - 1).max())
result["counters"] = {
    k: dict(v)
    for k, v in torch._dynamo.utils.counters.items()
    if k in ["stats", "graph_break", "inductor", "aot_autograd"]
}
np.save(args.out.with_suffix(".npy"), output)
if args.reference:
    ref = np.load(args.reference)
    delta = np.abs(output - ref)
    result["vs_eager"] = {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "argmax_agreement": float(np.mean(output.argmax(-1) == ref.argmax(-1))),
    }
assert result["all_finite"]
assert result["repeat_max_abs"] == 0
assert result["probability_sum_max_error"] < 1e-5
assert not result["counters"].get("graph_break")
result["complete"] = True
save()
print("RESULT", json.dumps(result), flush=True)
