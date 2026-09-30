#  Copyright (c) Prior Labs GmbH 2026.
"""Check graph reuse as dataset shapes change on the same released CUDA model."""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=ROOT)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--cache", type=Path)
parser.add_argument("--reuse-cache", action="store_true")
parser.add_argument("--kv-cache", action="store_true")
parser.add_argument(
    "--paths",
    nargs="+",
    choices=["full", "singleton", "chunked"],
    default=["full", "singleton", "chunked"],
)
args = parser.parse_args()
args.out = args.out.resolve()
args.repo = args.repo.resolve()
cache = (args.cache or args.out.parent / (args.out.stem + "-cache")).resolve()
if args.reuse_cache:
    assert cache.is_dir(), "A populated compiler cache is required"
else:
    cache.mkdir(parents=True, exist_ok=False)
assert not args.out.exists(), "Refusing to overwrite a shape report"
args.out.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.repo / "src"))
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"

import torch

import tabpfn
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.model_loading import load_model

torch.set_num_threads(8)
torch.manual_seed(0)
assert Path(tabpfn.__file__).resolve().is_relative_to(args.repo)
model, _, _, _ = load_model(path=args.checkpoint, estimator_type="classifier")
model = model.eval().cuda()
model.inference_chunk_cells = 65536
report = {
    "revision": subprocess.check_output(  # noqa: S603 -- fixed git command
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip(),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "tabpfn_source": tabpfn.__file__,
    "compiler_cache": str(cache),
    "reuse_disk_cache": args.reuse_cache,
    "kv_cache": args.kv_cache,
    "timing_scope": "Synchronized model forward; eager reference first; excludes loading and data generation.",
    "cases": [],
    "failures": [],
    "completed_paths": [],
}
shapes = [(1536, 1024, 2, 31), (1792, 1280, 3, 43), (2048, 1536, 4, 59)]
singletons = [(1536, 1024, 1, 31), (1792, 1280, 1, 43)]
torch._dynamo.utils.counters.clear()


def counters() -> dict:
    return {
        name: dict(values)
        for name, values in torch._dynamo.utils.counters.items()
        if name in ["stats", "inductor", "aot_autograd", "graph_break"]
    }


def save() -> None:
    report["counters"] = counters()
    args.out.write_text(json.dumps(report, indent=2) + "\n")


def compare_call(
    x: torch.Tensor,
    y: torch.Tensor,
    options: PerformanceOptions,
    metadata: dict,
    eager_kwargs: dict | None = None,
    compiled_kwargs: dict | None = None,
) -> tuple:
    report["running"] = metadata
    save()
    print("START", json.dumps(metadata), flush=True)
    before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        torch.cuda.synchronize()
        start = time.perf_counter()
        eager_result = model(
            x, y, "multiclass", performance_options=options, **(eager_kwargs or {})
        )
        torch.cuda.synchronize()
        eager_s = time.perf_counter() - start
        start = time.perf_counter()
        compiled_result = model(
            x,
            y,
            "multiclass",
            performance_options=dataclasses.replace(options, enable_torch_compile=True),
            **(compiled_kwargs or {}),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    eager = eager_result[0] if isinstance(eager_result, tuple) else eager_result
    compiled = (
        compiled_result[0] if isinstance(compiled_result, tuple) else compiled_result
    )
    ref = eager.float().softmax(-1)
    actual = compiled.float().softmax(-1)
    delta = (actual - ref).abs()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.02)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    case = {
        **metadata,
        "eager_s": eager_s,
        "prediction_s": elapsed,
        "cumulative_graphs": graphs,
        "new_graphs": graphs - before,
        "counters": counters(),
        "max_probability_difference": delta.max().item(),
        "mean_probability_difference": delta.mean().item(),
        "label_agreement": (actual.argmax(-1) == ref.argmax(-1)).float().mean().item(),
        "prediction_sha256": hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
    }
    report["cases"].append(case)
    print("RESULT", json.dumps(case), flush=True)
    save()
    return eager_result, compiled_result


for path in args.paths:
    cases = singletons if path == "singleton" else shapes
    call_started = time.perf_counter()
    # A failure stops this path, without retries or architecture changes. Other
    # paths are independent checks and can still provide useful measurements.
    try:
        for case_index, (rows, train, batch, features) in enumerate(cases):
            torch.manual_seed(
                100 * ["full", "singleton", "chunked"].index(path) + case_index
            )
            x = torch.randn(rows, batch, features, device="cuda")
            y = (
                (torch.arange(train, device="cuda") % 3)
                .float()
                .unsqueeze(1)
                .expand(-1, batch)
            )
            options = PerformanceOptions(use_chunkwise_inference=path == "chunked")
            metadata = {
                "path": path,
                "rows": rows,
                "train": train,
                "batch": batch,
                "features": features,
            }
            call_started = time.perf_counter()
            if not args.kv_cache:
                compare_call(x, y, options, {**metadata, "phase": "forward"})
                continue
            eager, compiled = compare_call(
                x,
                y,
                options,
                {**metadata, "phase": "prefill"},
                {"return_kv_cache": True},
                {"return_kv_cache": True},
            )
            extra = torch.randn(256, batch, features, device="cuda")
            test_x = torch.cat([x[train:], extra])
            for test_rows in (256, 512, 768):
                call_started = time.perf_counter()
                compare_call(
                    test_x[:test_rows],
                    y,
                    options,
                    {**metadata, "phase": "cached_predict", "test_rows": test_rows},
                    {"kv_cache": eager[1], "x_is_test_only": True},
                    {"kv_cache": compiled[1], "x_is_test_only": True},
                )
        report["completed_paths"].append(path)
    except Exception as exc:
        failure = {
            **report.get("running", {"path": path}),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": time.perf_counter() - call_started,
        }
        report["failures"].append(failure)
        print("FAILED", json.dumps(failure), flush=True)
    save()
report.pop("running", None)
report["complete"] = not report["failures"]
save()
