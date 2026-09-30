#  Copyright (c) Prior Labs GmbH 2026.
"""Warm sklearn API sweep, including one-row prediction from GPU KV caches."""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--mode", choices=["eager", "compiled"], required=True)
parser.add_argument("--reference", type=Path)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--single-repeats", type=int, default=25)
args = parser.parse_args()
args.repo = args.repo.resolve()
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(args.repo / "src"))
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.cache.resolve() / "inductor")
os.environ["TRITON_CACHE_DIR"] = str(args.cache.resolve() / "triton")
os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"
os.environ["MPLCONFIGDIR"] = str(args.out / "matplotlib")

import numpy as np
import torch
from sklearn.datasets import make_classification

import tabpfn
from tabpfn import TabPFNClassifier

torch.set_num_threads(8)
torch.manual_seed(0)
torch._dynamo.config.cache_size_limit = 64
assert Path(tabpfn.__file__).resolve().is_relative_to(args.repo)
report = {
    "revision": subprocess.check_output(  # noqa: S603 -- fixed git command
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "source": tabpfn.__file__,
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "mode": args.mode,
    "estimators": 4,
    "precision": "estimator default",
    "kv_cache_precision": "auto (computed dtype)",
    "keep_cache_on_device": True,
    "timing_scope": "CUDA-synchronized predict_proba, including preprocessing and ensemble overhead; fit and warmup separate; compiler caches may be populated.",
    "cases": [],
    "failures": [],
}
observed_shapes = set()
observe = False


def save() -> None:
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def counters() -> dict:
    return {
        k: dict(v)
        for k, v in torch._dynamo.utils.counters.items()
        if k in ["stats", "inductor", "aot_autograd", "graph_break"]
    }


def compilation_option(model: torch.nn.Module, inputs: tuple, kwargs: dict) -> tuple:
    if observe:
        x = inputs[0] if inputs else kwargs.get("x")
        y = inputs[1] if len(inputs) > 1 else kwargs.get("y")
        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        observed_shapes.add(
            (tuple(x.shape), len(y), kwargs.get("x_is_test_only", False))
        )
    if args.mode == "compiled":
        options = (
            kwargs.get("performance_options") or model.get_default_performance_options()
        )
        kwargs["performance_options"] = dataclasses.replace(
            options, enable_torch_compile=True
        )
    return inputs, kwargs


class BenchmarkClassifier(TabPFNClassifier):
    """Install the existing-flag adapter before fit can build a model KV cache."""

    def _initialize_model_variables(self) -> int:
        result = super()._initialize_model_variables()
        for model in self.models_:
            if compilation_option not in model._forward_pre_hooks.values():
                model.register_forward_pre_hook(compilation_option, with_kwargs=True)
        return result


def measure(clf: TabPFNClassifier, x: np.ndarray, metadata: dict) -> None:
    global observe  # noqa: PLW0603 -- shared observation switch for model hooks
    case = {
        **metadata,
        "test": len(x),
        "warmup_s": [],
        "times_s": [],
        "peak_allocated_gb": [],
    }
    report["cases"].append(case)
    report["running"] = {**metadata, "test": len(x)}
    print("START", json.dumps(report["running"]), flush=True)
    observed_shapes.clear()
    observe = True
    first = None
    for _ in range(2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = clf.predict_proba(x)
        torch.cuda.synchronize()
        case["warmup_s"].append(time.perf_counter() - start)
        if first is None:
            first = output.copy()
        save()
    observe = False
    case["observed_model_inputs"] = sorted(observed_shapes)
    assert observed_shapes
    assert all(s[1] == metadata["train"] for s in observed_shapes)
    if metadata["fit_mode"] == "fit_with_cache":
        assert all(s[0][0] == len(x) and s[2] for s in observed_shapes)
    before = counters()
    repeat_max = 0.0
    for _ in range(args.single_repeats if len(x) == 1 else args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = clf.predict_proba(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        case["times_s"].append(elapsed)
        case["peak_allocated_gb"].append(torch.cuda.max_memory_allocated() / 1e9)
        repeat_max = max(repeat_max, float(np.max(np.abs(output - first))))
        assert np.isfinite(output).all()
        save()
    after = counters()
    case.update(
        median_s=statistics.median(case["times_s"]),
        min_s=min(case["times_s"]),
        max_s=max(case["times_s"]),
        repeat_max_abs=repeat_max,
        new_graphs_during_measurement=after.get("stats", {}).get("unique_graphs", 0)
        - before.get("stats", {}).get("unique_graphs", 0),
        counters=after,
    )
    assert case["new_graphs_during_measurement"] == 0, (
        "Measured calls were not all warm"
    )
    assert repeat_max == 0, "Repeated prediction changed"
    assert np.max(np.abs(output.sum(-1) - 1)) < 1e-5
    name = f"{metadata['train']}x{metadata['features']}-{metadata['fit_mode']}-{len(x)}.npy"
    np.save(args.out / name, output)
    if args.reference and (args.reference / name).exists():
        ref = np.load(args.reference / name)
        delta = np.abs(output - ref)
        case["vs_eager"] = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "argmax_agreement": float(np.mean(output.argmax(-1) == ref.argmax(-1))),
        }
    case["complete"] = True
    save()
    print(
        "RESULT",
        json.dumps({k: v for k, v in case.items() if k != "counters"}),
        flush=True,
    )


for train, features in [
    (1000, 10),
    (100000, 10),
    (1000, 200),
    (50000, 200),
    (100000, 200),
]:
    x, y = make_classification(
        n_samples=train + 1024,
        n_features=features,
        n_informative=min(30, features - 2),
        n_classes=3,
        random_state=0,
    )
    x = x.astype(np.float32)
    for fit_mode in ("fit_preprocessors", "fit_with_cache"):
        metadata = {"train": train, "features": features, "fit_mode": fit_mode}
        clf = BenchmarkClassifier(
            model_path=args.checkpoint,
            device="cuda",
            n_estimators=4,
            categorical_features_indices=[],
            random_state=0,
            fit_mode=fit_mode,
            ignore_pretraining_limits=True,
            keep_cache_on_device=True,
            kv_cache_precision="auto",
            inference_config={"SUBSAMPLE_SAMPLES": None},
        )
        observe = True
        observed_shapes.clear()
        report["running"] = {**metadata, "phase": "fit"}
        save()
        print("FIT", json.dumps(metadata), flush=True)
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            clf.fit(x[:train], y[:train])
            torch.cuda.synchronize()
            metadata["fit_s"] = time.perf_counter() - start
            assert clf.n_estimators_ == 4
            assert clf.inference_config_.SUBSAMPLE_SAMPLES is None
            metadata["fit_model_inputs"] = sorted(observed_shapes)
            if fit_mode == "fit_with_cache":
                assert observed_shapes
                assert all(s[1] == train for s in observed_shapes)
                assert clf.executor_.kv_caches
                metadata["cache_groups"] = clf.executor_.cache_groups
            for test in [1, 1024] if fit_mode == "fit_with_cache" else [1024]:
                try:
                    measure(clf, x[train : train + test], metadata)
                except Exception as exc:
                    failure = {
                        **metadata,
                        "test": test,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    report["failures"].append(failure)
                    print("FAILED", json.dumps(failure), flush=True)
                    save()
        except Exception as exc:
            failure = {
                **metadata,
                "phase": "fit",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            report["failures"].append(failure)
            print("FAILED", json.dumps(failure), flush=True)
            save()
        finally:
            observe = False
            del clf
            gc.collect()
            torch.cuda.empty_cache()
    del x, y
report.pop("running", None)
report["complete"] = not report["failures"]
report["counters"] = counters()
save()
