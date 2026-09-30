#  Copyright (c) Prior Labs GmbH 2026.
"""Check graph reuse as dataset shapes change on the same released CUDA model."""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out = args.out.resolve()
cache = args.out.parent / (args.out.stem + "-cache")
cache.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT / "src"))
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"

import torch

from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.model_loading import load_model

torch.set_num_threads(8)
torch.manual_seed(0)
model, _, _, _ = load_model(path=args.checkpoint, estimator_type="classifier")
model = model.eval().cuda()
model.inference_chunk_cells = 65536
report = {
    "revision": subprocess.check_output(  # noqa: S603 -- fixed git command
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip(),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "checkpoint": str(args.checkpoint),
    "cases": [],
}
shapes = [(1536, 1024, 2, 31), (1792, 1280, 3, 43), (2048, 1536, 4, 59)]
singletons = [(1536, 1024, 1, 31), (1792, 1280, 1, 43)]
torch._dynamo.utils.counters.clear()
for path, cases in [("full", shapes), ("singleton", singletons), ("chunked", shapes)]:
    graph_counts = []
    for rows, train, batch, features in cases:
        x = torch.randn(rows, batch, features, device="cuda")
        y = (
            (torch.arange(train, device="cuda") % 3)
            .float()
            .unsqueeze(1)
            .expand(-1, batch)
        )
        options = PerformanceOptions(use_chunkwise_inference=path == "chunked")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            eager = model(x, y, "multiclass", performance_options=options)
            torch.cuda.synchronize()
            start = time.perf_counter()
            compiled = model(
                x,
                y,
                "multiclass",
                performance_options=dataclasses.replace(
                    options, enable_torch_compile=True
                ),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        ref = eager.float().softmax(-1)
        actual = compiled.float().softmax(-1)
        delta = (actual - ref).abs()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.02)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        graph_counts.append(graphs)
        case = {
            "path": path,
            "rows": rows,
            "train": train,
            "batch": batch,
            "features": features,
            "prediction_s": elapsed,
            "cumulative_graphs": graphs,
            "max_probability_difference": delta.max().item(),
            "mean_probability_difference": delta.mean().item(),
            "label_agreement": (actual.argmax(-1) == ref.argmax(-1))
            .float()
            .mean()
            .item(),
        }
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    assert len(set(graph_counts)) == 1, (path, graph_counts)
assert not torch._dynamo.utils.counters["graph_break"]
report["complete"] = True
report["counters"] = {
    name: dict(values) for name, values in torch._dynamo.utils.counters.items()
}
args.out.write_text(json.dumps(report, indent=2) + "\n")
