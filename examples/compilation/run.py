#  Copyright (c) Prior Labs GmbH 2026.
"""Serial fresh-process eager/compiled/cached checks, intended for one GPU job."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--train", type=int, default=50000)
parser.add_argument("--test", type=int, default=1024)
parser.add_argument("--features", type=int, default=200)
parser.add_argument("--estimators", type=int, default=4)
args = parser.parse_args()
OUT = args.out.resolve()
REPO = ROOT.parents[1]
# Refuse to overwrite measurements or accidentally reuse a cold-run cache.
OUT.mkdir(parents=True, exist_ok=False)
cache_root = OUT / "cache"
assert not (cache_root / "compiled").exists(), "Use a fresh cache for the cold check"


def status(state: str, **extra: object) -> None:
    (OUT / "status.json").write_text(
        json.dumps(
            {
                "state": state,
                "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "job_id": os.environ.get(
                    "SKYPILOT_JOB_ID", os.environ.get("SLURM_JOB_ID")
                ),
                **extra,
            },
            indent=2,
        )
        + "\n"
    )


status("starting")
for label, mode in [
    ("eager", "eager"),
    ("compiled", "compiled"),
    ("cached", "compiled"),
]:
    cache = cache_root / ("eager" if mode == "eager" else "compiled")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "check.py"),
        "--repo",
        str(REPO),
        "--mode",
        mode,
        "--out",
        str(OUT / (label + ".json")),
        "--cache",
        str(cache),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--train",
        str(args.train),
        "--test",
        str(args.test),
        "--features",
        str(args.features),
        "--estimators",
        str(args.estimators),
    ]
    if mode == "compiled":
        command += ["--reference", str(OUT / "eager.npy")]
    status("running", experiment=label)
    print("START", label, flush=True)
    with (OUT / (label + ".log")).open("w") as log:
        started = time.perf_counter()
        proc = subprocess.run(  # noqa: S603 -- fixed local entry point
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        process_s = time.perf_counter() - started
    if proc.returncode == 0:
        report_path = OUT / (label + ".json")
        report = json.loads(report_path.read_text())
        report["process_wall_s"] = process_s
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if proc.returncode:
        status("failed", experiment=label, exit_code=proc.returncode)
        print((OUT / (label + ".log")).read_text()[-6000:], flush=True)
        raise SystemExit(proc.returncode)
    print("DONE", label, flush=True)

compiled = json.loads((OUT / "compiled.json").read_text())
cached = json.loads((OUT / "cached.json").read_text())
assert cached["counters"].get("inductor", {}).get("fxgraph_cache_miss", 0) == 0
assert cached["counters"].get("inductor", {}).get("fxgraph_cache_hit", 0) > 0
hashes = [
    hashlib.sha256((OUT / (name + ".npy")).read_bytes()).hexdigest()
    for name in ["compiled", "cached"]
]
assert hashes[0] == hashes[1], "Fresh-process cache output changed"
status(
    "complete",
    audit="finite, repeatable, no graph breaks; cache hits only; exact cold/cached outputs",
)
print("CHECK PASSED", flush=True)
