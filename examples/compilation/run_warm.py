#  Copyright (c) Prior Labs GmbH 2026.
"""Compare eager, selected regions, and main through the same warm API sweep."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--main", type=Path, required=True)
parser.add_argument("--branch", type=Path, required=True)
parser.add_argument("--main-cache", type=Path)
parser.add_argument("--branch-cache", type=Path)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=False)
status = {"runs": []}


def save() -> None:
    (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


for name, repo, mode, seed in [
    ("eager", args.main, "eager", None),
    ("regions", args.branch, "compiled", args.branch_cache),
    ("main", args.main, "compiled", args.main_cache),
]:
    cache = args.out / "cache" / name
    if seed:
        shutil.copytree(seed, cache)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "warm.py"),
        "--repo",
        str(repo.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--mode",
        mode,
        "--out",
        str(args.out / name),
        "--cache",
        str(cache),
    ]
    if mode == "compiled":
        command += ["--reference", str(args.out / "eager")]
    status["running"] = name
    save()
    print("START", name, flush=True)
    started = time.perf_counter()
    with (args.out / (name + ".log")).open("w") as log:
        try:
            proc = subprocess.run(  # noqa: S603 -- fixed local benchmark script
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=1800,
            )
            outcome = {"exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            outcome = {"timeout_s": 1800}
    status["runs"].append(
        {"name": name, "wall_s": time.perf_counter() - started, **outcome}
    )
    save()
    print("DONE", name, json.dumps(outcome), flush=True)
status.pop("running", None)
status["finished"] = True
save()
