#  Copyright (c) Prior Labs GmbH 2026.
"""Run shape probes with fresh and reused compiler caches, with/without KV cache."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=600)
args = parser.parse_args()
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=False)
status = {"runs": []}


def save() -> None:
    (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


for kv_cache in (False, True):
    label = "kv" if kv_cache else "no-kv"
    paths = ["full", "singleton", "chunked"]
    for reuse in (False, True):
        name = label + ("-cached" if reuse else "-cold")
        if not paths:
            break
        command = [
            sys.executable,
            "-u",
            str(ROOT / "shapes.py"),
            "--repo",
            str(args.repo.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--out",
            str(args.out / (name + ".json")),
            "--cache",
            str(args.out / (label + "-cache")),
            "--paths",
            *paths,
        ]
        if reuse:
            command.append("--reuse-cache")
        if kv_cache:
            command.append("--kv-cache")
        status["running"] = name
        save()
        print("START", name, flush=True)
        started = time.perf_counter()
        with (args.out / (name + ".log")).open("w") as log:
            try:
                process = subprocess.run(  # noqa: S603 -- fixed local probe
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=args.timeout,
                )
                outcome = {"exit_code": process.returncode}
            except subprocess.TimeoutExpired:
                outcome = {"timeout_s": args.timeout}
        report_path = args.out / (name + ".json")
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        # Do not retry paths that failed. Disk-cache checks cover only paths
        # that completed in the cold process.
        paths = report.get("completed_paths", [])
        status["runs"].append(
            {
                "name": name,
                **outcome,
                "wall_s": time.perf_counter() - started,
                "completed_paths": paths,
                "failures": report.get("failures", []),
            }
        )
        save()
        print("DONE", name, json.dumps(outcome), flush=True)
status.pop("running", None)
status["finished"] = True
save()
