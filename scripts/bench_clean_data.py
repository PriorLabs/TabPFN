# ruff: noqa: T201
#  Copyright (c) Prior Labs GmbH 2026.

r"""Regression gate for `clean_data`'s transient RAM and wall time.

`clean_data` was measured holding 42.68 GB of transient RSS -- the single largest
contributor to the wrapper's host-RAM peak -- while fitting a 666,667 x 2,000
float32 table. That measurement came from the wrapper-wide `profiler.py` in
fomo-fitting's `experimental/arthur/res-2439-reduce-the-ram-footprint-of-the-wrapper/`;
this script isolates the same call so it can be iterated on without paying for a
whole fit + predict.

What it measures, on exactly the array that profiler hands to `clean_data`:

* **Transient RSS** -- the peak process RSS reached inside the call, minus the RSS
  on entry, sampled by a background thread so a spike that is allocated and freed
  between two boundaries is still caught. RSS rather than tracemalloc because the
  copies under suspicion are numpy/pandas buffers, invisible to Python's allocator.
* **Wall time** -- `torch.utils.benchmark`, median over `--timing-repeats` calls,
  after `--warmup-calls` untimed ones.

Both metrics, plus the full outputs, are written to a directory named after the
input shape. On a later run that finds a complete set of files there, the script
switches to gate mode: the new run must not use more transient RSS, must not take
longer, and must produce identical outputs, or it exits non-zero. Each check runs
as soon as its input exists, so a regression fails before the next (more expensive)
stage is paid for. Passing overwrites the baseline.

Smoke test locally (seconds, ~20 MB on disk):

    uv run scripts/bench_clean_data.py --small

Full shape -- needs ~55 GB of RAM and writes ~11 GB per run:

    srun -p gpuh100flex --gres=gpu:1 --mem=0 --time=01:00:00 \
        uv run scripts/bench_clean_data.py
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import pickle
import platform
import resource
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.benchmark import Measurement, Timer

from tabpfn.preprocessing.clean import clean_data
from tabpfn.preprocessing.modality_detection import detect_feature_modalities

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.steps.preprocessing_helpers import (
        OrderPreservingColumnTransformer,
    )

# ---------------------------------------------------------------------------
# The input `clean_data` receives from the wrapper profiler
# ---------------------------------------------------------------------------
#
# The profiler generates `--rows` rows and splits them the way BeyondArena does
# below 1.25M rows, so `fit` -- and therefore `clean_data` -- sees the train part
# only. Reproduced here down to the RNG call order, so the array is bit-identical
# to the profiled one.
DEFAULT_PROFILER_ROWS = 1_000_000
DEFAULT_COLS = 2_000
SMALL_PROFILER_ROWS = 20_000
SMALL_COLS = 100
BEYOND_ARENA_N_FOLDS = 3
SEED = 0

# Modality-detection thresholds from the profiler's 3.1_exp `InferenceConfig`.
# Standard-normal columns land on NUMERICAL for every one of these, but the
# detection has to run anyway: its `FeatureSchema` is what `clean_data` consumes.
MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE = 100
MAX_UNIQUE_FOR_CATEGORICAL_FEATURES = 30
MIN_UNIQUE_FOR_NUMERICAL_FEATURES = 4

# `InferenceConfig.PASSTHROUGH_INF` defaults to False and the profiler leaves it
# alone, so the +/-inf masking path is not exercised.
PASSTHROUGH_INF = False

# Bumped whenever the meaning of a recorded metric changes, so an old baseline is
# rejected rather than silently compared against.
SCHEMA_VERSION = 1

METRICS_FILE = "metrics.json"
ARRAY_FILE = "X_cleaned.npy"
AUX_FILE = "aux.pkl"

# Head and tail elements hashed into the input fingerprint. Enough to catch a
# changed generator or dtype without hashing gigabytes.
FINGERPRINT_ELEMENTS = 1_000_000

# Cells per block when comparing against a recorded array. Sized so the boolean
# temporaries the comparison builds stay in the tens of MB whatever the shape.
COMPARISON_CHUNK_CELLS = 32_000_000

_PAGE_SIZE = resource.getpagesize()
_GB = 1e9


# ---------------------------------------------------------------------------
# RSS sampling
# ---------------------------------------------------------------------------


def current_rss_bytes() -> int:
    """Resident set size of this process, read from /proc/self/statm."""
    with Path("/proc/self/statm").open() as handle:
        return int(handle.read().split()[1]) * _PAGE_SIZE


class RssSampler:
    """Background thread recording (time, RSS) so peaks between calls are caught.

    Boundary-only readings miss any allocation that is made and freed inside a
    single call, which is exactly the copy behaviour under investigation.
    """

    def __init__(self, interval_s: float) -> None:
        """Start nothing yet; call `start`."""
        self.interval_s = interval_s
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling in a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), current_rss_bytes()))
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        """Stop sampling and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def peak_between(self, start: float, end: float) -> int:
        """Highest sampled RSS in a time window, 0 if no sample landed in it.

        Snapshots the list first: the sampler thread appends to it concurrently.
        """
        values = [rss for t, rss in list(self.samples) if start <= t <= end]
        return max(values) if values else 0


@dataclasses.dataclass
class MemoryProfile:
    """RSS around one `clean_data` call."""

    rss_in_bytes: int
    peak_rss_bytes: int
    rss_out_bytes: int
    n_samples: int
    wall_s: float

    @property
    def transient_bytes(self) -> int:
        """Peak above the RSS on entry: memory allocated and freed inside."""
        return max(0, self.peak_rss_bytes - self.rss_in_bytes)

    @property
    def retained_bytes(self) -> int:
        """RSS still held after returning, relative to entry."""
        return self.rss_out_bytes - self.rss_in_bytes

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, including the derived quantities."""
        return {
            "rss_in_bytes": self.rss_in_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "rss_out_bytes": self.rss_out_bytes,
            "transient_rss_bytes": self.transient_bytes,
            "retained_rss_bytes": self.retained_bytes,
            "n_rss_samples": self.n_samples,
            "wall_s": self.wall_s,
        }


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def train_rows_for(profiler_rows: int) -> int:
    """Rows reaching `fit`, i.e. what the profiler's split leaves for training."""
    test_rows = max(1, round(profiler_rows / BEYOND_ARENA_N_FOLDS))
    return profiler_rows - test_rows


def generate_input(profiler_rows: int, cols: int, dtype: str) -> np.ndarray:
    """The train matrix the profiler hands to `fit`, and so to `clean_data`.

    Drawn as float32 and cast afterwards, matching the profiler's `generate_inputs`
    exactly -- the draw order fixes the values, so this is the same array.

    `ensure_compatible_fit_inputs` sits between `fit` and `clean_data`, but with
    `dtype=None` sklearn's `check_array` passes a C-contiguous float array straight
    through, so the array built here is what `clean_data` actually sees.
    """
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(
        (train_rows_for(profiler_rows), cols), dtype=np.float32
    ).astype(np.dtype(dtype), copy=False)


def fingerprint(X: np.ndarray) -> str:
    """Digest of the input's shape, dtype and edge values.

    Guards the comparison: a baseline is only meaningful against the same input,
    and the generator could drift (a numpy RNG change, a different seed).
    """
    digest = hashlib.sha256()
    digest.update(f"{X.shape}|{X.dtype}|{SEED}".encode())
    flat = X.ravel()
    digest.update(flat[:FINGERPRINT_ELEMENTS].tobytes())
    digest.update(flat[-FINGERPRINT_ELEMENTS:].tobytes())
    return digest.hexdigest()


def build_feature_schema(X: np.ndarray) -> FeatureSchema:
    """Run the wrapper's modality detection, as `fit` does just before cleaning."""
    return detect_feature_modalities(
        X=X,
        # A numpy input carries no column names, so the schema gets positional ones.
        feature_names=None,
        provided_categorical_indices=None,
        min_samples_for_inference=MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
        max_unique_for_category=MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
        min_unique_for_numerical=MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

CleanDataOutput = tuple[np.ndarray, "OrderPreservingColumnTransformer", "FeatureSchema"]


def measure_memory(
    X: np.ndarray,
    feature_schema: FeatureSchema,
    interval_s: float,
) -> tuple[CleanDataOutput, MemoryProfile]:
    """Run `clean_data` once, sampling RSS throughout, and keep its output.

    Deliberately un-warmed, and deliberately before the timing pass: once a call
    has run, the allocator holds on to the arenas it freed, so a second call can
    reach the same peak without growing RSS at all -- which would read as a
    phantom improvement.
    """
    gc.collect()
    sampler = RssSampler(interval_s)
    rss_in = current_rss_bytes()
    sampler.start()
    start = time.perf_counter()
    try:
        output = clean_data(
            X=X,
            feature_schema=feature_schema,
            passthrough_inf=PASSTHROUGH_INF,
        )
    finally:
        end = time.perf_counter()
        sampler.stop()
    rss_out = current_rss_bytes()
    profile = MemoryProfile(
        rss_in_bytes=rss_in,
        # A sampled peak can only err downwards, so floor it at the two readings
        # that are known to be real.
        peak_rss_bytes=max(sampler.peak_between(start, end), rss_in, rss_out),
        rss_out_bytes=rss_out,
        n_samples=len(sampler.samples),
        wall_s=end - start,
    )
    return output, profile


def measure_time(
    X: np.ndarray,
    feature_schema: FeatureSchema,
    repeats: int,
    warmup_calls: int,
) -> Measurement:
    """Median-of-`repeats` wall time for one `clean_data` call.

    `clean_data` does not mutate `X` -- `fix_dtypes` builds a new float64 frame from
    it -- so every call, warmup included, measures the same work.

    The warmup is what makes the repeats comparable: it settles the allocator, the
    page cache and the BLAS/OpenMP thread pools, all of which the first call pays
    for. `Timer.timeit` would run one for us, but it re-runs it on every call
    (`max(number // 100, 2)` executions each time), which at the default shape is
    two extra minutes per repeat -- so warm up once here and time the repeats
    through the same inner timer afterwards.
    """
    timer = Timer(
        stmt=(
            "clean_data(X=X, feature_schema=feature_schema, "
            "passthrough_inf=passthrough_inf)"
        ),
        globals={
            "clean_data": clean_data,
            "X": X,
            "feature_schema": feature_schema,
            "passthrough_inf": PASSTHROUGH_INF,
        },
        num_threads=torch.get_num_threads(),
        label="clean_data",
    )
    if warmup_calls:
        timer._timeit(number=warmup_calls)
    raw_times = [timer._timeit(number=1) for _ in range(repeats)]
    return Measurement(
        number_per_run=1,
        raw_times=raw_times,
        task_spec=timer._task_spec,
    )


def timing_as_dict(measurement: Measurement, warmup_calls: int) -> dict[str, Any]:
    """JSON-serialisable timing stats. `median_s` is the gated number."""
    times = sorted(measurement.times)
    return {
        "median_s": measurement.median,
        "mean_s": measurement.mean,
        "min_s": times[0],
        "max_s": times[-1],
        "repeats": len(times),
        "warmup_calls": warmup_calls,
        "raw_times_s": list(measurement.times),
        "num_threads": measurement.task_spec.num_threads,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def shape_dir_name(train_rows: int, cols: int, dtype: str) -> str:
    """Directory name identifying one input shape."""
    return f"rows{train_rows}_cols{cols}_{dtype}"


def baseline_paths(out_dir: Path) -> dict[str, Path]:
    """The three files that together make up a baseline."""
    return {
        "metrics": out_dir / METRICS_FILE,
        "array": out_dir / ARRAY_FILE,
        "aux": out_dir / AUX_FILE,
    }


def write_outputs(
    out_dir: Path,
    X_cleaned: np.ndarray,
    ord_encoder: OrderPreservingColumnTransformer,
    feature_schema: FeatureSchema,
) -> list[tuple[Path, Path]]:
    """Write the outputs beside their final names, as `.tmp` files.

    Returned as (temporary, final) pairs: nothing replaces a baseline until every
    check has passed, and the array is large enough that a half-written file would
    be an expensive thing to leave behind.

    `np.save` for the array -- a straight buffer write, and it can be re-read
    lazily with `mmap_mode` so a later comparison need not hold both copies -- and
    pickle protocol 5 for the two small objects.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = baseline_paths(out_dir)

    array_tmp = paths["array"].with_name(ARRAY_FILE + ".tmp")
    # Written through a handle: `np.save` appends `.npy` to any path that does not
    # already end in it, which would defeat the rename.
    with array_tmp.open("wb") as handle:
        np.save(handle, X_cleaned)

    aux_tmp = paths["aux"].with_name(AUX_FILE + ".tmp")
    with aux_tmp.open("wb") as handle:
        pickle.dump(
            {"ord_encoder": ord_encoder, "feature_schema": feature_schema},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return [(array_tmp, paths["array"]), (aux_tmp, paths["aux"])]


def commit(pending: list[tuple[Path, Path]]) -> None:
    """Move the temporary outputs onto their final names."""
    for temporary, final in pending:
        temporary.replace(final)


def discard(pending: list[tuple[Path, Path]]) -> None:
    """Delete any temporary output still around, leaving the baseline untouched."""
    for temporary, _ in pending:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_array(
    baseline_path: Path,
    X_cleaned: np.ndarray,
    chunk_rows: int | None,
) -> list[str]:
    """Check the cleaned array against the recorded one, in row chunks.

    Memory-mapped and chunked so the comparison itself does not double the
    footprint of a 10 GB array. NaN is a legitimate output value here -- it is what
    a missing cell becomes -- so NaNs compare equal.
    """
    if chunk_rows is None:
        chunk_rows = max(1, COMPARISON_CHUNK_CELLS // X_cleaned.shape[1])
    baseline = np.load(baseline_path, mmap_mode="r")
    if baseline.shape != X_cleaned.shape:
        return [f"cleaned array shape {X_cleaned.shape} != {baseline.shape}"]

    problems = []
    if baseline.dtype != X_cleaned.dtype:
        problems.append(f"cleaned array dtype {X_cleaned.dtype} != {baseline.dtype}")

    for start in range(0, X_cleaned.shape[0], chunk_rows):
        stop = min(start + chunk_rows, X_cleaned.shape[0])
        recorded = np.asarray(baseline[start:stop])
        current = X_cleaned[start:stop]
        if np.array_equal(recorded, current, equal_nan=True):
            continue
        differing = ~((recorded == current) | (np.isnan(recorded) & np.isnan(current)))
        rows, cols = np.nonzero(differing)
        row, col = int(rows[0]), int(cols[0])
        problems.append(
            f"cleaned array differs at [{start + row}, {col}]: "
            f"{current[row, col]!r} != recorded {recorded[row, col]!r} "
            f"({int(differing.sum())} cells differ in rows [{start}, {stop}))"
        )
        break

    return problems


def encoder_signature(
    ord_encoder: OrderPreservingColumnTransformer,
) -> dict[str, Any]:
    """Comparable summary of a fitted encoder.

    Sklearn estimators define no `__eq__`, and pickle bytes are too brittle to
    diff. What determines the encoder's behaviour is which columns each transformer
    took and the categories it learned for them.
    """
    encoder = ord_encoder.named_transformers_.get("encoder")
    return {
        "transformers": [
            [name, type(transformer).__name__, list(columns)]
            for name, transformer, columns in ord_encoder.transformers_
        ],
        "categories": [
            np.asarray(categories).tolist()
            for categories in getattr(encoder, "categories_", [])
        ],
    }


def compare_aux(
    baseline_path: Path,
    ord_encoder: OrderPreservingColumnTransformer,
    feature_schema: FeatureSchema,
) -> list[str]:
    """Check the encoder and feature schema against the recorded ones."""
    with baseline_path.open("rb") as handle:
        # Trusted input: written by a previous run of this script.
        recorded = pickle.load(handle)  # noqa: S301

    problems = []
    if recorded["feature_schema"] != feature_schema:
        problems.append("feature schema differs from the recorded one")
    if encoder_signature(recorded["ord_encoder"]) != encoder_signature(ord_encoder):
        problems.append("ordinal encoder differs from the recorded one")
    return problems


def regression(
    label: str,
    new: float,
    baseline: float,
    tolerance: float,
    unit: str,
) -> str | None:
    """A message if `new` is worse than `baseline`, else None."""
    if new <= baseline * (1.0 + tolerance):
        return None
    allowed = f", tolerance {tolerance:.1%}" if tolerance else ""
    return (
        f"{label} regressed: {new:.3f} {unit} vs {baseline:.3f} {unit} recorded "
        f"({change_pct(new, baseline)}{allowed})"
    )


def change_pct(new: float, baseline: float) -> str:
    """`new` against `baseline` as a signed percentage."""
    if not baseline:
        return "n/a"
    return f"{(new / baseline - 1.0) * 100:+.1f}%"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def meminfo_gb(field_name: str) -> float:
    """Read a /proc/meminfo field in GB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(f"{field_name}:"):
            return int(line.split()[1]) * 1024 / _GB
    return float("nan")


def describe_environment() -> dict[str, Any]:
    """Host and library versions. A baseline only compares within one of these."""
    return {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "host_memory_available_gb": round(meminfo_gb("MemAvailable"), 1),
    }


def environment_drift(recorded: dict[str, Any]) -> list[str]:
    """Fields whose drift makes the recorded numbers incomparable."""
    current = describe_environment()
    interesting = (
        "hostname",
        "python_version",
        "numpy_version",
        "pandas_version",
        "torch_version",
        "torch_num_threads",
    )
    return [
        f"{field}: {recorded.get(field)!r} when recorded, {current[field]!r} now"
        for field in interesting
        if recorded.get(field) != current[field]
    ]


def fail(problems: list[str]) -> None:
    """Print the problems and exit non-zero."""
    # Otherwise the banner overtakes the buffered progress output it refers to.
    sys.stdout.flush()
    print("\n" + "=" * 79, file=sys.stderr)
    print("FAIL", file=sys.stderr)
    print("=" * 79, file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    sys.exit(1)


def print_pass_summary(
    recorded: dict[str, Any],
    memory: MemoryProfile,
    timing: dict[str, Any],
) -> None:
    """Report both metrics against the baseline that is about to be replaced."""
    old_rss = recorded["memory"]["transient_rss_bytes"]
    old_median = recorded["timing"]["median_s"]
    print("\n" + "=" * 79)
    print("PASS: not worse than the baseline on either metric, outputs identical.")
    print(
        f"  transient RSS:    {old_rss / _GB:.2f} GB -> "
        f"{memory.transient_bytes / _GB:.2f} GB "
        f"({change_pct(memory.transient_bytes, old_rss)})"
    )
    print(
        f"  median wall time: {old_median:.3f} s -> {timing['median_s']:.3f} s "
        f"({change_pct(timing['median_s'], old_median)})"
    )
    print("=" * 79)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_input_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Everything identifying the input, for the directory name and the guard."""
    profiler_rows = args.rows
    if profiler_rows is None:
        profiler_rows = SMALL_PROFILER_ROWS if args.small else DEFAULT_PROFILER_ROWS
    cols = args.cols
    if cols is None:
        cols = SMALL_COLS if args.small else DEFAULT_COLS
    return {
        "profiler_rows": profiler_rows,
        "rows": train_rows_for(profiler_rows),
        "cols": cols,
        "dtype": args.input_dtype,
        "seed": SEED,
        "passthrough_inf": PASSTHROUGH_INF,
    }


def load_baseline(paths: dict[str, Path], spec: dict[str, Any]) -> dict[str, Any]:
    """Read the recorded metrics, rejecting a baseline built from another input.

    The cheapest check there is, so an unusable baseline costs nothing to find.
    """
    recorded = json.loads(paths["metrics"].read_text())
    problems = []
    if recorded.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"baseline schema_version {recorded.get('schema_version')!r} != "
            f"{SCHEMA_VERSION}; delete the directory to re-record it"
        )
    recorded_input = recorded.get("input", {})
    problems += [
        f"baseline was recorded with {field}={recorded_input.get(field)!r}, "
        f"this run uses {value!r}"
        for field, value in spec.items()
        if recorded_input.get(field) != value
    ]
    if problems:
        fail(problems)
    return recorded


def check_memory(
    recorded: dict[str, Any],
    memory: MemoryProfile,
    tolerance: float,
) -> None:
    """Gate the RSS, before the timing repeats cost another N full calls."""
    problem = regression(
        "transient RSS",
        memory.transient_bytes / _GB,
        recorded["memory"]["transient_rss_bytes"] / _GB,
        tolerance,
        "GB",
    )
    if problem:
        fail([problem])


def main(args: argparse.Namespace) -> None:
    """Measure `clean_data`, then either record a baseline or gate against one."""
    spec = resolve_input_spec(args)
    out_dir = args.out_root / shape_dir_name(spec["rows"], spec["cols"], spec["dtype"])
    paths = baseline_paths(out_dir)
    # A baseline is only a baseline once all three files are there; a partial one
    # (an interrupted run, a hand-deleted array) is re-recorded rather than trusted.
    comparing = (
        all(path.exists() for path in paths.values()) and not args.overwrite_baseline
    )

    environment = describe_environment()
    print(f"Output directory: {out_dir}")
    print(f"Mode: {'compare against baseline' if comparing else 'record baseline'}")
    for key, value in environment.items():
        print(f"  {key}: {value}")

    recorded = None
    if comparing:
        recorded = load_baseline(paths, spec)
        drift = environment_drift(recorded.get("environment", {}))
        if drift:
            print("\nWARNING: the baseline was recorded elsewhere; RSS and timings")
            print("are only comparable within one environment:")
            for line in drift:
                print(f"  - {line}")

    X = generate_input(spec["profiler_rows"], spec["cols"], spec["dtype"])
    input_fingerprint = fingerprint(X)
    print(
        f"\nInput to clean_data: {X.shape} {X.dtype} ({X.nbytes / _GB:.2f} GB), "
        f"fingerprint {input_fingerprint[:16]}"
    )
    if recorded is not None and recorded["input"]["fingerprint"] != input_fingerprint:
        fail(
            [
                "the generated input differs from the one the baseline was recorded "
                f"on ({input_fingerprint[:16]} != "
                f"{recorded['input']['fingerprint'][:16]}); comparing would be "
                "meaningless"
            ]
        )

    feature_schema = build_feature_schema(X)
    output, memory = measure_memory(X, feature_schema, args.sample_interval_ms / 1000)
    X_cleaned, ord_encoder, out_schema = output
    print(
        f"\nTransient RSS: {memory.transient_bytes / _GB:.2f} GB "
        f"(entry {memory.rss_in_bytes / _GB:.2f} GB, peak "
        f"{memory.peak_rss_bytes / _GB:.2f} GB, retained "
        f"{memory.retained_bytes / _GB:.2f} GB, {memory.n_samples} RSS samples, "
        f"{memory.wall_s:.3f} s cold)"
    )

    if recorded is not None:
        check_memory(recorded, memory, args.tolerance)
        problems = compare_array(paths["array"], X_cleaned, args.chunk_rows)
        problems += compare_aux(paths["aux"], ord_encoder, out_schema)
        if problems:
            fail(problems)
        print("Outputs are identical to the recorded ones.")

    pending = write_outputs(out_dir, X_cleaned, ord_encoder, out_schema)
    # Drop the output before timing, so the repeats run at the footprint the
    # measured call ran at rather than carrying an extra copy of it.
    del output, X_cleaned, ord_encoder, out_schema
    gc.collect()

    try:
        measurement = measure_time(
            X, feature_schema, args.timing_repeats, args.warmup_calls
        )
        timing = timing_as_dict(measurement, args.warmup_calls)
        print(
            f"\nWall time: median {timing['median_s']:.3f} s over "
            f"{timing['repeats']} repeats after {args.warmup_calls} warmup "
            f"(min {timing['min_s']:.3f} s, max {timing['max_s']:.3f} s)"
        )

        if recorded is not None:
            problem = regression(
                "median wall time",
                timing["median_s"],
                recorded["timing"]["median_s"],
                args.tolerance,
                "s",
            )
            if problem:
                fail([problem])
            print_pass_summary(recorded, memory, timing)

        paths["metrics"].write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "input": {**spec, "fingerprint": input_fingerprint},
                    "environment": environment,
                    "config": {"sample_interval_ms": args.sample_interval_ms},
                    "memory": memory.as_dict(),
                    "timing": timing,
                },
                indent=2,
            )
            + "\n"
        )
        commit(pending)
    finally:
        # A no-op once committed; the point is to leave nothing behind on a failure.
        discard(pending)

    print(f"\nWrote {METRICS_FILE}, {ARRAY_FILE} and {AUX_FILE} to {out_dir}")


def get_parser() -> argparse.ArgumentParser:
    """Get the parser for bench_clean_data.py."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Rows the profiler generates, of which the train split (rows minus "
        f"round(rows/{BEYOND_ARENA_N_FOLDS})) reaches clean_data. "
        f"Default {DEFAULT_PROFILER_ROWS}, or {SMALL_PROFILER_ROWS} with --small.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help=f"Default {DEFAULT_COLS}, or {SMALL_COLS} with --small.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a shape that runs in seconds, for debugging this script. Its "
        "baseline lives in its own directory, so it never touches the full one.",
    )
    parser.add_argument(
        "--input-dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="dtype of the generated matrix. The profiled run used float32.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="Timed calls. The median is the gated number.",
    )
    parser.add_argument(
        "--warmup-calls",
        type=int,
        default=1,
        help="Untimed calls before the timed ones, to settle the allocator and the "
        "thread pools.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=20.0,
        help="RSS sampling period. Shorter catches narrower spikes.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Fractional slack on both metrics before a run counts as a regression, "
        "e.g. 0.03 to absorb 3%% of run-to-run noise.",
    )
    parser.add_argument(
        "--overwrite-baseline",
        action="store_true",
        help="Record a new baseline even if one exists, skipping every check.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="Rows per block when comparing the cleaned array with the recorded one. "
        f"Defaults to {COMPARISON_CHUNK_CELLS:,} cells' worth for the column count.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("bench_out/clean_data"),
        help="Baselines go in a subdirectory of this, named after the input shape.",
    )
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
