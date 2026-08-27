# ruff: noqa: T201, S603, S607
#  Copyright (c) Prior Labs GmbH 2026.

r"""Regression gate for `clean_data`'s transient RAM and wall time.

`clean_data` was measured holding 42.68 GB of transient RSS -- the single largest
contributor to the wrapper's host-RAM peak -- while fitting a 666,667 x 2,000
float32 table. That measurement came from the wrapper-wide `profiler.py` in
fomo-fitting's `experimental/arthur/res-2439-reduce-the-ram-footprint-of-the-wrapper/`;
this script isolates the same call so it can be iterated on without paying for a
whole fit + predict.

By default it measures exactly the array that profiler hands to `clean_data`: an
all-numeric float32 table, which is the case the RAM spike was found in. `--mix`
swaps in one of two tables that reach `clean_data` as an object array, as mixed data
really does, so a change can be told apart from a float-only one:

* `half-string` alternates numeric and low-cardinality string columns, so the
  ordinal encoder is actually exercised rather than selecting nothing.
* `numeric-object` alternates float and integer columns. Nothing is encoded, but
  every column still has to be recast, which leaves the frame split into one block
  per column -- the layout pandas has to materialise before it can hand the values
  back, and so the one where a copy of that is worth avoiding.

Each mix keeps its own baseline, and `--mix all` runs all three of them.

What it measures:

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

Smoke test locally (seconds, ~20 MB on disk; `--mix` combines with it):

    uv run scripts/bench_clean_data.py --small

`clean_data` is pure CPU work -- pandas and numpy, no device involved -- so the
partition to use is a high-memory CPU one. `cpuhighmem16spot` has 300 nodes at ~123 GB,
so it is both big enough for the full shape and quick to get hold of:

    srun -p cpuhighmem16spot --mem=0 --time=01:00:00 \
        uv run scripts/bench_clean_data.py

Either object-array mix, on the same shape (see `--mix`):

    srun -p cpuhighmem16spot --mem=0 --time=01:00:00 \
        uv run scripts/bench_clean_data.py --mix half-string

Or every mix in one command -- a child process each, tabled together at the end, so a
change is answered for on all three tables rather than the one it was written against:

    srun -p cpuhighmem16spot --mem=0 --time=02:00:00 \
        uv run scripts/bench_clean_data.py --mix all --reference main

Every mix in such a run is measured whether the ones before it passed or not, and one
that trips a gate reports nothing for the checks after it -- so `--tolerance 0.05` is
worth passing when what is wanted is the whole table rather than the gate.

`--mix all` covers the column axis. The other axis worth covering before believing a
change is the supported dependency range, since what this costs -- and occasionally
what it returns -- depends on what numpy and pandas do underneath. `--environment`
takes that one: `lowest` and `highest` are its two ends, built here on first use the
way CI builds its own two legs, and `current` is this interpreter, named so it can sit
in the grid beside them.

    srun -p cpuhighmem16spot --mem=0 --time=04:00:00 \
        uv run scripts/bench_clean_data.py --mix all --reference main \
            --environment lowest --environment highest

The first use of `lowest` or `highest` resolves and downloads a virtualenv under
`--environment-root`, which takes a couple of minutes and a few GB of disk (torch
brings its CUDA wheels either way); later runs reuse it, and `--refresh-environments`
rebuilds it. `LABEL=PYTHON` measures under an interpreter of your own instead. Each
environment but `current` keeps its baselines in a subdirectory of `--out-root` of its
own, because one recorded against a different dependency set is not comparable with it.

The full numeric shape needs ~20 GB of RAM and writes ~11 GB per run. Pair either with
`scripts/srun_retry.py` when allocations are getting stuck CONFIGURING.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import gc
import hashlib
import json
import os
import pickle
import platform
import re
import resource
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.benchmark import Measurement, Timer

import tabpfn
from tabpfn.preprocessing.clean import clean_data
from tabpfn.preprocessing.modality_detection import detect_feature_modalities

if TYPE_CHECKING:
    from collections.abc import Callable

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

# How the columns are made up. The all-numeric mix is the profiled one; the other two
# exist to answer whether a change helps only float tables, and they split the two
# ways a table can be non-trivial: columns the encoder takes, and columns it does not
# but that still have to be converted.
MIX_NUMERIC = "numeric"
MIX_HALF_STRING = "half-string"
MIX_NUMERIC_OBJECT = "numeric-object"
MIXES = (MIX_NUMERIC, MIX_HALF_STRING, MIX_NUMERIC_OBJECT)

# Not a mix but a request for every one of them, a child process each; see
# `run_every_mix`.
MIX_ALL = "all"

# Distinct values per string column. Below MAX_UNIQUE_FOR_CATEGORICAL_FEATURES so
# the columns detect as CATEGORICAL rather than TEXT, and deliberately not
# numeric-looking: `_is_numeric_pandas_series` coerces columns whose strings parse
# as numbers, which would send them down the numerical branch instead.
STRING_LEVELS = 20

# Distinct values per integer column, well above MAX_UNIQUE_FOR_CATEGORICAL_FEATURES
# so those columns detect as NUMERICAL. A narrower range would have them picked up as
# categorical, which would send that mix down the encoder path and leave nothing
# measuring the conversion-only one.
INTEGER_LEVELS = 10_000

# Distinct values per generated non-float column, for the mixes that have any. Part
# of the recorded input description, since it decides which modality the columns
# detect as and so which path the mix measures.
MIX_LEVELS = {MIX_HALF_STRING: STRING_LEVELS, MIX_NUMERIC_OBJECT: INTEGER_LEVELS}

# Modality-detection thresholds from the profiler's 3.1_exp `InferenceConfig`.
# Standard-normal columns land on NUMERICAL for every one of these, but the
# detection has to run anyway: its `FeatureSchema` is what `clean_data` consumes.
MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE = 100
MAX_UNIQUE_FOR_CATEGORICAL_FEATURES = 30
MIN_UNIQUE_FOR_NUMERICAL_FEATURES = 4
# Above this many distinct values a string column detects as TEXT rather than
# CATEGORICAL. STRING_LEVELS sits under it, so the half-string mix stays on the
# encoder path it exists to exercise.
MIN_CARDINALITY_FOR_TEXT = 30

# `InferenceConfig.PASSTHROUGH_INF` defaults to False and the profiler leaves it
# alone, so the +/-inf masking path is not exercised.
PASSTHROUGH_INF = False

# Bumped whenever the meaning of a recorded metric, or the shape of the recorded
# input description, changes -- so an old baseline is rejected rather than silently
# compared against. 2: `input` gained `mix`. 3: its `string_levels` became `levels`,
# which the numeric-object mix has one of too.
SCHEMA_VERSION = 3

METRICS_FILE = "metrics.json"
ARRAY_FILE = "X_cleaned.npy"
AUX_FILE = "aux.pkl"

# Head and tail elements hashed into the input fingerprint. Enough to catch a
# changed generator or dtype without hashing gigabytes.
FINGERPRINT_ELEMENTS = 1_000_000
FINGERPRINT_OBJECT_ELEMENTS = 100_000

# Below this many RSS samples inside the measured call, the "peak" is essentially
# just the entry/exit readings: any spike between them was never looked for. Reported
# rather than silently passed off as a measured peak.
MIN_USEFUL_RSS_SAMPLES = 5

# Below this median, a call is too short for the timing comparison to mean anything:
# process-to-process quantisation is then a large fraction of the number itself. Same
# reasoning as MIN_USEFUL_RSS_SAMPLES -- say so rather than report a percentage that
# reads like a finding.
MIN_USEFUL_MEDIAN_S = 0.05

# Cells per block when comparing against a recorded array. Sized so the boolean
# temporaries the comparison builds stay in the tens of MB whatever the shape.
COMPARISON_CHUNK_CELLS = 32_000_000

REPO_ROOT = Path(__file__).resolve().parent.parent

_PAGE_SIZE = resource.getpagesize()
_GB = 1e9


# ---------------------------------------------------------------------------
# RSS sampling
# ---------------------------------------------------------------------------


def _rss_from_statm() -> int:
    """Resident set size of this process, read from /proc/self/statm."""
    with Path("/proc/self/statm").open() as handle:
        return int(handle.read().split()[1]) * _PAGE_SIZE


def _pick_rss_reader() -> Callable[[], int]:
    """How to read RSS here.

    `/proc` on Linux, which is where the numbers that matter are measured and which
    costs nothing to read. Off it -- a smoke test on a laptop -- fall back to psutil.
    That is not a dependency of this package, only of the dev environment, so it is
    imported lazily and only when there is no `/proc` to read instead.
    """
    if Path("/proc/self/statm").exists():
        return _rss_from_statm
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            f"no /proc/self/statm on {platform.system()} and psutil is not installed, "
            "so RSS cannot be sampled here"
        ) from None
    process = psutil.Process()
    return lambda: process.memory_info().rss


current_rss_bytes = _pick_rss_reader()


def release_free_memory() -> None:
    """Hand the allocator's free arenas back to the OS, so RSS reads as what is live.

    `gc.collect()` frees the objects; whether glibc then returns their pages is its own
    decision, and not a repeatable one. Measured on the `half-string` cell, one process
    in six entered the call ~0.6 GB above the others and so recorded a transient 5%
    lower -- landing on whichever side of a comparison happened to draw it, which reads
    as a 5% regression or improvement that no code change produced. Trimming first makes
    the entry reading the live floor in every process: the same six runs then agreed to
    0.05%.

    A no-op wherever `malloc_trim` is not there to call, which costs only the noise it
    would have removed.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        return


def pin_allocator_thresholds() -> bool:
    """Stop glibc deciding per run which allocations come back to the OS.

    Its mmap threshold is not a constant: freeing an mmapped chunk raises it to that
    chunk's size, up to 32 MB, after which same-sized allocations are served from the
    heap instead -- and the heap only shrinks from the top, so they stay resident. The
    ordinal encoder allocates one such buffer per column, so whether the threshold was
    raised before it ran decides ~0.5 GB of the peak.

    Which is not decided by the code under test. On `lowest`/`half-string` the cell came
    out as two clean states, 2.81 GB and 3.27 GB, and a gate comparing one against the
    other reported ±16%; the branch that drew the raised threshold more often was the
    one being measured, so the gate failed on an allocation order rather than on any
    memory it held. Pinned, six runs across both sides of that comparison all measured
    2.803 GB.

    Returns whether it took, which the environment block records: a run without it is
    not comparable to one with it.
    """
    # -3 is `M_MMAP_THRESHOLD`, and setting it is also what tells glibc to stop
    # adjusting it. 128 KB is its own documented default, so this pins rather than
    # retunes.
    try:
        return ctypes.CDLL("libc.so.6").mallopt(-3, 128 * 1024) == 1
    except (OSError, AttributeError):
        return False


# At import, which is before either harness has built anything: the threshold has to be
# pinned before the first large free moves it, and both scripts come through here.
_ALLOCATOR_PINNED = pin_allocator_thresholds()


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


def generate_input(spec: dict[str, Any]) -> np.ndarray:
    """The matrix `clean_data` is handed, for whichever column mix was asked for."""
    generator = {
        MIX_HALF_STRING: generate_half_string_input,
        MIX_NUMERIC_OBJECT: generate_numeric_object_input,
    }.get(spec["mix"], generate_numeric_input)
    return generator(spec["profiler_rows"], spec["cols"], spec["dtype"])


def generate_numeric_input(profiler_rows: int, cols: int, dtype: str) -> np.ndarray:
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


def generate_half_string_input(profiler_rows: int, cols: int, dtype: str) -> np.ndarray:
    """Half numeric columns, half low-cardinality string columns, as one object array.

    Object dtype is not a simplification -- it is how mixed data actually arrives.
    `ensure_compatible_fit_inputs` runs sklearn's `check_array(dtype=None)`, which
    collapses a frame of mixed dtypes into a single object array, so that is what
    `clean_data` sees.

    The two kinds of column alternate rather than sitting in two contiguous halves,
    so the numeric ones cannot be handled as one block by accident. With an odd
    `cols` the numeric half takes the extra column.
    """
    rows = train_rows_for(profiler_rows)
    numeric_cols = np.arange(0, cols, 2)
    string_cols = np.arange(1, cols, 2)

    rng = np.random.default_rng(SEED)
    X = np.empty((rows, cols), dtype=object)
    X[:, numeric_cols] = rng.standard_normal(
        (rows, len(numeric_cols)), dtype=np.float32
    ).astype(np.dtype(dtype), copy=False)
    levels = np.array([f"lvl_{i:02d}" for i in range(STRING_LEVELS)], dtype=object)
    X[:, string_cols] = levels[rng.integers(0, STRING_LEVELS, (rows, len(string_cols)))]
    return X


def generate_numeric_object_input(
    profiler_rows: int, cols: int, dtype: str
) -> np.ndarray:
    """Alternating float and integer columns, all numeric, as one object array.

    A frame of *mixed numeric* dtypes arrives this way for the same reason a
    half-string one does: `check_array(dtype=None)` collapses anything that is not of
    one dtype into an object array. Nothing here is categorical, so the ordinal
    encoder selects no columns and the encoding step is skipped altogether -- but
    every column still comes back from `convert_dtypes` as a nullable extension dtype
    and has to be recast, which leaves the frame holding one block per column.

    That layout is the point of this mix, and neither of the others reaches it: a
    float table is wrapped without a recast and stays one block, and a half-string
    one has columns the encoder takes, so it never gets as far as the block-layout
    question.

    The integers span `INTEGER_LEVELS` values so they detect as NUMERICAL rather than
    categorical. Alternating rather than in halves, as above, so the float columns
    cannot be handled as one block by accident.
    """
    rows = train_rows_for(profiler_rows)
    float_cols = np.arange(0, cols, 2)
    integer_cols = np.arange(1, cols, 2)

    rng = np.random.default_rng(SEED)
    X = np.empty((rows, cols), dtype=object)
    X[:, float_cols] = rng.standard_normal(
        (rows, len(float_cols)), dtype=np.float32
    ).astype(np.dtype(dtype), copy=False)
    X[:, integer_cols] = rng.integers(0, INTEGER_LEVELS, (rows, len(integer_cols)))
    return X


def fingerprint(X: np.ndarray) -> str:
    """Digest of the input's shape, dtype and edge values.

    Guards the comparison: a baseline is only meaningful against the same input,
    and the generator could drift (a numpy RNG change, a different seed).
    """
    digest = hashlib.sha256()
    digest.update(f"{X.shape}|{X.dtype}|{SEED}".encode())
    flat = X.ravel()
    if X.dtype == object:
        # `tobytes` on an object array digests pointer addresses, which differ on
        # every run and would fail the guard against the array they identify. The
        # elements' reprs are stable, and fewer of them are needed since building
        # each one costs a Python call.
        for edge in (
            flat[:FINGERPRINT_OBJECT_ELEMENTS],
            flat[-FINGERPRINT_OBJECT_ELEMENTS:],
        ):
            digest.update("|".join(map(repr, edge.tolist())).encode())
    else:
        digest.update(flat[:FINGERPRINT_ELEMENTS].tobytes())
        digest.update(flat[-FINGERPRINT_ELEMENTS:].tobytes())
    return digest.hexdigest()


def describe_input_size(X: np.ndarray) -> str:
    """`X`'s footprint, flagging what an object array's `nbytes` leaves out."""
    if X.dtype == object:
        return (
            f"{X.nbytes / _GB:.2f} GB of pointers, plus the Python objects they "
            "point at, which nbytes does not count"
        )
    return f"{X.nbytes / _GB:.2f} GB"


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
        min_cardinality_for_text=MIN_CARDINALITY_FOR_TEXT,
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
    release_free_memory()
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
    """JSON-serialisable timing stats. `median_s` is the gated number.

    The dispersion fields are recorded so a gate result can be read against the
    noise it sits in: a regression inside `spread_pct` of the run that recorded it
    says nothing. Within one process the repeats are typically tight (well under
    1%); the drift that matters is between runs, which no number of repeats removes
    -- that is what `--tolerance` is for.
    """
    times = sorted(measurement.times)
    median = measurement.median
    return {
        "median_s": median,
        "mean_s": measurement.mean,
        "min_s": times[0],
        "max_s": times[-1],
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "spread_pct": (times[-1] - times[0]) / median * 100 if median else 0.0,
        "repeats": len(times),
        "warmup_calls": warmup_calls,
        "raw_times_s": list(measurement.times),
        "num_threads": measurement.task_spec.num_threads,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def shape_dir_name(train_rows: int, cols: int, dtype: str, mix: str) -> str:
    """Directory name identifying one input, so each mix keeps its own baseline."""
    return f"rows{train_rows}_cols{cols}_{dtype}_{mix}"


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
# A/B against a git reference
# ---------------------------------------------------------------------------
#
# The baseline-on-disk gate compares across runs, which drags in every difference
# between two processes -- and, on a cluster, between two nodes. `--reference`
# removes that: it measures a reference commit and the working tree back to back on
# one machine, minutes apart, with the same harness.
#
# The reference package is selected for a whole child process, by putting the
# worktree's `src` on its `PYTHONPATH`. Two alternatives do not work:
#
# * Importing the reference under a second module name (`tabpfn_ref.clean`): the
#   codebase is full of absolute `from tabpfn...` imports, so the reference module
#   would pull its helpers from the *installed* package and silently run a mixture of
#   the two versions.
# * `uv run --project <worktree>`: hermetic, but the worktree has no `uv.lock` (it is
#   gitignored) and `exclude-newer` is a *relative* date, so it resolves its own
#   dependency set -- measured here as numpy 2.5.1 / pandas 3.0.5 / torch 2.13.0
#   against the working tree's 2.4.6 / 3.0.3 / 2.12.0. Since what is being measured
#   is how much copying pandas does, comparing across pandas versions answers a
#   different question than the one asked.
#
# `PYTHONPATH` wins over the editable install because its entries precede the ones
# site-processing appends for a `.pth` file. That is an assumption about the
# environment rather than a guarantee, so it is not trusted: each side records the
# `tabpfn` it actually imported and the comparison verifies both before believing any
# number, along with the library versions the two sides ran against.


def _git(*arguments: str) -> str:
    """Run git in this repository and return its stdout, stripped."""
    return subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def resolve_reference(revision: str) -> str:
    """The full commit SHA a revision names, so the worktree can be cached by it."""
    try:
        return _git("rev-parse", f"{revision}^{{commit}}")
    except subprocess.CalledProcessError:
        fail([f"{revision!r} is not a commit this repository knows"])
        raise  # unreachable; keeps the return type honest


def prepare_reference_worktree(sha: str, root: Path, *, refresh: bool) -> Path:
    """A worktree checked out at `sha`, reused across runs unless `refresh`.

    Reused because the first use of one pays for a `uv sync` of its environment;
    keyed by SHA so a different reference can never be served a stale checkout.
    """
    worktree = root / sha[:12]
    if refresh and worktree.exists():
        print(f"Removing the cached reference worktree at {worktree}")
        _git("worktree", "remove", "--force", str(worktree))
    if worktree.exists():
        # Trust it only if it really is at the requested commit.
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=worktree,
        ).stdout.strip()
        if head == sha:
            print(f"Reusing the reference worktree at {worktree}")
            return worktree
        print(f"Cached worktree at {worktree} is at {head[:12]}, not {sha[:12]}")
        _git("worktree", "remove", "--force", str(worktree))

    root.mkdir(parents=True, exist_ok=True)
    print(f"Creating a reference worktree for {sha[:12]} at {worktree}")
    _git("worktree", "add", "--detach", str(worktree), sha)
    return worktree


def child_arguments(args: argparse.Namespace, spec: dict[str, Any]) -> list[str]:
    """The measurement arguments both sides of the comparison are run with.

    Built from the resolved spec rather than forwarded from argv, so the `--small`
    and default shapes are already applied and the two children cannot disagree
    about what they measured.
    """
    forwarded = [
        "--rows",
        str(spec["profiler_rows"]),
        "--cols",
        str(spec["cols"]),
        "--input-dtype",
        spec["dtype"],
        "--mix",
        spec["mix"],
        "--timing-repeats",
        str(args.timing_repeats),
        "--warmup-calls",
        str(args.warmup_calls),
        "--sample-interval-ms",
        str(args.sample_interval_ms),
    ]
    if args.chunk_rows is not None:
        forwarded += ["--chunk-rows", str(args.chunk_rows)]
    return forwarded


def run_child(command: list[str], label: str, env: dict[str, str] | None = None) -> int:
    """Run one side of the comparison, letting it print straight through."""
    print("\n" + "-" * 79)
    print(f"{label}: {shlex.join(command)}")
    if env:
        print(f"  PYTHONPATH={env['PYTHONPATH']}")
    print("-" * 79, flush=True)
    return subprocess.run(
        command, check=False, env={**os.environ, **(env or {})}
    ).returncode


def read_recorded_environment(run_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """The `environment` block a side of the comparison just recorded."""
    metrics_path = (
        run_dir
        / shape_dir_name(spec["rows"], spec["cols"], spec["dtype"], spec["mix"])
        / METRICS_FILE
    )
    return json.loads(metrics_path.read_text())["environment"]


def verify_reference_ran_the_reference(
    reference_env: dict[str, Any],
    worktree: Path,
) -> None:
    """Check the reference side measured the worktree's package, in this environment.

    The whole comparison rests on this. If the child resolved the working tree's
    package instead -- a `PYTHONPATH` that did not take, a stale worktree -- it would
    quietly measure the same code twice and report no difference, which looks exactly
    like a change that does nothing. And if it resolved different library versions,
    any difference it *does* report might be pandas', not ours.

    Checked here, right after the reference side, so a broken comparison costs one
    run rather than two.
    """
    reference_path = Path(reference_env["tabpfn_path"])
    current_path = Path(describe_environment()["tabpfn_path"])
    print(f"\nreference package: {reference_path}")
    print(f"working tree:      {current_path}")

    problems = []
    if worktree.resolve() not in reference_path.parents:
        problems.append(
            f"the reference run imported {reference_path}, which is not inside the "
            f"reference worktree {worktree}, so it measured the wrong code"
        )
    if reference_path == current_path:
        problems.append(
            "both sides would import the same tabpfn, so this compares a commit "
            "against itself"
        )
    drift = environment_drift(reference_env)
    if drift:
        problems += [
            "the reference side did not run against this environment, so its numbers "
            "are not comparable:",
            *drift,
        ]
    if problems:
        fail(problems)


def run_reference_comparison(args: argparse.Namespace) -> int:
    """Measure a reference commit, then the working tree, and gate on the pair."""
    spec = resolve_input_spec(args)
    sha = resolve_reference(args.reference)
    worktree = prepare_reference_worktree(
        sha, args.reference_root, refresh=args.refresh_reference
    )
    subject = _git("log", "-1", "--format=%h %s", sha)
    print(f"Reference: {subject}")
    if not _git("status", "--porcelain", "--", "src/tabpfn") and sha == _git(
        "rev-parse", "HEAD"
    ):
        print(
            "\nWARNING: src/tabpfn has no uncommitted changes and the reference is "
            "HEAD,\nso this compares a commit against itself."
        )

    # Named for the mix as well as the commit, because `--mix all` runs three of
    # these in turn and each clears its run directory on the way in: one shared
    # directory would have every mix but the last lose the outputs
    # `--keep-reference-run` asked to keep.
    run_dir = args.out_root / "_reference_runs" / sha[:12] / spec["mix"]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    forwarded = child_arguments(args, spec)
    script = str(Path(__file__).resolve())

    try:
        # The reference side records the baseline. Same interpreter, same libraries,
        # same harness -- only `import tabpfn` differs.
        code = run_child(
            [
                sys.executable,
                script,
                *forwarded,
                "--out-root",
                str(run_dir),
                "--overwrite-baseline",
                "--record-as-reference",
            ],
            f"reference {sha[:12]}",
            env={"PYTHONPATH": str(worktree / "src")},
        )
        if code != 0:
            fail([f"the reference run failed with exit code {code}"])
        verify_reference_ran_the_reference(
            read_recorded_environment(run_dir, spec), worktree
        )

        # The working tree's side gates against it. `--strict-env` is belt and braces:
        # the environment was already checked above, and this catches anything that
        # changed between the two runs.
        return run_child(
            [
                sys.executable,
                script,
                *forwarded,
                "--out-root",
                str(run_dir),
                "--tolerance",
                str(args.tolerance),
                "--strict-env",
            ],
            "working tree",
        )
    finally:
        if args.keep_reference_run:
            print(f"\nKept the comparison's outputs in {run_dir}")
        elif run_dir.exists():
            shutil.rmtree(run_dir)


# ---------------------------------------------------------------------------
# The grid: every mix, under every interpreter
# ---------------------------------------------------------------------------
#
# A change wants answering for on more than the one table it was written against: on
# every column mix, since a float table exercises different code than one with
# categorical columns, and at both ends of the supported dependency range, since what
# this costs -- and occasionally what it returns -- depends on what numpy and pandas do
# underneath. `--mix all` takes the first axis, `--environment` the second, and
# together they are a grid of one child process per cell, tabled at the end.
#
# A child process per cell rather than one loop, because the gated metric is a
# *process* peak: a mix measured after another one starts on a heap the previous one
# already grew and fragmented, and would have its transient read against that. A
# second interpreter is a child process by definition. Under `--reference` each cell's
# child spawns the usual two grandchildren, so comparing every mix against a commit is
# six measured processes.
#
# The children are handed this run's own argv with `--mix` substituted, rather than a
# spec resolved here, so a cell is measured under exactly the flags this run was given.
# (`--reference`'s two children are the opposite case -- there the spec is resolved up
# front precisely so the two sides cannot disagree about what they measured.)
#
# The table is read back off what the children printed, which makes their report lines
# an interface. Nothing else in either script prints a metric in these shapes.
_COMPARED_METRIC = re.compile(
    r"^ +(transient RSS|median wall time): +([\d.]+) (?:GB|s) -> ([\d.]+) (?:GB|s)",
    re.MULTILINE,
)
_MEASURED_RSS = re.compile(r"^Transient RSS: ([\d.]+) GB", re.MULTILINE)
_MEASURED_MEDIAN = re.compile(r"^Wall time: median ([\d.]+) s", re.MULTILINE)

# Printed once every recorded output has been matched. Absent from a run that recorded
# a baseline rather than gating against one: there was nothing to match it against.
_OUTPUTS_MATCHED = re.compile(r"identical to the recorded ones")

# What each interpreter is asked for before the grid starts. Not an f-string: the
# braces are the probe's own, evaluated by the interpreter being probed.
_VERSION_PROBE = (
    "import sys, numpy, pandas, sklearn, torch; "
    "print(f'python {sys.version.split()[0]} numpy {numpy.__version__} "
    "pandas {pandas.__version__} sklearn {sklearn.__version__} "
    "torch {torch.__version__}')"
)

METRIC_COLUMNS = ("mix", "transient RSS", "median wall time", "status")

# `--environment current`: this interpreter, named so it can be asked for alongside the
# built ones rather than being the thing you get by asking for nothing.
CURRENT_ENVIRONMENT = "current"

# The two ends of the dependency range the project supports, as (python version, uv
# resolution) -- the same two legs CI runs, in `.github/workflows/ci.yml`: 3.10 with
# every direct dependency at its floor, and 3.14 with all of them at their newest.
# Asking for one of these builds it, so the range can be covered from one command
# without hand-building a virtualenv first.
BUILT_ENVIRONMENTS = {
    "lowest": ("3.10", "lowest-direct"),
    "highest": ("3.14", "highest"),
}


@dataclasses.dataclass(frozen=True)
class Environment:
    """One interpreter the grid is measured under."""

    # How it is named, in the report and in its cells' banners. None for the
    # interpreter this runs under when nothing named it: there is then nothing for it
    # to be told apart from.
    name: str | None
    interpreter: str
    # Subdirectory of `--out-root` its baselines go in, None to write into `--out-root`
    # itself. `current` writes there, where a run that passed no `--environment` at all
    # has always written, so the two gate against the same recorded numbers.
    baseline_dir: str | None
    # (python version, uv resolution) when this script builds the environment itself,
    # None when the interpreter is expected to be there already.
    recipe: tuple[str, str] | None = None

    def describe(self) -> str:
        """How this environment is spoken about in the report."""
        return self.name or "this interpreter"

    def qualify(self, mix: str) -> str:
        """The name of one of its cells: the mix, qualified by this environment."""
        return mix if self.name is None else f"{self.name}/{mix}"

    def venv(self) -> Path:
        """The virtualenv its interpreter lives in, for the ones built here."""
        return Path(self.interpreter).parent.parent


@dataclasses.dataclass(frozen=True)
class CellOutcome:
    """One cell's child process, as that child's own output described it."""

    environment: Environment
    mix: str
    exit_code: int
    outputs_identical: bool
    # (baseline, this run), or (None, this run) where the child had no baseline to
    # compare against, or None where it never got as far as printing the metric.
    rss_gb: tuple[float | None, float] | None
    median_s: tuple[float | None, float] | None

    @classmethod
    def from_output(
        cls,
        environment: Environment,
        mix: str,
        exit_code: int,
        output: str,
    ) -> CellOutcome:
        """Read one child's two gated metrics out of everything it printed."""
        compared = {
            match.group(1): (float(match.group(2)), float(match.group(3)))
            for match in _COMPARED_METRIC.finditer(output)
        }
        return cls(
            environment=environment,
            mix=mix,
            exit_code=exit_code,
            outputs_identical=bool(_OUTPUTS_MATCHED.search(output)),
            rss_gb=compared.get("transient RSS")
            or last_measured(_MEASURED_RSS, output),
            median_s=compared.get("median wall time")
            or last_measured(_MEASURED_MEDIAN, output),
        )

    @property
    def ok(self) -> bool:
        """Whether the child exited cleanly, i.e. nothing it checked regressed."""
        return self.exit_code == 0

    def status(self) -> str:
        """This cell's verdict, for the table's last column."""
        if not self.ok:
            return f"exit {self.exit_code}"
        return "ok" if self.outputs_identical else "recorded"

    def name(self) -> str:
        """How this cell is named in its banner and in the failure list."""
        return self.environment.qualify(self.mix)


def parse_environments(entries: list[str] | None, root: Path) -> list[Environment]:
    """The interpreters to measure under: this one, unless others were named.

    A label becomes a directory name, since every environment but `current` keeps its
    own baselines, so it is checked for being usable as one -- and for being unique --
    before an hour of measuring is spent writing into the wrong place.
    """
    if not entries:
        return [Environment(name=None, interpreter=sys.executable, baseline_dir=None)]

    environments: list[Environment] = []
    problems = []
    for entry in entries:
        name, separator, interpreter = entry.partition("=")
        problem = environment_problem(name, interpreter, named=bool(separator))
        if problem:
            problems.append(f"--environment {entry!r}: {problem}")
        elif any(name == existing.name for existing in environments):
            problems.append(
                f"--environment {name!r} was given twice, and the two of them would "
                "share a baseline directory"
            )
        else:
            environments.append(build_or_find(name, interpreter, root))
    if problems:
        fail(problems)
    return environments


def environment_problem(name: str, interpreter: str, *, named: bool) -> str | None:
    """Why one `--environment` cannot be honoured, or None if it can."""
    keywords = ", ".join([CURRENT_ENVIRONMENT, *BUILT_ENVIRONMENTS])
    if not named:
        if name in {CURRENT_ENVIRONMENT, *BUILT_ENVIRONMENTS}:
            return None
        return (
            f"{name!r} is not one of the environments this builds ({keywords}); give "
            "an interpreter of your own as LABEL=PYTHON"
        )
    if name in {CURRENT_ENVIRONMENT, *BUILT_ENVIRONMENTS}:
        return (
            f"{name!r} names an environment this script provides, so it takes no "
            "interpreter; label yours something else"
        )
    if not name or "/" in name or name in {".", ".."}:
        return (
            "does not begin with a label usable as a directory name, as in "
            "'mine=../.venv_mine/bin/python'"
        )
    if not interpreter:
        return "names no interpreter; write it as LABEL=PYTHON"
    return None


def build_or_find(name: str, interpreter: str, root: Path) -> Environment:
    """One `--environment`, resolved into where its interpreter is or will be."""
    if name == CURRENT_ENVIRONMENT:
        return Environment(name=name, interpreter=sys.executable, baseline_dir=None)
    if name in BUILT_ENVIRONMENTS:
        return Environment(
            name=name,
            interpreter=str(root / name / "bin" / "python"),
            baseline_dir=name,
            recipe=BUILT_ENVIRONMENTS[name],
        )
    return Environment(name=name, interpreter=interpreter, baseline_dir=name)


def provision_environments(environments: list[Environment], *, refresh: bool) -> None:
    """Build the virtualenvs for the environments this script owns.

    `uv venv` and `uv pip install` rather than the `uv sync` CI uses, because a sync at
    a non-default `--resolution` rewrites this checkout's `uv.lock` -- which would
    leave the next plain `uv run` here installing the floor of the dependency range
    too, silently, long after the benchmark was forgotten about.
    """
    for environment in environments:
        if environment.recipe is None:
            continue
        python_version, resolution = environment.recipe
        venv = environment.venv()
        if refresh and venv.exists():
            print(f"Removing the {environment.name} environment at {venv}")
            shutil.rmtree(venv)
        if Path(environment.interpreter).exists():
            print(f"Reusing the {environment.name} environment at {venv}")
            continue

        print(
            f"\nBuilding the {environment.name} environment at {venv}: python "
            f"{python_version}, --resolution {resolution}. Resolving and downloading "
            "it takes a couple of minutes and a few GB; later runs reuse it."
        )
        run_uv(["uv", "venv", str(venv), "--python", python_version])
        run_uv(
            [
                "uv",
                "pip",
                "install",
                "--python",
                environment.interpreter,
                "--resolution",
                resolution,
                "--group",
                "ci",
                "--editable",
                ".",
            ]
        )
        # `torch.utils.benchmark` imports `cpp_extension` -> `setuptools` on older
        # torch, and nothing in the dependency floor pulls it in. Deliberately not at
        # `resolution`: the floor of setuptools itself is not what is being measured.
        run_uv(
            ["uv", "pip", "install", "--python", environment.interpreter, "setuptools"]
        )


def run_uv(command: list[str]) -> None:
    """Run one uv command against this checkout, failing the run if it does not.

    Its output goes straight through: these are the slowest part of a grid, and a
    resolution that cannot be satisfied is worth reading in full.
    """
    print(f"  {shlex.join(command)}", flush=True)
    code = subprocess.run(command, check=False, cwd=REPO_ROOT).returncode
    if code:
        fail([f"{shlex.join(command)} failed with exit code {code}"])


def describe_interpreters(environments: list[Environment]) -> list[str]:
    """The library versions each interpreter resolves, for the report header.

    Probed before anything is measured, because an interpreter that cannot import what
    is measured here fails every cell it owns, and finding that out an hour in is worse
    than not having started. Skipped when `--environment` was not given: the versions
    are then the ones the single child prints for itself anyway.
    """
    lines = []
    problems = []
    for environment in environments:
        try:
            probe = subprocess.run(
                [environment.interpreter, "-c", _VERSION_PROBE],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            problems.append(f"{environment.interpreter} cannot be run: {error}")
            continue
        if probe.returncode == 0:
            lines.append(f"  {environment.describe()}: {probe.stdout.strip()}")
        else:
            detail = (probe.stderr.strip() or "it printed nothing").splitlines()[-1]
            problems.append(
                f"{environment.interpreter} cannot import what is measured here, so "
                f"every cell under it would fail: {detail}"
            )
    if problems:
        fail(problems)
    return lines


def last_measured(pattern: re.Pattern[str], output: str) -> tuple[None, float] | None:
    """The last absolute reading of a metric, as a pair with no baseline in it.

    The last rather than the first, because under `--reference` the reference side
    prints its readings before the working tree's, and it is the working tree's that is
    wanted when the comparison never got as far as a summary.
    """
    readings = pattern.findall(output)
    return (None, float(readings[-1])) if readings else None


def format_metric(
    pair: tuple[float | None, float] | None,
    unit: str,
    places: int,
) -> str:
    """One metric cell: `baseline -> this run (delta)`, or the reading on its own."""
    if pair is None:
        return "not reported"
    baseline, current = pair
    if baseline is None:
        return f"{current:.{places}f} {unit}"
    return (
        f"{baseline:.{places}f} -> {current:.{places}f} {unit} "
        f"({change_pct(current, baseline)})"
    )


def render_grid_table(outcomes: list[CellOutcome]) -> list[str]:
    """The grid as one markdown table, a cell per row.

    The environment column appears only when `--environment` named any, so a run over
    the mixes alone reads as the list of mixes it is.
    """
    header = METRIC_COLUMNS
    rows = [
        (
            outcome.mix,
            format_metric(outcome.rss_gb, "GB", 2),
            format_metric(outcome.median_s, "s", 3),
            outcome.status(),
        )
        for outcome in outcomes
    ]
    if any(outcome.environment.name is not None for outcome in outcomes):
        header = ("environment", *header)
        rows = [
            (outcome.environment.describe(), *row)
            for outcome, row in zip(outcomes, rows, strict=True)
        ]

    widths = [
        max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)
    ]

    def row(cells: tuple[str, ...]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return "| " + " | ".join(padded) + " |"

    return [
        row(header),
        row(tuple(":" + "-" * (width - 1) for width in widths)),
        *(row(cells) for cells in rows),
    ]


def argv_without(argv: list[str], flags: set[str]) -> list[str]:
    """This run's arguments with the given flags, and their values, taken back out.

    Only the long forms and their `=` spellings are dropped, which is every way these
    are spelled in practice. A `--mix` or `--out-root` that slipped through would be
    harmless anyway -- the child's own is appended last, and argparse lets the last one
    win -- but a surviving `--environment` would have that child fan out again, so the
    flags this drops are not all optional.
    """
    remaining = []
    skip_value = False
    for entry in argv:
        if skip_value:
            skip_value = False
        elif entry in flags:
            skip_value = True
        elif not any(entry.startswith(f"{flag}=") for flag in flags):
            remaining.append(entry)
    return remaining


def cell_command(
    args: argparse.Namespace,
    script: Path,
    argv: list[str],
    environment: Environment,
    mix: str,
) -> list[str]:
    """The child command for one cell: its own interpreter, mix and baselines."""
    dropped = {"--mix", "--environment"}
    if environment.baseline_dir is not None:
        dropped.add("--out-root")
    command = [environment.interpreter, str(script), *argv_without(argv, dropped)]
    if environment.baseline_dir is not None:
        # Its own baseline directory: one recorded against a different dependency set
        # is not comparable with it, and `environment_drift` would reject it anyway,
        # rightly.
        command += ["--out-root", str(args.out_root / environment.baseline_dir)]
    return [*command, "--mix", mix]


def run_cell_child(command: list[str], name: str) -> tuple[int, str]:
    """Run one cell to completion, streaming its output through and keeping a copy.

    The copy is what the summary table is read off; the streaming is so a run that
    takes an hour still says what it is doing while it does it. `PYTHONUNBUFFERED`
    because the child's stdout is a pipe here rather than a terminal, which would
    otherwise leave it block-buffered -- and under `--reference` its own children write
    to that same pipe, so their output would land ahead of the lines introducing them.
    """
    print("\n" + "=" * 79)
    print(f"{name}: {shlex.join(command)}")
    print("=" * 79, flush=True)
    captured = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ) as child:
        for line in child.stdout:
            print(line, end="")
            captured.append(line)
    return child.returncode, "".join(captured)


def print_grid_report(
    args: argparse.Namespace,
    outcomes: list[CellOutcome],
    described: list[str],
) -> None:
    """The consolidated table, and what to make of anything odd in it."""
    against = (
        f"vs {args.reference}"
        if args.reference is not None
        else "vs the recorded baselines"
    )
    print("\n" + "=" * 79)
    print(f"{len(outcomes)} cell(s) {against}, a process each")
    print("=" * 79 + "\n")
    if described:
        print("\n".join(described) + "\n")
    print("\n".join(render_grid_table(outcomes)))
    print(
        "\nEvery mix runs on the same shape, so the cells are directly comparable; "
        "each still gates against a baseline of its own."
    )
    if any(outcome.ok and not outcome.outputs_identical for outcome in outcomes):
        print(
            "'recorded' means that cell had no baseline to gate against, so it wrote "
            "one instead of checking one."
        )
    failed = [outcome.name() for outcome in outcomes if not outcome.ok]
    if failed:
        print(
            f"\n{len(failed)} of {len(outcomes)} cells failed: {', '.join(failed)}. "
            "Each one's FAIL banner is above, in its own output; a cell that fails one "
            "check reports nothing for the checks after it."
        )


def run_grid(args: argparse.Namespace, script: Path, argv: list[str]) -> int:
    """Run a child per cell of the grid and table what they all reported.

    Every cell is measured whether the ones before it passed or not -- a failure is a
    result like any other here -- and the exit code is the worst of them.
    """
    environments = parse_environments(args.environment, args.environment_root)
    mixes = MIXES if args.mix == MIX_ALL else (args.mix,)
    provision_environments(environments, refresh=args.refresh_environments)
    described = describe_interpreters(environments) if args.environment else []
    if described:
        print("\nMeasuring under:")
        print("\n".join(described))

    cells = [(environment, mix) for environment in environments for mix in mixes]
    outcomes = []
    for index, (environment, mix) in enumerate(cells):
        command = cell_command(args, script, argv, environment, mix)
        # Only the first cell may rebuild the reference worktree. The rest share it --
        # it is a checkout of source, the same whichever interpreter reads it -- and
        # would tear down and re-add the same commit again.
        if index > 0 and "--refresh-reference" in command:
            command.remove("--refresh-reference")
        exit_code, output = run_cell_child(command, environment.qualify(mix))
        outcomes.append(CellOutcome.from_output(environment, mix, exit_code, output))
    print_grid_report(args, outcomes, described)
    return 0 if all(outcome.ok for outcome in outcomes) else 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def meminfo_gb(field_name: str) -> float:
    """Read a /proc/meminfo field in GB, NaN where there is no /proc to read.

    Reported only, never compared, so a host that cannot answer costs nothing.
    """
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("nan")
    for line in meminfo.read_text().splitlines():
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
        "allocator_pinned": _ALLOCATOR_PINNED,
        # Which tabpfn actually got imported. Deliberately outside the drift check
        # below -- under `--reference` the two sides *must* differ here -- but
        # recorded so `verify_packages_differed` can prove they did.
        "tabpfn_path": str(Path(tabpfn.__file__).resolve().parent),
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
        "allocator_pinned",
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


def resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    """Rows to generate and columns, from whichever of --rows/--cols/--small.

    An explicit --rows/--cols wins, then --small (debugging beats everything it is
    combined with), then the profiled default. Every mix is measured on the same
    shape, so their numbers sit in one table without a mental rescaling. Shared with
    `profile_clean_data.py` so the two cannot drift into measuring different tables
    under the same flags.
    """
    if args.rows is not None:
        profiler_rows = args.rows
    elif args.small:
        profiler_rows = SMALL_PROFILER_ROWS
    else:
        profiler_rows = DEFAULT_PROFILER_ROWS

    if args.cols is not None:
        cols = args.cols
    elif args.small:
        cols = SMALL_COLS
    else:
        cols = DEFAULT_COLS
    return profiler_rows, cols


def resolve_input_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Everything identifying the input, for the directory name and the guard."""
    profiler_rows, cols = resolve_shape(args)
    return {
        "profiler_rows": profiler_rows,
        "rows": train_rows_for(profiler_rows),
        "cols": cols,
        "dtype": args.input_dtype,
        "mix": args.mix,
        "levels": MIX_LEVELS.get(args.mix),
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


def describe_role(comparing: bool, *, record_as_reference: bool) -> str:  # noqa: FBT001
    """What this run is doing, for the header."""
    if comparing:
        return "compare against baseline"
    if record_as_reference:
        return "record baseline (as the reference side of a comparison)"
    return "record baseline"


def load_baseline_for_comparison(
    paths: dict[str, Path],
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The recorded baseline, having checked it is comparable with this run."""
    recorded = load_baseline(paths, spec)
    drift = environment_drift(recorded.get("environment", {}))
    if not drift:
        return recorded
    if args.strict_env:
        fail(
            [
                "the two runs did not share an environment, so their numbers are not "
                "comparable:",
                *drift,
                "under --reference this usually means the reference worktree resolved "
                "its own dependency versions (it has no uv.lock)",
            ]
        )
    print("\nWARNING: the baseline was recorded elsewhere; RSS and timings")
    print("are only comparable within one environment:")
    for line in drift:
        print(f"  - {line}")
    return recorded


def delegated_exit_code(args: argparse.Namespace) -> int | None:
    """The exit code of whichever child-process mode was asked for, if either was.

    `--mix all`, `--environment` and `--reference` measure nothing themselves: all
    three resolve to child runs of this script, and their verdict is the children's.
    """
    if args.mix == MIX_ALL or args.environment:
        return run_grid(args, Path(__file__).resolve(), sys.argv[1:])
    if args.reference is not None:
        return run_reference_comparison(args)
    return None


def main(args: argparse.Namespace) -> None:
    """Measure `clean_data`, and gate it against a reference or a recorded baseline."""
    delegated = delegated_exit_code(args)
    if delegated is not None:
        sys.exit(delegated)

    spec = resolve_input_spec(args)
    out_dir = args.out_root / shape_dir_name(
        spec["rows"], spec["cols"], spec["dtype"], spec["mix"]
    )
    paths = baseline_paths(out_dir)
    # A baseline is only a baseline once all three files are there; a partial one
    # (an interrupted run, a hand-deleted array) is re-recorded rather than trusted.
    comparing = (
        all(path.exists() for path in paths.values()) and not args.overwrite_baseline
    )

    environment = describe_environment()
    print(f"Output directory: {out_dir}")
    role = describe_role(comparing, record_as_reference=args.record_as_reference)
    print(f"Mode: {role}")
    for key, value in environment.items():
        print(f"  {key}: {value}")

    recorded = load_baseline_for_comparison(paths, spec, args) if comparing else None

    X = generate_input(spec)
    input_fingerprint = fingerprint(X)
    print(
        f"\nInput to clean_data: {X.shape} {X.dtype} ({describe_input_size(X)})"
        f"\n  mix: {spec['mix']}, fingerprint {input_fingerprint[:16]}"
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
    if memory.n_samples < MIN_USEFUL_RSS_SAMPLES:
        print(
            f"WARNING: only {memory.n_samples} RSS sample(s) landed inside a "
            f"{memory.wall_s * 1000:.0f} ms call, so the transient above is the "
            "entry/exit readings rather than a sampled peak. Lower "
            "--sample-interval-ms, or measure a bigger shape."
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
            f"(min {timing['min_s']:.3f} s, max {timing['max_s']:.3f} s, "
            f"spread {timing['spread_pct']:.1f}%, "
            f"stdev {timing['stdev_s'] * 1000:.1f} ms)"
        )
        if timing["median_s"] < MIN_USEFUL_MEDIAN_S:
            print(
                f"WARNING: a {timing['median_s'] * 1000:.0f} ms median is too short "
                "to compare across processes; treat this shape as a correctness "
                "check and measure timing on a bigger one."
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
                    "config": {
                        "sample_interval_ms": args.sample_interval_ms,
                        "tolerance": args.tolerance,
                    },
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
        "--mix",
        default=MIX_NUMERIC,
        choices=[*MIXES, MIX_ALL],
        help="Column make-up. 'numeric' is the profiled float table. 'half-string' "
        f"makes every other column a low-cardinality string ({STRING_LEVELS} distinct "
        "values, so it detects as categorical), which is what exercises the ordinal "
        "encoder. 'numeric-object' makes every other column an integer instead, which "
        "encodes nothing but forces a recast of every column. The latter two arrive "
        "as object arrays, far heavier per cell, but all three run on the profiled "
        "shape, so their numbers compare directly. Each keeps its own baseline "
        "directory. 'all' runs the three of them in sequence, a child process each, "
        "and prints what they reported as one table.",
    )
    parser.add_argument(
        "--environment",
        action="append",
        metavar="LABEL=PYTHON",
        default=None,
        help="An interpreter to measure under, repeatable. 'lowest' and 'highest' are "
        "the two ends of the supported dependency range, built under "
        "--environment-root on first use the way CI builds its own legs; 'current' is "
        "this interpreter, named so it can sit in the grid beside them; anything else "
        "is LABEL=PYTHON, an interpreter of your own. Each of them but 'current' keeps "
        "its baselines in a subdirectory of --out-root of its own, since one recorded "
        "against a different dependency set is not comparable with it. Combines with "
        "--mix all into the whole grid, a child process per cell. Defaults to just "
        "this interpreter, with its baselines in --out-root itself.",
    )
    parser.add_argument(
        "--environment-root",
        type=Path,
        default=Path("bench_out/environments"),
        help="Where the virtualenvs for 'lowest' and 'highest' are built, one per "
        "environment. The first run against one pays for its resolution and download; "
        "later ones reuse it.",
    )
    parser.add_argument(
        "--refresh-environments",
        action="store_true",
        help="Rebuild the virtualenvs of any built environment this run asks for, "
        "before using them.",
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
        help="Timed calls. The median is the gated number; more of them make it "
        "harder for one slow call to bias a run, and `spread_pct` in the metrics "
        "records how tight they were.",
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
        default=5.0,
        help="RSS sampling period. Shorter catches narrower spikes; a call shorter "
        "than a few periods cannot be sampled meaningfully at all, which the run "
        "warns about.",
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
        "--reference",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="REV",
        help="Compare against a commit/branch/tag instead of a recorded baseline: it "
        "is checked out in a cached worktree, measured with its own tabpfn via "
        "`uv run --project`, and then the working tree is measured and gated against "
        "it -- both on this machine, minutes apart. Defaults to HEAD when given "
        "without a value, i.e. 'what do my uncommitted changes do'.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("bench_out/references"),
        help="Where reference worktrees are cached, one per commit. The first run "
        "against a commit pays for a `uv sync` of its environment; later ones reuse "
        "it.",
    )
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="Rebuild the cached reference worktree before using it.",
    )
    parser.add_argument(
        "--keep-reference-run",
        action="store_true",
        help="Keep the two sides' metrics and outputs instead of deleting them when "
        "the comparison ends.",
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Treat an environment difference from the baseline as a failure rather "
        "than a warning. Set for the working-tree side of a --reference comparison, "
        "where the two runs share a machine and must share their libraries too.",
    )
    parser.add_argument(
        "--record-as-reference",
        action="store_true",
        help="Label this run as the reference side of a comparison. Only affects "
        "reporting.",
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
