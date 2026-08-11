# ruff: noqa: T201
#  Copyright (c) Prior Labs GmbH 2026.

r"""Attribute `clean_data`'s host RAM to the calls inside it, for a mixed table.

Vendored from the wrapper-wide `profiler.py` in fomo-fitting's
`experimental/arthur/res-2439-reduce-the-ram-footprint-of-the-wrapper/` and narrowed
to `clean_data`: the same RSS-sampling machinery and nested-stage report, but the
instrumented boundaries are the ones inside the cleaning step, and no model is built.

Why the mixed case in particular. On an all-numeric table `clean_data` now allocates
just the array it hands back -- peak equals exit RSS, nothing transient. A table that
is half low-cardinality strings still peaks at ~2.6x its output, and the boundaries
in `clean_data` itself do not say where that goes, because most of it happens inside
pandas' dtype conversion and sklearn's column transformer. So those are instrumented
too.

RSS rather than tracemalloc: the copies under suspicion are numpy/pandas buffers,
which allocate outside Python's allocator and so are invisible to tracemalloc.

Two numbers per stage are worth separating:
  * peak_rss - rss_in  : transient overhead, freed before the stage returns
  * rss_out - rss_in   : retained growth, still live afterwards
A large transient means an avoidable copy; a large retained means a genuinely bigger
representation.

This tool is for RAM attribution, not timing. Every stage boundary runs a
`gc.collect()` so a reading is not polluted by garbage that happens to be pending,
and the wrapping itself costs something, so the wall times here are inflated and only
good for ranking. `scripts/bench_clean_data.py` is the one that measures time.

Mixed columns at the default shape (~7 GB of RAM, a couple of minutes). This is pure
CPU work, so a high-memory CPU partition is the one to ask for -- `cpuhighmem16spot`
has 300 nodes at ~123 GB, so it is easy to get hold of:

    srun -p cpuhighmem16spot --mem=0 --time=01:00:00 \
        uv run scripts/profile_clean_data.py

Smoke test locally in seconds:

    uv run scripts/profile_clean_data.py --small

The all-numeric case, for contrast:

    uv run scripts/profile_clean_data.py --mix numeric
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import importlib
import inspect
import json
import platform
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# Same directory, and deliberately not duplicated: the input generators must be the
# ones the regression gate uses, or this would attribute a different table's RAM.
from bench_clean_data import (
    _GB,
    DEFAULT_COLS,
    DEFAULT_PROFILER_ROWS,
    MIN_USEFUL_RSS_SAMPLES,
    MIX_HALF_STRING,
    MIX_NUMERIC,
    PASSTHROUGH_INF,
    SMALL_COLS,
    SMALL_PROFILER_ROWS,
    STRING_COLS,
    STRING_LEVELS,
    STRING_PROFILER_ROWS,
    RssSampler,
    build_feature_schema,
    current_rss_bytes,
    describe_input_size,
    generate_input,
    meminfo_gb,
    train_rows_for,
)

# The module, not the function: `patch_functions` rebinds the attribute on it, and a
# `from ... import clean_data` here would hold the original and quietly call that --
# the same by-value trap the patching itself exists to work around.
from tabpfn.preprocessing import clean as clean_module
from tabpfn.preprocessing.datamodel import FeatureModality

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# ---------------------------------------------------------------------------
# Instrumentation targets
# ---------------------------------------------------------------------------

# Module-level functions, as (module, name). Patched in every module that binds the
# name, not just the defining one: tabpfn imports these by value
# (`from ...clean import fix_dtypes`), so patching only the definition site would
# leave the actual call sites untouched.
FUNCTION_TARGETS: tuple[tuple[str, str], ...] = (
    ("tabpfn.preprocessing.clean", "clean_data"),
    ("tabpfn.preprocessing.clean", "fix_dtypes"),
    ("tabpfn.preprocessing.clean", "process_text_na_dataframe"),
    ("tabpfn.preprocessing.clean", "_apply_ordinal_encoder"),
    ("tabpfn.preprocessing.clean", "_owned_float64_values"),
    # Only reached with PASSTHROUGH_INF; a module-level alias, so patching the
    # attribute is what the call site sees.
    ("tabpfn.preprocessing.clean", "inf_masks_dataframe"),
)

# The stage every other one has to nest inside. Recording is off outside it, so the
# pandas methods below are not also charged for building the input or the report.
ROOT_LABEL = "clean_data"

# Where the mixed-column peak actually lives: the ordinal encoder builds each
# transformer's output and then stacks them, holding two full-size arrays at once,
# and our subclass reorders the result afterwards.
SKLEARN_METHOD_TARGETS: tuple[tuple[str, str, str], ...] = (
    (
        "tabpfn.preprocessing.steps.preprocessing_helpers",
        "OrderPreservingColumnTransformer",
        "fit_transform",
    ),
    (
        "tabpfn.preprocessing.steps.preprocessing_helpers",
        "OrderPreservingColumnTransformer",
        "_preserve_order",
    ),
    ("sklearn.compose", "ColumnTransformer", "fit_transform"),
    ("sklearn.compose", "ColumnTransformer", "_hstack"),
    ("sklearn.preprocessing", "OrdinalEncoder", "fit_transform"),
)

# The pandas calls `fix_dtypes` leans on. `__init__` is included because building the
# frame from an object array is itself one of the larger allocations, and pandas 3
# constructs its internal frames through `_from_mgr` rather than `__init__`, so this
# stays close to the calls the cleaning code actually writes.
PANDAS_METHOD_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("pandas", "DataFrame", "__init__"),
    ("pandas", "DataFrame", "convert_dtypes"),
    ("pandas", "DataFrame", "astype"),
    ("pandas", "DataFrame", "to_numpy"),
    ("pandas", "DataFrame", "copy"),
    ("pandas", "DataFrame", "__setitem__"),
)

# A patched pandas method called per column would bury the report in stages; past
# this many, recording stops and the report says so.
DEFAULT_MAX_STAGES = 2_000

# Columns of the text RSS timeline that replaces the original's matplotlib plot
# (neither matplotlib nor tabulate is a dependency of this repository).
TIMELINE_WIDTH = 72
TIMELINE_BLOCKS = " ▁▂▃▄▅▆▇█"

_PAGE_SIZE = resource.getpagesize()


def peak_rss_bytes() -> int:
    """Kernel high-water RSS for this process (never decreases)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# ---------------------------------------------------------------------------
# Stage recording
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    """One instrumented call."""

    name: str
    depth: int
    index: int
    t_start: float
    t_end: float = 0.0
    rss_in_bytes: int = 0
    rss_out_bytes: int = 0
    peak_rss_bytes: int = 0
    arg_bytes: int = 0
    ret_bytes: int = 0
    arg_desc: str = ""
    ret_desc: str = ""
    # False when a generator stage was abandoned before exhaustion, so its timings
    # never closed and must not be read as a measurement.
    completed: bool = False

    @property
    def wall_s(self) -> float:
        """Wall-clock seconds spent in the call, nan if it never completed."""
        if not self.completed:
            return float("nan")
        return self.t_end - self.t_start

    @property
    def transient_bytes(self) -> int:
        """Peak above the RSS on entry: memory allocated and freed inside."""
        return max(0, self.peak_rss_bytes - self.rss_in_bytes)

    @property
    def retained_bytes(self) -> int:
        """RSS still held after returning, relative to entry."""
        return self.rss_out_bytes - self.rss_in_bytes


def describe_payload(value: Any) -> tuple[int, str]:
    """Byte size and a short dtype description of an array/frame, else (0, "").

    The dtype description is the point: a float32 input arriving as float64 doubles
    the footprint before any copy is made. For an object frame or array the size is
    pointers only -- the Python objects behind them are not counted, and dominate.
    """
    if isinstance(value, np.ndarray):
        return value.nbytes, f"ndarray{value.shape} {value.dtype}"
    if isinstance(value, pd.DataFrame):
        nbytes = int(value.memory_usage(index=True, deep=False).sum())
        dtypes = sorted({str(d) for d in value.dtypes})
        summary = "/".join(dtypes[:3]) + ("/..." if len(dtypes) > 3 else "")
        return nbytes, f"DataFrame{value.shape} {summary}"
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=False)), (
            f"Series{value.shape} {value.dtype}"
        )
    if isinstance(value, (list, tuple)) and value:
        sizes = [describe_payload(item) for item in value]
        total = sum(size for size, _ in sizes)
        if total:
            return total, f"{type(value).__name__}[{len(value)}] of {sizes[0][1]}"
    return 0, ""


def safe_describe_payload(value: Any) -> tuple[int, str]:
    """`describe_payload`, but a payload that cannot be measured is not fatal.

    Sizing a payload must never be able to break the code being profiled, and half
    the patched boundaries are pandas internals whose arguments can be in states
    pandas' own accessors reject.
    """
    try:
        return describe_payload(value)
    except Exception as error:  # noqa: BLE001
        return 0, f"<unmeasurable: {type(error).__name__}>"


class Profiler:
    """Collects stages and the RSS trace for one `clean_data` call.

    Recording is gated on being inside the root stage: the patched pandas methods
    are called all over the place -- generating the input, writing the report -- and
    only the ones `clean_data` makes are of any interest.
    """

    def __init__(self, sampler: RssSampler, max_stages: int) -> None:
        """Wrap a started sampler."""
        self.sampler = sampler
        self.max_stages = max_stages
        self.stages: list[Stage] = []
        self.skipped = 0
        self._depth = 0

    @property
    def active(self) -> bool:
        """Whether the root stage is currently open."""
        return self._depth > 0

    @contextmanager
    def stage(self, name: str, payload: Any = None) -> Iterator[Stage | None]:
        """Record one call's timing, RSS window, and payload size."""
        if not self.active and name != ROOT_LABEL:
            yield None
            return
        if len(self.stages) >= self.max_stages:
            self.skipped += 1
            yield None
            return

        gc.collect()
        record = Stage(
            name=name,
            depth=self._depth,
            index=len(self.stages),
            t_start=time.perf_counter(),
            rss_in_bytes=current_rss_bytes(),
        )
        record.arg_bytes, record.arg_desc = safe_describe_payload(payload)
        self.stages.append(record)
        self._depth += 1
        try:
            yield record
        finally:
            self._depth -= 1
            gc.collect()
            record.t_end = time.perf_counter()
            record.rss_out_bytes = current_rss_bytes()
            record.peak_rss_bytes = max(
                self.sampler.peak_between(record.t_start, record.t_end),
                record.rss_in_bytes,
                record.rss_out_bytes,
            )
            record.completed = True


# ---------------------------------------------------------------------------
# Monkeypatching
# ---------------------------------------------------------------------------


def _first_payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The argument most likely to be the feature matrix.

    A DataFrame with no `_mgr` is the half-built `self` of `DataFrame.__init__`, so
    it is skipped in favour of the `data` argument that follows it -- which is the
    thing worth sizing there anyway.
    """
    for key in ("X", "X_train", "X_test", "data"):
        if key in kwargs:
            return kwargs[key]
    for arg in args:
        if isinstance(arg, pd.DataFrame) and getattr(arg, "_mgr", None) is None:
            continue
        if isinstance(arg, (np.ndarray, pd.DataFrame)):
            return arg
    return None


def _wrap(profiler: Profiler, label: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callable so its call is recorded as a stage.

    Generator functions get a wrapper that is itself a generator, so the stage spans
    the whole consumption rather than just the cheap call that builds the generator.
    """
    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def generator_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
            with profiler.stage(label, _first_payload(args, kwargs)):
                yield from fn(*args, **kwargs)

        return generator_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with profiler.stage(label, _first_payload(args, kwargs)) as record:
            result = fn(*args, **kwargs)
            if record is not None:
                record.ret_bytes, record.ret_desc = safe_describe_payload(result)
            return result

    return wrapper


def patch_functions(
    profiler: Profiler,
    targets: tuple[tuple[str, str], ...],
) -> list[Callable[[], None]]:
    """Instrument module-level functions in every module that binds them."""
    undo: list[Callable[[], None]] = []
    for module_name, attribute in targets:
        module = importlib.import_module(module_name)
        original = getattr(module, attribute)
        wrapped = _wrap(profiler, attribute, original)
        for other in list(sys.modules.values()):
            if other is None or not getattr(other, "__name__", "").startswith("tabpfn"):
                continue
            if getattr(other, attribute, None) is original:
                setattr(other, attribute, wrapped)
                undo.append(functools.partial(setattr, other, attribute, original))
    return undo


def patch_methods(
    profiler: Profiler,
    targets: tuple[tuple[str, str, str], ...],
) -> list[Callable[[], None]]:
    """Instrument methods by shadowing them on the class named.

    The method may be inherited -- `OrdinalEncoder.fit_transform` comes from
    `TransformerMixin` -- in which case the wrapper is set on the subclass and the
    undo deletes it again, rather than leaving the base's function copied onto the
    subclass for good.
    """
    undo: list[Callable[[], None]] = []
    for module_name, class_name, method_name in targets:
        cls = getattr(importlib.import_module(module_name), class_name)
        original = getattr(cls, method_name, None)
        if original is None:
            raise AttributeError(f"{class_name} has no {method_name}")
        label = f"{class_name}.{method_name}"
        was_own = method_name in cls.__dict__
        setattr(cls, method_name, _wrap(profiler, label, original))
        undo.append(
            functools.partial(setattr, cls, method_name, original)
            if was_own
            else functools.partial(delattr, cls, method_name)
        )
    return undo


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def tree_prefixes(depths: list[int]) -> list[str]:
    """`tree`-style connector prefixes for a pre-order list of depths.

    Stages are recorded in call order, which is a pre-order traversal, so the
    parent/child structure is recoverable from the depths alone: a stage is its
    parent's last child when no later stage returns to the same depth before dipping
    below it.
    """
    count = len(depths)
    is_last = [True] * count
    for i, depth in enumerate(depths):
        for j in range(i + 1, count):
            if depths[j] < depth:
                break
            if depths[j] == depth:
                is_last[i] = False
                break

    prefixes: list[str] = []
    # ancestor_is_last[d] is the flag of this stage's ancestor at depth d; a non-last
    # ancestor still has siblings below, so its column stays drawn.
    ancestor_is_last: list[bool] = []
    for i, depth in enumerate(depths):
        del ancestor_is_last[depth:]
        # Depth 0 is the root, which contributes no column of its own.
        columns = ["    " if ancestor_is_last[d] else "│   " for d in range(1, depth)]
        connector = ("└── " if is_last[i] else "├── ") if depth else ""
        prefixes.append("".join(columns) + connector)
        ancestor_is_last.append(is_last[i])
    return prefixes


def collapse_repeats(stages: list[Stage]) -> list[tuple[Stage, int]]:
    """Fold runs of identical leaf siblings into one row each.

    pandas calls some of the patched methods once per column, which at 400 columns
    would bury the tree in hundreds of identical rows. Only adjacent same-name,
    same-depth stages are folded, so a stage with children of its own is never
    merged away, and the CSV and JSON keep every individual call.
    """
    folded: list[tuple[Stage, int]] = []
    for stage in stages:
        if folded:
            previous, count = folded[-1]
            adjacent = stage.index == previous.index + count
            if (
                adjacent
                and stage.name == previous.name
                and stage.depth == previous.depth
            ):
                previous.peak_rss_bytes = max(
                    previous.peak_rss_bytes, stage.peak_rss_bytes
                )
                previous.rss_out_bytes = stage.rss_out_bytes
                previous.t_end += stage.wall_s
                folded[-1] = (previous, count + 1)
                continue
        folded.append((dataclasses.replace(stage), 1))
    return folded


def stages_frame(stages: list[Stage]) -> pd.DataFrame:
    """One row per instrumented call, in call order.

    The stage name is bare, so the CSV stays parseable; `depth` carries the nesting
    and `tree_prefixes` renders it for the human-facing outputs.
    """
    return pd.DataFrame(
        [
            {
                "stage": stage.name + ("" if stage.completed else " [never completed]"),
                "depth": stage.depth,
                "wall_s": round(stage.wall_s, 3),
                "rss_in_gb": round(stage.rss_in_bytes / _GB, 3),
                "peak_rss_gb": round(stage.peak_rss_bytes / _GB, 3),
                "transient_gb": round(stage.transient_bytes / _GB, 3),
                "retained_gb": round(stage.retained_bytes / _GB, 3),
                "arg_gb": round(stage.arg_bytes / _GB, 3),
                "ret_gb": round(stage.ret_bytes / _GB, 3),
                "arg": stage.arg_desc,
                "ret": stage.ret_desc,
            }
            for stage in stages
        ]
    )


def render_markdown_table(frame: pd.DataFrame) -> str:
    """A pipe table, since `to_markdown` would need tabulate."""
    columns = [str(column) for column in frame.columns]
    right_aligned = {
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False)]
    widths = [
        max([len(name), *(len(row[index]) for row in rows)])
        for index, name in enumerate(columns)
    ]

    def render_row(cells: list[str]) -> str:
        padded = [
            cell.rjust(widths[index])
            if columns[index] in right_aligned
            else cell.ljust(widths[index])
            for index, cell in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    separator = [
        ("-" * (width - 1) + ":")
        if columns[index] in right_aligned
        else (":" + "-" * (width - 1))
        for index, width in enumerate(widths)
    ]
    return "\n".join(
        [render_row(columns), render_row(separator), *map(render_row, rows)]
    )


def render_rss_timeline(sampler: RssSampler) -> list[str]:
    """A text sketch of the RSS trace, in place of the original's plot."""
    if not sampler.samples:
        return ["(no RSS samples collected)"]
    values = [rss for _, rss in sampler.samples]
    low, high = min(values), max(values)
    span = high - low
    buckets: list[str] = []
    for index in range(TIMELINE_WIDTH):
        start = index * len(values) // TIMELINE_WIDTH
        stop = max(start + 1, (index + 1) * len(values) // TIMELINE_WIDTH)
        level = max(values[start:stop])
        position = (
            0 if span == 0 else int((level - low) / span * (len(TIMELINE_BLOCKS) - 1))
        )
        buckets.append(TIMELINE_BLOCKS[position])
    return [
        f"peak {high / _GB:.2f} GB  " + "".join(buckets) + f"  {high / _GB:.2f} GB",
        f"base {low / _GB:.2f} GB  "
        + f"{'':<{TIMELINE_WIDTH}}".replace(" ", "-")
        + f"  {len(values)} samples",
    ]


LEGEND = (
    "`calls` folds runs of identical sibling calls (pandas invokes some of these "
    "once per column); the folded row keeps the largest peak and the last exit "
    "reading, and `stages.csv` keeps every call separately. "
    "`transient_gb` is the peak RSS reached inside the call above the RSS on entry, "
    "so it does **not** add up across children: a parent's transient is often just "
    "the largest peak one of its children reached. `retained_gb` does add up -- it is "
    "the RSS still held on return. A stage whose transient exceeds every child's is "
    "allocating that memory itself. `arg_gb`/`ret_gb` size the frame or array going "
    "in and coming out, counting only pointers for object dtypes."
)


def write_report(
    stages: list[Stage],
    sampler: RssSampler,
    environment: dict[str, Any],
    summary: dict[str, Any],
    out_dir: Path,
) -> None:
    """Write the CSV, JSON and markdown, and print the table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = stages_frame(stages)
    frame.to_csv(out_dir / "stages.csv", index=False)

    (out_dir / "profile.json").write_text(
        json.dumps(
            {
                "environment": environment,
                "summary": summary,
                "stages": [dataclasses.asdict(stage) for stage in stages],
                "rss_trace": [
                    {"t": round(t, 4), "rss_gb": round(rss / _GB, 3)}
                    for t, rss in sampler.samples
                ],
            },
            indent=2,
            default=str,
        )
    )

    folded = collapse_repeats(stages)
    display = stages_frame([stage for stage, _ in folded])
    display.insert(2, "calls", [count for _, count in folded])
    prefixes = tree_prefixes([stage.depth for stage, _ in folded])
    markdown_frame = display.drop(columns=["depth"]).copy()
    # Backticked so markdown does not collapse the whitespace, which would flatten
    # the tree connectors and the shape in "ndarray(333333, 400) object".
    markdown_frame["stage"] = [
        f"`{prefix}{name}`"
        for prefix, name in zip(prefixes, display["stage"], strict=True)
    ]
    for column in ("arg", "ret"):
        markdown_frame[column] = markdown_frame[column].map(
            lambda value: f"`{value}`" if value else ""
        )

    lines = [f"# clean_data host-RAM profile ({summary['mix']})", ""]
    lines += ["## Environment", ""]
    lines += [f"- `{key}`: {value}" for key, value in environment.items()]
    lines += ["", "## Summary", ""]
    lines += [f"- `{key}`: {value}" for key, value in summary.items()]
    lines += ["", "## RSS over the call", "", "```"]
    lines += render_rss_timeline(sampler)
    lines += [
        "```",
        "",
        "## Stages",
        "",
        LEGEND,
        "",
        render_markdown_table(markdown_frame),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        250,
        "display.expand_frame_repr",
        False,  # noqa: FBT003
    ):
        console_frame = display.drop(columns=["depth"])
        # Padded to a common width because pandas right-justifies object columns,
        # which would stagger the connectors and destroy the tree.
        tree_lines = [
            prefix + name
            for prefix, name in zip(prefixes, display["stage"], strict=True)
        ]
        width = max(len(line) for line in tree_lines)
        console_frame["stage"] = [line.ljust(width) for line in tree_lines]
        print("\n" + "=" * 100)
        print("STAGES")
        print("=" * 100)
        print(console_frame.to_string(index=False))

    print("\n".join(["", *render_rss_timeline(sampler)]))
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nWrote profile to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def describe_environment() -> dict[str, Any]:
    """Host details to record alongside the profile."""
    return {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "host_memory_total_gb": round(meminfo_gb("MemTotal"), 1),
        "host_memory_available_gb": round(meminfo_gb("MemAvailable"), 1),
    }


def resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    """Rows and columns, mirroring the benchmark's precedence."""
    if args.rows is not None:
        profiler_rows = args.rows
    elif args.small:
        profiler_rows = SMALL_PROFILER_ROWS
    elif args.mix == MIX_HALF_STRING:
        profiler_rows = STRING_PROFILER_ROWS
    else:
        profiler_rows = DEFAULT_PROFILER_ROWS

    if args.cols is not None:
        cols = args.cols
    elif args.small:
        cols = SMALL_COLS
    elif args.mix == MIX_HALF_STRING:
        cols = STRING_COLS
    else:
        cols = DEFAULT_COLS
    return profiler_rows, cols


def main(args: argparse.Namespace) -> None:
    """Profile one `clean_data` call and write the report."""
    environment = describe_environment()
    print("Environment:")
    for key, value in environment.items():
        print(f"  {key}: {value}")

    profiler_rows, cols = resolve_shape(args)
    spec = {
        "profiler_rows": profiler_rows,
        "cols": cols,
        "dtype": args.input_dtype,
        "mix": args.mix,
    }
    X = generate_input(spec)
    feature_schema = build_feature_schema(X)
    n_categorical = len(feature_schema.indices_for(FeatureModality.CATEGORICAL))
    print(
        f"\nInput to clean_data: {X.shape} {X.dtype} ({describe_input_size(X)})"
        f"\n  mix: {args.mix}, {n_categorical} of {cols} columns "
        "detected categorical"
    )

    sampler = RssSampler(args.sample_interval_ms / 1000.0)
    profiler = Profiler(sampler, args.max_stages)
    targets = FUNCTION_TARGETS
    method_targets: tuple[tuple[str, str, str], ...] = ()
    if not args.no_sklearn_detail:
        method_targets += SKLEARN_METHOD_TARGETS
    if not args.no_pandas_detail:
        method_targets += PANDAS_METHOD_TARGETS
    undo = patch_functions(profiler, targets) + patch_methods(profiler, method_targets)

    gc.collect()
    # Every stage boundary runs a `gc.collect()` to quiesce the reading, and a
    # gc pass walks the whole heap -- which for the mixed case holds one Python float
    # per numeric cell, tens of millions of them. Freezing moves everything alive now
    # (the input included) into a generation collections skip, so those passes only
    # walk what `clean_data` itself allocates. Without this the profile takes hours.
    gc.freeze()
    baseline_rss_gb = current_rss_bytes() / _GB
    sampler.start()
    try:
        X_cleaned, _, _ = clean_module.clean_data(
            X=X, feature_schema=feature_schema, passthrough_inf=PASSTHROUGH_INF
        )
    finally:
        sampler.stop()
        for restore in reversed(undo):
            restore()

    if not profiler.stages:
        print("No stages recorded; the patches did not take.", file=sys.stderr)
        sys.exit(1)

    root = profiler.stages[0]
    output_gb = X_cleaned.nbytes / _GB
    peak_gb = max(rss for _, rss in sampler.samples) / _GB if sampler.samples else 0.0
    summary = {
        "mix": args.mix,
        "profiler_rows": profiler_rows,
        "rows": train_rows_for(profiler_rows),
        "cols": cols,
        "string_levels": STRING_LEVELS if args.mix == MIX_HALF_STRING else None,
        "input_dtype": args.input_dtype,
        "input_pointer_gb": round(X.nbytes / _GB, 3),
        "output_gb": round(output_gb, 3),
        "baseline_rss_gb": round(baseline_rss_gb, 3),
        "peak_rss_gb": round(peak_gb, 3),
        "kernel_peak_rss_gb": round(peak_rss_bytes() / _GB, 3),
        "transient_gb": round(root.transient_bytes / _GB, 3),
        "retained_gb": round(root.retained_bytes / _GB, 3),
        # The number the mixed case exists to show: an all-numeric clean sits at 1.0
        # (it allocates only what it returns), so anything above that is copying.
        "transient_over_output": (
            round(root.transient_bytes / _GB / output_gb, 2) if output_gb else None
        ),
        "stages_recorded": len(profiler.stages),
        "stages_skipped_over_cap": profiler.skipped,
        "rss_samples": len(sampler.samples),
        "wall_s_with_profiling_overhead": round(root.wall_s, 3),
    }
    if len(sampler.samples) < MIN_USEFUL_RSS_SAMPLES:
        print(
            f"\nWARNING: only {len(sampler.samples)} RSS sample(s) landed in the "
            "call, so the peaks are entry/exit readings rather than sampled ones. "
            "Lower --sample-interval-ms or profile a bigger shape."
        )
    if profiler.skipped:
        print(
            f"\nWARNING: {profiler.skipped} stage(s) went unrecorded past the "
            f"--max-stages cap of {args.max_stages}; the tree below is incomplete."
        )
    write_report(profiler.stages, sampler, environment, summary, args.out_dir)


def get_parser() -> argparse.ArgumentParser:
    """Get the parser for profile_clean_data.py."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mix",
        default=MIX_HALF_STRING,
        choices=[MIX_HALF_STRING, MIX_NUMERIC],
        help="Column make-up, generated by the benchmark's own generators. The mixed "
        "one is the case with RAM left to explain.",
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument(
        "--small",
        action="store_true",
        help="A shape that runs in seconds, for debugging this script.",
    )
    parser.add_argument(
        "--input-dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
    )
    parser.add_argument(
        "--no-sklearn-detail",
        action="store_true",
        help="Skip the ordinal-encoder boundaries. They are where the mixed-column "
        "peak lives, so this is mainly for checking they are not what perturbs it.",
    )
    parser.add_argument(
        "--no-pandas-detail",
        action="store_true",
        help="Skip the DataFrame method boundaries, leaving only tabpfn's own "
        "functions.",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=DEFAULT_MAX_STAGES,
        help="Stop recording past this many stages, so a per-column call cannot bury "
        "the report.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=5.0,
        help="RSS sampling period. Shorter catches narrower spikes.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to profile_out/clean_data_<mix>.",
    )
    return parser


if __name__ == "__main__":
    parsed = get_parser().parse_args()
    if parsed.out_dir is None:
        parsed.out_dir = Path("profile_out") / f"clean_data_{parsed.mix}"
    main(parsed)
