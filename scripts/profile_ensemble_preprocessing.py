# ruff: noqa: T201
#  Copyright (c) Prior Labs GmbH 2026.

r"""Attribute the ensemble preprocessor's host RAM to the calls inside it.

`TabPFNEnsemblePreprocessor.fit_transform_ensemble_members_iterator` is, now that
`clean_data` has been brought down, the largest transient in the wrapper's host-RAM
profile: 42.64 GB reached inside one call, against the 10.67 GB float64 table it is
handed, fitting one estimator on 666,667 x 2,000. This narrows the wrapper-wide
`profiler.py` from fomo-fitting's
`experimental/arthur/res-2439-reduce-the-ram-footprint-of-the-wrapper/` to that call:
the same RSS-sampling machinery and nested-stage report, but the instrumented
boundaries are the ones underneath it, and no model is built.

The boundaries are chosen for where a full-size copy can hide:

* `_fit_preprocessing_one` copies `X_train` before touching it, and slices it again
  when a member subsamples rows or features.
* `PreprocessingPipeline._process_steps` copies once more, to keep the caller's array
  immutable, then reassembles per step.
* each pipeline step, and inside them sklearn's `ColumnTransformer`, which builds every
  transformer's output before stacking them -- two full-size arrays at once.

Inputs, the report machinery and the mixes are shared with the two `clean_data`
harnesses (`scripts/bench_ensemble_preprocessing.py` builds the inputs;
`scripts/profile_clean_data.py` owns the stage recording), so the three cannot drift
into measuring different tables or reporting them differently.

RSS rather than tracemalloc: the copies under suspicion are numpy buffers, which
allocate outside Python's allocator and so are invisible to tracemalloc.

The torch transforms this preprocessor configures do not run here, so nothing below is
a GPU measurement: with ENABLE_GPU_PREPROCESSING on, the quantile/squashing scaler, the
SVD, the fingerprint and the shuffle are dropped from the CPU pipeline and
`create_gpu_preprocessing_pipeline` only *builds* the `TorchPreprocessingPipeline` that
replaces them, which the inference engine runs later via
`_maybe_run_gpu_preprocessing`. Every stage still closes on a device synchronise, so
that stays true rather than merely assumed: if any of that work moves in here, the
per-stage times keep meaning what they say, and `device_peak_allocated_gb` in the
summary reports it.

Two numbers per stage are worth separating:
  * peak_rss - rss_in  : transient overhead, freed before the stage returns
  * rss_out - rss_in   : retained growth, still live afterwards
A large transient means an avoidable copy; a large retained means a genuinely bigger
representation -- and every member does retain its own preprocessed table.

This tool is for RAM attribution, not timing. Every stage boundary runs a
`gc.collect()` so a reading is not polluted by garbage that happens to be pending, and
the wrapping itself costs something, so the wall times here are inflated and only good
for ranking. `scripts/bench_ensemble_preprocessing.py` is the one that measures time.

The profiled shape needs ~55 GB of RAM. This is pure CPU work -- the GPU-scheduled
steps only run at inference time -- so a high-memory CPU partition is the one to ask
for; `cpuhighmem16spot` has 300 nodes at ~123 GB, so it is easy to get hold of:

    srun -p cpuhighmem16spot --mem=0 --time=01:00:00 \
        uv run scripts/profile_ensemble_preprocessing.py

Smoke test locally in seconds:

    uv run scripts/profile_ensemble_preprocessing.py --small

The mixed cases, whose cleaned tables still carry categorical columns and so feed the
ordinal encoder inside the ensemble pipelines:

    uv run scripts/profile_ensemble_preprocessing.py --mix half-string
    uv run scripts/profile_ensemble_preprocessing.py --mix numeric-object

And the legacy CPU-only pipelines, which put the squashing scaler, the SVD, the
fingerprint and the shuffle back on this side:

    uv run scripts/profile_ensemble_preprocessing.py --no-gpu-preprocessing
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# All from this directory, and deliberately not duplicated: the inputs must be the ones
# the regression gate builds, or this would attribute a different table's RAM, and the
# stage recording, monkeypatching and report are `profile_clean_data`'s, so a fix to
# either profiler lands in both.
from bench_clean_data import (
    _GB,
    MIN_USEFUL_RSS_SAMPLES,
    MIX_HALF_STRING,
    MIX_LEVELS,
    MIX_NUMERIC,
    MIX_NUMERIC_OBJECT,
    RssSampler,
    current_rss_bytes,
    describe_input_size,
    meminfo_gb,
)
from bench_ensemble_preprocessing import (
    DEFAULT_N_ESTIMATORS,
    DEFAULT_PARALLEL_MODE,
    PARALLEL_MODES,
    build_ensemble_inputs,
    build_preprocessor,
    describe_device_activity,
    device_activity,
    reset_device_activity,
    resolve_input_spec,
    synchronize_device,
)
from profile_clean_data import (
    Profiler,
    describe_environment,
    patch_functions,
    patch_methods,
    peak_rss_bytes,
    write_report,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Instrumentation targets
# ---------------------------------------------------------------------------

# The stage every other one has to nest inside. Recording is off outside it, so the
# sklearn and pipeline boundaries below are not also charged for building the input.
# `fit_transform_ensemble_members` is deliberately not instrumented: it is a one-line
# `list(...)` around this, and making it the root would push the call this profile is
# about down a level for nothing.
ROOT_LABEL = "TabPFNEnsemblePreprocessor.fit_transform_ensemble_members_iterator"

# Module-level functions, as (module, name). Patched in every module that binds the
# name, not just the defining one: tabpfn imports these by value, so patching only the
# definition site would leave the actual call sites untouched.
FUNCTION_TARGETS: tuple[tuple[str, str], ...] = (
    # The generator the root iterates; with parallel_mode="block" its first `next`
    # covers every member's preprocessing.
    ("tabpfn.preprocessing.transform", "fit_preprocessing"),
    # One call per member: copies X_train, and slices it first when the member
    # subsamples rows or features.
    ("tabpfn.preprocessing.transform", "_fit_preprocessing_one"),
    ("tabpfn.preprocessing.transform", "_transform_labels_one"),
    # Cheap, but it is the other thing the root does per member, so its absence from
    # the RAM total is worth showing rather than assuming.
    ("tabpfn.preprocessing.torch.factory", "create_gpu_preprocessing_pipeline"),
    # Only allocates when the table carries infinities, which the profiled one does
    # not; recorded so that stays visible.
    ("tabpfn.preprocessing.pipeline_interface", "_extract_inf_masks"),
)

# Methods, as (module, class, method). Looked up dynamically, so shadowing the class
# attribute is what the call sites see -- including for the steps' `fit_transform`,
# which most of them inherit from `PreprocessingStep`.
METHOD_TARGETS: tuple[tuple[str, str, str], ...] = (
    (
        "tabpfn.preprocessing.ensemble",
        "TabPFNEnsemblePreprocessor",
        "fit_transform_ensemble_members_iterator",
    ),
    (
        "tabpfn.preprocessing.pipeline_interface",
        "PreprocessingPipeline",
        "fit_transform",
    ),
    # Where the per-call copy of the whole table is made, and where each step's output
    # is written back into it.
    (
        "tabpfn.preprocessing.pipeline_interface",
        "PreprocessingPipeline",
        "_process_steps",
    ),
    # `np.concatenate` of the table and a step's added columns: two full-size arrays.
    (
        "tabpfn.preprocessing.pipeline_interface",
        "PreprocessingPipeline",
        "_maybe_append_added_columns",
    ),
)

# The steps a 3.1_exp classifier pipeline can hold. Which ones actually run depends on
# ENABLE_GPU_PREPROCESSING: with it on (the profiled configuration) the CPU pipeline is
# just the first three, and the quantile/squashing transform, the SVD, the fingerprint
# and the shuffle are deferred to the GPU pipeline. The rest are listed so
# `--no-gpu-preprocessing` is attributed too; a step that never runs simply never
# appears in the report.
STEP_METHOD_TARGETS: tuple[tuple[str, str, str], ...] = tuple(
    ("tabpfn.preprocessing.steps", class_name, method)
    for class_name, methods in (
        ("RemoveConstantFeaturesStep", ("fit_transform", "_fit", "_transform")),
        # Overrides `fit_transform` to fit and transform in one ColumnTransformer pass,
        # so `_fit`/`_transform` are the test-time path rather than this one.
        ("ReshapeFeatureDistributionsStep", ("fit_transform", "_transform")),
        ("EncodeCategoricalFeaturesStep", ("fit_transform", "_fit_transform_internal")),
        ("AddSVDFeaturesStep", ("fit_transform", "_fit", "_transform")),
        ("AddFingerprintFeaturesStep", ("fit_transform", "_transform")),
        ("ShuffleFeaturesStep", ("fit_transform", "_transform")),
    )
    for method in methods
)

# The sklearn machinery the steps lean on. `ColumnTransformer` is the important one: it
# materialises each transformer's output and then stacks them, so the peak inside it is
# two copies of the columns it was given.
SKLEARN_METHOD_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("sklearn.compose", "ColumnTransformer", "fit_transform"),
    ("sklearn.compose", "ColumnTransformer", "transform"),
    ("sklearn.compose", "ColumnTransformer", "_hstack"),
    ("sklearn.preprocessing", "OrdinalEncoder", "fit_transform"),
    ("sklearn.preprocessing", "StandardScaler", "fit_transform"),
    ("sklearn.decomposition", "TruncatedSVD", "fit_transform"),
)

# A patched method called per column or per member would bury the report in stages;
# past this many, recording stops and the report says so.
DEFAULT_MAX_STAGES = 2_000


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_targets(
    args: argparse.Namespace,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str, str], ...],
]:
    """The function and method targets this run instruments."""
    method_targets = METHOD_TARGETS
    if not args.no_step_detail:
        method_targets += STEP_METHOD_TARGETS
    if not args.no_sklearn_detail:
        method_targets += SKLEARN_METHOD_TARGETS
    return FUNCTION_TARGETS, method_targets


def main(args: argparse.Namespace) -> None:
    """Profile one full consumption of the iterator and write the report."""
    environment = describe_environment()
    print("Environment:")
    for key, value in environment.items():
        print(f"  {key}: {value}")

    spec = resolve_input_spec(args)
    inputs = build_ensemble_inputs(spec)
    X = inputs.X_train
    print(
        f"\nInput to the iterator: {X.shape} {X.dtype} ({describe_input_size(X)})"
        f"\n  mix: {spec['mix']}, {inputs.n_categorical} of "
        f"{inputs.feature_schema.num_columns} columns categorical after cleaning"
        f"\n  {spec['n_estimators']} estimator(s), "
        f"{'GPU-scheduled' if spec['enable_gpu_preprocessing'] else 'CPU-only'} "
        f"pipelines, parallel_mode={spec['parallel_mode']}"
    )
    available = meminfo_gb("MemAvailable")
    if X.nbytes * 5 > available * _GB:
        print(
            f"WARNING: a {X.nbytes / _GB:.1f} GB input has been seen to reach ~5x in "
            f"RSS here, above the {available:.0f} GB available. A host OOM is a "
            "SIGKILL -- run this on a high-memory node or reduce --rows/--cols."
        )
    if spec["n_preprocessing_jobs"] != 1:
        print(
            f"WARNING: n_preprocessing_jobs={spec['n_preprocessing_jobs']} dispatches "
            "members to worker processes. Neither their RSS nor their stages are "
            "visible here, so the tree below would be empty of the per-member work."
        )

    preprocessor = build_preprocessor(inputs, spec)

    sampler = RssSampler(args.sample_interval_ms / 1000.0)
    profiler = Profiler(
        sampler,
        args.max_stages,
        root_label=ROOT_LABEL,
        # Every stage closes on a synchronise. No stage queues device work today -- the
        # torch pipeline this call builds is run later, by the inference engine -- so
        # this costs nothing now and keeps the per-stage attribution correct if any of
        # it moves in. `device` in the summary records whether it stayed a no-op.
        synchronize=synchronize_device,
    )
    function_targets, method_targets = resolve_targets(args)
    undo: list[Callable[[], None]] = patch_functions(
        profiler, function_targets
    ) + patch_methods(profiler, method_targets)

    gc.collect()
    # Every stage boundary runs a `gc.collect()` to quiesce the reading, and a gc pass
    # walks the whole heap -- which for the object mixes holds one Python object per
    # cell, tens of millions of them. Freezing moves everything alive now (the input
    # included) into a generation collections skip, so those passes only walk what the
    # measured call allocates. Without this the profile takes hours.
    gc.freeze()
    reset_device_activity()
    baseline_rss_gb = current_rss_bytes() / _GB
    sampler.start()
    try:
        members = list(
            preprocessor.fit_transform_ensemble_members_iterator(
                X_train=inputs.X_train,
                y_train=inputs.y_train,
                parallel_mode=spec["parallel_mode"],
            )
        )
        synchronize_device()
    finally:
        sampler.stop()
        for restore in reversed(undo):
            restore()

    if not profiler.stages:
        print("No stages recorded; the patches did not take.", file=sys.stderr)
        sys.exit(1)

    root = profiler.stages[0]
    device = device_activity()
    print(f"\nDevice: {describe_device_activity(device)}")
    member_gb = sum(member.X_train.nbytes for member in members) / _GB
    peak_gb = max(rss for _, rss in sampler.samples) / _GB if sampler.samples else 0.0
    summary: dict[str, Any] = {
        "mix": spec["mix"],
        "profiler_rows": spec["profiler_rows"],
        "rows": spec["rows"],
        "cols": spec["cols"],
        "levels": MIX_LEVELS.get(spec["mix"]),
        "input_dtype": spec["dtype"],
        "n_estimators": spec["n_estimators"],
        "enable_gpu_preprocessing": spec["enable_gpu_preprocessing"],
        "parallel_mode": spec["parallel_mode"],
        "feature_cap": spec["feature_cap"],
        "input_gb": round(X.nbytes / _GB, 3),
        "members_gb": round(member_gb, 3),
        "member_shapes": [list(member.X_train.shape) for member in members],
        "baseline_rss_gb": round(baseline_rss_gb, 3),
        "peak_rss_gb": round(peak_gb, 3),
        "kernel_peak_rss_gb": round(peak_rss_bytes() / _GB, 3),
        "transient_gb": round(root.transient_bytes / _GB, 3),
        "retained_gb": round(root.retained_bytes / _GB, 3),
        # The headline ratio: the members are what the call is for, so anything above
        # 1.0 here is memory held at the peak beyond what it hands back.
        "transient_over_members": (
            round(root.transient_bytes / _GB / member_gb, 2) if member_gb else None
        ),
        # Zero unless torch work has moved into this call, in which case the stage wall
        # times below include waiting for it -- every stage closes on a synchronise --
        # and the RSS columns still count host memory only.
        "device_cuda_available": device["cuda_available"],
        "device_peak_allocated_gb": (
            round(device["peak_allocated_bytes"] / _GB, 3)
            if device["peak_allocated_bytes"] is not None
            else None
        ),
        "stages_recorded": len(profiler.stages),
        "stages_skipped_over_cap": profiler.skipped,
        "rss_samples": len(sampler.samples),
        "wall_s_with_profiling_overhead": round(root.wall_s, 3),
    }
    if len(sampler.samples) < MIN_USEFUL_RSS_SAMPLES:
        print(
            f"\nWARNING: only {len(sampler.samples)} RSS sample(s) landed in the call, "
            "so the peaks are entry/exit readings rather than sampled ones. Lower "
            "--sample-interval-ms or profile a bigger shape."
        )
    if profiler.skipped:
        print(
            f"\nWARNING: {profiler.skipped} stage(s) went unrecorded past the "
            f"--max-stages cap of {args.max_stages}; the tree below is incomplete."
        )
    # The members hold a preprocessed copy of the table each, and the report builds
    # frames of its own; nothing below needs them.
    del members
    gc.collect()
    write_report(
        profiler.stages,
        sampler,
        environment,
        summary,
        args.out_dir,
        title=f"Ensemble preprocessing host-RAM profile ({spec['mix']})",
    )


def get_parser() -> argparse.ArgumentParser:
    """Get the parser for profile_ensemble_preprocessing.py."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mix",
        default=MIX_NUMERIC,
        choices=[MIX_NUMERIC, MIX_HALF_STRING, MIX_NUMERIC_OBJECT],
        help="Column make-up, generated by the benchmark's own generators and then "
        "cleaned, which also decide the default shape. The all-numeric table is the "
        "one the 42.64 GB transient was measured on, so it is the default here.",
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
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Ensemble members to fit. The profiled run used one; more of them repeat "
        "the same per-member stages and retain a preprocessed table each.",
    )
    parser.add_argument(
        "--feature-cap",
        type=int,
        default=None,
        help="max_features_per_estimator. Defaults to the column count, as the "
        "profiler's does, which leaves feature subsampling switched off.",
    )
    parser.add_argument(
        "--no-gpu-preprocessing",
        action="store_true",
        help="Build CPU-only pipelines. The profiled run had GPU preprocessing on, "
        "which defers the squashing scaler, the SVD, the fingerprint and the shuffle; "
        "this puts all four back on the CPU pipeline.",
    )
    parser.add_argument(
        "--parallel-mode",
        default=DEFAULT_PARALLEL_MODE,
        choices=list(PARALLEL_MODES),
        help="How joblib returns the members. 'block' is what the wrapper uses.",
    )
    parser.add_argument(
        "--n-preprocessing-jobs",
        type=int,
        default=1,
        help="joblib workers. Only 1 is useful here: other processes' stages and RSS "
        "are invisible to this one.",
    )
    parser.add_argument(
        "--no-step-detail",
        action="store_true",
        help="Skip the pipeline steps' own boundaries, leaving the pipeline and the "
        "per-member function around them.",
    )
    parser.add_argument(
        "--no-sklearn-detail",
        action="store_true",
        help="Skip the ColumnTransformer/encoder boundaries. They are where a step's "
        "peak usually lives, so this is mainly for checking they are not what perturbs "
        "it.",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=DEFAULT_MAX_STAGES,
        help="Stop recording past this many stages, so a per-column or per-member call "
        "cannot bury the report.",
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
        help="Defaults to profile_out/ensemble_preprocessing_<mix>.",
    )
    return parser


if __name__ == "__main__":
    parsed = get_parser().parse_args()
    if parsed.out_dir is None:
        parsed.out_dir = Path("profile_out") / f"ensemble_preprocessing_{parsed.mix}"
    main(parsed)
