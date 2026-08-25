# ruff: noqa: T201
#  Copyright (c) Prior Labs GmbH 2026.

r"""Regression gate for the ensemble preprocessor's transient RAM and wall time.

With `clean_data` brought down, the largest transient in the wrapper's host-RAM
profile is `TabPFNEnsemblePreprocessor.fit_transform_ensemble_members_iterator`:
42.64 GB reached inside one call, against the 10.67 GB float64 table handed to it,
on a 666,667 x 2,000 fit with one estimator. That measurement comes from the
wrapper-wide `profiler.py` in fomo-fitting's
`experimental/arthur/res-2439-reduce-the-ram-footprint-of-the-wrapper/`; this script
isolates the same call so it can be iterated on without paying for a whole
fit + predict, and gates it the way `scripts/bench_clean_data.py` gates `clean_data`.

The inputs are the ones the profiler's `fit` produces at that boundary, reconstructed
step for step rather than approximated:

* `X_train` is `clean_data`'s output for the profiled table -- so the generators are
  imported from `bench_clean_data`, and `--mix` selects the same three column make-ups
  -- or all of them at once (see that script for what each one exercises).
* `feature_schema` is the schema `clean_data` returns, not the one detection produced.
* `y_train` is label-encoded by `TabPFNLabelEncoder`, as `fit` does.
* the ensemble configs come from `generate_classification_ensemble_configs` driven by
  the profiler's 3.1_exp `InferenceConfig`, and the preprocessor is constructed with
  exactly the arguments `TabPFNClassifier.fit` passes it.

What it measures:

* **Transient RSS** -- the peak process RSS reached inside the call, minus the RSS on
  entry, sampled by a background thread so a spike that is allocated and freed between
  two boundaries is still caught. RSS rather than tracemalloc because the copies under
  suspicion are numpy buffers, invisible to Python's allocator.
* **Wall time** -- `torch.utils.benchmark`, median over `--timing-repeats` calls, after
  `--warmup-calls` untimed ones. Every clock stops after a device synchronise, and the
  device's allocation high-water mark across the call is recorded: the torch transforms
  this preprocessor configures are *built* here and *run* by the inference engine, so
  today that peak is zero, and if it ever is not the timings are still honest (see
  "Device work" below).

Both metrics, plus every ensemble member's preprocessed `X_train` and a signature of
the rest of the member, are written to a directory named after the input. On a later
run that finds a complete set of files there, the script switches to gate mode: the new
run must not use more transient RSS, must not take longer, and must produce identical
members, or it exits non-zero. Each check runs as soon as its input exists, so a
regression fails before the next (more expensive) stage is paid for. Passing overwrites
the baseline.

Smoke test locally (seconds, ~10 MB on disk; `--mix` combines with it):

    uv run scripts/bench_ensemble_preprocessing.py --small

This is pure CPU work -- the GPU-scheduled steps only run at inference time -- so the
partition to use is a high-memory CPU one. The profiled shape holds ~11 GB of input and
reaches ~55 GB of RSS, so `cpuhighmem16spot` (300 nodes at ~123 GB) is both big enough
and quick to get hold of:

    srun -p cpuhighmem16spot --mem=0 --time=01:00:00 \
        uv run scripts/bench_ensemble_preprocessing.py

Either object-array mix, at the smaller shape they force:

    srun -p cpuhighmem16spot --mem=0 --time=01:00:00 \
        uv run scripts/bench_ensemble_preprocessing.py --mix half-string

Or every mix in one command -- a child process each, tabled together at the end, so a
change is answered for on all three tables rather than the one it was written against:

    srun -p cpuhighmem16spot --mem=0 --time=02:00:00 \
        uv run scripts/bench_ensemble_preprocessing.py --mix all --reference main

Every mix in such a run is measured whether the ones before it passed or not, and one
that trips a gate reports nothing for the checks after it -- so `--tolerance 0.05` is
worth passing when what is wanted is the whole table rather than the gate. That covers
the column axis; `--environment` takes the other one, and the two multiply into a grid
of a child process per cell. `lowest` and `highest` are the two ends of the supported
dependency range, built here on first use the way CI builds its own two legs:

    srun -p cpuhighmem16spot --mem=0 --time=04:00:00 \
        uv run scripts/bench_ensemble_preprocessing.py --mix all --reference main \
            --environment lowest --environment highest

See `bench_clean_data.py`'s header for what those cost and where they are kept.

The full numeric shape writes ~11 GB per estimator per run. Pair either with
`scripts/srun_retry.py` when allocations are getting stuck CONFIGURING.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

# Same directory, and deliberately not duplicated: the input generators, the RSS
# sampler and the reference-worktree machinery must be the ones `clean_data`'s gate
# uses, or the two scripts would measure different tables and report their numbers
# against differently-built baselines.
from bench_clean_data import (
    _GB,
    MIN_USEFUL_MEDIAN_S,
    MIN_USEFUL_RSS_SAMPLES,
    MIX_ALL,
    MIX_LEVELS,
    MIX_NUMERIC,
    MIXES,
    PASSTHROUGH_INF,
    SEED,
    MemoryProfile,
    RssSampler,
    _git,
    build_feature_schema,
    change_pct,
    commit,
    compare_array,
    current_rss_bytes,
    describe_environment,
    describe_input_size,
    discard,
    environment_drift,
    fail,
    fingerprint,
    generate_input,
    prepare_reference_worktree,
    regression,
    release_free_memory,
    resolve_reference,
    resolve_shape,
    run_child,
    run_grid,
    timing_as_dict,
    train_rows_for,
    verify_reference_ran_the_reference,
)
from torch.utils.benchmark import Measurement, Timer

from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing.clean import clean_data
from tabpfn.preprocessing.configs import FeatureSubsamplingMethod, PreprocessorConfig
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.ensemble import (
    TabPFNEnsemblePreprocessor,
    generate_classification_ensemble_configs,
)
from tabpfn.preprocessing.label_encoder import TabPFNLabelEncoder

if TYPE_CHECKING:
    from collections.abc import Callable

    from tabpfn.preprocessing.configs import ClassifierEnsembleConfig
    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.ensemble import TabPFNEnsembleMember

# ---------------------------------------------------------------------------
# What the profiler's `fit` hands the ensemble preprocessor
# ---------------------------------------------------------------------------
#
# The shape, mix and dtype come from `bench_clean_data` (imported above). Everything
# else here is the profiler's estimator configuration, reproduced so the measured call
# sees the same configs, the same pipelines and the same seeds.

# `TabPFNClassifier(random_state=0)`. `fit` derives `static_seed` from it via
# `infer_random_state`, which for an int returns the int itself, and passes that same
# value both to the config generation and to the preprocessor.
RANDOM_STATE = 0

# The profiler builds one random-weight architecture, so `num_models=1`, and asks for
# one estimator with `auto_scale_n_estimators=False` -- hence no coverage scaling here.
NUM_MODELS = 1
DEFAULT_N_ESTIMATORS = 1

# The profiler's labels: three balanced classes.
N_CLASSES = 3

# `fit_mode="fit_preprocessors"`, so the GPU-side fitted cache is not kept, and
# `n_preprocessing_jobs` is left at the constructor default.
KEEP_FITTED_CACHE = False
DEFAULT_N_PREPROCESSING_JOBS = 1

# `fit_transform_ensemble_members` -- the caller in `create_inference_engine` -- passes
# "block", i.e. joblib returns every member before the first one is yielded.
DEFAULT_PARALLEL_MODE = "block"
PARALLEL_MODES = ("block", "as-ready", "in-order")

ESTIMATOR_TYPE: Literal["classifier"] = "classifier"

# Bumped whenever the meaning of a recorded metric, or the shape of the recorded input
# description, changes -- so an old baseline is rejected rather than silently compared
# against.
SCHEMA_VERSION = 1

METRICS_FILE = "metrics.json"
MEMBERS_FILE = "members.pkl"


def member_array_name(index: int) -> str:
    """File holding one ensemble member's preprocessed training matrix."""
    return f"member{index:02d}_X_train.npy"


# ---------------------------------------------------------------------------
# The inference config the profiled run was configured with
# ---------------------------------------------------------------------------


def build_inference_config(
    cols: int,
    feature_cap: int,
    *,
    enable_gpu_preprocessing: bool,
) -> InferenceConfig:
    """The 3.1_exp inference config, as the wrapper profiler builds it.

    Vendored from that profiler, which in turn mirrors `_3p1_exp_config` / `_v3_config`
    in `training.inference_config_presets` -- kept in sync by hand so neither has to
    depend on that package. Every field the measured call depends on is read back off
    this object rather than restated, so there is one source of truth for what the
    ensemble sees: the preprocessor configs, the shift and fingerprint settings, the
    subsampling method and the outlier-removal std.
    """
    return InferenceConfig(
        MAX_UNIQUE_FOR_CATEGORICAL_FEATURES=30,
        MIN_UNIQUE_FOR_NUMERICAL_FEATURES=4,
        MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE=100,
        OUTLIER_REMOVAL_STD="auto",
        FEATURE_SHIFT_METHOD="shuffle",
        CLASS_SHIFT_METHOD="shuffle",
        FINGERPRINT_FEATURE=True,
        POLYNOMIAL_FEATURES="no",
        SUBSAMPLE_SAMPLES=None,
        ENABLE_GPU_PREPROCESSING=enable_gpu_preprocessing,
        FEATURE_SUBSAMPLING_METHOD="auto",
        FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT=50,
        FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT=150,
        PREPROCESS_TRANSFORMS=[
            PreprocessorConfig(
                name="squashing_scaler_max10",
                append_original=False,
                categorical_name="ordinal_very_common_categories_shuffled",
                global_transformer_name="svd_quarter_components",
                max_features_per_estimator=feature_cap,
            ),
            PreprocessorConfig(
                name="none",
                append_original=False,
                categorical_name="numeric",
                global_transformer_name=None,
                max_features_per_estimator=feature_cap,
            ),
        ],
        REGRESSION_Y_PREPROCESS_TRANSFORMS=(None, "safepower"),
        USE_SKLEARN_16_DECIMAL_PRECISION=False,
        MAX_NUMBER_OF_CLASSES=160,
        MAX_NUMBER_OF_FEATURES=max(cols, feature_cap) + 1,
        MAX_NUMBER_OF_SAMPLES=2**62,
        FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM=True,
        _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD=None,
        _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD=12.0,
    )


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def generate_labels(rows: int) -> np.ndarray:
    """The training labels, three balanced classes over `rows` rows.

    Not bit-identical to the profiler's `y_train`: that one is drawn third, after the
    test matrix, so reproducing its stream would mean generating (and holding) another
    2.7 GB of test features. Nothing on the measured path depends on the values --
    `subsample_samples` is None, so there is no stratified row subsampling, and the
    labels are only label-encoded and class-permuted -- so an independent draw of the
    same distribution measures the same work.
    """
    return np.random.default_rng(SEED).integers(0, N_CLASSES, size=rows)


@dataclasses.dataclass
class EnsembleInputs:
    """Everything `fit` has ready by the time it reaches the measured call."""

    X_train: np.ndarray
    y_train: np.ndarray
    feature_schema: FeatureSchema
    configs: list[ClassifierEnsembleConfig]
    inference_config: InferenceConfig
    n_classes: int
    raw_fingerprint: str
    cleaned_fingerprint: str
    n_categorical: int


def build_ensemble_inputs(spec: dict[str, Any]) -> EnsembleInputs:
    """Reproduce `fit`'s state at the point it calls the ensemble preprocessor.

    `ensure_compatible_fit_inputs` is skipped for the same reason
    `bench_clean_data` skips it: with `dtype=None` sklearn's `check_array` passes a
    C-contiguous float array (or an object array) straight through, and the labels
    generated here are already a 1-d numpy integer array.
    """
    X_raw = generate_input(spec)
    raw_fingerprint = fingerprint(X_raw)
    detected_schema = build_feature_schema(X_raw)
    X_train, _ordinal_encoder, feature_schema = clean_data(
        X=X_raw,
        feature_schema=detected_schema,
        passthrough_inf=PASSTHROUGH_INF,
    )
    # The raw table is dead once cleaned, and at the profiled shape it is 5.3 GB of
    # the RSS the measured call would otherwise start from.
    del X_raw, detected_schema
    gc.collect()

    inference_config = build_inference_config(
        spec["cols"],
        spec["feature_cap"],
        enable_gpu_preprocessing=spec["enable_gpu_preprocessing"],
    )
    y_train, label_metadata = TabPFNLabelEncoder().fit_transform(
        y=generate_labels(spec["rows"]),
        max_num_classes=inference_config.MAX_NUMBER_OF_CLASSES,
    )
    configs = generate_classification_ensemble_configs(
        num_estimators=spec["n_estimators"],
        add_fingerprint_feature=inference_config.FINGERPRINT_FEATURE,
        feature_shift_decoder=inference_config.FEATURE_SHIFT_METHOD,
        polynomial_features=inference_config.POLYNOMIAL_FEATURES,
        preprocessor_configs=inference_config.PREPROCESS_TRANSFORMS,
        class_shift_method=inference_config.CLASS_SHIFT_METHOD,
        n_classes=label_metadata.n_classes,
        random_state=RANDOM_STATE,
        num_models=NUM_MODELS,
        outlier_removal_std=inference_config.get_resolved_outlier_removal_std(
            estimator_type=ESTIMATOR_TYPE
        ),
        passthrough_inf=inference_config.PASSTHROUGH_INF,
    )
    return EnsembleInputs(
        X_train=X_train,
        y_train=y_train,
        feature_schema=feature_schema,
        configs=configs,
        inference_config=inference_config,
        n_classes=label_metadata.n_classes,
        raw_fingerprint=raw_fingerprint,
        cleaned_fingerprint=fingerprint(X_train),
        n_categorical=len(feature_schema.indices_for(FeatureModality.CATEGORICAL)),
    )


def build_preprocessor(
    inputs: EnsembleInputs,
    spec: dict[str, Any],
) -> TabPFNEnsemblePreprocessor:
    """Construct the preprocessor exactly as `TabPFNClassifier.fit` does.

    Kept outside every measurement: `__init__` is where the pipelines and the
    subsample indices are built, and the wrapper profile timed it at 5 ms against the
    42 s of the call this script is about.
    """
    config = inputs.inference_config
    return TabPFNEnsemblePreprocessor(
        configs=inputs.configs,
        n_samples=inputs.X_train.shape[0],
        feature_schema=inputs.feature_schema,
        random_state=RANDOM_STATE,
        n_preprocessing_jobs=spec["n_preprocessing_jobs"],
        keep_fitted_cache=KEEP_FITTED_CACHE,
        enable_gpu_preprocessing=config.ENABLE_GPU_PREPROCESSING,
        feature_subsampling_method=FeatureSubsamplingMethod(
            config.FEATURE_SUBSAMPLING_METHOD
        ),
        constant_feature_count=config.FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT,
        subsample_samples=config.SUBSAMPLE_SAMPLES,
        importance_top_k_count=config.FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT,
        X_train=inputs.X_train,
        y_train=inputs.y_train,
        task_type=ESTIMATOR_TYPE,
    )


# ---------------------------------------------------------------------------
# Device work
# ---------------------------------------------------------------------------
#
# The ensemble preprocessor does have torch transforms -- the quantile/squashing
# scaler, the SVD, the fingerprint and the shuffle -- but with
# ENABLE_GPU_PREPROCESSING on `create_preprocessing_pipeline` *removes* those four from
# the CPU pipeline and `create_gpu_preprocessing_pipeline` only *builds* the
# `TorchPreprocessingPipeline` that replaces them. The measured call stores that
# pipeline on each member and returns; the inference engine runs it later, through
# `_maybe_run_gpu_preprocessing`. The CPU steps do reach for torch, but only in their
# `isinstance(X, torch.Tensor)` branches, and the table here is a numpy array. So no
# kernel is launched inside the measured call -- which is also why these runs work at
# all on a host with no CUDA runtime.
#
# That is an invariant, not a law, so it is both guarded and measured: every clock is
# stopped after a synchronise, and the device's allocation high-water mark is recorded
# across the call. If work ever moves in here, the timings stay honest and
# `device.peak_allocated_bytes` says so instead of the report quietly meaning something
# else.


def _no_synchronize() -> None:
    """Nothing to wait for: there is no device to queue work on."""


def _resolve_device_sync() -> Callable[[], None]:
    """How to wait for queued device work before a clock is read."""
    if torch.cuda.is_available():
        return torch.cuda.synchronize
    return _no_synchronize


synchronize_device = _resolve_device_sync()


def reset_device_activity() -> None:
    """Settle the device and zero its allocation high-water mark.

    Called before the entry RSS reading, not after, for a reason beyond the high-water
    mark: this is where torch creates its CUDA context if nothing has touched the device
    yet, and a context costs host RSS. Paid here it is part of the baseline; paid on the
    first synchronise inside the measured call, it would read as that call's transient.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def device_activity() -> dict[str, Any]:
    """Whether the device was touched since the last reset, and how much of it.

    Recorded rather than asserted: a non-zero peak is not a failure, it is the signal
    that this call now does device work and that its wall time is only meaningful
    because of the synchronise above.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False, "peak_allocated_bytes": None}
    return {
        "cuda_available": True,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_memory(
    preprocessor: TabPFNEnsemblePreprocessor,
    inputs: EnsembleInputs,
    parallel_mode: str,
    interval_s: float,
) -> tuple[list[TabPFNEnsembleMember], MemoryProfile]:
    """Consume the iterator once, sampling RSS throughout, and keep the members.

    Consumed into a list because that is what `fit_transform_ensemble_members` does,
    and so what the profiled run measured: the members are all live at once by the
    time the call returns.

    Deliberately un-warmed, and deliberately before the timing pass: once a call has
    run, the allocator holds on to the arenas it freed, so a second call can reach the
    same peak without growing RSS at all -- which would read as a phantom improvement.
    """
    gc.collect()
    release_free_memory()
    reset_device_activity()
    sampler = RssSampler(interval_s)
    rss_in = current_rss_bytes()
    sampler.start()
    start = time.perf_counter()
    try:
        members = list(
            preprocessor.fit_transform_ensemble_members_iterator(
                X_train=inputs.X_train,
                y_train=inputs.y_train,
                parallel_mode=parallel_mode,  # type: ignore[arg-type]
            )
        )
        synchronize_device()
    finally:
        end = time.perf_counter()
        sampler.stop()
    rss_out = current_rss_bytes()
    profile = MemoryProfile(
        rss_in_bytes=rss_in,
        # A sampled peak can only err downwards, so floor it at the two readings that
        # are known to be real.
        peak_rss_bytes=max(sampler.peak_between(start, end), rss_in, rss_out),
        rss_out_bytes=rss_out,
        n_samples=len(sampler.samples),
        wall_s=end - start,
    )
    return members, profile


def measure_time(
    preprocessor: TabPFNEnsemblePreprocessor,
    inputs: EnsembleInputs,
    parallel_mode: str,
    repeats: int,
    warmup_calls: int,
) -> Measurement:
    """Median-of-`repeats` wall time for one full consumption of the iterator.

    Every call measures the same work. The call does not mutate its input --
    `_fit_preprocessing_one` copies `X_train` before touching it -- and re-fitting the
    same preprocessor is idempotent, because each pipeline's seed is fixed in
    `TabPFNEnsemblePreprocessor.__init__` and every step re-creates its transformer
    from that seed on each `fit_transform`.

    The warmup is what makes the repeats comparable: it settles the allocator, the
    page cache and the BLAS/OpenMP thread pools, all of which the first call pays for.
    `Timer.timeit` would run one for us, but it re-runs it on every call
    (`max(number // 100, 2)` executions each time), which at the default shape is
    minutes per repeat -- so warm up once here and time the repeats through the same
    inner timer afterwards.

    The synchronise is inside the timed statement, not around it: outside, it would
    fall between two repeats and leave each one crediting its device work to its
    successor.
    """
    timer = Timer(
        stmt=(
            "list(iterator(X_train=X, y_train=y, parallel_mode=parallel_mode))\n"
            "synchronize()"
        ),
        globals={
            "iterator": preprocessor.fit_transform_ensemble_members_iterator,
            "X": inputs.X_train,
            "y": inputs.y_train,
            "parallel_mode": parallel_mode,
            "synchronize": synchronize_device,
        },
        num_threads=torch.get_num_threads(),
        label="fit_transform_ensemble_members_iterator",
    )
    if warmup_calls:
        timer._timeit(number=warmup_calls)
    raw_times = [timer._timeit(number=1) for _ in range(repeats)]
    return Measurement(
        number_per_run=1,
        raw_times=raw_times,
        task_spec=timer._task_spec,
    )


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


def _comparable(value: Any) -> Any:
    """A value that `==` can be trusted on, for the member signature.

    Arrays are the reason this exists: comparing two dicts that hold ndarrays raises
    rather than answering, so they become lists. Anything not JSON-shaped falls back
    to its repr, which is enough to notice it changed.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def member_record(member: TabPFNEnsembleMember) -> dict[str, Any]:
    """Everything about one member except its (separately stored) `X_train`.

    The pipelines themselves are summarised by their step classes rather than
    pickled: sklearn estimators define no `__eq__`, and what a changed step list would
    mean -- a different sequence of transforms -- is exactly what needs catching.
    Deliberately plain dicts and arrays, so the pickle does not depend on this module
    being imported under any particular name.
    """
    return {
        "signature": {
            "config": _comparable(dataclasses.asdict(member.config)),
            "cpu_steps": [
                type(step).__name__ for step, _ in member.cpu_preprocessor.steps
            ],
            "gpu_steps": _gpu_step_names(member),
            "X_train_shape": list(member.X_train.shape),
            "X_train_dtype": str(member.X_train.dtype),
        },
        "y_train": np.asarray(member.y_train),
        "feature_schema": member.feature_schema,
        "feature_indices": member.feature_indices,
    }


def _gpu_step_names(member: TabPFNEnsembleMember) -> list[str] | None:
    """Step classes of the member's torch pipeline, None when it has none."""
    pipeline = member.gpu_preprocessor
    if pipeline is None:
        return None
    return [type(step).__name__ for step, _ in pipeline.steps]


def compare_members(
    out_dir: Path,
    members: list[TabPFNEnsembleMember],
    chunk_rows: int | None,
) -> list[str]:
    """Check the members against the recorded ones, cheapest comparison first."""
    with (out_dir / MEMBERS_FILE).open("rb") as handle:
        # Trusted input: written by a previous run of this script.
        recorded = pickle.load(handle)  # noqa: S301

    if len(recorded) != len(members):
        return [f"produced {len(members)} ensemble members, recorded {len(recorded)}"]

    problems: list[str] = []
    for index, (was, member) in enumerate(zip(recorded, members, strict=True)):
        now = member_record(member)
        if was["signature"] != now["signature"]:
            problems += [
                f"member {index} {field} differs from the recorded one: "
                f"{now['signature'][field]!r} != {was['signature'][field]!r}"
                for field in was["signature"]
                if was["signature"][field] != now["signature"][field]
            ]
        if not np.array_equal(was["y_train"], now["y_train"]):
            problems.append(f"member {index} y_train differs from the recorded one")
        if was["feature_schema"] != now["feature_schema"]:
            problems.append(
                f"member {index} feature schema differs from the recorded one"
            )
        if not _indices_equal(was["feature_indices"], now["feature_indices"]):
            problems.append(
                f"member {index} feature indices differ from the recorded ones"
            )
        # Left last, and only when the rest matches: it is the one comparison that
        # reads gigabytes off disk.
        if not problems:
            problems += [
                f"member {index}: {problem}"
                for problem in compare_array(
                    out_dir / member_array_name(index),
                    np.asarray(member.X_train),
                    chunk_rows,
                )
            ]
        if problems:
            break

    return problems


def _indices_equal(was: np.ndarray | None, now: np.ndarray | None) -> bool:
    """Whether two per-member feature-index selections agree, None included."""
    if was is None or now is None:
        return was is None and now is None
    return np.array_equal(was, now)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def spec_dir_name(spec: dict[str, Any]) -> str:
    """Directory name identifying one run, so each variant keeps its own baseline.

    Everything that changes *what* is measured has to be in here, not only in the guard
    `load_baseline` applies: unlike `clean_data`'s gate, where the spec beyond the shape
    is all constants, several of these are flags. Two variants sharing a directory would
    reject each other's baseline instead of each keeping one. The defaults are left out
    of the name so the profiled configuration stays legible.
    """
    name = (
        f"rows{spec['rows']}_cols{spec['cols']}_{spec['dtype']}_{spec['mix']}"
        f"_est{spec['n_estimators']}"
        f"_{'gpu' if spec['enable_gpu_preprocessing'] else 'cpu'}"
    )
    if spec["parallel_mode"] != DEFAULT_PARALLEL_MODE:
        name += f"_{spec['parallel_mode']}"
    if spec["feature_cap"] != spec["cols"]:
        name += f"_cap{spec['feature_cap']}"
    if spec["n_preprocessing_jobs"] != DEFAULT_N_PREPROCESSING_JOBS:
        name += f"_jobs{spec['n_preprocessing_jobs']}"
    return name


def baseline_paths(out_dir: Path, n_estimators: int) -> dict[str, Path]:
    """The files that together make up a baseline."""
    paths = {
        "metrics": out_dir / METRICS_FILE,
        "members": out_dir / MEMBERS_FILE,
    }
    for index in range(n_estimators):
        paths[f"member{index}"] = out_dir / member_array_name(index)
    return paths


def write_outputs(
    out_dir: Path,
    members: list[TabPFNEnsembleMember],
) -> list[tuple[Path, Path]]:
    """Write the members beside their final names, as `.tmp` files.

    Returned as (temporary, final) pairs: nothing replaces a baseline until every
    check has passed, and the arrays are large enough that a half-written file would
    be an expensive thing to leave behind.

    `np.save` for the matrices -- a straight buffer write, and it can be re-read
    lazily with `mmap_mode` so a later comparison need not hold both copies -- and
    pickle protocol 5 for the rest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path]] = []

    for index, member in enumerate(members):
        final = out_dir / member_array_name(index)
        temporary = final.with_name(final.name + ".tmp")
        # Written through a handle: `np.save` appends `.npy` to any path that does not
        # already end in it, which would defeat the rename.
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(member.X_train))
        pending.append((temporary, final))

    members_final = out_dir / MEMBERS_FILE
    members_tmp = members_final.with_name(MEMBERS_FILE + ".tmp")
    with members_tmp.open("wb") as handle:
        pickle.dump(
            [member_record(member) for member in members],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    pending.append((members_tmp, members_final))

    return pending


# ---------------------------------------------------------------------------
# A/B against a git reference
# ---------------------------------------------------------------------------
#
# The reference-worktree machinery is `bench_clean_data`'s, imported wholesale; only
# the three functions that name this script and its baseline layout live here. See that
# module for why the reference package is selected per child process via `PYTHONPATH`
# rather than by importing it under a second name or by `uv run --project`.


def child_arguments(args: argparse.Namespace, spec: dict[str, Any]) -> list[str]:
    """The measurement arguments both sides of the comparison are run with.

    Built from the resolved spec rather than forwarded from argv, so the `--small` and
    per-mix shape defaults are already applied and the two children cannot disagree
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
        "--n-estimators",
        str(spec["n_estimators"]),
        "--feature-cap",
        str(spec["feature_cap"]),
        "--parallel-mode",
        spec["parallel_mode"],
        "--n-preprocessing-jobs",
        str(spec["n_preprocessing_jobs"]),
        "--timing-repeats",
        str(args.timing_repeats),
        "--warmup-calls",
        str(args.warmup_calls),
        "--sample-interval-ms",
        str(args.sample_interval_ms),
    ]
    if not spec["enable_gpu_preprocessing"]:
        forwarded.append("--no-gpu-preprocessing")
    if args.chunk_rows is not None:
        forwarded += ["--chunk-rows", str(args.chunk_rows)]
    return forwarded


def read_recorded_environment(run_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """The `environment` block a side of the comparison just recorded."""
    metrics_path = run_dir / spec_dir_name(spec) / METRICS_FILE
    return json.loads(metrics_path.read_text())["environment"]


def run_reference_comparison(args: argparse.Namespace) -> int:
    """Measure a reference commit, then the working tree, and gate on the pair."""
    spec = resolve_input_spec(args)
    sha = resolve_reference(args.reference)
    worktree = prepare_reference_worktree(
        sha, args.reference_root, refresh=args.refresh_reference
    )
    print(f"Reference: {_git('log', '-1', '--format=%h %s', sha)}")
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
# Reporting
# ---------------------------------------------------------------------------


def describe_device_activity(device: dict[str, Any]) -> str:
    """One line on what the measured call did to the device, if anything."""
    if not device["cuda_available"]:
        return (
            "no CUDA here, so the torch transforms could not have run even if the "
            "call queued them"
        )
    peak = device["peak_allocated_bytes"]
    if not peak:
        return (
            "CUDA present and synchronised, but the call allocated nothing on it -- "
            "the torch pipeline it builds is run later, by the inference engine"
        )
    return (
        f"the call allocated {peak / _GB:.2f} GB on the device, so torch work now runs "
        "inside it. The timings are synchronised and so still comparable, but the RSS "
        "above counts host memory only"
    )


def print_pass_summary(
    recorded: dict[str, Any],
    memory: MemoryProfile,
    timing: dict[str, Any],
) -> None:
    """Report both metrics against the baseline that is about to be replaced."""
    old_rss = recorded["memory"]["transient_rss_bytes"]
    old_median = recorded["timing"]["median_s"]
    print("\n" + "=" * 79)
    print("PASS: not worse than the baseline on either metric, members identical.")
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
    """Everything identifying the input, for the directory name and the guard.

    `resolve_shape` is `bench_clean_data`'s, so `--rows/--cols/--small/--mix` pick the
    same table here as they do there.
    """
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
        "n_estimators": args.n_estimators,
        # Defaults to every column, as the profiler's does, which leaves feature
        # subsampling switched off.
        "feature_cap": cols if args.feature_cap is None else args.feature_cap,
        "enable_gpu_preprocessing": not args.no_gpu_preprocessing,
        "parallel_mode": args.parallel_mode,
        "n_preprocessing_jobs": args.n_preprocessing_jobs,
        "n_classes": N_CLASSES,
        "random_state": RANDOM_STATE,
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


def check_input_fingerprints(
    recorded: dict[str, Any],
    inputs: EnsembleInputs,
) -> None:
    """Refuse to compare two runs that measured different tables.

    The cleaned table is the one that matters: it is the argument the measured call
    receives. It can drift without the generated table drifting at all -- a change to
    `clean_data` is enough -- which under `--reference` means the two sides are not
    measuring the same input and no difference between them can be attributed.
    """
    was = recorded["input"]
    problems = []
    if was["raw_fingerprint"] != inputs.raw_fingerprint:
        problems.append(
            "the generated table differs from the one the baseline was recorded on "
            f"({inputs.raw_fingerprint[:16]} != {was['raw_fingerprint'][:16]}); "
            "comparing would be meaningless"
        )
    if was["cleaned_fingerprint"] != inputs.cleaned_fingerprint:
        problems.append(
            "clean_data's output -- the argument the measured call receives -- differs "
            f"from the recorded one ({inputs.cleaned_fingerprint[:16]} != "
            f"{was['cleaned_fingerprint'][:16]}), so the two runs did not measure the "
            "same input"
        )
    if problems:
        fail(problems)


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


def print_inputs(inputs: EnsembleInputs, spec: dict[str, Any]) -> None:
    """What the measured call is about to be handed."""
    X = inputs.X_train
    print(
        f"\nInput to fit_transform_ensemble_members_iterator: {X.shape} {X.dtype} "
        f"({describe_input_size(X)})"
        f"\n  mix: {spec['mix']}, {inputs.n_categorical} of "
        f"{inputs.feature_schema.num_columns} columns categorical after cleaning"
        f"\n  {spec['n_estimators']} estimator(s), "
        f"{'GPU-scheduled' if spec['enable_gpu_preprocessing'] else 'CPU-only'} "
        f"pipelines, parallel_mode={spec['parallel_mode']}, "
        f"n_preprocessing_jobs={spec['n_preprocessing_jobs']}"
        f"\n  fingerprints: generated {inputs.raw_fingerprint[:16]}, "
        f"cleaned {inputs.cleaned_fingerprint[:16]}"
    )
    if spec["n_preprocessing_jobs"] != 1:
        print(
            f"WARNING: n_preprocessing_jobs={spec['n_preprocessing_jobs']} dispatches "
            "members to worker processes, whose RSS this process cannot see. The "
            "transient below then measures only what stays in this process."
        )


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
    """Measure the ensemble preprocessor, gating it against a reference or baseline."""
    delegated = delegated_exit_code(args)
    if delegated is not None:
        sys.exit(delegated)

    spec = resolve_input_spec(args)
    out_dir = args.out_root / spec_dir_name(spec)
    paths = baseline_paths(out_dir, spec["n_estimators"])
    # A baseline is only a baseline once every file is there; a partial one (an
    # interrupted run, a hand-deleted array) is re-recorded rather than trusted.
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

    inputs = build_ensemble_inputs(spec)
    print_inputs(inputs, spec)
    if recorded is not None:
        check_input_fingerprints(recorded, inputs)

    preprocessor = build_preprocessor(inputs, spec)
    members, memory = measure_memory(
        preprocessor, inputs, spec["parallel_mode"], args.sample_interval_ms / 1000
    )
    device = device_activity()
    print(
        f"\nTransient RSS: {memory.transient_bytes / _GB:.2f} GB "
        f"(entry {memory.rss_in_bytes / _GB:.2f} GB, peak "
        f"{memory.peak_rss_bytes / _GB:.2f} GB, retained "
        f"{memory.retained_bytes / _GB:.2f} GB, {memory.n_samples} RSS samples, "
        f"{memory.wall_s:.3f} s cold)"
    )
    print(f"Device: {describe_device_activity(device)}")
    if memory.n_samples < MIN_USEFUL_RSS_SAMPLES:
        print(
            f"WARNING: only {memory.n_samples} RSS sample(s) landed inside a "
            f"{memory.wall_s * 1000:.0f} ms call, so the transient above is the "
            "entry/exit readings rather than a sampled peak. Lower "
            "--sample-interval-ms, or measure a bigger shape."
        )

    if recorded is not None:
        check_memory(recorded, memory, args.tolerance)
        problems = compare_members(out_dir, members, args.chunk_rows)
        if problems:
            fail(problems)
        print(f"All {len(members)} member(s) are identical to the recorded ones.")

    pending = write_outputs(out_dir, members)
    # Drop the members before timing, so the repeats run at the footprint the measured
    # call ran at rather than carrying an extra copy of every one of them.
    del members
    gc.collect()

    try:
        measurement = measure_time(
            preprocessor,
            inputs,
            spec["parallel_mode"],
            args.timing_repeats,
            args.warmup_calls,
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
                    "input": {
                        **spec,
                        "raw_fingerprint": inputs.raw_fingerprint,
                        "cleaned_fingerprint": inputs.cleaned_fingerprint,
                        "n_categorical_after_cleaning": inputs.n_categorical,
                    },
                    "environment": environment,
                    "config": {
                        "sample_interval_ms": args.sample_interval_ms,
                        "tolerance": args.tolerance,
                    },
                    "memory": memory.as_dict(),
                    "device": device,
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

    print(
        f"\nWrote {METRICS_FILE}, {MEMBERS_FILE} and the member matrices to {out_dir}"
    )


def get_parser() -> argparse.ArgumentParser:
    """Get the parser for bench_ensemble_preprocessing.py."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Rows the profiler generates, of which the train split reaches fit and so "
        "the ensemble preprocessor. Same meaning and defaults as in "
        "bench_clean_data.py.",
    )
    parser.add_argument("--cols", type=int, default=None)
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
        help="Column make-up, generated by bench_clean_data's generators and then "
        "cleaned, which is what the measured call receives. 'numeric' is the profiled "
        "float table, and the case the 42.64 GB transient was measured on. The other "
        "two leave categorical columns in the cleaned schema, so the encoder-fed path "
        "through the ensemble pipelines is exercised. Each keeps its own baseline. "
        "'all' runs the three of them in sequence, a child process each, and prints "
        "what they reported as one table.",
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
        help="dtype of the generated table, before cleaning casts it. The profiled run "
        "used float32.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Ensemble members to fit. The profiled run used one. Every member holds "
        "its own preprocessed copy of the table, so this multiplies both the retained "
        "RSS and what a run writes to disk.",
    )
    parser.add_argument(
        "--feature-cap",
        type=int,
        default=None,
        help="max_features_per_estimator. Defaults to the column count, as the "
        "profiler's does, which leaves feature subsampling switched off. A smaller "
        "value turns it on, and with the 'auto' method on a table this large that "
        "means LightGBM importance fits inside the preprocessor's constructor.",
    )
    parser.add_argument(
        "--no-gpu-preprocessing",
        action="store_true",
        help="Build CPU-only pipelines (ENABLE_GPU_PREPROCESSING=False). The profiled "
        "run had it on, which moves the quantile/squashing transform, the SVD, the "
        "fingerprint and the shuffle off the CPU pipeline; turning it off puts all "
        "four back, so this measures the legacy path rather than the profiled one.",
    )
    parser.add_argument(
        "--parallel-mode",
        default=DEFAULT_PARALLEL_MODE,
        choices=list(PARALLEL_MODES),
        help="How joblib returns the members. 'block' is what "
        "fit_transform_ensemble_members -- the caller in create_inference_engine -- "
        "passes.",
    )
    parser.add_argument(
        "--n-preprocessing-jobs",
        type=int,
        default=DEFAULT_N_PREPROCESSING_JOBS,
        help="joblib workers. Anything but 1 dispatches members to other processes, "
        "whose RSS this process cannot see, which the run warns about.",
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
        help="Rows per block when comparing a member's matrix with the recorded one. "
        "Defaults to a fixed cell budget for the column count.",
    )
    parser.add_argument(
        "--reference",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="REV",
        help="Compare against a commit/branch/tag instead of a recorded baseline: it "
        "is checked out in a cached worktree, measured with its own tabpfn on its own "
        "PYTHONPATH, and then the working tree is measured and gated against it -- "
        "both on this machine, minutes apart. Defaults to HEAD when given without a "
        "value, i.e. 'what do my uncommitted changes do'. Note that both sides also "
        "run their own clean_data, so a reference whose cleaning differs is rejected "
        "rather than compared.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("bench_out/references"),
        help="Where reference worktrees are cached, one per commit. Shared with "
        "bench_clean_data.py, so a worktree either script created is reused by both.",
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
        default=Path("bench_out/ensemble_preprocessing"),
        help="Baselines go in a subdirectory of this, named after the input.",
    )
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
