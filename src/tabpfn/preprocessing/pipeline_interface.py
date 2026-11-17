#  Copyright (c) Prior Labs GmbH 2026.

"""Interfaces for creating preprocessing pipelines."""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import os
import time
from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias

import numpy as np
import torch

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.utils import as_tensor_to_read

_STEP_OWNS_INPUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tabpfn_step_owns_input", default=False
)


PERMUTE_BLOCK_BYTES = 1 << 26
"""Bytes of rows `permute_columns_in_place` moves at a time."""

GATHER_BLOCK_BYTES = 1 << 26
"""Bytes of columns `gather_columns` reads at a time."""


def gather_columns(X: np.ndarray, indices: np.ndarray, dtype: Any) -> np.ndarray:
    """`X[:, indices].astype(dtype, order="C")`, with no full-size intermediate.

    The columns are gathered a block at a time straight into the C-ordered result.
    """
    out = np.empty((X.shape[0], len(indices)), dtype=dtype, order="C")
    width = max(1, GATHER_BLOCK_BYTES // max(1, X.shape[0] * X.itemsize))
    for start in range(0, len(indices), width):
        out[:, start : start + width] = X[:, indices[start : start + width]]
    return out


def step_owns_input() -> bool:
    """Whether the running step may write into the array it was handed.

    True only inside `PreprocessingPipeline`, for an array nobody outside the
    pipeline holds.
    """
    return _STEP_OWNS_INPUT.get()


def permute_columns_in_place(X: np.ndarray, order: list[int]) -> np.ndarray:
    """`X[:, order]` for a permutation `order`, written into `X` a block of rows at a
    time; returns `X`.
    """
    if not X.flags.c_contiguous:
        return X[:, order]
    index = np.asarray(order, dtype=np.intp)
    rows = max(1, PERMUTE_BLOCK_BYTES // max(1, X.shape[1] * X.itemsize))
    for start in range(0, X.shape[0], rows):
        block = X[start : start + rows]
        block[:] = block[:, index]
    return X


StepWithModalities: TypeAlias = tuple["PreprocessingStep", set[FeatureModality]]

InfMasks: TypeAlias = dict[str, np.ndarray | torch.Tensor]
"""Mapping from an *input* feature name to a 1d array recording that feature's
+/-inf values (0 where finite).  Used to pass infinities through preprocessing:
they are NaN'd before the steps run and written back afterwards."""


def _extract_inf_masks(
    X: np.ndarray | torch.Tensor,
    feature_schema: FeatureSchema,
) -> InfMasks:
    """Record per-feature +/-inf values and replace them with NaN in ``X``.

    Returns a ``{feature_name: signed-inf column}`` mapping for the features that
    contained infinities (empty when ``X`` is finite, leaving ``X`` untouched).
    ``X`` is mutated in place. The scan reads blocks of rows on threads, so no
    table-sized boolean mask is held.
    """
    xp = torch if isinstance(X, torch.Tensor) else np
    (has_inf,) = any_per_column(X, [xp.isinf])
    masks: InfMasks = {}
    for idx, feat in enumerate(feature_schema.features):
        if has_inf[idx]:
            column = X[:, idx]
            col_inf = xp.isinf(column)
            masks[feat.name] = xp.where(col_inf, column, 0)
            column[col_inf] = float("nan")
    return masks


ROW_SCAN_BLOCK_BYTES = 1 << 26
"""Bytes of rows `any_per_column` tests at a time on one thread."""

ROW_SCAN_MAX_THREADS = 16
"""Threads `any_per_column` runs at most."""


def _usable_cpus() -> int:
    """The CPUs this process may run on."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclasses.dataclass(frozen=True, eq=False)
class SharedColumnsTable:
    """A 2-D table whose columns are columns of `base`, but for a few of its own.

    Column `j` is `base[:, sources[j]]` where `sources[j] >= 0`, and
    `own[:, -1 - sources[j]]` otherwise. The ensemble members preprocessed from one
    input share it as `base` and each hold only the columns they rewrote; tables
    made from one another with `take` share `own` too.

    Rows come out as a new array, or as a view of `base` while the table is `base`
    itself; `np.asarray` builds the whole table.
    """

    base: np.ndarray
    sources: np.ndarray
    own: np.ndarray

    @classmethod
    def of(cls, X: np.ndarray) -> SharedColumnsTable:
        """`X` itself, as a table."""
        return cls(X, np.arange(X.shape[1]), np.empty((X.shape[0], 0), X.dtype))

    @property
    def shape(self) -> tuple[int, int]:
        """Rows and columns."""
        return self.base.shape[0], len(self.sources)

    @property
    def dtype(self) -> np.dtype:
        """`base`'s dtype, which the own columns share."""
        return self.base.dtype

    @property
    def itemsize(self) -> int:
        """Bytes per value."""
        return self.base.itemsize

    def __len__(self) -> int:
        return self.base.shape[0]

    def is_base(self) -> bool:
        """Whether the table is `base` itself, column for column."""
        return self.shape == self.base.shape and bool(
            np.array_equal(self.sources, np.arange(self.base.shape[1]))
        )

    def __getitem__(self, key: Any) -> Any:
        """A slice of rows as an array, or `[:, columns]` as a table."""
        if isinstance(key, tuple):
            rows, columns = key
            if rows != slice(None):
                raise IndexError("A table selects rows or columns, not both.")
            selected = np.arange(self.shape[1])[columns]
            return SharedColumnsTable(self.base, self.sources[selected], self.own)
        rows = range(self.shape[0])[key]
        if not isinstance(rows, range) or rows.step != 1:
            raise IndexError("A table reads a contiguous slice of rows.")
        if self.is_base():
            return self.base[rows.start : rows.stop]
        out = np.empty((len(rows), self.shape[1]), dtype=self.dtype)
        self.copy_rows_into(rows.start, rows.stop, torch.from_numpy(out))
        return out

    def own_column(self, column: int) -> np.ndarray:
        """Column `column`, which the table holds itself, as a writeable view."""
        source = int(self.sources[column])
        if source >= 0:
            raise ValueError(f"Column {column} is shared, not the table's own.")
        return self.own[:, -1 - source]

    def with_columns(
        self,
        values: np.ndarray,
        at: slice | list[int],
        passthrough: list[tuple[int, int]],
    ) -> SharedColumnsTable:
        """A table of `values` at positions `at` and, per `(position, column)` in
        `passthrough`, this table's `column` at `position`.

        `values`, in `base`'s dtype, become the table's own columns; a table takes
        own columns once.
        """
        if self.own.shape[1] or values.dtype != self.dtype:
            raise ValueError("A table takes its own columns once, in its base's dtype.")
        sources = np.empty(len(passthrough) + values.shape[1], dtype=np.intp)
        for position, column in passthrough:
            sources[position] = self.sources[column]
        sources[at] = -1 - np.arange(values.shape[1])
        return SharedColumnsTable(self.base, sources, values)

    def copy_rows_into(self, start: int, stop: int, out: torch.Tensor) -> None:
        """Writes rows `start:stop` into `out`, a contiguous CPU tensor of their
        shape in `base`'s dtype.
        """
        from_base, own_at, own_from = self._gather_indices
        rows = stop - start
        # Every column from `base` in one gather, then the own ones over theirs.
        base = as_tensor_to_read(self.base[start:stop])
        torch.gather(base, 1, from_base.expand(rows, -1), out=out)
        if own_at.numel():
            own = torch.from_numpy(self.own[start:stop])
            values = torch.gather(own, 1, own_from.expand(rows, -1))
            out.scatter_(1, own_at.expand(rows, -1), values)

    @functools.cached_property
    def _gather_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per column, its source in `base` (column 0 for an own column); then where
        the own columns go, and which of `own` they are.
        """

        def index(values: np.ndarray) -> torch.Tensor:
            # int64: torch 2.5 rejects the int32 that numpy 1 gives on Windows.
            return torch.as_tensor(values, dtype=torch.long).view(1, -1)

        shared = self.sources >= 0
        return (
            index(np.where(shared, self.sources, 0)),
            index(np.flatnonzero(~shared)),
            index(-1 - self.sources[~shared]),
        )

    def __array__(self, dtype: Any = None, *, copy: bool | None = None) -> np.ndarray:
        if self.is_base():
            # Not `np.array(..., copy=copy)`: numpy 1 rejects `copy=None`.
            if copy:
                return np.array(self.base, dtype=dtype)
            return np.asarray(self.base, dtype=dtype)
        if copy is False:
            raise ValueError("A table is built anew; it cannot be read without a copy.")
        out = np.empty(self.shape, dtype=self.dtype)
        rows = max(1, ROW_SCAN_BLOCK_BYTES // max(1, self.shape[1] * self.itemsize))
        for start in range(0, self.shape[0], rows):
            stop = min(start + rows, self.shape[0])
            self.copy_rows_into(start, stop, torch.from_numpy(out[start:stop]))
        return out if dtype is None else out.astype(dtype, copy=False)


def any_per_column(
    X: np.ndarray | torch.Tensor,
    tests: list[Callable[[Any], Any]],
) -> list[np.ndarray]:
    """For each elementwise boolean `test`, per column of 2D `X` whether any cell
    passes it.

    Reads `X` by blocks of rows, which are contiguous in a C-order table, on up to
    `ROW_SCAN_MAX_THREADS` threads.
    """
    n_rows, n_columns = X.shape
    itemsize = X.element_size() if isinstance(X, torch.Tensor) else X.itemsize
    rows = max(1, ROW_SCAN_BLOCK_BYTES // max(1, n_columns * itemsize))
    starts = range(0, n_rows, rows)

    def scan(start: int) -> list[np.ndarray]:
        block = X[start : start + rows]
        found = [test(block).any(0) for test in tests]
        return [
            f.cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
            for f in found
        ]

    found = [np.zeros(n_columns, dtype=bool) for _ in tests]
    threads = min(len(starts), ROW_SCAN_MAX_THREADS, _usable_cpus())
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for part in pool.map(scan, starts):
            for index, flags in enumerate(part):
                found[index] |= flags
    return found


def _restore_inf_masks(
    X: np.ndarray | torch.Tensor,
    feature_schema: FeatureSchema,
    inf_masks: InfMasks,
) -> None:
    """Write recorded infinities back into the columns that still carry them.

    A column is matched to a recorded mask by its own name, or by its
    :attr:`Feature.ancestor` when a step renamed/derived it (e.g. the
    ``reshape_{k}`` outputs of :class:`ReshapeFeatureDistributionsStep`). One
    input feature may map to several output columns; each gets the infinities
    restored. ``X`` is mutated in place.
    """
    xp = torch if isinstance(X, torch.Tensor) else np
    for idx, feat in enumerate(feature_schema.features):
        source = feat.name if feat.name in inf_masks else feat.ancestor
        mask = inf_masks.get(source) if source is not None else None
        if mask is None:
            continue
        # A step may change the array kind (e.g. sklearn steps return numpy even
        # for torch input), so coerce the recorded mask to ``X`` before indexing.
        mask = _coerce_like(mask, X)
        bool_mask = xp.isinf(mask)
        X[bool_mask, idx] = mask[bool_mask]


def _flag_non_constant_with_infs(
    feature_schema: FeatureSchema,
    inf_masks: InfMasks,
) -> FeatureSchema:
    """Flag inf-carrying features that are genuinely non-constant."""
    if not inf_masks:
        return feature_schema

    new_features = list(feature_schema.features)
    changed = False
    for idx, feat in enumerate(feature_schema.features):
        mask = inf_masks.get(feat.name)
        if mask is None:
            continue
        if bool((mask != mask[0]).any()):
            new_features[idx] = dataclasses.replace(feat, non_constant_with_inf=True)
            changed = True

    if not changed:
        return feature_schema
    return dataclasses.replace(feature_schema, features=new_features)


def _coerce_like(
    arr: np.ndarray | torch.Tensor,
    reference: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Return ``arr`` as ``reference``'s array kind (and torch dtype/device)."""
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(arr, dtype=reference.dtype, device=reference.device)
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr


@dataclasses.dataclass
class PreprocessingStepResult:
    """Result of a feature preprocessing step.

    Attributes:
        X: Transformed array. For steps registered with specific modalities,
            this is only the transformed columns (not the full array).
            The shape should match the input shape unless columns are removed.
        feature_schema: Feature schema for the columns this step processed.
            Contains 0-based indices relative to the step's input.
            Should NOT include added_columns - the pipeline handles that.
        X_added: Optional new features to append (e.g., fingerprint features).
            These are handled by the pipeline, which concatenates them and
            updates the schema accordingly. Steps should NOT concatenate
            these internally.
        modality_added: Modality for the added features. Required if X_added
            is provided.
    """

    X: np.ndarray | torch.Tensor
    feature_schema: FeatureSchema
    X_added: np.ndarray | torch.Tensor | None = None
    modality_added: FeatureModality | None = None

    def __post_init__(self) -> None:
        """Validate that modality_added is provided when X_added is set."""
        if self.X_added is not None and self.modality_added is None:
            raise ValueError("modality_added must be provided when X_added is not None")


@dataclasses.dataclass
class PreprocessingPipelineResult:
    """Result from the preprocessing pipeline.

    Attributes:
        X: The transformed array, or a table sharing the input's columns.
        feature_schema: Updated feature schema (may have new columns added).
    """

    X: np.ndarray | torch.Tensor | SharedColumnsTable
    feature_schema: FeatureSchema


class PreprocessingStep:
    """Base class for feature preprocessing steps.

    Steps can be registered with specific feature modalities, and the pipeline
    will handle slicing the data to only pass the relevant columns to the step.

    Subclasses should implement `_fit` and `_transform` methods. The `_fit` method
    receives the sliced data and schema, and should return the schema after
    transformation (for the transformed columns only, NOT including added_columns).

    The `_transform` method receives the sliced data and returns the transformed
    array plus new columns and new modality separately. The pipeline handles
    concatenation.

    Design principle: Steps should NOT internally handle passthrough of columns
    they don't transform. The pipeline handles column slicing and reassembly.
    """

    feature_schema_updated_: FeatureSchema
    """Schema describing this step's *main* transformed output.

    This schema corresponds to the `X` returned by `_transform(...)` (i.e. the
    transformed columns for this step) and **must not** include any features
    produced via the optional `X_added` / `modality_added` return values.

    Expected usage patterns (for backwards compatibility):
    - If the step adds features via `X_added` (e.g. fingerprint features), then
      `_fit(...)` should describe only the main output `X` with potentially permuted
      features. The pipeline will append `X_added` and update the overall
      feature schema itself.
    - If the step changes the column semantics/shape of its main output `X`
      (e.g. encoding, reordering, dropping columns) and does **not** rely on the
      `X_added` pathway, then `_fit(...)` should update this schema to match that
      main output.
    """

    n_added_columns_: int | None
    """Number of added columns from `_transform`. Set during first transform call."""

    modality_added_: FeatureModality | None
    """Modality of added columns from `_transform`. Set during first transform call."""

    no_arithmetic: bool = False
    """Whether the step only selects, reorders, compares and encodes values.

    Such a step gives the same float32 results from a float32 copy of its input as
    from a float64 one. It may be handed the caller's array, so it writes into its
    input only when `step_owns_input()`, or a `SharedColumnsTable` over it, which
    supports `shape`, `dtype`, `len`, slices of rows, `[:, columns]` and
    `np.asarray`.
    """

    @abstractmethod
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        """Underlying method of the preprocessor to implement by subclasses.

        Args:
            X: 2d array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            feature_schema: feature schema for the input columns.

        Returns:
            Feature schema after the transform.
        """
        ...

    @abstractmethod
    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, FeatureModality | None]:
        """Underlying method of the preprocessor to implement by subclasses.

        Args:
            X: array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            is_test: Whether this is test data (used for AddFingerPrint step).

        Returns:
            Tuple of (transformed_columns, added_columns, added_modality).
            added_columns and added_modality can be None if no columns are added.
        """
        ...

    def fit_transform(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> PreprocessingStepResult:
        """Fits the preprocessor and transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
            feature_schema: feature schema.

        Returns:
            PreprocessingStepResult with transformed data and updated feature schema.
        """
        # Reset cached validation state so re-fitting is safe.
        if hasattr(self, "n_added_columns_"):
            del self.n_added_columns_
        if hasattr(self, "modality_added_"):
            del self.modality_added_
        self.feature_schema_updated_ = self._fit(X, feature_schema)
        return self.transform(X, is_test=False)

    def transform(
        self,
        X: np.ndarray,
        *,
        is_test: bool = True,
    ) -> PreprocessingStepResult:
        """Transforms the data.

        Args:
            X: array of shape (n_samples, n_features). For steps registered
                with specific modalities, this is only the relevant columns.
            is_test: Whether this is test data (used for AddFingerPrint step).

        Returns:
            PreprocessingStepResult with transformed data and feature schema.
        """
        result, X_added, modality_added = self._transform(X, is_test=is_test)
        self._validate_added_data(X_added=X_added, modality_added=modality_added)

        return PreprocessingStepResult(
            X=result,
            feature_schema=self.feature_schema_updated_,
            X_added=X_added,
            modality_added=modality_added,
        )

    def num_added_features(self, n_samples: int, feature_schema: FeatureSchema) -> int:
        """Return the number of added features.

        This needs to be overridden by subclasses that add features.
        """
        del n_samples, feature_schema
        return 0

    def added_feature_prefix(self) -> str:
        """Name prefix for features this step appends via ``X_added``.

        Used by the pipeline to generate self-describing, unique names for
        appended columns (e.g. ``svd_0``). Steps that add features should
        override this with a short identifier of the transform.
        """
        return self.__class__.__name__

    def has_data_dependent_feature_expansion(self) -> bool:
        """Return True if this step's feature expansion depends on data values.

        Override to return True for steps where ``num_added_features()`` is an
        approximation because the true count depends on fitting data (e.g.
        one-hot encoding cardinality).
        """
        return False

    def _validate_added_data(
        self,
        X_added: np.ndarray | None,
        modality_added: FeatureModality | None,
    ) -> None:
        """Validate consistency of added columns across train/test transforms."""
        n_added = X_added.shape[1] if X_added is not None else None

        store_expected_values_on_first_call = not hasattr(self, "n_added_columns_")
        if store_expected_values_on_first_call:
            self.n_added_columns_ = n_added
            self.modality_added_ = modality_added
        else:
            if n_added != self.n_added_columns_:
                raise ValueError(
                    f"Inconsistent number of added columns: expected "
                    f"{self.n_added_columns_}, got {n_added}"
                )
            if modality_added != self.modality_added_:
                raise ValueError(
                    f"Inconsistent modality for added columns: expected "
                    f"{self.modality_added_}, got {modality_added}"
                )


class PreprocessingPipeline:
    """Modality-aware preprocessing pipeline that handles column slicing.

    This pipeline applies a sequence of preprocessing steps to data,
    where each step can be registered to target specific feature modalities.
    The pipeline handles slicing columns based on registered modalities,
    passing only relevant columns to each step, reassembling data after each
    step, and tracking feature schema updates.

    For backwards compatibility, steps can be registered as (step, modalities)
    tuples where the step receives only columns matching the specified modalities,
    or as bare steps that receive all columns. In the latter case, the column modalities
    need to be tracked by the step itself. Going forward, only the former format will be
    supported.
    """

    initial_feature_schema_: FeatureSchema | None = None
    """The feature schema before the pipeline has been fit."""

    final_feature_schema_: FeatureSchema | None = None
    """The feature schema after the pipeline has been fit."""

    def __init__(
        self,
        steps: list[PreprocessingStep | StepWithModalities],
    ) -> None:
        """Initialize the pipeline with preprocessing steps.

        Args:
            steps: List of preprocessing steps. Each can be a PreprocessingStep
                (receives all columns) or a tuple of (PreprocessingStep,
                set[FeatureModality]) where the step receives only columns
                matching the specified modalities.
        """
        super().__init__()
        self._raw_steps = steps
        self.steps = self._validate_steps(steps)

    @property
    def keep_float32(self) -> bool:
        """Whether a float32 input is copied as float32 rather than float64, and not
        copied at all when nothing would write into it and the caller lets the
        result keep it.

        True when every step only selects, reorders, compares and encodes values:
        their float32 results equal the float64 ones cast back.
        """
        return all(step.no_arithmetic for step, _ in self.steps)

    step_timings_: dict[str, float] | None
    """Per-step wall-clock time (seconds) from the last ``fit_transform`` or
    ``transform`` call.  Keyed by ``<index>_<ClassName>``.
    Only populated when ``record_timings=True``."""

    record_timings: bool = False
    """Set to ``True`` to collect per-step wall-clock timings."""

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
        *,
        feature_indices: np.ndarray | None = None,
        may_keep_input: bool = False,
    ) -> PreprocessingPipelineResult:
        """Fit and transform the data using the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
            feature_schema: feature schema, of the columns the pipeline reads.
            feature_indices: The columns of `X` the pipeline reads, gathered into
                its working copy; None for all of them.
            may_keep_input: Whether the result may be `X` itself, or a
                `SharedColumnsTable` over it. A fitted model keeps its training
                data, so this is only for a caller that hands `X` over.

        Returns:
            PreprocessingPipelineResult with transformed data and updated schema.
        """
        self.initial_feature_schema_ = feature_schema
        X, feature_schema = self._process_steps(
            X,
            feature_schema,
            is_fitting=True,
            feature_indices=feature_indices,
            may_keep_input=may_keep_input,
        )
        self.final_feature_schema_ = feature_schema
        return PreprocessingPipelineResult(X=X, feature_schema=feature_schema)

    def transform(
        self, X: np.ndarray | torch.Tensor, *, may_keep_input: bool = False
    ) -> PreprocessingPipelineResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
            may_keep_input: Whether the result may be `X` itself, or a
                `SharedColumnsTable` over it, which the caller then must not write
                into.

        Returns:
            PreprocessingPipelineResult with transformed data and feature schema.
        """
        assert self.initial_feature_schema_ is not None, (
            "The pipeline must be fit before it can be used to transform."
        )
        assert self.final_feature_schema_ is not None

        X, updated_schema = self._process_steps(
            X,
            self.initial_feature_schema_,
            is_fitting=False,
            may_keep_input=may_keep_input,
        )
        return PreprocessingPipelineResult(X=X, feature_schema=updated_schema)

    def num_added_features(self, n_samples: int, feature_schema: FeatureSchema) -> int:
        """Return the number of added features.

        Threads an evolving feature schema through the steps so that each step
        sees the feature count that includes columns added by prior steps.
        """
        total_added = 0
        current_schema = feature_schema
        for step, _ in self.steps:
            added = step.num_added_features(n_samples, current_schema)
            total_added += added
            if added > 0:
                current_schema = current_schema.append_columns(
                    FeatureModality.NUMERICAL, added
                )
        return total_added

    def has_data_dependent_feature_expansion(self) -> bool:
        """Return True if any step has data-dependent feature expansion."""
        return any(
            step.has_data_dependent_feature_expansion() for step, _ in self.steps
        )

    def _working_copy(
        self,
        X: np.ndarray | torch.Tensor,
        feature_indices: np.ndarray | None,
        *,
        may_keep_input: bool,
    ) -> np.ndarray | torch.Tensor | SharedColumnsTable:
        """The array the steps run on.

        This is a copy of `X`, or of its `feature_indices` columns. It is `X` itself,
        as a `SharedColumnsTable`, when nothing would write into it and
        `may_keep_input`: a step that would copy `X` to move or rewrite its columns
        then shares the columns it leaves alone.

        One copy preserves the caller's array and saves each step its own; the cast
        is fused with it.
        """
        if not isinstance(X, np.ndarray):
            if feature_indices is None:
                return X.clone()
            index = torch.as_tensor(feature_indices, dtype=torch.long, device=X.device)
            return X.index_select(-1, index)
        keep = self.keep_float32 and X.dtype == np.float32
        if (
            keep
            and feature_indices is None
            and may_keep_input
            # The copy is C-ordered, and `torch.as_tensor` rejects negative strides.
            and X.flags.c_contiguous
            and self._writes_only_owned_input(X)
        ):
            return SharedColumnsTable.of(X)
        dtype = np.float32 if keep else np.float64
        if feature_indices is None:
            return X.astype(dtype, order="C", copy=True)
        return gather_columns(X, feature_indices, dtype)

    def _writes_only_owned_input(self, X: np.ndarray) -> bool:
        """Whether the steps can run on `X` itself without anything writing into it.

        That is not the case when a step has modalities, because the pipeline writes
        that step's output back into its array, or when `X` holds infinities, which
        are replaced in place. The steps themselves only write into arrays they own.
        """
        if any(modalities for _, modalities in self.steps):
            return False
        (has_inf,) = any_per_column(X, [np.isinf])
        return not has_inf.any()

    def _process_steps(
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
        *,
        is_fitting: bool,
        may_keep_input: bool,
        feature_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray | torch.Tensor, FeatureSchema]:
        """Process all pipeline steps.

        Args:
            X: Input array of shape (n_samples, n_features).
            feature_schema: Feature schema.
            is_fitting: If True, call fit_transform on steps; otherwise transform.
            may_keep_input: Whether the result may be `X` itself.
            feature_indices: Columns of `X` to gather into the working copy.

        Returns:
            Tuple of (transformed array, updated feature schema).
        """
        given = X
        X = self._working_copy(X, feature_indices, may_keep_input=may_keep_input)
        owns_input = not isinstance(X, SharedColumnsTable)

        # Record any +/-inf positions and replace them with NaN so the steps
        # (which assume finite/NaN input) can run, then write them back at the
        # end.  Computed per call so train and test each restore their own
        # pattern.  A no-op when the input is already finite.
        # Without a copy, `_working_copy` has already found `X` free of infinities.
        inf_masks = _extract_inf_masks(X, feature_schema) if owns_input else {}
        if is_fitting:
            feature_schema = _flag_non_constant_with_infs(feature_schema, inf_masks)

        self.step_timings_ = {} if self.record_timings else None
        for step_idx, (step, modalities) in enumerate(self.steps):
            if self.record_timings:
                t0 = time.perf_counter()

            if modalities:
                indices = feature_schema.indices_for_modalities(modalities)
                if not indices:
                    continue

                X_slice = X[:, indices]
                result = (
                    step.fit_transform(
                        X_slice, feature_schema.slice_for_indices(indices)
                    )
                    if is_fitting
                    else step.transform(X_slice)
                )
                if result.X.shape[1] != len(indices):
                    raise ValueError(
                        f"Step {step.__class__.__name__} registered with modalities "
                        f"{modalities} received {len(indices)} columns but returned "
                        f"{result.X.shape[1]} columns. Steps registered with "
                        f"modalities must return the same number of columns."
                    )

                self._validate_expected_dtype(pre_x=X, post_x=result.X, step=step)

                X[:, indices] = result.X

                X, feature_schema = self._maybe_append_added_columns(
                    X, feature_schema, result, step.added_feature_prefix()
                )
                feature_schema = feature_schema.update_from_preprocessing_step_result(
                    indices, result.feature_schema
                )
            else:
                # We still have preprocessing steps that don't change the columns
                # internally (will be deprecated going forward). For backwards
                # compatibility, we still handle these here.
                owned = _STEP_OWNS_INPUT.set(owns_input)
                try:
                    result = (
                        step.fit_transform(X, feature_schema)
                        if is_fitting
                        else step.transform(X)
                    )
                finally:
                    _STEP_OWNS_INPUT.reset(owned)
                X = result.X
                feature_schema = result.feature_schema
                X, feature_schema = self._maybe_append_added_columns(
                    X, feature_schema, result, step.added_feature_prefix()
                )
                # A new array from a step is the pipeline's to hand on.
                owns_input = owns_input or (
                    isinstance(X, np.ndarray) and not np.may_share_memory(X, given)
                )

            if self.record_timings:
                step_key = f"{step_idx}_{step.__class__.__name__}"
                self.step_timings_[step_key] = time.perf_counter() - t0

        if inf_masks:
            _restore_inf_masks(X, feature_schema, inf_masks)

        if isinstance(X, SharedColumnsTable) and X.is_base():
            X = X.base
        return X, feature_schema

    def _validate_expected_dtype(
        self, pre_x: np.ndarray, post_x: np.ndarray, step: PreprocessingStep
    ) -> None:
        """Validate that the input and output dtypes are as expected."""
        is_dtype_preserving_step = step.__class__.__name__ != "TokenizeTextStep"
        if is_dtype_preserving_step:
            assert pre_x.dtype == post_x.dtype
        else:
            assert pre_x.dtype != post_x.dtype
            assert pre_x.dtype in ("U", "O")
            assert post_x.dtype in ("float", "int")

    def _maybe_append_added_columns(
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
        result: PreprocessingStepResult,
        name_prefix: str,
    ) -> tuple[np.ndarray | torch.Tensor, FeatureSchema]:
        """Append added columns from a step result and update schema."""
        if result.X_added is not None:
            if isinstance(X, np.ndarray):
                X = np.concatenate([X, result.X_added], axis=1)
            else:
                assert isinstance(result.X_added, torch.Tensor)
                X = torch.cat([X, result.X_added], dim=1)
            feature_schema = feature_schema.append_columns(
                result.modality_added or FeatureModality.NUMERICAL,
                result.X_added.shape[1],
                name_prefix=name_prefix,
            )
        return X, feature_schema

    def _validate_steps(
        self,
        steps: list[PreprocessingStep | StepWithModalities],
    ) -> list[StepWithModalities]:
        """Convert steps to normalized (step, modalities) format."""
        normalized: list[StepWithModalities] = []
        if len(steps) == 0:
            raise ValueError("The pipeline must have at least one step.")
        for step in steps:
            if isinstance(step, tuple):
                if len(step) != 2:
                    raise ValueError(
                        f"Step tuple must be (step, modalities), got {step}"
                    )
                normalized.append(step)
            else:
                normalized.append((step, set()))
        return normalized

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)
