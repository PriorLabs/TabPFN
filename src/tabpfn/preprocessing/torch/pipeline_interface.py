#  Copyright (c) Prior Labs GmbH 2026.

"""Interfaces for torch preprocessing pipeline."""

from __future__ import annotations

import abc
import dataclasses
import functools
import inspect
import time
from typing_extensions import override

import torch

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema

IN_PLACE_BLOCK_BYTES = 1 << 28
"""Bytes of selected columns a columnwise step copies at a time when in place."""

DEVICE_BLOCK_BYTES = 1 << 31
"""Bytes of an input held elsewhere that the pipeline moves to its compute device at a
time."""


class TorchPreprocessingStep(abc.ABC):
    """Base class for preprocessing steps that can operate on specific columns.

    These steps are designed to be stateless and can be easily used in the forward pass
    of the model during training. The fitted state is returned explicitly and can be
    used in the transform step.

    Subclasses should implement `_fit` and `_transform` to define the actual
    transformation logic. The base class handles column selection, tensor
    cloning, and reassignment.
    """

    columnwise: bool = False
    """Whether each column's fit and transform read only that column.

    Such a step fits and transforms an input it may write into a block of
    columns at a time, its cache tensors' last dimension indexing the columns. A
    column's fitted values are bitwise the same whichever other columns it is
    fitted with.
    """

    rowwise: bool = False
    """Whether the step fits without reading cell values and transforms each row
    alone.

    The pipeline runs such a step on an input held off its compute device a block
    of rows at a time.
    """

    def fit_transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
        fitted_cache: dict[str, torch.Tensor] | None = None,
        *,
        in_place: bool = False,
    ) -> TorchPreprocessingStepResult:
        """Fit on training data for the specified columns.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should fit on.
            num_train_rows: Number of training rows (fit on x[:num_train_rows]).
            fitted_cache: Fitted cache from the previous step. If None, the step will
                fit the cache on the training data.
            in_place: Whether the step may write its result into `x`.
        """
        if in_place and self.columnwise and column_indices:
            return self._fit_transform_column_blocks(
                x, column_indices, num_train_rows, fitted_cache
            )
        x_cols_selected = x[:, :, column_indices]

        if fitted_cache is None:
            fitted_cache = self._fit(x_cols_selected[:num_train_rows])

        transformed, added_columns, added_modality = self._transform(
            x_cols_selected, fitted_cache=fitted_cache
        )

        if not in_place:
            x = x.clone()
        x[:, :, column_indices] = transformed
        return TorchPreprocessingStepResult(
            x=x,
            added_columns=added_columns,
            added_modality=added_modality,
            fitted_cache=fitted_cache,
        )

    def _fit_transform_column_blocks(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
        fitted_cache: dict[str, torch.Tensor] | None,
    ) -> TorchPreprocessingStepResult:
        """`fit_transform` of a columnwise step into `x`, a block of columns at a time.

        Bounds the selected columns' copies to one block.
        """
        columns = torch.as_tensor(column_indices, device=x.device)
        row_bytes = x.shape[0] * x.shape[1] * x.element_size()
        width = max(1, IN_PLACE_BLOCK_BYTES // max(1, row_bytes))
        blocks = [
            slice(start, min(start + width, len(column_indices)))
            for start in range(0, len(column_indices), width)
        ]
        if fitted_cache is None:
            parts = [self._fit(x[:num_train_rows, :, columns[b]]) for b in blocks]
            fitted_cache = {
                key: torch.cat([part[key] for part in parts], dim=-1)
                for key in parts[0]
            }
        for block in blocks:
            cache = {key: value[..., block] for key, value in fitted_cache.items()}
            transformed, added_columns, _ = self._transform(
                x[:, :, columns[block]], fitted_cache=cache
            )
            assert added_columns is None, "A columnwise step adds no columns."
            x[:, :, columns[block]] = transformed
        return TorchPreprocessingStepResult(x=x, fitted_cache=fitted_cache)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def target_columns(
        self,
        feature_schema: FeatureSchema,
        modalities: set[FeatureModality] | None,
    ) -> list[int]:
        """The columns this step transforms, which the pipeline passes as
        `column_indices`.

        These are the columns of `modalities`, or all columns if it is None.
        """
        if modalities is None:
            return list(range(feature_schema.num_columns))
        return feature_schema.indices_for_modalities(modalities)

    def added_feature_prefix(self) -> str:
        """Name prefix for features this step appends via ``added_columns``.

        Mirrors :meth:`PreprocessingStep.added_feature_prefix` on the CPU side so
        appended columns get self-describing, unique names.
        """
        return self.__class__.__name__

    @abc.abstractmethod
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit on the selected columns (training rows only) and return a cache.

        Args:
            x: Tensor of selected columns [num_train_rows, batch_size, num_cols].

        Returns:
            Cache dictionary with the cache for the transform step.
        """
        ...

    @abc.abstractmethod
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform the selected columns using the cache.

        Args:
            x: Tensor of selected columns [num_rows, batch_size, num_cols].
            fitted_cache: Cache returned by _fit.

        Returns:
            Tuple of (transformed_columns, added_columns, added_modality).
            added_columns and added_modality can be None if no columns are added.
        """
        ...


class TorchPreprocessingPipeline:
    """Modality-aware preprocessing pipeline with explicit state management.

    This pipeline applies a sequence of stateless preprocessing steps to a tensor,
    where each step targets specific feature modalities. Steps can target
    multiple modalities at once (e.g., StandardScaler for both NUMERICAL
    and CATEGORICAL features).

    Steps registered with ``modalities=None`` receive ALL column indices. This is
    used for steps that must operate on the full tensor (e.g., shuffle).
    """

    def __init__(
        self,
        steps: list[tuple[TorchPreprocessingStep, set[FeatureModality] | None]],
        *,
        keep_fitted_cache: bool = False,
    ) -> None:
        """Initialize with list of (step, target_modalities) pairs.

        Args:
            steps: List of (step, modalities) where modalities is a set of
                FeatureModality values the step should be applied to, or None
                to apply the step to all columns.
            keep_fitted_cache: Whether to keep the state of the individual steps
                between calls. If True, the fitted state of all steps will be kept
                inside the fitted_cache attribute. It can be re-used when parsing
                `use_fitted_cache=True` in the __call__ method.
                If False, the cache will not be saved and the steps are refit
                on the training data.
        """
        super().__init__()

        self._validate_steps(steps)
        self.steps = steps
        self.keep_fitted_cache = keep_fitted_cache
        self.fitted_cache: list[dict[str, torch.Tensor] | None] = [None] * len(
            self.steps
        )
        self.record_timings: bool = False
        """Set to ``True`` to collect per-step wall-clock timings."""
        self.step_timings_: dict[str, float] | None = None
        """Per-step wall-clock time (seconds) from the last ``__call__``.
        Keyed by ``<index>_<ClassName>``.  Only populated when
        ``record_timings=True``."""

    def __call__(  # noqa: C901
        self,
        x: torch.Tensor,
        feature_schema: FeatureSchema,
        num_train_rows: int | None = None,
        *,
        use_fitted_cache: bool = False,
        consume_input: bool = False,
        compute_device: torch.device | None = None,
    ) -> TorchPreprocessingPipelineOutput:
        """Apply all steps to the input tensor.

        Args:
            x: Input tensor [num_rows, batch_size, num_columns] or
                [num_rows, num_columns]. If 2D, a batch dimension is added
                and removed after processing.
            feature_schema: Feature schema.
            num_train_rows: If provided, fit steps on x[:num_train_rows]. If
                not provided, fits on the entire input tensor.
            use_fitted_cache: Whether to use the fitted cache from the previous call
                of the pipeline. If False, the processors are refit on the provided
                data.
            consume_input: Whether the steps may write into `x`, which the caller
                no longer reads.
            compute_device: The device to compute on when it is not `x`'s. With
                `consume_input`, the infinity scan and the columnwise steps then
                move `x` there a block at a time and write the results back into
                `x`; the other steps run on `x`'s device.

        Returns:
            PipelineOutput with transformed tensor and updated schema.
        """
        self._validate_use_fitted_cache(use_fitted_cache=use_fitted_cache)

        squeeze_batch_dim = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze_batch_dim = True

        num_columns = x.shape[-1]
        self._validate_metadata(feature_schema=feature_schema, num_columns=num_columns)

        if num_train_rows is None:
            num_train_rows = x.shape[0]

        # Record any +/-inf positions (passed through from CPU preprocessing) and
        # replace them with NaN so SVD/quantile/scaler steps don't crash on or
        # corrupt from them; they are written back at the end.  No-op when finite.
        staged = (
            consume_input
            and compute_device is not None
            and not _same_device(torch.device(compute_device), x.device)
        )
        inf_masks = _extract_inf_masks(
            x, feature_schema, scan_device=compute_device if staged else None
        )

        self.step_timings_ = {} if self.record_timings else None
        for i, (step, modalities) in enumerate(self.steps):
            if self.record_timings:
                t0 = time.perf_counter()

            indices = step.target_columns(feature_schema, modalities)
            if not indices:
                continue

            fitted_cache = self.fitted_cache[i] if use_fitted_cache else None
            if fitted_cache is not None:
                fitted_cache = _move_cache_to_device(fitted_cache, x.device)

            result = _run_step(
                step,
                x,
                indices,
                num_train_rows,
                fitted_cache,
                consume_input=consume_input,
                compute_device=compute_device if staged else None,
            )
            x = result.x

            if result.added_columns is not None:
                x = torch.cat([x, result.added_columns], dim=-1)
                feature_schema = feature_schema.append_columns(
                    result.added_modality or FeatureModality.NUMERICAL,
                    result.added_columns.shape[-1],
                    name_prefix=step.added_feature_prefix(),
                )

            if result.schema_permutation is not None:
                feature_schema = feature_schema.apply_permutation(
                    result.schema_permutation
                )

            self._maybe_update_fitted_cache(i, result)

            if self.record_timings:
                self.step_timings_[f"{i}_{step.__class__.__name__}"] = (
                    time.perf_counter() - t0
                )

        # Write the recorded infinities back into the columns that still carry
        # them (matched by name; torch steps don't rename surviving columns,
        # appended SVD/fingerprint columns get none).
        # NOTE: since no GPU pipelines currently duplicate features, we don't
        # need to track ancestry
        _restore_inf_masks(x, feature_schema, inf_masks)

        if squeeze_batch_dim:
            x = x.squeeze(1)

        # Clear GPU transform marks — all GPU transforms have been applied.
        feature_schema = feature_schema.clear_gpu_transform_marks()

        return TorchPreprocessingPipelineOutput(x=x, feature_schema=feature_schema)

    def _maybe_update_fitted_cache(
        self, i: int, result: TorchPreprocessingStepResult
    ) -> None:
        if self.keep_fitted_cache:
            # Store on CPU so the cache is decoupled from the device the
            # pipeline ran on.
            cache = result.fitted_cache
            if cache is not None:
                cache = _move_cache_to_device(cache, torch.device("cpu"))
            self.fitted_cache[i] = cache

    @override
    def __repr__(self) -> str:
        return f"TorchPreprocessingPipeline(steps={self.steps})"

    def _validate_steps(
        self,
        steps: list[tuple[TorchPreprocessingStep, set[FeatureModality] | None]],
    ) -> None:
        for step in steps:
            if len(step) != 2:
                raise ValueError(
                    f"Each step must be a tuple of (step, modalities), but got `{step}`"
                )

    def _validate_use_fitted_cache(self, *, use_fitted_cache: bool) -> None:
        if use_fitted_cache and not self.keep_fitted_cache:
            raise ValueError(
                "use_fitted_cache=True is only supported if keep_fitted_cache=True "
                "during initialization."
            )

    def _validate_metadata(
        self, feature_schema: FeatureSchema, num_columns: int
    ) -> None:
        if num_columns != feature_schema.num_columns:
            raise ValueError(
                f"Number of columns in input tensor ({num_columns}) does not match "
                f"number of columns in schema ({feature_schema.num_columns})"
            )


@functools.cache
def _accepts_in_place(step_type: type) -> bool:
    """Whether `step_type.fit_transform` takes an `in_place` keyword."""
    parameters = inspect.signature(step_type.fit_transform).parameters
    return "in_place" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _in_place_kwargs(step: TorchPreprocessingStep) -> dict[str, bool]:
    """`in_place=True` for a step whose `fit_transform` takes it, else nothing."""
    return {"in_place": True} if _accepts_in_place(type(step)) else {}


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Whether `a` and `b` name one device, an index of None meaning the current
    one of its type.
    """
    if a.type != b.type:
        return False
    if a.type == "cpu" or a.index == b.index:
        return True
    return _device_index(a) == _device_index(b)


def _device_index(device: torch.device) -> int:
    """`device`'s index, the current one of its type when it names none."""
    if device.index is not None:
        return device.index
    current = getattr(getattr(torch, device.type, None), "current_device", None)
    return int(current()) if callable(current) else 0


def _move_cache_to_device(
    cache: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move cache tensors to ``device``, keeping float64 on CPU for MPS.

    MPS has no float64; steps that need it fall back to CPU themselves.
    """

    def move(v: torch.Tensor) -> torch.Tensor:
        if device.type == "mps" and v.dtype == torch.float64:
            return v.cpu()
        return v.to(device)

    return {k: move(v) if isinstance(v, torch.Tensor) else v for k, v in cache.items()}


def _run_step(
    step: TorchPreprocessingStep,
    x: torch.Tensor,
    column_indices: list[int],
    num_train_rows: int,
    fitted_cache: dict[str, torch.Tensor] | None,
    *,
    consume_input: bool,
    compute_device: torch.device | None,
) -> TorchPreprocessingStepResult:
    """`step.fit_transform`, by blocks moved to `compute_device` when one is given
    and the step reads its input by columns or by rows.
    """
    if compute_device is not None and step.columnwise:
        return _fit_transform_on_device(
            step, x, column_indices, num_train_rows, fitted_cache, compute_device
        )
    if compute_device is not None and step.rowwise and x.shape[0] > 0:
        return _fit_transform_rows_on_device(
            step, x, column_indices, num_train_rows, fitted_cache, compute_device
        )
    in_place = _in_place_kwargs(step) if consume_input else {}
    return step.fit_transform(
        x,
        column_indices=column_indices,
        num_train_rows=num_train_rows,
        fitted_cache=fitted_cache,
        **in_place,
    )


def _fit_transform_on_device(
    step: TorchPreprocessingStep,
    x: torch.Tensor,
    column_indices: list[int],
    num_train_rows: int,
    fitted_cache: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> TorchPreprocessingStepResult:
    """A columnwise step's `fit_transform` into `x`, a block of columns at a time on
    `device`.

    Returns the fitted cache on `device`.
    """
    row_bytes = x.shape[0] * x.shape[1] * x.element_size()
    width = max(1, DEVICE_BLOCK_BYTES // max(1, row_bytes))
    parts: list[dict[str, torch.Tensor] | None] = []
    for start in range(0, len(column_indices), width):
        selected = column_indices[start : start + width]
        index = _column_index(selected)
        cache = None
        if fitted_cache is not None:
            cache = {
                key: value[..., start : start + len(selected)].to(device)
                for key, value in fitted_cache.items()
            }
        result = step.fit_transform(
            x[:, :, index].to(device),
            column_indices=list(range(len(selected))),
            num_train_rows=num_train_rows,
            fitted_cache=cache,
            **_in_place_kwargs(step),
        )
        assert result.added_columns is None, "A columnwise step adds no columns."
        x[:, :, index] = result.x.to(x.device)
        parts.append(result.fitted_cache)
    fitted = None
    if parts and parts[0] is not None:
        fitted = {
            key: torch.cat([part[key] for part in parts if part is not None], dim=-1)
            for key in parts[0]
        }
    return TorchPreprocessingStepResult(x=x, fitted_cache=fitted)


def _fit_transform_rows_on_device(
    step: TorchPreprocessingStep,
    x: torch.Tensor,
    column_indices: list[int],
    num_train_rows: int,
    fitted_cache: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> TorchPreprocessingStepResult:
    """A rowwise step's `fit_transform` into `x`, a block of rows at a time on
    `device`.

    The first block fits the step when `fitted_cache` is None; the others reuse it.
    """
    row_bytes = max(1, x[:1].numel() * x.element_size())
    rows = max(1, DEVICE_BLOCK_BYTES // row_bytes)
    cache = None
    if fitted_cache is not None:
        cache = _move_cache_to_device(fitted_cache, device)
    permutation = None
    for start in range(0, x.shape[0], rows):
        result = step.fit_transform(
            x[start : start + rows].to(device),
            column_indices=column_indices,
            num_train_rows=num_train_rows,
            fitted_cache=cache,
            **_in_place_kwargs(step),
        )
        assert result.added_columns is None, "A rowwise step adds no columns."
        x[start : start + rows] = result.x.to(x.device)
        cache = result.fitted_cache
        permutation = result.schema_permutation
    return TorchPreprocessingStepResult(
        x=x, fitted_cache=cache, schema_permutation=permutation
    )


def _column_index(columns: list[int]) -> slice | list[int]:
    """`columns` as a slice when they are a contiguous ascending range."""
    if columns == list(range(columns[0], columns[0] + len(columns))):
        return slice(columns[0], columns[0] + len(columns))
    return columns


def _extract_inf_masks(
    x: torch.Tensor,
    feature_schema: FeatureSchema,
    scan_device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Record per-feature +/-inf values and replace them with NaN in ``x``.

    The torch counterpart of the CPU pipeline's helper, operating on the column
    (last) axis of an ``[num_rows, batch_size, num_columns]`` tensor. Returns a
    ``{feature_name: signed-inf slice}`` mapping for the columns that contained
    infinities (empty when ``x`` is finite, leaving ``x`` untouched). ``x`` is
    mutated in place. Rows are scanned a block at a time, so no table-sized boolean
    mask is held. With `scan_device`, the scan moves the blocks there.
    """
    n_columns = x.shape[-1]
    if n_columns == 0:
        return {}
    block_bytes = _INF_SCAN_BLOCK_BYTES if scan_device is None else DEVICE_BLOCK_BYTES
    row_bytes = max(1, x[:1].numel() * x.element_size())
    rows = max(1, block_bytes // row_bytes)
    found: torch.Tensor | None = None
    for start in range(0, x.shape[0], rows):
        block = x[start : start + rows]
        if scan_device is not None:
            block = block.to(scan_device)
        block_inf = torch.isinf(block)
        if block_inf.any():
            columns_inf = block_inf.flatten(0, -2).any(0)
            found = columns_inf if found is None else found | columns_inf
    if found is None:
        return {}
    has_inf = found.cpu().tolist()
    masks: dict[str, torch.Tensor] = {}
    for idx, feat in enumerate(feature_schema.features):
        if has_inf[idx]:
            col = x[..., idx]
            col_inf = torch.isinf(col)
            masks[feat.name] = torch.where(col_inf, col, 0)
            col[col_inf] = float("nan")
    return masks


_INF_SCAN_BLOCK_BYTES = 1 << 28
"""Bytes of rows `_extract_inf_masks` scans for infinities at a time."""


def _restore_inf_masks(
    x: torch.Tensor,
    feature_schema: FeatureSchema,
    inf_masks: dict[str, torch.Tensor],
) -> None:
    """Write recorded infinities back into the columns that still carry them.

    Columns are matched by name. Torch steps don't rename surviving columns
    (shuffle only permutes, SVD/fingerprint only append), so a name match maps
    each mask to its final column position. ``x`` is mutated in place.
    """
    if not inf_masks:
        return
    for idx, feat in enumerate(feature_schema.features):
        mask = inf_masks.get(feat.name)
        if mask is None:
            continue
        bool_mask = torch.isinf(mask)
        col = x[..., idx]  # integer index on the last axis -> view into ``x``
        col[bool_mask] = mask[bool_mask].to(col.dtype)


@dataclasses.dataclass
class TorchPreprocessingStepResult:
    """Result from a preprocessing step's transform.

    Attributes:
        x: Full tensor with columns modified in-place.
        added_columns: Optional new columns to append (e.g., NaN indicators).
        added_modality: Modality for the added columns.
        fitted_cache: Fitted cache from the step.
        schema_permutation: Optional column permutation to apply to the feature
            schema. Used by steps like shuffle that reorder all columns.
    """

    x: torch.Tensor
    added_columns: torch.Tensor | None = None
    added_modality: FeatureModality | None = None
    fitted_cache: dict[str, torch.Tensor] | None = None
    schema_permutation: list[int] | None = None


@dataclasses.dataclass
class TorchPreprocessingPipelineOutput:
    """Output from the preprocessing pipeline.

    Attributes:
        x: The transformed tensor.
        feature_schema: Updated feature schema (may have new columns added).
    """

    x: torch.Tensor
    feature_schema: FeatureSchema
