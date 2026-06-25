#  Copyright (c) Prior Labs GmbH 2026.

"""Interfaces for torch preprocessing pipeline."""

from __future__ import annotations

import abc
import dataclasses
import time
from typing_extensions import override

import torch

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema


class TorchPreprocessingStep(abc.ABC):
    """Base class for preprocessing steps that can operate on specific columns.

    These steps are designed to be stateless and can be easily used in the forward pass
    of the model during training. The fitted state is returned explicitly and can be
    used in the transform step.

    Subclasses should implement `_fit` and `_transform` to define the actual
    transformation logic. The base class handles column selection, tensor
    cloning, and reassignment.
    """

    def fit_transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
        fitted_cache: dict[str, torch.Tensor] | None = None,
    ) -> TorchPreprocessingStepResult:
        """Fit on training data for the specified columns.

        Args:
            x: Full input tensor [num_rows, batch_size, num_columns].
            column_indices: Which columns this step should fit on.
            num_train_rows: Number of training rows (fit on x[:num_train_rows]).
            fitted_cache: Fitted cache from the previous step. If None, the step will
                fit the cache on the training data.
        """
        x_cols_selected = x[:, :, column_indices]

        if fitted_cache is None:
            fitted_cache = self._fit(x_cols_selected[:num_train_rows])

        transformed, added_columns, added_modality = self._transform(
            x_cols_selected, fitted_cache=fitted_cache
        )

        x = x.clone()
        x[:, :, column_indices] = transformed
        return TorchPreprocessingStepResult(
            x=x,
            added_columns=added_columns,
            added_modality=added_modality,
            fitted_cache=fitted_cache,
        )

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

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
        inf_masks = _extract_inf_masks(x, feature_schema)

        self.step_timings_ = {} if self.record_timings else None
        for i, (step, modalities) in enumerate(self.steps):
            if self.record_timings:
                t0 = time.perf_counter()

            if modalities is None:
                indices = list(range(x.shape[-1]))
            else:
                indices = feature_schema.indices_for_modalities(modalities)
            if not indices:
                continue

            fitted_cache = self.fitted_cache[i] if use_fitted_cache else None
            if fitted_cache is not None:
                fitted_cache = _move_cache_to_device(fitted_cache, x.device)

            result = step.fit_transform(
                x,
                column_indices=indices,
                num_train_rows=num_train_rows,
                fitted_cache=fitted_cache,
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


def _move_cache_to_device(
    cache: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cache.items()
    }


def _extract_inf_masks(
    x: torch.Tensor,
    feature_schema: FeatureSchema,
) -> dict[str, torch.Tensor]:
    """Record per-feature +/-inf values and replace them with NaN in ``x``.

    The torch counterpart of the CPU pipeline's helper, operating on the column
    (last) axis of an ``[num_rows, batch_size, num_columns]`` tensor. Returns a
    ``{feature_name: signed-inf slice}`` mapping for the columns that contained
    infinities (empty when ``x`` is finite, leaving ``x`` untouched). ``x`` is
    mutated in place.
    """
    inf_bool = torch.isinf(x)
    if not bool(inf_bool.any()):
        return {}

    masks: dict[str, torch.Tensor] = {}
    for idx, feat in enumerate(feature_schema.features):
        col_inf = inf_bool[..., idx]
        if bool(col_inf.any()):
            col = x[..., idx]
            masks[feat.name] = torch.where(col_inf, col, 0)
    x[inf_bool] = float("nan")
    return masks


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
