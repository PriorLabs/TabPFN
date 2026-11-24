#  Copyright (c) Prior Labs GmbH 2026.

"""The model input of an ensemble member, produced one block of cells at a time.

A member's model input is its preprocessed table, train rows then test rows, after
GPU preprocessing. Held whole in host memory, it is as large as the cleaned table,
for each member in turn. `MemberRows` fits the GPU preprocessing on blocks of columns
and applies it only to the rows and columns that are read, so the whole input never
exists. `MemberRowSource` gives the members of one forward pass to a model that reads
its input this way.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch

from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
    _join_column_caches,
    column_blocks,
    runs_staged,
)
from tabpfn.preprocessing.torch.steps import (
    TorchRandomSignFlipStep,
    TorchShuffleFeaturesStep,
)
from tabpfn.utils import as_tensor_to_read

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.preprocessing.datamodel import FeatureSchema

Table = np.ndarray | SharedColumnsTable


@dataclasses.dataclass(frozen=True)
class _ColumnwiseOp:
    """A columnwise step fitted on `columns`, input columns in its cache's order."""

    step: TorchPreprocessingStep
    columns: np.ndarray
    cache: dict[str, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class _SignsOp:
    """Signs to multiply into `columns`, input columns in the order of `signs`."""

    columns: np.ndarray
    signs: torch.Tensor


class MemberRows:
    """The model input of one member, `(train rows + test rows, columns)`, by blocks.

    Its cells equal those of the member's table after `gpu_preprocessor` runs on the
    whole table. The steps are fitted on the same blocks of columns. Each step
    transforms the cells that are read as it would transform every cell.
    """

    def __init__(
        self,
        tables: tuple[Table, Table],
        order: np.ndarray,
        ops: list[_ColumnwiseOp | _SignsOp],
        feature_schema: FeatureSchema,
    ) -> None:
        self.tables = tables
        self.order = order
        self.ops = ops
        self.feature_schema = feature_schema
        self.num_train = len(tables[0])

    @property
    def shape(self) -> tuple[int, int]:
        """`(rows, columns)` of the model input."""
        return self.num_train + len(self.tables[1]), len(self.order)

    @classmethod
    def fit(
        cls,
        X_train: Table,
        X_test: Table,
        gpu_preprocessor: TorchPreprocessingPipeline | None,
        feature_schema: FeatureSchema,
        device: torch.device,
    ) -> MemberRows | None:
        """Fit `gpu_preprocessor` on `X_train` one block of columns at a time.

        Returns None if a step is not columnwise, a sign flip or a shuffle. Such a step
        reads whole rows, so the member's input must be built whole.
        """
        member = cls((X_train, X_test), np.arange(X_train.shape[1]), [], feature_schema)
        if gpu_preprocessor is None:
            return member
        # The pipeline stages an input held off `device`, as the host tables are. The
        # other columns in a block can change how a column's fitted values round, so
        # each columnwise step fits on the pipeline's blocks.
        staged = runs_staged(torch.device("cpu"), device)
        itemsize = np.dtype(np.float32).itemsize
        schema = feature_schema
        for step, modalities in gpu_preprocessor.steps:
            targets = step.target_columns(schema, modalities)
            if not targets:
                continue
            if isinstance(step, TorchShuffleFeaturesStep):
                permutation = step.permutation(len(member.order))
                member.order = member.order[permutation]
                schema = schema.apply_permutation(permutation)
            elif isinstance(step, TorchRandomSignFlipStep):
                signs = step.signs(1, len(targets)).reshape(-1).to(device)
                member.ops.append(_SignsOp(member.order[targets], signs))
            elif step.columnwise:
                columns = member.order[targets]
                plan = column_blocks(
                    step,
                    len(columns),
                    member.shape[0],
                    member.num_train,
                    itemsize,
                    staged=staged,
                )
                parts = [
                    step._fit(
                        member._read_input(
                            slice(0, member.num_train),
                            columns[block],
                            device,
                            infinities=False,
                        )
                    )
                    for block in plan.fit_blocks()
                ]
                cache = _join_column_caches(parts, step.shared_cache_keys)
                member.ops.append(_ColumnwiseOp(step, columns, cache))
            else:
                return None
        member.feature_schema = schema.clear_gpu_transform_marks()
        return member

    def read(
        self, rows: slice | np.ndarray, columns: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """The input's `columns` at `rows`, `(rows, columns)` float32 on `device`."""
        return self._read_input(rows, self.order[columns], device)[:, 0]

    def _read_input(
        self,
        rows: slice | np.ndarray,
        columns: np.ndarray,
        device: torch.device,
        *,
        infinities: bool = True,
    ) -> torch.Tensor:
        """Read input `columns` at `rows` after the steps fitted so far.

        The steps see an infinity as NaN. With `infinities`, the infinities are written
        back after the steps. Without it, they stay NaN, as the next step to fit sees
        them.

        Returns:
            A `(rows, 1, columns)` tensor.
        """
        cells = _rows_of(self.tables, self.num_train, rows, columns).to(device)
        cells = cells.unsqueeze(1)
        infinite = torch.isinf(cells)
        has_inf = bool(infinite.any())
        if has_inf:
            values = cells[infinite]
            cells[infinite] = float("nan")
        for op in self.ops:
            at, of = _positions(columns, op.columns)
            if not len(at):
                continue
            if isinstance(op, _SignsOp):
                cells[:, :, at] *= op.signs[of].to(cells)
                continue
            cache = {
                key: value if key in op.step.shared_cache_keys else value[..., of]
                for key, value in op.cache.items()
            }
            transformed, _, _ = op.step._transform(cells[:, :, at], fitted_cache=cache)
            cells[:, :, at] = transformed.to(cells)
        if has_inf and infinities:
            cells[infinite] = values
        return cells


class MemberRowSource:
    """The model input `(rows, members, columns)` of members with one input shape.

    It is read one block of cells at a time, as a `HostRowSource` for the model's
    `x_host`.
    """

    def __init__(self, members: Sequence[MemberRows]) -> None:
        shapes = {member.shape for member in members}
        if len(shapes) != 1:
            raise ValueError(f"The members' inputs differ in shape: {shapes}.")
        (rows, columns) = shapes.pop()
        self.members = list(members)
        self.shape = (rows, len(self.members), columns)
        self.dtype = torch.float32

    def read(
        self,
        rows: slice | torch.Tensor,
        columns: slice | torch.Tensor,
        device: torch.device,
        batch: int | None = None,
    ) -> torch.Tensor:
        """See `HostRowSource.read`."""
        column_index = np.arange(self.shape[2])[
            columns if isinstance(columns, slice) else columns.cpu().numpy()
        ]
        if batch is not None:
            return self.members[batch].read(_rows(rows), column_index, device)
        return torch.stack(
            [
                member.read(
                    _rows(rows if isinstance(rows, slice) else rows[b]),
                    column_index,
                    device,
                )
                for b, member in enumerate(self.members)
            ]
        )


def _rows(rows: slice | torch.Tensor) -> slice | np.ndarray:
    return rows if isinstance(rows, slice) else rows.cpu().numpy()


def _positions(columns: np.ndarray, of: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The positions in `columns` of the columns that are in `of`, and their positions
    in `of`.
    """
    order = np.argsort(of, kind="stable")
    found = np.searchsorted(of, columns, sorter=order)
    found = np.minimum(found, len(of) - 1)
    hit = of[order[found]] == columns
    return np.flatnonzero(hit), order[found[hit]]


def _rows_of(
    tables: tuple[Table, Table],
    num_train: int,
    rows: slice | np.ndarray,
    columns: np.ndarray,
) -> torch.Tensor:
    """`columns` at `rows` of the train table followed by the test table, float32."""
    if isinstance(rows, slice):
        train = slice(min(rows.start, num_train), min(rows.stop, num_train))
        test = slice(
            max(rows.start, num_train) - num_train,
            max(rows.stop, num_train) - num_train,
        )
        parts = [
            _cells(table, part, columns)
            for table, part in zip(tables, (train, test), strict=True)
            if part.stop > part.start
        ]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts) if parts else torch.empty((0, len(columns)))
    out = torch.empty((len(rows), len(columns)), dtype=torch.float32)
    in_train = rows < num_train
    out[torch.from_numpy(in_train)] = _cells(tables[0], rows[in_train], columns)
    out[torch.from_numpy(~in_train)] = _cells(
        tables[1], rows[~in_train] - num_train, columns
    )
    return out


def _cells(table: Table, rows: slice | np.ndarray, columns: np.ndarray) -> torch.Tensor:
    """`columns` at `rows` of `table`, as a new row-major float32 CPU tensor.

    The blocks of the built input are row-major too. A fit's reductions sum in an
    order that follows the memory layout, so another layout can round a fitted value
    differently.
    """
    if isinstance(table, SharedColumnsTable):
        selected = table[:, columns]
        if isinstance(rows, slice):
            out = torch.from_numpy(
                np.empty((rows.stop - rows.start, len(columns)), dtype=table.dtype)
            )
            selected.copy_rows_into(rows.start, rows.stop, out)
        else:
            out = selected.rows_at(rows)
    else:
        # Torch gathers with many threads. NumPy fancy indexing uses one thread.
        whole = as_tensor_to_read(table)
        picked = (
            whole[rows]
            if isinstance(rows, slice)
            else whole.index_select(0, torch.from_numpy(rows))
        )
        out = picked.index_select(1, torch.from_numpy(columns))
    return out.to(torch.float32).contiguous()
