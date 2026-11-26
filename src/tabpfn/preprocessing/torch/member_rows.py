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

from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable, gather_cells
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingPipeline,
    runs_staged,
)
from tabpfn.utils import as_tensor_to_read

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tabpfn.preprocessing.datamodel import FeatureSchema

Table = np.ndarray | SharedColumnsTable


@dataclasses.dataclass(frozen=True)
class _Op:
    """A fitted step's transform of some input columns."""

    transform: Callable[[torch.Tensor, np.ndarray], torch.Tensor]
    position: np.ndarray
    """Per input column, its position among the step's target columns, or -1."""


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
        ops: list[_Op],
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

        Returns None if a step cannot fit by blocks. Such a step reads whole rows, so
        the member's input must be built whole.
        """
        member = cls((X_train, X_test), np.arange(X_train.shape[1]), [], feature_schema)
        if gpu_preprocessor is None:
            return member
        if any(
            not step.fits_by_blocks and step.target_columns(feature_schema, modalities)
            for step, modalities in gpu_preprocessor.steps
        ):
            # A reordering keeps the set of columns each step targets, so the schema
            # before the steps tells which steps run.
            return None
        # The pipeline stages an input held off `device`, as the host tables are.
        staged = runs_staged(torch.device("cpu"), device)
        schema = feature_schema
        for step, modalities in gpu_preprocessor.steps:
            targets = step.target_columns(schema, modalities)
            if not targets:
                continue
            columns = member.order[targets]

            def read_train(block: slice, columns: np.ndarray = columns) -> torch.Tensor:
                return member._read_input(
                    slice(0, member.num_train), columns[block], device, infinities=False
                )

            fitted = step.fit_by_blocks(
                read_train,
                len(columns),
                member.shape[0],
                member.num_train,
                cell_bytes=np.dtype(np.float32).itemsize,
                staged=staged,
            )
            if fitted.transform is not None:
                position = np.full(len(member.order), -1)
                position[columns] = np.arange(len(columns))
                member.ops.append(_Op(fitted.transform, position))
            if fitted.permutation is not None:
                member.order = member.order[fitted.permutation]
                schema = schema.apply_permutation(fitted.permutation)
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
            of = op.position[columns]
            at = np.flatnonzero(of >= 0)
            if not len(at):
                continue
            if len(at) == len(columns):
                # The op reads every column: it transforms the cells themselves.
                out = op.transform(cells, of)
                if out is not cells:
                    cells.copy_(out)
            else:
                cells[:, :, at] = op.transform(cells[:, :, at], of[at]).to(cells)
        if has_inf and infinities:
            cells[infinite] = values
        return cells


class MemberRowSource:
    """The model input `(rows, members, columns)` of members with one input shape,
    read one block of cells at a time, for a model's `x_host`.
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
        """The cells at `rows` and `columns` of each member, or of member `batch`.

        Args:
            rows: A slice of rows with step 1, or an index of rows: `(members, P)`,
                the rows of each member, when `batch` is None, else `(P,)`.
            columns: A slice of columns with step 1, or a `(K,)` index.
            device: The device of the result.
            batch: The member to read, or None to read all of them.

        Returns:
            `(members, rows, columns)`, the members first unlike in `shape`, or
            `(rows, columns)` when `batch` is set.
        """
        column_index = np.arange(self.shape[2])[
            columns if isinstance(columns, slice) else columns.cpu().numpy()
        ]
        n_rows = self.shape[0]
        if batch is not None:
            return self.members[batch].read(_rows(rows, n_rows), column_index, device)
        return torch.stack(
            [
                member.read(
                    _rows(rows if isinstance(rows, slice) else rows[b], n_rows),
                    column_index,
                    device,
                )
                for b, member in enumerate(self.members)
            ]
        )


def _rows(rows: slice | torch.Tensor, n_rows: int) -> slice | np.ndarray:
    """`rows` as a slice with int bounds or as an index array."""
    if isinstance(rows, slice):
        start, stop, _ = rows.indices(n_rows)
        return slice(start, stop)
    return rows.cpu().numpy()


def _rows_of(
    tables: tuple[Table, Table],
    num_train: int,
    rows: slice | np.ndarray,
    columns: np.ndarray,
) -> torch.Tensor:
    """`columns` at `rows` of the train table followed by the test table, float32."""
    if isinstance(rows, slice):
        if rows.stop <= num_train:
            return _cells(tables[0], rows, columns)
        test = slice(max(rows.start, num_train) - num_train, rows.stop - num_train)
        if rows.start >= num_train:
            return _cells(tables[1], test, columns)
        return torch.cat(
            [
                _cells(tables[0], slice(rows.start, num_train), columns),
                _cells(tables[1], test, columns),
            ]
        )
    in_train = rows < num_train
    if in_train.all():
        return _cells(tables[0], rows, columns)
    if not in_train.any():
        return _cells(tables[1], rows - num_train, columns)
    # The train and test rows can interleave: gather each into its places.
    out = torch.empty((len(rows), len(columns)), dtype=torch.float32)
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
    elif (
        isinstance(rows, slice) and len(columns) and bool((np.diff(columns) == 1).all())
    ):
        # A run of columns is a slice, which copies much faster than a gather.
        # A copy: the slice is a view of the table, which the steps write into.
        whole = as_tensor_to_read(table)
        out = whole[rows, int(columns[0]) : int(columns[-1]) + 1].to(
            torch.float32, memory_format=torch.contiguous_format, copy=True
        )
    else:
        out = gather_cells(as_tensor_to_read(table), rows, columns)
    return out.to(torch.float32).contiguous()
