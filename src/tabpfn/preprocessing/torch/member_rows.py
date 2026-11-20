#  Copyright (c) Prior Labs GmbH 2026.

"""An ensemble member's model input, produced a block of cells at a time.

A member's model input is its preprocessed table, train rows then test rows, after
its GPU preprocessing. Held whole in host memory it is as large as the cleaned
table, once per member in turn. `MemberRows` fits the GPU preprocessing by blocks of
columns instead, and applies it to whatever rows and columns are read, so the input
never exists whole. `MemberRowSource` serves the members of one forward to a model
that reads its input this way.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch

from tabpfn.preprocessing.pipeline_interface import SharedColumnsTable
from tabpfn.preprocessing.torch.pipeline_interface import (
    DEVICE_BLOCK_BYTES,
    IN_PLACE_BLOCK_BYTES,
    TorchPreprocessingPipeline,
    TorchPreprocessingStep,
    _join_column_caches,
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
    """A columnwise step, fitted on `columns`, input columns in its cache's order."""

    step: TorchPreprocessingStep
    columns: np.ndarray
    cache: dict[str, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class _SignsOp:
    """Signs multiplied into `columns`, input columns in the order of `signs`."""

    columns: np.ndarray
    signs: torch.Tensor


class MemberRows:
    """One member's model input, `(train rows + test rows, columns)`, by blocks.

    Its cells equal those of the member's table after `gpu_preprocessor` runs on it
    whole: the steps are fitted on the same blocks of columns, and each is applied to
    the cells read as it is to every cell.
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
        """Fit `gpu_preprocessor` on `X_train`'s rows by blocks of columns on `device`.

        None when a step is neither columnwise nor a sign flip nor a shuffle: such a
        step reads whole rows, and the member's input has to be built whole.
        """
        member = cls((X_train, X_test), np.arange(X_train.shape[1]), [], feature_schema)
        if gpu_preprocessor is None:
            return member
        # The blocks a staged pipeline fits a columnwise step on: groups of
        # `DEVICE_BLOCK_BYTES` copied to the device, each fitted in blocks of
        # `IN_PLACE_BLOCK_BYTES`. A block's columns can round a column's fit.
        row_bytes = member.shape[0] * np.dtype(np.float32).itemsize
        group = max(1, DEVICE_BLOCK_BYTES // row_bytes)
        width = max(1, IN_PLACE_BLOCK_BYTES // row_bytes)
        schema = feature_schema
        for step, modalities in gpu_preprocessor.steps:
            targets = step.target_columns(schema, modalities)
            if not targets:
                continue
            if isinstance(step, TorchShuffleFeaturesStep):
                empty = torch.empty((0, 1, len(member.order)))
                permutation = step._fit(empty)["permutation"].tolist()
                member.order = member.order[permutation]
                schema = schema.apply_permutation(permutation)
            elif isinstance(step, TorchRandomSignFlipStep):
                signs = step._fit(torch.empty((0, 1, len(targets))))["signs"]
                member.ops.append(
                    _SignsOp(member.order[targets], signs.reshape(-1).to(device))
                )
            elif step.columnwise:
                columns = member.order[targets]
                parts = []
                for start in range(0, len(columns), group):
                    stop = min(start + group, len(columns))
                    for block in range(start, stop, width):
                        cells = member._read_input(
                            slice(0, member.num_train),
                            columns[block : min(block + width, stop)],
                            device,
                            infinities=False,
                        )
                        parts.append(step._fit(cells))
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
        """The table's input `columns` at `rows` after the steps fitted so far,
        `(rows, 1, columns)`.

        The steps see an infinity as missing; it is written back after them unless
        not `infinities`, as the steps after them see it.
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
    """The model input `(rows, members, columns)` of members that share its shape,
    read a block of cells at a time: a `HostRowSource` for the model's `x_host`.
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
    """Where `columns` holds a column of `of`, and that column's place in `of`."""
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
    """`table`'s `columns` at `rows`, as a new row-major float32 CPU tensor.

    Row-major as the built input's blocks are: a fit's reductions add up in an order
    that follows the layout, so another one can round a fitted value differently.
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
    elif isinstance(rows, slice):
        out = as_tensor_to_read(table[rows][:, columns])
    else:
        out = as_tensor_to_read(table[np.ix_(rows, columns)])
    return out.to(torch.float32).contiguous()
