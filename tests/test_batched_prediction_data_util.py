#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

from unittest import mock

import pytest
import torch

from tabpfn.finetuning.data_util import (
    ClassifierBatch,
    _batch_shape_signature,
    _collate_same_shape_for_batched_inference,
    _group_batches_by_shape,
    meta_dataset_collator,
)


def _item(
    features: tuple[int, ...] = (3,),
    *,
    context_rows: int = 5,
    query_rows: int = 2,
) -> ClassifierBatch:
    return ClassifierBatch(
        X_context=[torch.zeros(context_rows, width) for width in features],
        X_query=[torch.zeros(query_rows, width) for width in features],
        y_context=[torch.zeros(context_rows) for _ in features],
        y_query=torch.zeros(query_rows),
        cat_indices=[[] for _ in features],
        configs=[object() for _ in features],
    )


@pytest.mark.parametrize(
    "changed",
    [
        _item((4, 3)),
        _item((3, 4)),
        _item(context_rows=6),
        _item(query_rows=3),
    ],
)
def test_batch_shape_signature_covers_all_model_inputs(
    changed: ClassifierBatch,
) -> None:
    assert _batch_shape_signature(_item((3, 3))) != _batch_shape_signature(changed)


def test_group_batches_by_shape_is_stable() -> None:
    items = [_item(), _item((4,)), _item(), _item((5,)), _item((4,))]
    groups = _group_batches_by_shape(items)
    assert [[index for index, _ in group] for group in groups] == [[0, 2], [1, 4], [3]]


def test_inference_collator_rejects_padding() -> None:
    with (
        mock.patch("tabpfn.finetuning.data_util.pad_tensors") as pad,
        pytest.raises(RuntimeError, match="heterogeneous"),
    ):
        _collate_same_shape_for_batched_inference([_item(), _item((4,))])
    pad.assert_not_called()


def test_inference_collator_rejects_empty_group() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        _collate_same_shape_for_batched_inference([])


def test_inference_collator_matches_general_collator() -> None:
    items = [_item(), _item()]
    safe = _collate_same_shape_for_batched_inference(items)
    general = meta_dataset_collator(items)
    for safe_values, general_values in zip(
        safe.X_context + safe.X_query + safe.y_context,
        general.X_context + general.X_query + general.y_context,
        strict=True,
    ):
        torch.testing.assert_close(safe_values, general_values)


def test_general_collator_still_pads() -> None:
    batch = meta_dataset_collator([_item(), _item((4,))])
    assert batch.X_context[0].shape == (2, 5, 4)
