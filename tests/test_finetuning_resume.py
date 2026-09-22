#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the finetuning checkpoint-resume opt-out (issue #814)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest import mock

import torch

from tabpfn.finetuning.finetuned_base import (
    FinetunedTabPFNBase,
    _resolve_resume_checkpoint,
)
from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
from tabpfn.finetuning.train_util import get_checkpoint_name


def _write_numbered_checkpoint(output_dir: Path, train_size: int, epoch: int) -> Path:
    path = output_dir / get_checkpoint_name(train_size, epoch)
    torch.save({"epoch": epoch}, path)
    return path


def test__resume_enabled__finds_existing_checkpoint(tmp_path: Path) -> None:
    train_size, epoch = 64, 3
    expected = _write_numbered_checkpoint(tmp_path, train_size, epoch)

    path, found_epoch = _resolve_resume_checkpoint(
        tmp_path, train_size, resume_from_checkpoint=True
    )

    assert path == expected
    assert found_epoch == epoch


def test__resume_disabled__ignores_existing_checkpoint(tmp_path: Path) -> None:
    train_size, epoch = 64, 3
    _write_numbered_checkpoint(tmp_path, train_size, epoch)

    with mock.patch(
        "tabpfn.finetuning.finetuned_base.get_checkpoint_path_and_epoch_from_output_dir"
    ) as lookup:
        path, found_epoch = _resolve_resume_checkpoint(
            tmp_path, train_size, resume_from_checkpoint=False
        )

    lookup.assert_not_called()
    assert path is None
    assert found_epoch == 0


def test__resume_enabled__empty_dir_starts_fresh(tmp_path: Path) -> None:
    path, found_epoch = _resolve_resume_checkpoint(
        tmp_path, 64, resume_from_checkpoint=True
    )

    assert path is None
    assert found_epoch == 0


def test__no_output_dir__starts_fresh() -> None:
    assert _resolve_resume_checkpoint(None, 64, resume_from_checkpoint=True) == (
        None,
        0,
    )
    assert _resolve_resume_checkpoint(None, 64, resume_from_checkpoint=False) == (
        None,
        0,
    )


def test__fit_signatures__offer_resume_opt_out() -> None:
    for cls in (
        FinetunedTabPFNBase,
        FinetunedTabPFNClassifier,
        FinetunedTabPFNRegressor,
    ):
        param = inspect.signature(cls.fit).parameters["resume_from_checkpoint"]
        assert param.default is True
