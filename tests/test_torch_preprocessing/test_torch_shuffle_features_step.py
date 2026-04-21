"""Tests for TorchShuffleFeaturesStep."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
import torch

from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.steps.shuffle_features_step import ShuffleFeaturesStep
from tabpfn.preprocessing.torch.steps import TorchShuffleFeaturesStep


@pytest.mark.parametrize("method", ["shuffle", "rotate", None])
def test__shuffle_features_step__same_permutation_as_cpu(
    method: Literal["shuffle", "rotate"] | None,
) -> None:
    seed: int = 42
    n_cols: int = 25
    shift_index: int = 7

    cpu_step = ShuffleFeaturesStep(
        shuffle_method=method,
        shuffle_index=shift_index,
        random_state=seed,
    )
    cpu_X: npt.NDArray[np.float64] = np.arange(n_cols).reshape(1, -1).astype(np.float64)
    cpu_schema = FeatureSchema(
        features=[Feature(name=None, modality=FeatureModality.NUMERICAL)] * n_cols
    )
    cpu_step._fit(cpu_X, cpu_schema)
    cpu_perm: list[int] = list(cpu_step.index_permutation_)

    gpu_step = TorchShuffleFeaturesStep(
        shuffle_method=method,
        shuffle_index=shift_index,
        random_state=seed,
    )
    gpu_X: torch.Tensor = torch.arange(n_cols, dtype=torch.float32).reshape(
        1, 1, n_cols
    )
    cache: dict[str, torch.Tensor] = gpu_step._fit(gpu_X)
    gpu_perm: list[int] = cache["permutation"].tolist()

    assert cpu_perm == gpu_perm
