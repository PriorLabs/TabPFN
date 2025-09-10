"""Test the inference engines."""

from __future__ import annotations

from typing import Literal, overload
from typing_extensions import override

import torch
from numpy.random import default_rng
from torch import Tensor

from tabpfn.architectures.interface import Architecture
from tabpfn.inference import InferenceEngineCachePreprocessing, InferenceEngineOnDemand
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    PreprocessorConfig,
)


class TestModel(Architecture):
    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[True] = True,
        categorical_inds: list[list[int]] | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[False],
        categorical_inds: list[list[int]] | None = None,
    ) -> dict[str, Tensor]: ...

    @override
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
    ) -> Tensor | dict[str, Tensor]:
        return torch.zeros(size=(10, 1, 10))

    @property
    def ninp(self) -> int:
        return 2

    @property
    def features_per_group(self) -> int:
        return 2

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        pass


def test__cache_preprocessing__result_equal_in_serial_and_in_parallel() -> None:
    rng = default_rng(seed=0)
    n_train = 10
    X_train = rng.standard_normal(size=(n_train, 1, 2))
    y_train = rng.standard_normal(size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, 1, 2))

    ensemble_config = ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig(name="power", categorical_name="none"),
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_count=0,
        feature_shift_decoder="shuffle",
        subsample_ix=None,
        class_permutation=None,
    )
    engine = InferenceEngineCachePreprocessing.prepare(
        X_train,
        y_train,
        cat_ix=[0] * n_train,
        model=TestModel(),
        ensemble_configs=[ensemble_config] * 2,
        n_workers=0,
        rng=rng,
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem=True,
        inference_mode=True,
    )

    outputs_sequential = list(
        engine.iter_outputs(X_test, devices=[torch.device("cpu")], autocast=False)
    )
    outputs_parallel = list(
        engine.iter_outputs(
            X_test, devices=[torch.device("cpu"), torch.device("cpu")], autocast=False
        )
    )
    assert len(outputs_sequential) == len(outputs_parallel)
    for (seq_output, seq_config), (par_output, par_config) in zip(
        outputs_sequential, outputs_parallel
    ):
        assert isinstance(seq_output, Tensor)
        assert isinstance(par_output, Tensor)
        assert torch.allclose(seq_output, par_output)
        assert seq_config == par_config


def test__on_demand__result_equal_in_serial_and_in_parallel() -> None:
    rng = default_rng(seed=0)
    n_train = 10
    n_estimators = 5
    X_train = rng.standard_normal(size=(n_train, 1, 2))
    y_train = rng.standard_normal(size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, 1, 2))

    ensemble_config = ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig(name="power", categorical_name="none"),
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_count=0,
        feature_shift_decoder="shuffle",
        subsample_ix=None,
        class_permutation=None,
    )
    engine = InferenceEngineOnDemand.prepare(
        X_train,
        y_train,
        cat_ix=[0] * n_train,
        model=TestModel(),
        ensemble_configs=[ensemble_config] * n_estimators,
        n_workers=0,
        rng=rng,
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem=True,
    )

    outputs_sequential = list(
        engine.iter_outputs(X_test, devices=[torch.device("cpu")], autocast=False)
    )
    outputs_parallel = list(
        engine.iter_outputs(
            X_test, devices=[torch.device("cpu"), torch.device("cpu")], autocast=False
        )
    )
    assert len(outputs_sequential) == len(outputs_parallel)
    for (seq_output, seq_config), (par_output, par_config) in zip(
        outputs_sequential, outputs_parallel
    ):
        assert isinstance(seq_output, Tensor)
        assert isinstance(par_output, Tensor)
        assert torch.allclose(seq_output, par_output)
        assert seq_config == par_config
