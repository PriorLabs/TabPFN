"""Test the inference engines."""

from __future__ import annotations

from typing import Literal, overload
from typing_extensions import override

import pytest
import torch
from numpy.random import default_rng
from torch import Tensor

from tabpfn.architectures.interface import Architecture
from tabpfn.inference import InferenceEngineCachePreprocessing, InferenceEngineOnDemand
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    PreprocessorConfig,
    generate_classification_ensemble_configs,
)
from tabpfn.preprocessing.ensemble import TabPFNEnsembleFactory
from tabpfn.preprocessing.torch import FeatureSchema


class TestModel(Architecture):
    def __init__(self) -> None:
        """Create a new instance."""
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))

    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[True] = True,
        categorical_inds: list[list[int]] | None = None,
        force_recompute_layer: bool = False,
        save_peak_memory_factor: int | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[False],
        categorical_inds: list[list[int]] | None = None,
        force_recompute_layer: bool = False,
        save_peak_memory_factor: int | None = None,
    ) -> dict[str, Tensor]: ...

    @override
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        force_recompute_layer: bool = False,
        save_peak_memory_factor: int | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Perform a forward pass, see doc string of `Architecture`."""
        assert isinstance(x, Tensor)
        assert isinstance(y, Tensor)
        n_train_test, _, _ = x.shape
        n_train, _ = y.shape
        test_rows = n_train_test - n_train
        return x.sum(-2, keepdim=True).sum(-1, keepdim=True).reshape(-1, test_rows)

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
    n_train = 100
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    ensemble_preprocessor = TabPFNEnsembleFactory(
        configs=_create_test_ensemble_configs(
            n_configs=5,
            n_classes=3,
            num_models=1,
        ),
        rng=rng,
        # We want to test n_preprocessing_jobs>1 as this might mean the outputs are not
        # in the same order as the input configs, and we want to check that the parallel
        # evaluation code behaves correctly in this scenario.
        n_preprocessing_jobs=5,
    )
    engine = InferenceEngineCachePreprocessing(
        X_train,
        y_train,
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        ensemble_preprocessor=ensemble_preprocessor,
        models=[TestModel()],
        devices=[torch.device("cpu")],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem=True,
        inference_mode=True,
    )

    engine.to([torch.device("cpu")], force_inference_dtype=None, dtype_byte_size=4)
    outputs_sequential = list(engine.iter_outputs(X_test, autocast=False))
    engine.to(
        [torch.device("cpu"), torch.device("cpu")],
        force_inference_dtype=None,
        dtype_byte_size=4,
    )
    outputs_parallel = list(engine.iter_outputs(X_test, autocast=False))

    assert len(outputs_sequential) == len(outputs_parallel)
    for par_output, par_config in outputs_parallel:
        seq_output = _find_seq_output(par_config, outputs_sequential)
        assert isinstance(seq_output, Tensor)
        assert isinstance(par_output, Tensor)
        assert torch.allclose(seq_output, par_output)


def test__cache_preprocessing__with_outlier_removal() -> None:
    def get_outputs(
        outlier_removal_std: float | None = None,
    ) -> list[tuple[torch.Tensor | dict, EnsembleConfig]]:
        rng = default_rng(seed=0)
        n_train = 50
        n_features = 4
        n_classes = 3
        X_train = rng.standard_normal(size=(n_train, n_features))
        X_train[0:10] = 500  # outliers
        y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
        X_test = rng.standard_normal(size=(2, n_features))

        num_models = 1
        models = [TestModel() for _ in range(num_models)]
        ensemble_preprocessor = TabPFNEnsembleFactory(
            configs=_create_test_ensemble_configs(
                n_configs=5,
                n_classes=3,
                num_models=num_models,
                outlier_removal_std=outlier_removal_std,
            ),
            rng=rng,
            n_preprocessing_jobs=1,
        )
        engine = InferenceEngineOnDemand(
            X_train,
            y_train,
            feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
            ensemble_preprocessor=ensemble_preprocessor,
            models=models,
            devices=[torch.device("cpu")],
            dtype_byte_size=4,
            force_inference_dtype=None,
            save_peak_mem=True,
        )
        engine.to([torch.device("cpu")], force_inference_dtype=None, dtype_byte_size=4)
        return list(engine.iter_outputs(X_test, autocast=False))

    outputs_outlier_removed = get_outputs(outlier_removal_std=1.0)
    outputs_outlier_not_removed = get_outputs(outlier_removal_std=None)

    assert len(outputs_outlier_removed) == len(outputs_outlier_not_removed)
    for outlier_removed_output, outlier_not_removed_output in zip(
        outputs_outlier_removed, outputs_outlier_not_removed
    ):
        assert isinstance(outlier_removed_output[0], Tensor)
        assert isinstance(outlier_not_removed_output[0], Tensor)
        assert not torch.allclose(
            outlier_removed_output[0], outlier_not_removed_output[0]
        )


def test__on_demand__result_equal_in_serial_and_in_parallel() -> None:
    rng = default_rng(seed=0)
    n_train = 100
    n_features = 4
    n_classes = 3
    X_train = rng.standard_normal(size=(n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
    X_test = rng.standard_normal(size=(2, n_features))

    num_models = 3
    models = [TestModel() for _ in range(num_models)]
    ensemble_preprocessor = TabPFNEnsembleFactory(
        configs=_create_test_ensemble_configs(
            n_configs=5,
            n_classes=3,
            num_models=num_models,
        ),
        rng=rng,
        # We want to test n_preprocessing_jobs>1 as this might mean the outputs are not
        # in the same order as the input configs, and we want to check that the parallel
        # evaluation code behaves correctly in this scenario.
        n_preprocessing_jobs=5,
    )
    engine = InferenceEngineOnDemand(
        X_train,
        y_train,
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        ensemble_preprocessor=ensemble_preprocessor,
        models=models,
        devices=[torch.device("cpu")],
        dtype_byte_size=4,
        force_inference_dtype=None,
        save_peak_mem=True,
    )

    engine.to([torch.device("cpu")], force_inference_dtype=None, dtype_byte_size=4)
    outputs_sequential = list(engine.iter_outputs(X_test, autocast=False))
    engine.to(
        [torch.device("cpu"), torch.device("cpu")],
        force_inference_dtype=None,
        dtype_byte_size=4,
    )
    outputs_parallel = list(engine.iter_outputs(X_test, autocast=False))

    assert len(outputs_sequential) == len(outputs_parallel)
    for par_output, par_config in outputs_parallel:
        seq_output = _find_seq_output(par_config, outputs_sequential)
        assert isinstance(seq_output, Tensor)
        assert isinstance(par_output, Tensor)
        assert torch.allclose(seq_output, par_output)


def _create_test_ensemble_configs(
    n_configs: int,
    n_classes: int,
    num_models: int,
    outlier_removal_std: float | None = None,
) -> list[ClassifierEnsembleConfig]:
    preprocessor_configs = [
        PreprocessorConfig(
            "quantile_uni_coarse",
            append_original="auto",
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
            max_features_per_estimator=500,
        ),
        PreprocessorConfig(
            "none",
            categorical_name="numeric",
            max_features_per_estimator=500,
        ),
    ]
    return generate_classification_ensemble_configs(
        num_estimators=n_configs,
        subsample_samples=None,
        max_index=n_classes - 1,
        add_fingerprint_feature=True,
        polynomial_features="all",
        feature_shift_decoder="shuffle",
        preprocessor_configs=preprocessor_configs,
        class_shift_method=None,
        n_classes=n_classes,
        random_state=0,
        num_models=num_models,
        outlier_removal_std=outlier_removal_std,
    )


def _find_seq_output(
    config: EnsembleConfig,
    outputs_sequential: list[tuple[Tensor | dict, EnsembleConfig]],
) -> Tensor | dict:
    """Find the sequential output corresponding to the given config.

    The configs are not hashable, so we have to resort to this search method.
    """
    for output, trial_config in outputs_sequential:
        if trial_config == config:
            return output

    return pytest.fail(f"Parallel config was not found in sequential configs: {config}")
