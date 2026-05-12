#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, overload
from typing_extensions import override
from unittest.mock import patch

import pytest
import torch
from pydantic.dataclasses import dataclass
from torch import Tensor

from tabpfn import model_loading
from tabpfn.architectures import ARCHITECTURES, base, tabpfn_v3
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.architectures.base.config import ModelConfig
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    ArchitectureModule,
)
from tabpfn.architectures.tabpfn_v3 import TabPFNV3Config
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import PreprocessorConfig


def test__load_model__no_architecture_name_in_checkpoint__loads_base_architecture(
    tmp_path: Path,
) -> None:
    config = _get_minimal_base_architecture_config()
    model = base.get_architecture(config, cache_trainset_representation=True)
    checkpoint = {"state_dict": model.state_dict(), "config": asdict(config)}
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    loaded_model, _, loaded_config, _ = model_loading.load_model(path=checkpoint_path)
    assert isinstance(loaded_model, PerFeatureTransformer)
    assert isinstance(loaded_config, ModelConfig)


def _get_minimal_base_architecture_config() -> ModelConfig:
    return ModelConfig(
        emsize=8,
        features_per_group=1,
        max_num_classes=10,
        nhead=2,
        nlayers=2,
        remove_duplicate_features=True,
        num_buckets=1000,
    )


class FakeArchitectureModule(ArchitectureModule):
    @override
    def parse_config(
        self, config: dict[str, Any]
    ) -> tuple[ArchitectureConfig, dict[str, Any]]:
        return FakeConfig(**config), {}

    @override
    def get_architecture(
        self,
        config: ArchitectureConfig,
        *,
        cache_trainset_representation: bool,
    ) -> Architecture:
        return DummyArchitecture()


@dataclass
class FakeConfig(ArchitectureConfig):
    key_a: str = "a_value"


class DummyArchitecture(Architecture):
    """The interface that all architectures must implement.

    Architectures are PyTorch modules, which is then wrapped by e.g.
    TabPFNClassifier or TabPFNRegressor to form the complete model.
    """

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
        raise NotImplementedError()

    @property
    @override
    def embedding_dim(self) -> int:
        raise NotImplementedError()


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_model__architecture_name_in_checkpoint__loads_specified_architecture(
    tmp_path: Path,
) -> None:
    config_dict = {
        "max_num_classes": 10,
        "num_buckets": 100,
    }
    checkpoint = {
        "state_dict": {},
        "config": config_dict,
        "architecture_name": "fake_arch",
    }
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    loaded_model, _, loaded_config, _ = model_loading.load_model(path=checkpoint_path)
    assert isinstance(loaded_model, DummyArchitecture)
    assert isinstance(loaded_config, FakeConfig)


def test__save_tabpfn_model__stores_v3_architecture_and_inference_config(
    tmp_path: Path,
) -> None:
    config = TabPFNV3Config(max_num_classes=10, num_buckets=100)
    inference_config = InferenceConfig.get_default("multiclass", ModelVersion.V2_5)
    estimator = SimpleNamespace(
        models_=[torch.nn.Linear(1, 1)],
        configs_=[config],
        inference_config_=inference_config,
    )
    checkpoint_path = tmp_path / "checkpoint.ckpt"

    model_loading.save_tabpfn_model(estimator, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["architecture_name"] == "tabpfn_v3"
    assert checkpoint["config"]["name"] == "TabPFN-v3"
    assert checkpoint["inference_config"] == asdict(inference_config)


def test__load_v2_checkpoint__returns_v2_preprocessings(
    tmp_path: Path,
) -> None:
    architecture_config = _get_minimal_base_architecture_config()
    model = base.get_architecture(
        architecture_config, cache_trainset_representation=True
    )
    # v2 checkpoints have no "architecture_name" key
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": asdict(architecture_config),
    }
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    _, _, _, inference_config = model_loading.load_model_criterion_config(
        model_path=[checkpoint_path, checkpoint_path],
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2",
        download_if_not_exists=False,
    )

    assert len(inference_config.PREPROCESS_TRANSFORMS) == 2
    assert inference_config.PREPROCESS_TRANSFORMS[0].name == "quantile_uni_coarse"
    assert inference_config.PREPROCESS_TRANSFORMS[0].append_original == "auto"
    assert (
        inference_config.PREPROCESS_TRANSFORMS[0].categorical_name
        == "ordinal_very_common_categories_shuffled"
    )
    assert inference_config.PREPROCESS_TRANSFORMS[0].global_transformer_name == "svd"
    assert (
        inference_config.PREPROCESS_TRANSFORMS[0].max_features_per_estimator
        == 1_000_000
    )
    assert inference_config.PREPROCESS_TRANSFORMS[1].name == "none"
    assert inference_config.PREPROCESS_TRANSFORMS[1].categorical_name == "numeric"
    assert (
        inference_config.PREPROCESS_TRANSFORMS[1].max_features_per_estimator
        == 1_000_000
    )


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_v2_5_classification_ckpt__returns_v2_5_preprocessing(
    tmp_path: Path,
) -> None:
    # v2.5 checkpoints have a architecture_name but no inference_config
    # classification checkpoints have max_num_classes > 0
    architecture_config = {"max_num_classes": 10, "num_buckets": 100}
    checkpoint = {
        "state_dict": {},
        "config": architecture_config,
        "architecture_name": "fake_arch",
    }
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    _, _, _, inference_config = model_loading.load_model_criterion_config(
        model_path=[checkpoint_path, checkpoint_path],
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2.5",
        download_if_not_exists=False,
    )

    assert len(inference_config.PREPROCESS_TRANSFORMS) == 2
    assert inference_config.PREPROCESS_TRANSFORMS[0].name == "squashing_scaler_default"
    assert inference_config.PREPROCESS_TRANSFORMS[0].append_original is False
    assert (
        inference_config.PREPROCESS_TRANSFORMS[0].categorical_name
        == "ordinal_very_common_categories_shuffled"
    )
    assert (
        inference_config.PREPROCESS_TRANSFORMS[0].global_transformer_name
        == "svd_quarter_components"
    )
    assert inference_config.PREPROCESS_TRANSFORMS[0].max_features_per_estimator == 500
    assert inference_config.PREPROCESS_TRANSFORMS[1].name == "none"
    assert inference_config.PREPROCESS_TRANSFORMS[1].categorical_name == "numeric"
    assert inference_config.PREPROCESS_TRANSFORMS[1].max_features_per_estimator == 500


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_v2_5_regression_ckpt__returns_v2_5_preprocessing(
    tmp_path: Path,
) -> None:
    # v2.5 checkpoints have a architecture_name but no inference_config
    # regression checkpoints have max_num_classes 0
    architecture_config = {"max_num_classes": 0, "num_buckets": 100}
    checkpoint = {
        "state_dict": {
            "criterion.borders": torch.arange(101),
            "criterion.losses_per_bucket": torch.randn((100,)),
        },
        "config": architecture_config,
        "architecture_name": "fake_arch",
    }
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    _, _, _, inference_config = model_loading.load_model_criterion_config(
        model_path=[checkpoint_path, checkpoint_path],
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2.5",
        download_if_not_exists=False,
    )

    assert len(inference_config.PREPROCESS_TRANSFORMS) == 2
    assert inference_config.PREPROCESS_TRANSFORMS[0].name == "quantile_uni_coarse"
    assert inference_config.PREPROCESS_TRANSFORMS[0].append_original == "auto"
    assert inference_config.PREPROCESS_TRANSFORMS[0].categorical_name == "numeric"
    assert inference_config.PREPROCESS_TRANSFORMS[0].global_transformer_name is None
    assert inference_config.PREPROCESS_TRANSFORMS[1].name == "squashing_scaler_default"
    assert (
        inference_config.PREPROCESS_TRANSFORMS[1].categorical_name
        == "ordinal_very_common_categories_shuffled"
    )


def _build_small_v3_checkpoint(
    inference_config: InferenceConfig, *, max_num_classes: int
) -> dict:
    # v3 checkpoints store the inference_config directly; loading must round-trip
    # it without falling back to a version-specific default.
    config = TabPFNV3Config(
        max_num_classes=max_num_classes,
        num_buckets=5,
        embed_dim=48,
        nlayers=1,
        icl_num_heads=3,
        dist_embed_num_heads=3,
        feat_agg_num_heads=3,
    )
    model = tabpfn_v3.get_architecture(config, cache_trainset_representation=False)
    return {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "architecture_name": "tabpfn_v3",
        "inference_config": asdict(inference_config),
    }


def test__load_v3_classification_ckpt__returns_inference_config_from_checkpoint(
    tmp_path: Path,
) -> None:
    inference_config = InferenceConfig(
        PREPROCESS_TRANSFORMS=[
            PreprocessorConfig(
                "squashing_scaler_default",
                append_original=False,
                categorical_name="ordinal_very_common_categories_shuffled",
                global_transformer_name="svd_quarter_components",
                max_features_per_estimator=500,
            ),
        ]
    )
    checkpoint = _build_small_v3_checkpoint(inference_config, max_num_classes=10)
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    _, _, _, loaded_inference_config = model_loading.load_model_criterion_config(
        model_path=[checkpoint_path, checkpoint_path],
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v3",
        download_if_not_exists=False,
    )

    assert loaded_inference_config == inference_config


def test__load_v3_regression_ckpt__returns_bar_distribution_from_model_borders(
    tmp_path: Path,
) -> None:
    # v3 stores the loss criterion inside the model and returns no external
    # criterion from load_model. The regression path wraps the model's
    # `regression_borders` buffer into a FullSupportBarDistribution.
    inference_config = InferenceConfig(
        PREPROCESS_TRANSFORMS=[
            PreprocessorConfig("quantile_uni_coarse"),
        ]
    )
    checkpoint = _build_small_v3_checkpoint(inference_config, max_num_classes=0)
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)

    _, criterion, _, loaded_inference_config = (
        model_loading.load_model_criterion_config(
            model_path=[checkpoint_path],
            check_bar_distribution_criterion=True,
            cache_trainset_representation=False,
            which="regressor",
            version="v3",
            download_if_not_exists=False,
        )
    )

    assert isinstance(criterion, FullSupportBarDistribution)
    assert loaded_inference_config == inference_config


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_checkpoints_with_inference_configs__returns_inference_config(
    tmp_path: Path,
) -> None:
    architecture_config = {"max_num_classes": 10, "num_buckets": 100}
    inference_config = InferenceConfig(
        PREPROCESS_TRANSFORMS=[
            PreprocessorConfig(
                "quantile_uni_coarse",
                append_original="auto",
                categorical_name="ordinal_very_common_categories_shuffled",
                global_transformer_name="svd",
                max_features_per_estimator=-1,
            )
        ]
    )

    checkpoint_1 = {
        "state_dict": {},
        "config": architecture_config,
        "architecture_name": "fake_arch",
        "inference_config": asdict(inference_config),
    }
    checkpoint_1_path = tmp_path / "checkpoint1.ckpt"
    torch.save(checkpoint_1, checkpoint_1_path)
    checkpoint_2 = {
        "state_dict": {},
        "config": architecture_config,
        "architecture_name": "fake_arch",
        "inference_config": asdict(inference_config),
    }
    checkpoint_2_path = tmp_path / "checkpoint2.ckpt"
    torch.save(checkpoint_2, checkpoint_2_path)

    loaded_models, _, _, loaded_config = model_loading.load_model_criterion_config(
        model_path=[checkpoint_1_path, checkpoint_2_path],
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2",
        download_if_not_exists=False,
    )
    assert len(loaded_models) == 2
    assert loaded_config == inference_config


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_multiple_models_with_difference_inference_configs__raises(
    tmp_path: Path,
) -> None:
    architecture_config = {"max_num_classes": 10, "num_buckets": 100}
    checkpoint_1 = {
        "state_dict": {},
        "config": architecture_config,
        "architecture_name": "fake_arch",
        "inference_config": asdict(
            InferenceConfig(
                PREPROCESS_TRANSFORMS=[
                    PreprocessorConfig(
                        "quantile_uni_coarse",
                        append_original="auto",
                        categorical_name="ordinal_very_common_categories_shuffled",
                        global_transformer_name="svd",
                        max_features_per_estimator=-1,
                    )
                ]
            )
        ),
    }
    checkpoint_1_path = tmp_path / "checkpoint1.ckpt"
    torch.save(checkpoint_1, checkpoint_1_path)
    checkpoint_2 = {
        "state_dict": {},
        "config": architecture_config,
        "architecture_name": "fake_arch",
        "inference_config": asdict(
            InferenceConfig(
                PREPROCESS_TRANSFORMS=[
                    PreprocessorConfig(
                        "none",
                        categorical_name="numeric",
                        max_features_per_estimator=-1,
                    )
                ]
            )
        ),
    }
    checkpoint_2_path = tmp_path / "checkpoint2.ckpt"
    torch.save(checkpoint_2, checkpoint_2_path)

    with pytest.raises(ValueError, match="Inference configs for different models"):
        model_loading.load_model_criterion_config(
            model_path=[checkpoint_1_path, checkpoint_2_path],
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download_if_not_exists=False,
        )


def test__prepend_cache_path__single_path__filename_unchanged() -> None:
    full_path = model_loading.prepend_cache_path("my_path.test")
    assert Path(full_path).name == "my_path.test"


def test__prepend_cache_path__multiple_paths__filename_unchanged() -> None:
    full_paths = model_loading.prepend_cache_path(["my_dir/my_path.test", "another"])
    assert Path(full_paths[0]).name == "my_path.test"
    assert Path(full_paths[1]).name == "another"


@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
def test__load_model_criterion_config__parallel_downloads_do_not_crash(
    tmp_path: Path,
) -> None:
    """Test that parallel model downloads are properly synchronized by the file lock.

    This test verifies that when multiple threads attempt to download the same
    non-existent model simultaneously, only one download proceeds while the others
    wait for the lock.
    """
    # Track download attempts
    download_attempts: int = 0
    download_lock = threading.Lock()

    def mock_download_model(
        to: Path, **_kwargs: Any
    ) -> Literal["ok"] | list[Exception]:
        """Mock download that tracks concurrent access."""
        nonlocal download_attempts
        with download_lock:
            download_attempts += 1

        # Simulate a slow download to ensure overlap if locking doesn't work
        time.sleep(1)
        # Create a fake checkpoint to simulate a successful download
        architecture_config = {"max_num_classes": 10, "num_buckets": 100}
        fake_checkpoint = {
            "state_dict": {},
            "config": architecture_config,
            "architecture_name": "fake_arch",
        }

        # Write the fake checkpoint
        torch.save(fake_checkpoint, to)
        return "ok"

    def attempt_load_model() -> None:
        """Attempt to load a model, raising any exceptions that occur."""
        # Use the same model path for all threads to test locking
        shared_checkpoint_path: Path = tmp_path / "shared_model.ckpt"

        _, _, _, _ = model_loading.load_model_criterion_config(
            model_path=shared_checkpoint_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download_if_not_exists=True,
        )

    with patch.object(
        model_loading, "_download_model", side_effect=mock_download_model
    ):
        num_threads = 5
        completed = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(attempt_load_model) for _ in range(num_threads)]

            for future in as_completed(futures):
                future.result()  # Raises exception if the thread failed
                completed += 1

    # Verify that all threads completed successfully
    assert completed == num_threads, "Some threads failed to load model"

    # asserts only one download happened across 5 thread.
    assert download_attempts == 1, (
        f"Expected at most 1 concurrent download, got {download_attempts}. "
        "The file lock is not working correctly."
    )
