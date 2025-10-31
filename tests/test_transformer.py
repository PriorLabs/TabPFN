from __future__ import annotations

from dataclasses import replace

import torch

from tabpfn.architectures.base.config import ModelConfig
from tabpfn.architectures.base.transformer import PerFeatureTransformer


def test__forward__thinking_rows_enabled__output_has_correct_shape() -> None:
    config = replace(_minimal_config(), num_thinking_rows=5)
    model = PerFeatureTransformer(config=config)
    batch = 2
    train_rows = 10
    eval_rows = 3
    total_rows = train_rows + eval_rows
    x_features = 2
    y_features = 1
    x = {"main": torch.randn(total_rows, batch, x_features)}
    y = {"main": torch.randn(train_rows, batch, y_features)}
    output = model(x, y)
    assert output.shape == (eval_rows, batch, 1)


def _minimal_config() -> ModelConfig:
    return ModelConfig(
        emsize=8,
        features_per_group=1,
        max_num_classes=10,
        nhead=2,
        nlayers=2,
        remove_duplicate_features=True,
        num_buckets=1000,
    )
