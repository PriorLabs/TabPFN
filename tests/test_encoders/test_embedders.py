from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.encoders import (
    LinearFeatureGroupEmbedder,
    MLPFeatureGroupEmbedder,
)


def test_linear_embedder_forward():
    N, B, num_features, emsize = 10, 3, 7, 5
    x = torch.randn([N, B, num_features])
    projection = LinearFeatureGroupEmbedder(
        num_features_per_group_with_metadata=num_features, emsize=emsize
    )
    out = projection(x)
    assert out.shape == (N, B, emsize)


@pytest.mark.parametrize("num_layers", [2, 3])
def test_mlp_projection_forward(num_layers: int):
    N, B, num_features, emsize = 10, 3, 4, 8
    x = torch.randn([N, B, num_features])
    projection = MLPFeatureGroupEmbedder(
        num_features_per_group_with_metadata=num_features,
        emsize=emsize,
        num_layers=num_layers,
    )
    out = projection(x)
    assert out.shape == (N, B, emsize)
