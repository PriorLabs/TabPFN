from __future__ import annotations

import pytest
import torch

from tabpfn.model import encoders
from tabpfn.model.config import ModelConfig
from tabpfn.model.transformer import PerFeatureTransformer


@pytest.mark.parametrize(
    "multiquery_item_attention_for_test_set",
    [False, True],
)
@torch.inference_mode()
def test_separate_train_inference(multiquery_item_attention_for_test_set: bool):
    emsize = 16
    model = PerFeatureTransformer(
        config=ModelConfig(
            emsize=emsize,
            nhead=2,
            nhid_factor=1,
            nlayers=4,
            features_per_group=1,
            max_num_classes=1,
            remove_duplicate_features=False,
            num_buckets=1000,
            max_num_features=85,
        ),
        encoder=encoders.SequentialEncoder(
            encoders.InputNormalizationEncoderStep(
                normalize_on_train_only=True,
                normalize_to_ranking=False,
                normalize_x=True,
                remove_outliers=True,
            ),  # makes it more interesting
            encoders.LinearInputEncoderStep(
                num_features=1,
                emsize=emsize,
                in_keys=["main"],
                out_keys=["output"],
            ),
        ),
    )

    for p in model.parameters():
        p.add_(0.01)  # make it more interesting, not anymore mean 0

    model.feature_positional_embedding = None  # 'subspace'
    for layer in model.transformer_encoder.layers:
        layer.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )

    model.cache_trainset_representation = True
    model.reset_save_peak_mem_factor(None)
    model.empty_trainset_representation_cache()

    device = "cpu"

    n_train = 10
    n_features = 10
    n_test = 3
    batch_size = 2
    x_train = torch.normal(
        0.0,
        2.0,
        size=(n_train, batch_size, n_features),
        device=device,
    )
    y = (x_train[:, :, :1] > 1.0).float().to(device).to(torch.float)
    x_test = torch.normal(
        0.0,
        1.0,
        size=(n_test, batch_size, n_features),
        device=device,
    )

    torch.manual_seed(12345)
    model(x_train, y, single_eval_pos=n_train)
    logits1 = model(x_test, None, single_eval_pos=0)

    torch.manual_seed(12345)
    logits1a = model(train_x=x_train, train_y=y, test_x=x_test)

    assert logits1.float() == pytest.approx(
        logits1a.float(), abs=1e-5
    ), f"{logits1} != {logits1a}"
