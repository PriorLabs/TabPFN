"""Encoder step to encode categorical input features per feature."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tabpfn.architectures.encoders import SeqEncStep


class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
    """Expects input of size 1."""

    def __init__(
        self,
        num_features: int,
        emsize: int,
        base_encoder,  # noqa: ANN001
        num_embs: int = 1_000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert num_features == 1
        self.num_features = num_features
        self.emsize = emsize
        self.num_embs = num_embs
        self.embedding = nn.Embedding(num_embs, emsize)
        self.base_encoder = base_encoder

    def _fit(
        self, x: torch.Tensor, single_eval_pos: int, categorical_inds: list[int]
    ) -> None:
        pass

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        categorical_inds: list[int],
    ) -> tuple[torch.Tensor]:
        if categorical_inds is None:
            is_categorical = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        else:
            assert all(ci in ([0], []) for ci in categorical_inds), categorical_inds
            is_categorical = torch.tensor(
                [ci == [0] for ci in categorical_inds],
                device=x.device,
            )

        if is_categorical.any():
            lx = x[:, is_categorical]
            nan_mask = torch.isnan(lx) | torch.isinf(lx)
            lx = lx.long().clamp(0, self.num_embs - 2)
            lx[nan_mask] = self.num_embs - 1
            categorical_embs = self.embedding(lx.squeeze(-1))
        else:
            categorical_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        if (~is_categorical).any():
            lx = x[:, ~is_categorical]
            continuous_embs = self.base_encoder(lx, single_eval_pos=single_eval_pos)[0]
        else:
            continuous_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        # return (torch.cat((continuous_embs, categorical_embs), dim=1),)
        # above is wrong as we need to preserve order in the batch dimension
        embs = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.emsize,
            device=x.device,
            dtype=torch.float,
        )
        embs[:, is_categorical] = categorical_embs.float()
        embs[:, ~is_categorical] = continuous_embs.float()
        return (embs,)
