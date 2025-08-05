#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Literal, Optional
from typing_extensions import Self

import pydantic
from pydantic.dataclasses import dataclass

from tabpfn.architectures.interface import ArchitectureConfig

logger = logging.getLogger(__name__)

FeaturePositionalEmbedding = Optional[
    Literal["normal_rand_vec", "uni_rand_vec", "learned", "subspace"]
]


@dataclass
class ModelConfig(ArchitectureConfig):
    """Configuration for the base architecture."""

    # ------ Actual variation across configs
    emsize: int = 192
    """The embedding dimension."""
    features_per_group: Literal[1, 2] = 2
    """If > 1, the features will be grouped into groups of this size and the attention
    is across groups."""
    nhead: int = 6
    """Number of attention heads for both between-item and between-feature attention."""
    remove_duplicate_features: bool = False

    # --------

    # --- Constant across all configs and used
    dropout: float = 0.0
    encoder_use_bias: bool = False
    feature_positional_embedding: FeaturePositionalEmbedding = "subspace"
    multiquery_item_attention: Literal[False] = False
    """When True, uses multiquery for attention between items."""
    nan_handling_enabled: Literal[True] = True
    nan_handling_y_encoder: Literal[True] = True
    nhid_factor: int = 4
    """Hidden dimension in the MLP layers is ninp * nhid_factor."""
    nlayers: int = 12
    """Number of layers in the encoder, each consisting of
    a multi-head attention and an MLP layer."""
    normalize_by_used_features: Literal[True] = True
    normalize_on_train_only: Literal[True] = True
    normalize_to_ranking: Literal[False] = False
    normalize_x: Literal[True] = True
    recompute_attn: bool = False
    """If True, enables activation checkpointing for each attention  layer **and each
    MLP layer** in the encoder. This saves memory. recompute_layer is a related flag
    which checkpoints the input to each PerFeatureEncoderLayer."""
    recompute_layer: bool = True
    """If True, enables activation checkpointing for each PerFeatureEncoderLayer in the
    encoder. This saves memory. recompute_attn is a related flag which checkpoints the
    attention and mlp layers individually."""
    remove_empty_features: Literal[True] = True
    remove_outliers: Literal[False] = False
    use_separate_decoder: Literal[False] = False
    """If True, the decoder will be separate from the encoder."""

    multiquery_item_attention_for_test_set: Literal[True] = True
    """If true, uses multiquery attention on the test set."""

    attention_init_gain: float = 1.0
    """The gain when initializing the attention parameters. If None, then 1.0 is
    used."""
    # --------

    dag_pos_enc_dim: int | None = None

    item_attention_type: Literal["full"] = "full"
    feature_attention_type: Literal["full"] = "full"
    seed: int = 0
    """The seed to use for the model. The default 0 is chosen to match
    the default random_state of 0 in the TabPFN estimator,
    which was used to set this seed before
    (though I'm not sure it makes a difference for a trained model).
    """

    @classmethod
    def upgrade_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Upgrade old configs to match the current config.

        This allows backwards compatibility with  checkpoints.
        Raises a ValueError if the config is not compatible with the current code.
        """
        # The dates are to help us remove upgrades when they get very old.
        config = deepcopy(config)

        # Config changed on unknown date
        try:
            del config["use_flash_attention"]
            logger.debug(
                "`use_flash_attention` was specified in the config. This will be "
                "ignored and the attention implementation selected automatically."
            )
        except KeyError:
            pass

        # Config changed on 2025-05-22
        # Some keys were previously allowed to be None, and replaced with a default
        # value when they were used. Now we keep the default value in the configs and
        # None isn't allowed, so replace None with the default value.
        if "attention_init_gain" in config and config["attention_init_gain"] is None:
            config["attention_init_gain"] = cls._get_default("attention_init_gain")

        # Config changed on 2025-06-03
        if "attention_type" in config:
            if "item_attention_type" in config or "feature_attention_type" in config:
                raise ValueError("Can't have both old and new attention types set")
            config["item_attention_type"] = config["attention_type"]
            config["feature_attention_type"] = config["attention_type"]
            del config["attention_type"]

        # Config changed on 2025-06-04
        if config.get("canonical_y_encoder", False) is not False:
            raise ValueError("Current version only supports canonical_y_encoder=False")
        if config.get("bias", False) is not False:
            raise ValueError("Current version only supports bias=False")

        # Config changed on 2025-07-09
        if config.pop("two_sets_of_queries", False):
            raise ValueError("`two_sets_of_queries` is no longer supported in config")

        return config

    @classmethod
    def _get_default(cls, field: str) -> Any:
        return cls.__dataclass_fields__[field].default

    @pydantic.model_validator(mode="after")
    def validate_consistent(self) -> Self:
        if self.emsize % self.nhead != 0:
            raise ValueError("emsize must be divisible by nhead")
        return self
