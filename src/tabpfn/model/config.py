#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import dataclasses
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


# TODO(eddiebergman): Anything with a default value basically has every config have it
#   to the same value, could consider removing those. In some cases, the code asserts
#   that it should be that value.
@dataclass
class ModelConfig:
    """Configuration for the TabPFN model."""

    # ------ Actual variation across configs
    emsize: int
    """The embedding dimension."""
    features_per_group: Literal[1, 2]
    """If > 1, the features will be grouped into groups of this size and the attention
    is across groups."""
    max_num_classes: int
    nhead: int
    """Number of attention heads for both between-item and between-feature attention."""
    remove_duplicate_features: bool
    # Only seems used in `get_loss` which transitively gets
    # used through bar_dist.num_buckets later
    num_buckets: Literal[1000, 5000]
    max_num_features: Literal[85, 90, 95]

    two_sets_of_queries: bool = False
    # --------

    # --- Constant across all configs and used
    dropout: float = 0.0
    encoder_use_bias: bool = False
    feature_positional_embedding: Literal["subspace"] = "subspace"
    multiquery_item_attention: Literal[False] = False
    """When True, uses multiquery for attention between items."""
    nan_handling_enabled: Literal[True] = True
    nan_handling_y_encoder: Literal[True] = True
    nhid_factor: Literal[4] = 4
    """Hidden dimension in the MLP layers is ninp * nhid_factor."""
    nlayers: Literal[12] = 12
    """Number of layers, each consisting of a multi-head attention + MLP layer."""
    normalize_by_used_features: Literal[True] = True
    normalize_on_train_only: Literal[True] = True
    normalize_to_ranking: Literal[False] = False
    normalize_x: Literal[True] = True
    recompute_attn: Literal[False] = False
    """If True, enables activation checkpointing for each attention  layer **and each
    MLP layer** in the encoder. This saves memory. recompute_layer is a related flag
    which checkpoints the input to each PerFeatureEncoderLayer."""
    recompute_layer: Literal[True] = True
    """If True, enables activation checkpointing for each PerFeatureEncoderLayer in the
    encoder. This saves memory. recompute_attn is a related flag which checkpoints the
    attention and mlp layers individually."""
    remove_empty_features: Literal[True] = True
    remove_outliers: Literal[False] = False
    use_separate_decoder: Literal[False] = False
    """If True, the decoder will be separate from the encoder."""

    # This seems to no longer be used, and the multi-head-attention class
    # always uses it if it's available, there's no option to pass down
    use_flash_attention: Literal[False] = False  # asserted False

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

    # TODO(eddiebergman): Remove, we can just unpack directly
    # into the `Config` cls once we have fixed the stored model configs.
    @classmethod
    def from_dict(cls, config: dict) -> ModelConfig:
        """Create a Config object from a dictionary.

        This method also does some sanity checking initially.
        """
        cls_fields = {field.name for field in dataclasses.fields(cls)}
        config_keys = set(config.keys())

        fields_in_config_not_in_cls = config_keys - cls_fields

        if (
            any(fields_in_config_not_in_cls) and False
        ):  # disabled in public release for prints
            warnings.warn(
                f"Fields in config not in Config class: {fields_in_config_not_in_cls}",
                stacklevel=2,
            )

        present_fields = config_keys.intersection(cls_fields)
        selected_config = {field: config[field] for field in present_fields}

        return cls(**selected_config)

    @classmethod
    def upgrade_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Upgrade old configs to match the current config.

        This allows backwards compatibility with  checkpoints.
        Raises a ValueError if the config is not compatible with the current code.
        """
        # The dates are to help us remove upgrades when they get very old.

        config = deepcopy(config)

        # Config changed on 2025-05-22
        # Some keys were previously allowed to be None, and replaced with a default
        # value when they were used. Now we keep the default value in the configs and
        # None isn't allowed, so replace None with the default value.
        if "attention_init_gain" in config and config["attention_init_gain"] is None:
            config["attention_init_gain"] = cls._get_default("attention_init_gain")
        if "two_sets_of_queries" in config and config["two_sets_of_queries"] is None:
            config["two_sets_of_queries"] = cls._get_default("two_sets_of_queries")

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

        return config

    @classmethod
    def _get_default(cls, field: str) -> Any:
        return cls.__dataclass_fields__[field].default
