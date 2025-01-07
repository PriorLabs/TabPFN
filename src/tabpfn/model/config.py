#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Literal


# TODO(eddiebergman):
# * Some parameters effect model architecture which only makes sense when training
# * Some parameters are used for training such as `epochs` (`batch_size`?)
# TODO(eddiebergman): Remove inheritance from `MutableMapping` once problems fixed
# TODO(eddiebergman): Anything with a default value basically has every config have it
#   to the same value, could consider removing those. In some cases, the code asserts
#   that it should be that value.
@dataclass
class InferenceConfig:
    """Configuration for the TabPFN model."""

    # ------ Actual variation across configs
    adaptive_max_seq_len_to_max_full_table_size: Literal[75000, 150000, 300000]
    batch_size: Literal[2, 4, 8]
    emsize: Literal[128, 192]
    features_per_group: Literal[1, 2]
    max_num_classes: Literal[0, 10]
    nhead: Literal[4, 6]
    remove_duplicate_features: bool
    seq_len: Literal[2000, 4000]
    task_type: Literal["multiclass", "regression"]
    # Only seems used in `get_loss` which transitively gets
    # used through bar_dist.num_buckets later
    num_buckets: Literal[1000, 5000]
    max_num_features: Literal[85, 90, 95]

    two_sets_of_queries: bool | None = None  # Defaulted to False when None in config
    # --------

    # --- Constant across all configs and used
    aggregate_k_gradients: Literal[1] = 1
    differentiable_hps_as_style: Literal[False] = False
    dropout: float = 0.0
    encoder_use_bias: Literal[False] = False
    feature_positional_embedding: Literal["subspace"] = "subspace"
    multiquery_item_attention: Literal[False] = False
    nan_handling_enabled: Literal[True] = True
    nan_handling_y_encoder: Literal[True] = True
    nhid_factor: Literal[4] = 4
    nlayers: Literal[12] = 12
    normalize_by_used_features: Literal[True] = True
    normalize_on_train_only: Literal[True] = True
    normalize_to_ranking: Literal[False] = False
    normalize_x: Literal[True] = True
    num_global_att_tokens: Literal[0] = 0
    progress_bar: Literal[False] = False
    recompute_attn: Literal[False] = False
    recompute_layer: Literal[True] = True
    remove_empty_features: Literal[True] = True
    remove_outliers: Literal[False] = False
    semisupervised_enabled: Literal[False] = False
    timing: Literal[False] = False
    use_separate_decoder: Literal[False] = False

    # This seems to no longer be used, and the multi-head-attention class
    # always uses it if it's available, there's no option to pass down
    use_flash_attention: Literal[False] = False  # asserted False

    # Seems to just set the config value "multiquery_item_attention_for_test_set"
    # to True. However this never triggers as multi_query_factor is always None
    # in all configs.
    #
    # > if (mqf := config.get("multi_query_factor")) and mqf > 1:
    # >   assert mqf == config["nhead"], "multi_query_factor must be equal to nhead"
    # >  config["multiquery_item_attention_for_test_set"] = True
    #
    # Also multiquery_item_attention_for_test_set is always True in all configs
    multi_query_factor: Literal[None] = None
    multiquery_item_attention_for_test_set: Literal[True] = True

    # Missing in some configs but the parameter default is set to 1.0 when this is None
    # Basically constant at 1.0
    attention_init_gain: float | None = None  # 1.0
    # --------

    # TODO(eddiebergman): Remove, we can just unpack directly
    # into the `Config` cls once we have fixed the stored model configs.
    @classmethod
    def from_dict(cls, config: dict) -> InferenceConfig:
        """Create a Config object from a dictionary.

        This method also does some sanity checking initially.
        """
        cls_fields = {field.name for field in dataclasses.fields(cls)}
        config_keys = set(config.keys())

        fields_in_config_not_in_cls = config_keys - cls_fields

        if any(fields_in_config_not_in_cls):
            warnings.warn(
                f"Fields in config not in Config class: {fields_in_config_not_in_cls}",
                stacklevel=2,
            )

        present_fields = config_keys.intersection(cls_fields)
        selected_config = {field: config[field] for field in present_fields}

        return cls(**selected_config)
