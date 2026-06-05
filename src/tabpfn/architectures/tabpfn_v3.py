# ruff: noqa: PLR0912, C901,
"""TabPFN v3 architecture.


Shape suffix convention:

B: batch size
R: total rows (train + test)
Ri: input rows, could be either train + test or test.
Rj: Chunked rows (<= R)
N: train rows
M: test rows
C: total columns
Cj: Chunked columns (<= C)
E: embedding dimension
T: Target dim (e.g. number of classes).
Cl: number of CLS tokens

D: head dimension
H: num heads
S: sequence length

Copyright (c) Prior Labs GmbH 2026.
"""

from __future__ import annotations

import dataclasses
import logging as _logging
import math
import operator
import time
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from tabpfn.architectures.shared.mlp_residual_block import MLPResidualBlock
from tabpfn.architectures.shared.multihead_attention import MultiHeadAttention
from tabpfn.architectures.shared.normalization import Normalization
from tabpfn.preprocessing.torch.torch_standard_scaler import (
    TorchStandardScalerState,
)

if TYPE_CHECKING:
    from tabpfn.architectures.architecture_config import (
        ArchitectureConfig,
        PerformanceOptions,
    )
    from tabpfn.architectures.kv_cache import KVCache, KVCacheEntry, QuantizedKVCacheEntry
    from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace
    from tabpfn.architectures.shared.scaled_dot_product_attention import (
        scaled_dot_product_attention,
    )

from tabpfn.errors import is_oom_error
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler

if TYPE_CHECKING:
    from torch.nn.attention import SDPBackend
