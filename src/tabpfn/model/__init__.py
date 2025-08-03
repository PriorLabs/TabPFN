"""Contains references to the base architecture, for backwards compatability.

DEPRECATED: import tabpfn.architectures.base instead

Previously tabpfn only supported a single architecture, which was in this tabpfn.model
module. Now we support multiple architectures, stored in tabpfn.architectures, and
tabpfn.model has moved to tabpfn.architectures.base .
"""

import warnings

from tabpfn import model_loading as loading
from tabpfn.architectures.base import (
    attention,
    bar_distribution,
    config,
    encoders,
    layer,
    memory,
    mlp,
    preprocessing,
    transformer,
)

__all__ = [
    "attention",
    "bar_distribution",
    "config",
    "encoders",
    "layer",
    "loading",
    "memory",
    "mlp",
    "preprocessing",
    "transformer",
]

warnings.warn(
    "tabpfn.model has moved to tabpfn.architectures.base. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
