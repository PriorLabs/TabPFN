"""Compatibility shim for legacy imports of :mod:`tabpfn.preprocessing`.

DEPRECATED: Preprocessing logic has been reorganized into :mod:`tabpfn.preprocessors`.
Please update your imports to use the new location.
"""

from __future__ import annotations

import warnings

from tabpfn import preprocessors

# Re-export all members from the new package location
from tabpfn.preprocessors import *  # noqa: F403

warnings.warn(
    "The 'tabpfn.preprocessing' module is deprecated and has been moved to "
    "'tabpfn.preprocessors'. Please update your imports as this shim will "
    "be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Mirror the __all__ from the new location to maintain API parity and IDE support
__all__ = preprocessors.__all__
