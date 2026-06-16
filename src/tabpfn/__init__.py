#  Copyright (c) Prior Labs GmbH 2026.

import sys
import types
import warnings
from importlib.metadata import version

from tabpfn import model_loading
from tabpfn.classifier import TabPFNClassifier
from tabpfn.errors import TabPFNCUDAOutOfMemoryError, TabPFNMPSOutOfMemoryError
from tabpfn.misc.debug_versions import display_debug_info
from tabpfn.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from tabpfn.regressor import TabPFNRegressor


def _install_legacy_model_aliases() -> None:
    """Keep the removed ``tabpfn.model.loading`` import path working.

    This was listed as deprecated but without a warning in the code.
    Stillf used by tabpfn-extensions via autogluon.
    """
    model_module = types.ModuleType("tabpfn.model")
    model_module.__doc__ = "Deprecated alias namespace; see tabpfn.model_loading."
    loading_module = types.ModuleType("tabpfn.model.loading")

    def _forward(name: str) -> object:
        if hasattr(model_loading, name):
            warnings.warn(
                "Importing from 'tabpfn.model.loading' is deprecated and will be "
                "removed in a future release; import from 'tabpfn.model_loading' "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(model_loading, name)
        raise AttributeError(
            f"module 'tabpfn.model.loading' has no attribute {name!r}",
        )

    loading_module.__getattr__ = _forward  # type: ignore[attr-defined]
    model_module.loading = loading_module  # type: ignore[attr-defined]

    sys.modules.setdefault("tabpfn.model", model_module)
    sys.modules.setdefault("tabpfn.model.loading", loading_module)


_install_legacy_model_aliases()


try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNCUDAOutOfMemoryError",
    "TabPFNClassifier",
    "TabPFNMPSOutOfMemoryError",
    "TabPFNRegressor",
    "__version__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
