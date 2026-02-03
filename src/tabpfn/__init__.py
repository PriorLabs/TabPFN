from importlib.metadata import version

# Configure MPS memory limits early to prevent macOS system crashes
# This must happen before any MPS tensor operations
import torch

if torch.backends.mps.is_available():
    from tabpfn.architectures.base.memory import configure_mps_memory_limits

    configure_mps_memory_limits()

from tabpfn.classifier import TabPFNClassifier
from tabpfn.misc.debug_versions import display_debug_info
from tabpfn.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from tabpfn.regressor import TabPFNRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
