"""Contains a collection of different model architectures.

"Architecture" refers to a PyTorch module, which is then wrapped by e.g.
TabPFNClassifier or TabPFNRegressor to form the complete model.

Each submodule in this module should contain an architecture. Each may be a directory,
or just a single file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import base

if TYPE_CHECKING:
    from tabpfn.architectures.interface import ArchitectureModule

ARCHITECTURES: dict[str, ArchitectureModule] = {"base": base}
"""Map from architecture names to the corresponding module."""
