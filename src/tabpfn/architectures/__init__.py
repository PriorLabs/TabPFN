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


def register_architecture(name: str, module: ArchitectureModule) -> None:
    """Add an architecture, from an external source, to the available architectures.

    This allows checkpoints containing this architecture to be loaded by tabpfn.

    Raises:
        ValueError: If a module different from the one specified is already registered
            for the given name.
    """
    if name in ARCHITECTURES and ARCHITECTURES[name] is not module:
        raise ValueError(
            f"There is already a different architecture registered for '{name}': "
            f"{ARCHITECTURES[name]}"
        )
    ARCHITECTURES[name] = module
