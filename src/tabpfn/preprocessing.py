"""Compatibility shim for the preprocessing API."""
from __future__ import annotations

from tabpfn.preprocessors import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
