"""Compatibility shim for legacy imports of :mod:`tabpfn.preprocessing`."""

from __future__ import annotations

from tabpfn import preprocessors as _preprocessors
from tabpfn.preprocessors import *  # noqa: F401,F403

__all__ = _preprocessors.__all__
