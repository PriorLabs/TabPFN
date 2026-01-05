"""A collection of random utilities for the TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations


class TabPFNError(Exception):
    """Base class for all TabPFN-specific exceptions."""


class TabPFNUserError(TabPFNError):
    """Base class for errors caused by invalid user input (safe to map to HTTP 400)."""


class TabPFNValidationError(ValueError, TabPFNUserError):
    """User provided invalid data (shape, NaNs, categories, etc.)."""
