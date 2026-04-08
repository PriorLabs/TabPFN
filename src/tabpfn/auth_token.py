"""Token cache I/O for TabPFN authentication.

Pure I/O helpers with no dependencies on other TabPFN modules, so they
can be imported from both ``browser_auth`` and ``telemetry`` without
creating a circular import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "tabpfn"
_TOKEN_FILE = _CACHE_DIR / "auth_token"

# tabpfn-client stores its token here — we read it as a fallback.
_CLIENT_TOKEN_FILE = Path.home() / ".tabpfn" / "token"


def get_cached_token() -> str | None:
    """Return a cached token.

    Checks (in priority order):

    1. ``TABPFN_TOKEN`` environment variable
    2. ``~/.cache/tabpfn/auth_token``
    3. ``~/.tabpfn/token`` (tabpfn-client's cache)
    """
    env_token = os.environ.get("TABPFN_TOKEN")
    if env_token:
        return env_token.strip() or None

    for path in (_TOKEN_FILE, _CLIENT_TOKEN_FILE):
        if path.is_file():
            token = path.read_text().strip()
            if len(token) > 0:
                return token

    return None


def save_token(token: str) -> None:
    """Persist *token* to ``~/.cache/tabpfn/auth_token``."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(token)
    logger.debug("Token saved to %s", _TOKEN_FILE)


def delete_cached_token() -> None:
    """Remove the cached token file (if it exists)."""
    _TOKEN_FILE.unlink(missing_ok=True)
