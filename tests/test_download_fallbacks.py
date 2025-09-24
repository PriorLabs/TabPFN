from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from tabpfn.model_loading import (
    FALLBACK_S3_BASE_URL,
    ModelSource,
    _try_direct_downloads,
)


class DummyResponse:
    """Simple context manager mimicking ``urllib`` responses."""

    def __init__(self, status: int = 200, data: bytes = b"ok") -> None:
        """Create a dummy response with a given status and payload."""
        self.status = status
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def test_direct_download_fallback(tmp_path: Path):
    src = ModelSource.get_classifier_v2()
    base_path = tmp_path / src.default_filename

    attempted_urls: list[str] = []

    def fake_urlopen(url: str, *_args, **_kwargs) -> DummyResponse:
        attempted_urls.append(url)
        if "huggingface.co" in url:
            raise urllib.error.URLError("HF down")
        return DummyResponse()

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        _try_direct_downloads(base_path, src)

    assert any(url.startswith("https://huggingface.co") for url in attempted_urls)
    assert any(url.startswith(FALLBACK_S3_BASE_URL) for url in attempted_urls)
