from __future__ import annotations

import urllib.request
from pathlib import Path
from urllib.error import URLError

from tabpfn.model.loading import (
    FALLBACK_S3_BASE_URL,
    ModelSource,
    _try_direct_downloads,
)


class DummyResponse:
    def __init__(self, data: bytes = b"data", status: int = 200) -> None:
        """Store dummy response information."""
        self.data = data
        self.status = status

    def read(self) -> bytes:
        """Return response data."""
        return self.data

    def __enter__(self) -> DummyResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_try_direct_downloads_fallback(monkeypatch, tmp_path: Path) -> None:
    src = ModelSource.get_classifier_v2()
    dest = tmp_path / src.default_filename
    calls: list[str] = []

    def fake_urlopen(url: str):
        calls.append(url)
        if url.startswith("https://huggingface.co/"):
            raise URLError("hf down")
        return DummyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    _try_direct_downloads(dest, src)

    assert dest.exists()
    assert any(url.startswith(FALLBACK_S3_BASE_URL) for url in calls)
