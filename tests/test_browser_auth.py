"""Tests for the browser-based license acceptance module."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from tabpfn.browser_auth import (
    delete_cached_token,
    get_cached_token,
    save_token,
    verify_token,
)
from tabpfn.errors import TabPFNLicenseError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_token_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all token file paths to tmp_path so tests don't touch $HOME."""
    cache_dir = tmp_path / "cache" / "tabpfn"
    token_file = cache_dir / "auth_token"
    client_file = tmp_path / ".tabpfn" / "token"

    monkeypatch.setattr("tabpfn.browser_auth._CACHE_DIR", cache_dir)
    monkeypatch.setattr("tabpfn.browser_auth._TOKEN_FILE", token_file)
    monkeypatch.setattr("tabpfn.browser_auth._CLIENT_TOKEN_FILE", client_file)

    # Reset in-process cache so tests don't leak state.
    monkeypatch.setattr("tabpfn.browser_auth._accepted_repos", set())

    # Stub out HF API calls so tests don't make network requests.
    monkeypatch.setattr(
        "tabpfn.browser_auth._get_license_name",
        lambda repo_id: f"{repo_id}-license-v1.0",
    )

    # Clear env vars that could interfere.
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.delenv("TABPFN_NO_BROWSER", raising=False)


# ---------------------------------------------------------------------------
# get_cached_token
# ---------------------------------------------------------------------------


class TestGetCachedToken:
    def test_returns_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TABPFN_TOKEN", "  tok-from-env  ")
        assert get_cached_token() == "tok-from-env"

    def test_returns_from_token_file(self, tmp_path: Path):
        token_file = tmp_path / "cache" / "tabpfn" / "auth_token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("tok-from-file")
        assert get_cached_token() == "tok-from-file"

    def test_returns_from_client_file(self, tmp_path: Path):
        client_file = tmp_path / ".tabpfn" / "token"
        client_file.parent.mkdir(parents=True, exist_ok=True)
        client_file.write_text("tok-from-client")
        assert get_cached_token() == "tok-from-client"

    def test_env_var_takes_priority(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("TABPFN_TOKEN", "env-wins")
        token_file = tmp_path / "cache" / "tabpfn" / "auth_token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("file-token")
        assert get_cached_token() == "env-wins"

    def test_own_cache_takes_priority_over_client(self, tmp_path: Path):
        token_file = tmp_path / "cache" / "tabpfn" / "auth_token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("own-token")

        client_file = tmp_path / ".tabpfn" / "token"
        client_file.parent.mkdir(parents=True, exist_ok=True)
        client_file.write_text("client-token")

        assert get_cached_token() == "own-token"

    def test_returns_none_when_nothing_cached(self):
        assert get_cached_token() is None

    def test_skips_empty_files(self, tmp_path: Path):
        token_file = tmp_path / "cache" / "tabpfn" / "auth_token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("   ")
        assert get_cached_token() is None


# ---------------------------------------------------------------------------
# save_token / delete_cached_token
# ---------------------------------------------------------------------------


class TestSaveAndDeleteToken:
    def test_save_creates_file(self, tmp_path: Path):
        save_token("my-token")
        token_file = tmp_path / "cache" / "tabpfn" / "auth_token"
        assert token_file.read_text() == "my-token"

    def test_delete_removes_file(self, tmp_path: Path):
        save_token("my-token")
        token_file = tmp_path / "cache" / "tabpfn" / "auth_token"
        assert token_file.exists()
        delete_cached_token()
        assert not token_file.exists()

    def test_delete_is_idempotent(self):
        delete_cached_token()  # no file — should not raise
        delete_cached_token()


# ---------------------------------------------------------------------------
# verify_token
# ---------------------------------------------------------------------------


class _DummyHTTPResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def read(self) -> bytes:
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestVerifyToken:
    def test_valid_token(self):
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=_DummyHTTPResponse(200),
        ):
            assert verify_token("good-tok", "https://api.example.com") is True

    def test_invalid_token_401(self):
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=urllib.error.HTTPError(
                url="",
                code=401,
                msg="",
                hdrs=None,
                fp=None,  # type: ignore[arg-type]
            ),
        ):
            assert verify_token("bad-tok", "https://api.example.com") is False

    def test_invalid_token_403(self):
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=urllib.error.HTTPError(
                url="",
                code=403,
                msg="",
                hdrs=None,
                fp=None,  # type: ignore[arg-type]
            ),
        ):
            assert verify_token("bad-tok", "https://api.example.com") is False

    def test_server_unreachable(self):
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            assert verify_token("tok", "https://api.example.com") is None

    def test_unexpected_http_error(self):
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=urllib.error.HTTPError(
                url="",
                code=500,
                msg="",
                hdrs=None,
                fp=None,  # type: ignore[arg-type]
            ),
        ):
            assert verify_token("tok", "https://api.example.com") is None

    def test_url_construction(self):
        """Verify the endpoint URL is built correctly."""
        called_with: list[str] = []

        def capture_url(req, **_kw) -> _DummyHTTPResponse:
            called_with.append(req.full_url)
            return _DummyHTTPResponse(200)

        with patch.object(urllib.request, "urlopen", side_effect=capture_url):
            verify_token("tok", "https://api.example.com")

        assert called_with[0] == "https://api.example.com/protected/"

    def test_url_construction_trailing_slash(self):
        called_with: list[str] = []

        def capture_url(req, **_kw) -> _DummyHTTPResponse:
            called_with.append(req.full_url)
            return _DummyHTTPResponse(200)

        with patch.object(urllib.request, "urlopen", side_effect=capture_url):
            verify_token("tok", "https://api.example.com/")

        assert called_with[0] == "https://api.example.com/protected/"


# ---------------------------------------------------------------------------
# ensure_license_accepted
# ---------------------------------------------------------------------------


class TestEnsureLicenseAccepted:
    """Test the main entry point with various scenarios."""

    def _import_ensure(self):  # noqa: ANN202
        from tabpfn.browser_auth import ensure_license_accepted  # noqa: PLC0415

        return ensure_license_accepted

    def test_valid_cached_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TABPFN_TOKEN", "valid-tok")
        with (
            patch("tabpfn.browser_auth.verify_token", return_value=True),
            patch(
                "tabpfn.browser_auth.check_license_accepted",
                return_value=True,
            ),
        ):
            assert self._import_ensure()("tabpfn_2_6") is True

    def test_cached_token_server_unreachable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Server unreachable + cached token -> raise."""
        monkeypatch.setenv("TABPFN_TOKEN", "cached-tok")
        with (
            patch("tabpfn.browser_auth.verify_token", return_value=None),
            pytest.raises(TabPFNLicenseError, match="verify"),
        ):
            self._import_ensure()("tabpfn_2_6")

    def test_invalid_cached_token_triggers_browser(self, tmp_path: Path):
        """Invalid token should delete cache and attempt browser login."""
        token_file = tmp_path / "cache" / "tabpfn" / "auth_token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("expired-tok")

        with (
            patch(
                "tabpfn.browser_auth.verify_token",
                side_effect=[False, True],
            ),
            patch(
                "tabpfn.browser_auth.try_browser_login",
                return_value="new-valid-tok",
            ),
            patch(
                "tabpfn.browser_auth.check_license_accepted",
                return_value=True,
            ),
        ):
            assert self._import_ensure()("tabpfn_2_6") is True
            assert not token_file.read_text().startswith("expired")

    def test_no_browser_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        """TABPFN_NO_BROWSER=1 without token -> error."""
        monkeypatch.setenv("TABPFN_NO_BROWSER", "1")
        with pytest.raises(TabPFNLicenseError, match="TABPFN_NO_BROWSER"):
            self._import_ensure()("tabpfn_2_6")

    def test_no_browser_false_values_dont_block(self, monkeypatch: pytest.MonkeyPatch):
        """TABPFN_NO_BROWSER=0/false/no should NOT block browser login."""
        for val in ("0", "false", "no", "off"):
            monkeypatch.setenv("TABPFN_NO_BROWSER", val)
            with (
                patch(
                    "tabpfn.browser_auth.try_browser_login",
                    return_value="tok",
                ),
                patch(
                    "tabpfn.browser_auth.verify_token",
                    return_value=True,
                ),
                patch(
                    "tabpfn.browser_auth.check_license_accepted",
                    return_value=True,
                ),
            ):
                assert self._import_ensure()("tabpfn_2_6") is True

    def test_browser_login_returns_none_raises(self):
        """Browser login failure -> error."""
        with (
            patch(
                "tabpfn.browser_auth.try_browser_login",
                return_value=None,
            ),
            pytest.raises(TabPFNLicenseError, match="headless"),
        ):
            self._import_ensure()("tabpfn_2_6")

    def test_browser_token_rejected_raises(self):
        """Token from browser rejected by server -> error."""
        with (
            patch(
                "tabpfn.browser_auth.try_browser_login",
                return_value="bad-browser-tok",
            ),
            patch(
                "tabpfn.browser_auth.verify_token",
                return_value=False,
            ),
            pytest.raises(TabPFNLicenseError, match="rejected"),
        ):
            self._import_ensure()("tabpfn_2_6")
