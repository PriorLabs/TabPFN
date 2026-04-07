"""Tests for the browser-based license acceptance module."""

from __future__ import annotations

import contextlib
import io
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from tabpfn.browser_auth import (
    _has_display,
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
            pytest.raises(TabPFNLicenseError, match="no interactive terminal"),
        ):
            self._import_ensure()("tabpfn_2_6")

    def test_browser_login_returns_none_error_includes_steps(self):
        """Non-interactive error should include step-by-step instructions."""
        with (
            patch(
                "tabpfn.browser_auth.try_browser_login",
                return_value=None,
            ),
            pytest.raises(TabPFNLicenseError, match="TABPFN_TOKEN"),
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


# ---------------------------------------------------------------------------
# _has_display
# ---------------------------------------------------------------------------


class TestHasDisplay:
    def test_windows_always_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tabpfn.browser_auth.sys.platform", "win32")
        assert _has_display() is True

    def test_macos_local_session(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tabpfn.browser_auth.sys.platform", "darwin")
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        assert _has_display() is True

    def test_macos_ssh_without_x_forwarding(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tabpfn.browser_auth.sys.platform", "darwin")
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5678 5.6.7.8 22")
        monkeypatch.delenv("DISPLAY", raising=False)
        assert _has_display() is False

    def test_macos_ssh_with_x_forwarding(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tabpfn.browser_auth.sys.platform", "darwin")
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5678 5.6.7.8 22")
        monkeypatch.setenv("DISPLAY", "localhost:10.0")
        assert _has_display() is True

    def test_linux_with_x11(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tabpfn.browser_auth.sys.platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _has_display() is True

    def test_linux_with_wayland(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tabpfn.browser_auth.sys.platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert _has_display() is True

    def test_linux_headless(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tabpfn.browser_auth.sys.platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _has_display() is False


# ---------------------------------------------------------------------------
# _headless_interactive_login
# ---------------------------------------------------------------------------


class TestHeadlessCbreakLoop:
    """Tests for _headless_cbreak_loop (cbreak-mode input)."""

    def _import_cbreak_loop(self):  # noqa: ANN202
        from tabpfn.browser_auth import _headless_cbreak_loop  # noqa: PLC0415

        return _headless_cbreak_loop

    def _fake_stdin(self, chars: str) -> io.StringIO:
        """Create a fake stdin backed by a StringIO with a no-op fileno."""
        fake = io.StringIO(chars)
        fake.fileno = lambda: 0  # type: ignore[assignment]
        return fake

    @contextlib.contextmanager
    def _patch_termios(self):
        """Patch termios/tty so cbreak mode doesn't touch the real terminal."""
        import termios as _termios  # noqa: PLC0415
        import tty as _tty  # noqa: PLC0415

        with (
            patch.object(_termios, "tcgetattr", return_value=[]),
            patch.object(_termios, "tcsetattr"),
            patch.object(_tty, "setcbreak"),
        ):
            yield

    def test_returns_pasted_token(self, monkeypatch: pytest.MonkeyPatch):
        """Simulates pasting a full token followed by Enter."""
        cbreak_loop = self._import_cbreak_loop()
        fake = self._fake_stdin("eyJhbGciOiJIUzI1NiJ9\r")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)

        with self._patch_termios():
            token = cbreak_loop("https://ux.priorlabs.ai/login?hf_repo_id=tabpfn_2_6")

        assert token == "eyJhbGciOiJIUzI1NiJ9"

    def test_copy_then_paste(self, monkeypatch: pytest.MonkeyPatch):
        """Press c to copy, then paste a token."""
        cbreak_loop = self._import_cbreak_loop()
        # 'c' for copy, then token chars, then Enter
        fake = self._fake_stdin("cmytok\r")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)

        with (
            self._patch_termios(),
            patch("tabpfn.browser_auth._copy_osc52") as mock_osc52,
        ):
            token = cbreak_loop("https://ux.priorlabs.ai/login")

        assert token == "mytok"
        mock_osc52.assert_called_once_with("https://ux.priorlabs.ai/login")

    def test_eof_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """EOF on first read returns None."""
        cbreak_loop = self._import_cbreak_loop()
        fake = self._fake_stdin("")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)

        with self._patch_termios():
            assert cbreak_loop("https://ux.priorlabs.ai/login") is None

    def test_ctrl_c_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """Ctrl+C character returns None."""
        cbreak_loop = self._import_cbreak_loop()
        fake = self._fake_stdin("\x03")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)

        with self._patch_termios():
            assert cbreak_loop("https://ux.priorlabs.ai/login") is None

    def test_keyboard_interrupt_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """KeyboardInterrupt during read returns None."""
        cbreak_loop = self._import_cbreak_loop()
        fake = self._fake_stdin("")
        fake.read = lambda n: (_ for _ in ()).throw(KeyboardInterrupt)  # type: ignore[assignment,method-assign]
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)

        with self._patch_termios():
            assert cbreak_loop("https://ux.priorlabs.ai/login") is None

    def test_backspace_erases_char(self, monkeypatch: pytest.MonkeyPatch):
        """Backspace removes the previous character."""
        cbreak_loop = self._import_cbreak_loop()
        # Type 'ab', backspace, 'c', Enter → token is 'ac'
        fake = self._fake_stdin("ab\x7fc\r")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)

        with self._patch_termios():
            assert cbreak_loop("https://ux.priorlabs.ai/login") == "ac"


class TestHeadlessReadlineLoop:
    """Tests for _headless_readline_loop (fallback when termios unavailable)."""

    def _import_readline_loop(self):  # noqa: ANN202
        from tabpfn.browser_auth import _headless_readline_loop  # noqa: PLC0415

        return _headless_readline_loop

    def test_returns_token(self, monkeypatch: pytest.MonkeyPatch):
        readline_loop = self._import_readline_loop()
        fake = io.StringIO("my-token\n")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)
        assert readline_loop("https://ux.priorlabs.ai/login") == "my-token"

    def test_copy_then_token(self, monkeypatch: pytest.MonkeyPatch):
        readline_loop = self._import_readline_loop()
        fake = io.StringIO("c\nmy-token\n")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)
        with patch("tabpfn.browser_auth._copy_osc52") as mock_osc52:
            token = readline_loop("https://ux.priorlabs.ai/login")
        assert token == "my-token"
        mock_osc52.assert_called_once()

    def test_eof_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        readline_loop = self._import_readline_loop()
        fake = io.StringIO("")
        monkeypatch.setattr("tabpfn.browser_auth.sys.stdin", fake)
        assert readline_loop("https://ux.priorlabs.ai/login") is None


class TestHeadlessInteractiveLogin:
    """Integration tests for _headless_interactive_login routing."""

    def _import_headless(self):  # noqa: ANN202
        from tabpfn.browser_auth import _headless_interactive_login  # noqa: PLC0415

        return _headless_interactive_login

    def test_routes_to_cbreak_when_termios_available(self):
        headless_login = self._import_headless()
        with patch(
            "tabpfn.browser_auth._headless_cbreak_loop",
            return_value="tok",
        ) as mock_cbreak:
            token = headless_login("https://ux.priorlabs.ai", hf_repo_id="tabpfn_2_6")
        assert token == "tok"
        assert "tabpfn_2_6" in mock_cbreak.call_args[0][0]

    def test_routes_to_readline_without_termios(self):
        headless_login = self._import_headless()

        import builtins  # noqa: PLC0415

        _real_import = builtins.__import__

        def block_termios(name, *args, **kwargs):
            if name == "termios":
                raise ImportError("no termios")
            return _real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=block_termios),
            patch(
                "tabpfn.browser_auth._headless_readline_loop",
                return_value="tok",
            ) as mock_readline,
        ):
            token = headless_login("https://ux.priorlabs.ai")
        assert token == "tok"
        mock_readline.assert_called_once()

    def test_login_url_includes_hf_repo_id(
        self, capsys: pytest.CaptureFixture[str]
    ):
        headless_login = self._import_headless()
        with patch(
            "tabpfn.browser_auth._headless_cbreak_loop",
            return_value=None,
        ):
            headless_login("https://ux.priorlabs.ai", hf_repo_id="tabpfn_2_6")
        captured = capsys.readouterr()
        assert "hf_repo_id=tabpfn_2_6" in captured.out


# ---------------------------------------------------------------------------
# try_browser_login routing
# ---------------------------------------------------------------------------


class TestTryBrowserLoginRouting:
    def _import_try_login(self):  # noqa: ANN202
        from tabpfn.browser_auth import try_browser_login  # noqa: PLC0415

        return try_browser_login

    def test_non_interactive_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        """Non-TTY stdin → returns None without attempting any login."""
        try_login = self._import_try_login()
        with patch("tabpfn.browser_auth.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            assert try_login("https://ux.priorlabs.ai") is None

    def test_headless_routes_to_headless_login(self, monkeypatch: pytest.MonkeyPatch):
        """TTY + no display → delegates to _headless_interactive_login."""
        try_login = self._import_try_login()
        with (
            patch("tabpfn.browser_auth.sys.stdin") as mock_stdin,
            patch("tabpfn.browser_auth._has_display", return_value=False),
            patch(
                "tabpfn.browser_auth._headless_interactive_login",
                return_value="headless-tok",
            ) as mock_headless,
        ):
            mock_stdin.isatty.return_value = True
            result = try_login("https://ux.priorlabs.ai", hf_repo_id="tabpfn_2_6")

        assert result == "headless-tok"
        mock_headless.assert_called_once_with(
            "https://ux.priorlabs.ai", hf_repo_id="tabpfn_2_6"
        )

    def test_graphical_opens_browser(self, monkeypatch: pytest.MonkeyPatch):
        """TTY + display → opens browser (existing flow)."""
        try_login = self._import_try_login()
        with (
            patch("tabpfn.browser_auth.sys.stdin") as mock_stdin,
            patch("tabpfn.browser_auth._has_display", return_value=True),
            patch("tabpfn.browser_auth.webbrowser.open") as mock_browser,
            patch("tabpfn.browser_auth._poll_for_token", return_value="browser-tok"),
        ):
            mock_stdin.isatty.return_value = True
            result = try_login("https://ux.priorlabs.ai")

        assert result == "browser-tok"
        mock_browser.assert_called_once()
