"""Browser-based license acceptance for TabPFN.

Opens a browser to the PriorLabs login page so the user can accept the
license.  The resulting JWT is cached locally for subsequent runs.

No dependency on tabpfn-client.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import select
import socketserver
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from tabpfn.errors import TabPFNLicenseError
from tabpfn.settings import settings

if TYPE_CHECKING:
    from typing import Literal

logger = logging.getLogger(__name__)

# In-process cache: tracks which HF repos have been confirmed this session.
# Short-circuits repeated calls within the same Python process.
_accepted_repos: set[str] = set()


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def _has_display() -> bool:
    """Heuristic: is a graphical display likely available for opening a browser?

    Returns ``True`` when it is reasonable to call :func:`webbrowser.open`.
    """
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        # macOS has a display unless we are in a pure SSH session
        # without X forwarding.
        return not (os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"))
    # Linux / other Unix: require X11 or Wayland.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ---------------------------------------------------------------------------
# Token cache helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------


def verify_token(token: str, api_url: str) -> bool | None:
    """Verify *token* against the PriorLabs API.

    Returns:
    -------
    True
        Token is valid.
    False
        Token is invalid / expired (server returned 401/403).
    None
        Server is unreachable — cannot verify.
    """
    url = f"{api_url.rstrip('/')}/protected/"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False
        logger.warning("Unexpected HTTP %s from token verification endpoint", exc.code)
        return None
    except Exception:
        logger.error("Token verification endpoint unreachable", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# HuggingFace model metadata
# ---------------------------------------------------------------------------


def _get_license_name(hf_repo_id: str) -> str | None:
    """Fetch the license_name from the HuggingFace API for a Prior-Labs repo.

    Returns the license_name string (e.g. ``"tabpfn-2.6-license-v1.0"``)
    or ``None`` if the request fails.
    """
    url = f"https://huggingface.co/api/models/Prior-Labs/{hf_repo_id}"
    try:
        req = urllib.request.Request(url)  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
            return data.get("cardData", {}).get("license_name")
    except Exception:  # noqa: BLE001
        logger.debug("Could not fetch license_name for %s", hf_repo_id, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# License acceptance check
# ---------------------------------------------------------------------------


def check_license_accepted(token: str, api_url: str, version: str) -> bool | None:
    """Check whether the user has accepted the TabPFN license.

    Returns:
    -------
    True
        License has been accepted.
    False
        License has not been accepted (or token invalid).
    None
        Server is unreachable — cannot verify.
    """
    encoded_version = urllib.parse.quote(version)
    url = f"{api_url.rstrip('/')}/account/license/?version={encoded_version}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
            return data.get("accepted", False)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False
        logger.warning("Unexpected HTTP %s from license check endpoint", exc.code)
        return None
    except Exception:  # noqa: BLE001
        logger.debug("License check endpoint unreachable", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Terminal helpers (headless-interactive flow)
# ---------------------------------------------------------------------------


def _copy_osc52(text: str) -> None:
    """Copy *text* to the system clipboard via the OSC 52 terminal escape.

    Works over SSH when the terminal emulator supports it (iTerm2, kitty,
    Windows Terminal, most modern terminals).
    """
    import base64  # noqa: PLC0415

    encoded = base64.b64encode(text.encode()).decode()
    sys.stdout.write(f"\033]52;c;{encoded}\a")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Browser login flow
# ---------------------------------------------------------------------------


def _create_callback_server(
    gui_url: str,
    auth_event: threading.Event,
    received_token: list[str | None],
) -> tuple[socketserver.TCPServer, int]:
    """Create a TCP server on an ephemeral port to receive the login callback.

    Returns ``(httpd, port)``.
    """

    class _CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if "token" in query:
                received_token[0] = query["token"][0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            page_style = (
                "<style>"
                "*{margin:0;padding:0;box-sizing:border-box}"
                "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
                "Roboto,Helvetica,Arial,sans-serif;background:#f3f4f6;"
                "color:#101075;display:flex;align-items:center;"
                "justify-content:center;min-height:100vh;"
                "-webkit-font-smoothing:antialiased}"
                ".card{background:#fff;border-radius:12px;padding:48px 40px;"
                "max-width:440px;width:90%;text-align:center;"
                "box-shadow:0 1px 3px rgba(0,0,0,.08)}"
                "h2{font-size:1.4rem;margin-bottom:12px}"
                "p{color:#6b7280;font-size:.95rem;line-height:1.6;"
                "margin-top:8px}"
                "a{color:#4d65ff;text-decoration:none}"
                "a:hover{text-decoration:underline}"
                ".logo{font-weight:700;font-size:1.1rem;letter-spacing:-.02em;"
                "color:#101075;margin-bottom:24px}"
                ".check{font-size:2.4rem;margin-bottom:16px}"
                ".warn{font-size:2.4rem;margin-bottom:16px}"
                "</style>"
            )

            if received_token[0] is not None:
                html = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,"
                    "initial-scale=1'><title>Prior Labs</title>"
                    f"{page_style}</head><body><div class='card'>"
                    "<div class='logo'>Prior Labs</div>"
                    "<div class='check'>&#10003;</div>"
                    "<h2>Login successful</h2>"
                    "<p>You can close this tab and return to your terminal.</p>"
                    "</div>"
                    "<script>window.location.href="
                    f'"{gui_url}/redirect-success";</script>'
                    "</body></html>"
                )
                self.wfile.write(html.encode())
                auth_event.set()
            else:
                html = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,"
                    "initial-scale=1'><title>Prior Labs</title>"
                    f"{page_style}</head><body><div class='card'>"
                    "<div class='logo'>Prior Labs</div>"
                    "<div class='warn'>&#9888;</div>"
                    "<h2>No token received</h2>"
                    "<p>Please paste your token in the terminal, or visit "
                    f'<a href="{gui_url}/account">{gui_url}/account</a> '
                    "to copy your Access Token.</p>"
                    "</div></body></html>"
                )
                self.wfile.write(html.encode())

        def log_message(self, format: str, *args: object) -> None:
            pass  # silence request logs

    httpd = socketserver.TCPServer(("", 0), _CallbackHandler)
    port = httpd.server_address[1]
    return httpd, port


def _serve_until_event(
    httpd: socketserver.TCPServer, auth_event: threading.Event
) -> None:
    """Handle HTTP requests until *auth_event* is set.

    Meant to run in a daemon thread.
    """
    httpd.timeout = 0.5
    while not auth_event.is_set():
        try:
            httpd.handle_request()
        except Exception:  # noqa: BLE001
            break


def _poll_for_token(
    auth_event: threading.Event, received_token: list[str | None]
) -> str | None:
    """Read token from stdin or wait for browser callback, whichever comes first."""
    sys.stdout.write("Token (or press Enter to keep waiting): ")
    sys.stdout.flush()
    while not auth_event.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:  # EOF
            return None
        token = line.strip()
        if token:
            return token
        sys.stdout.write("Token (or press Enter to keep waiting): ")
        sys.stdout.flush()
    return received_token[0]


def _headless_interactive_login(
    gui_url: str, hf_repo_id: str | None = None
) -> str | None:
    """Token acquisition for headless but interactive environments (e.g. SSH).

    Shows the login URL, offers single-keypress clipboard copy via OSC 52,
    and waits for the user to paste a token.

    Returns the JWT on success, or ``None`` on abort / EOF.
    """
    login_url = f"{gui_url}/login"
    if hf_repo_id:
        login_url += f"?hf_repo_id={urllib.parse.quote(hf_repo_id)}"

    print(  # noqa: T201
        "\nTabPFN requires a one-time license acceptance.\n"
        "\nNo display detected. Open this URL in a browser on another device:\n"
        f"\n  {login_url}\n"
        f"\nAfter logging in, accept the license on the Licenses tab,\n"
        f"then copy your Access Token from\n"
        f"  {gui_url}/account\n"
    )

    try:
        import termios  # noqa: PLC0415
    except ImportError:
        termios = None  # type: ignore[assignment]

    if termios is not None:
        return _headless_cbreak_loop(login_url)

    # Fallback when termios is unavailable (shouldn't happen on Unix,
    # but be safe).
    return _headless_readline_loop(login_url)


def _read_token_cbreak(first_char: str) -> str | None:
    """Read token characters in cbreak mode, echoing manually.

    *first_char* is the character that was already read (and not ``c``).
    Returns the completed token string, or ``None`` on EOF / Ctrl+C.
    """
    chars = [first_char]
    sys.stdout.write(first_char)
    sys.stdout.flush()
    while True:
        ch = sys.stdin.read(1)
        if not ch or ch == "\x03":
            sys.stdout.write("\n")
            return None
        if ch in ("\r", "\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(chars).strip() or None
        if ch in ("\x7f", "\x08"):  # Backspace / Delete
            if chars:
                chars.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        chars.append(ch)
        sys.stdout.write(ch)
        sys.stdout.flush()


def _headless_cbreak_loop(login_url: str) -> str | None:
    """Headless input loop using cbreak mode (single-keypress, no Enter)."""
    import termios  # noqa: PLC0415
    import tty  # noqa: PLC0415

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            sys.stdout.write(
                "  [c] Copy URL to clipboard    Paste your token to continue\n\n> "
            )
            sys.stdout.flush()

            ch = sys.stdin.read(1)
            if not ch or ch == "\x03":  # EOF / Ctrl+C
                sys.stdout.write("\n")
                return None
            if ch in ("c", "C"):
                _copy_osc52(login_url)
                sys.stdout.write("\r> \u2713 Copied to clipboard\n\n")
                sys.stdout.flush()
                continue

            token = _read_token_cbreak(ch)
            if token:
                return token
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _headless_readline_loop(login_url: str) -> str | None:
    """Headless input loop using readline (Enter required, termios unavailable)."""
    try:
        while True:
            sys.stdout.write("  Type [c]+Enter to copy URL, or paste your token:\n\n> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                return None
            text = line.strip()
            if text.lower() == "c":
                _copy_osc52(login_url)
                sys.stdout.write("\u2713 Copied to clipboard\n\n")
                sys.stdout.flush()
                continue
            if text:
                return text
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return None


def try_browser_login(gui_url: str, hf_repo_id: str | None = None) -> str | None:
    """Obtain a token via browser callback and/or manual paste concurrently.

    Chooses the right strategy based on the environment:

    * **Non-interactive** (no TTY): returns ``None`` immediately.
    * **Headless interactive** (TTY but no display): shows the URL and waits
      for the user to paste a token.
    * **Graphical** (TTY + display): opens the browser and runs a local
      callback server alongside a paste prompt.

    Returns the JWT on success, or ``None`` on failure / non-TTY environments.
    """
    if not sys.stdin.isatty():
        return None

    if not _has_display():
        return _headless_interactive_login(gui_url, hf_repo_id=hf_repo_id)

    auth_event = threading.Event()
    received_token: list[str | None] = [None]

    # --- callback server ---
    try:
        httpd, port = _create_callback_server(gui_url, auth_event, received_token)
    except Exception:  # noqa: BLE001
        logger.debug("Could not create callback server", exc_info=True)
        return None

    callback_url = f"http://localhost:{port}"
    login_url = f"{gui_url}/login?callback={callback_url}"
    if hf_repo_id:
        login_url += f"&hf_repo_id={urllib.parse.quote(hf_repo_id)}"

    server_thread = threading.Thread(
        target=_serve_until_event, args=(httpd, auth_event), daemon=True
    )
    server_thread.start()

    # --- open browser ---
    webbrowser.open(login_url)

    # --- print unified instructions ---
    print(  # noqa: T201
        "\nTabPFN requires a one-time license acceptance."
        "\nOpening your browser to complete login/registration…\n"
        f"\n  {login_url}\n"
        "\nWaiting for login to complete…\n"
        "\nHaving trouble? You can also authenticate manually:\n"
        f"  1. Open {gui_url}/account in a browser"
        " (log in or register if needed)\n"
        "  2. Accept the license at"
        f" {gui_url}/account/licenses\n"
        "  3. Copy your Access Token\n"
        "  4. Paste the token below\n"
    )

    # --- main thread: poll stdin while waiting for callback ---
    try:
        token = _poll_for_token(auth_event, received_token)
    except KeyboardInterrupt:
        token = None

    httpd.server_close()
    return token


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ensure_license_accepted(hf_repo_id: str) -> Literal[True]:  # noqa: C901
    """Ensure the user has accepted the TabPFN license.

    Checks for a cached token, verifies it, and falls back to browser login
    if needed.

    Returns ``True`` on success.

    Raises:
    ------
    TabPFNLicenseError
        If the license cannot be accepted (no browser, server unreachable
        without cached token, etc.).
    """
    # In-process cache: skip API call if already confirmed this session.
    if hf_repo_id in _accepted_repos:
        return True

    gui_url = settings.tabpfn.auth_gui_url
    api_url = settings.tabpfn.auth_api_url

    # Resolve the canonical license version string from HF; fall back to repo ID.
    license_version = _get_license_name(hf_repo_id) or hf_repo_id

    token = get_cached_token()
    if token is not None:
        status = verify_token(token, api_url)
        if status is True:
            # Token is valid — now check license acceptance.
            license_status = check_license_accepted(token, api_url, license_version)
            if license_status is True:
                save_token(token)
                _accepted_repos.add(hf_repo_id)
                return True
            if license_status is None:
                raise TabPFNLicenseError(
                    "Could not reach the license server to verify acceptance.\n\n"
                    "Please check your internet connection and try again."
                )
            # license_status is False — license not yet accepted.
            # Fall through to browser login so the GUI can show the acceptance form.
            logger.info(
                "Token valid but license not accepted; opening browser for acceptance.",
            )
        elif status is None:
            raise TabPFNLicenseError(
                "Could not reach the license server to verify your token.\n\n"
                "Please check your internet connection and try again."
            )
        else:
            # status is False — invalid/expired token.
            logger.info("Cached token is invalid; deleting and re-authenticating.")
            delete_cached_token()

    # No valid cached token — need browser login.
    no_browser = os.environ.get("TABPFN_NO_BROWSER", "").strip()
    if no_browser and no_browser not in ("0", "false", "no", "off"):
        raise TabPFNLicenseError(
            "TabPFN requires license acceptance, but browser login is\n"
            "disabled (TABPFN_NO_BROWSER is set).\n\n"
            "Set the TABPFN_TOKEN environment variable with a valid token\n"
            "obtained from https://ux.priorlabs.ai"
        )

    token = try_browser_login(gui_url, hf_repo_id=hf_repo_id)
    if token is None:
        raise TabPFNLicenseError(
            "TabPFN requires license acceptance, but no interactive terminal\n"
            "is available.\n\n"
            "To authenticate in a non-interactive environment:\n"
            f"  1. Open {gui_url} in a browser and log in (or register)\n"
            f"  2. Accept the license on the Licenses tab\n"
            f"  3. Copy your Access Token from {gui_url}/account\n"
            '  4. Set the environment variable: export TABPFN_TOKEN="<your-token>"\n'
            "     or in Python (before calling .fit()):"
            ' import os; os.environ["TABPFN_TOKEN"] = "<your-token>"'
        )

    # Verify the token we just received from the browser.
    status = verify_token(token, api_url)
    if status is False:
        raise TabPFNLicenseError(
            "The token received from the browser login was rejected by the\n"
            "server.  Please try again or contact support@priorlabs.ai"
        )

    # Token is valid (status is True or None/unreachable — save it regardless).
    save_token(token)

    license_status = check_license_accepted(token, api_url, license_version)
    if license_status is True:
        print("License accepted — token cached for future sessions.\n")  # noqa: T201
        _accepted_repos.add(hf_repo_id)
        return True
    if license_status is None:
        raise TabPFNLicenseError(
            "Could not reach the license server to verify acceptance.\n\n"
            "Please check your internet connection and try again."
        )
    # license_status is False
    encoded = urllib.parse.quote(hf_repo_id)
    raise TabPFNLicenseError(
        "License not yet accepted. Please complete the acceptance form at\n"
        f"{gui_url}/accept-license?hf_repo_id={encoded} and try again."
    )
