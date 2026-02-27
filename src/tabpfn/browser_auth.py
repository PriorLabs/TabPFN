"""Browser-based license acceptance for TabPFN v2.5.

Opens a browser to the PriorLabs login page so the user can accept the
license.  The resulting JWT is cached locally for subsequent runs.

Stdlib-only (no dependency on tabpfn-client).
"""

from __future__ import annotations

import http.server
import logging
import os
import select
import socketserver
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token cache helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "tabpfn"
_TOKEN_FILE = _CACHE_DIR / "auth_token"

# tabpfn-client stores its token here — we read it as a fallback.
_CLIENT_TOKEN_FILE = Path.home() / ".tabpfn" / "token"


def get_cached_token() -> str | None:
    """Return a cached token, checking (in priority order):

    1. ``TABPFN_TOKEN`` environment variable
    2. ``~/.cache/tabpfn/auth_token``
    3. ``~/.tabpfn/token`` (tabpfn-client's cache)
    """
    env_token = os.environ.get("TABPFN_TOKEN")
    if env_token:
        return env_token.strip()

    for path in (_TOKEN_FILE, _CLIENT_TOKEN_FILE):
        if path.is_file():
            token = path.read_text().strip()
            if token:
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

    Returns
    -------
    True
        Token is valid.
    False
        Token is invalid / expired (server returned 401/403).
    None
        Server is unreachable — cannot verify.
    """
    url = f"{api_url.rstrip('/')}/protected/"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False
        logger.warning("Unexpected HTTP %s from token verification endpoint", exc.code)
        return None
    except Exception:  # noqa: BLE001
        logger.debug("Token verification endpoint unreachable", exc_info=True)
        return None


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
        def do_GET(self) -> None:  # noqa: N802
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
                    f'<script>window.location.href="{gui_url}/redirect-success";</script>'
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

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # silence request logs

    httpd = socketserver.TCPServer(("", 0), _CallbackHandler)
    port = httpd.server_address[1]
    return httpd, port


def _serve_until_event(
    httpd: socketserver.TCPServer, auth_event: threading.Event
) -> None:
    """Handle HTTP requests until *auth_event* is set.  Meant to run in a daemon thread."""
    httpd.timeout = 0.5
    while not auth_event.is_set():
        try:
            httpd.handle_request()
        except Exception:
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


def try_browser_login(gui_url: str) -> str | None:
    """Obtain a token via browser callback and/or manual paste concurrently.

    Both the local callback server and the paste prompt run at the same time
    so that neither blocks the other.

    Returns the JWT on success, or ``None`` on failure / non-TTY environments.
    """
    if not sys.stdin.isatty():
        return None

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

    server_thread = threading.Thread(
        target=_serve_until_event, args=(httpd, auth_event), daemon=True
    )
    server_thread.start()

    # --- open browser ---
    webbrowser.open(login_url)

    # --- print unified instructions ---
    print(
        "\nTabPFN v2.5 requires a one-time license acceptance."
        "\nOpening your browser to complete login/registration…\n"
        f"\n  {login_url}\n"
        "\nWaiting for login to complete…\n"
        "\nHaving trouble? You can also authenticate manually:\n"
        f"  1. Open {gui_url}/account in a browser (log in or register if needed)\n"
        "  2. Copy your Access Token\n"
        "  3. Paste the token below\n"
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


def ensure_license_accepted() -> Literal[True]:
    """Ensure the user has accepted the TabPFN v2.5 license.

    Checks for a cached token, verifies it, and falls back to browser login
    if needed.

    Returns ``True`` on success.

    Raises
    ------
    TabPFNLicenseError
        If the license cannot be accepted (no browser, server unreachable
        without cached token, etc.).
    """
    from tabpfn.errors import TabPFNLicenseError  # noqa: PLC0415
    from tabpfn.settings import settings  # noqa: PLC0415

    gui_url = settings.tabpfn.auth_gui_url
    api_url = settings.tabpfn.auth_api_url

    token = get_cached_token()
    if token is not None:
        status = verify_token(token, api_url)
        if status is True:
            # Token from env var or file is valid — make sure it's persisted.
            save_token(token)
            return True
        if status is None:
            # Server unreachable — accept optimistically.
            logger.info(
                "Could not reach the license server; accepting cached token."
            )
            return True
        # status is False — invalid/expired token.
        logger.info("Cached token is invalid; deleting and re-authenticating.")
        delete_cached_token()

    # No valid cached token — need browser login.
    no_browser = os.environ.get("TABPFN_NO_BROWSER", "").strip()
    if no_browser and no_browser not in ("0", "false", "no", "off"):
        raise TabPFNLicenseError(
            "TabPFN v2.5 requires license acceptance, but browser login is\n"
            "disabled (TABPFN_NO_BROWSER is set).\n\n"
            "Set the TABPFN_TOKEN environment variable with a valid token\n"
            "obtained from https://ux.priorlabs.ai"
        )

    token = try_browser_login(gui_url)
    if token is None:
        raise TabPFNLicenseError(
            "Browser login did not complete successfully.\n\n"
            "If you are in a headless environment, set the TABPFN_TOKEN\n"
            "environment variable with a valid token obtained from\n"
            "https://ux.priorlabs.ai"
        )

    # Verify the token we just received.
    status = verify_token(token, api_url)
    if status is False:
        raise TabPFNLicenseError(
            "The token received from the browser login was rejected by the\n"
            "server.  Please try again or contact support@priorlabs.ai"
        )

    save_token(token)
    print("License accepted — token cached for future sessions.\n")
    return True
