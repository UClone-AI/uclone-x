"""Telling a person's decision from a program's on the local API (#1589 item 6).

The dashboard's API answers any program on this computer, and that includes the model's own
shell: Clone's `bash_run` could `curl` a story's proposal digest and post the story view's
approve route, and the decision was then recorded as made in the story view by a person.
Every route that records a person's decision therefore asks for proof that the request came
from a dashboard window this server opened.

**The proof is a secret made when the server starts** (`secrets.token_urlsafe`), held in this
object and nowhere else: not in the environment, so no tool subprocess can inherit it, not on
disk and not in a log. A window receives it as an `HttpOnly` cookie in exchange for a one-time
pairing code, and the browser sends it back on its own.

**The pairing code reaches the window without passing through the API.** The server itself
opens the browser (`webbrowser.open`) at `/#pair=<code>`. The part after `#` is never sent
in a request, so the code is in no request line and no log; the page reads it, removes it
from the address bar and exchanges it (`POST /api/person/pair`), which spends it.

Why not the simpler channels, which a program on this computer can read:

* **The secret written into the served page, sent back as a header.** Anything the server
  serves, `curl http://127.0.0.1:<port>/` receives too.
* **An `Origin` check with that header.** `Origin` is a header like any other to `curl`; it
  constrains browsers, not programs.
* **A desktop shell (Electron, Tauri) handing it over.** This product has none: the
  dashboard is a page in the person's own browser.

A program can ask for a window (`POST /api/person/window`); the window opens in the person's
browser, and the program learns nothing it could use. It can ask again, so a window opens at
most once every `WINDOW_INTERVAL_SECONDS`, however it is asked for: the model cannot bury the
person in windows. What this does not stop: a program
running as the same user can read what the browser keeps on disk (its cookie store and its
history, which records the address with the code until the window spends it), and on Linux
the code is briefly on `xdg-open`'s command line. On macOS `webbrowser` hands the address to
`osascript` on its standard input, not its command line.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

__all__ = [
    "PAIRING_REFUSAL",
    "PERSON_REFUSAL",
    "WINDOW_FAILURE",
    "WINDOW_INTERVAL_SECONDS",
    "WINDOW_TOO_SOON",
    "PersonGate",
    "register_person_routes",
]

#: What a decision without the proof is answered with. No mechanism is named: the person is
#: told what to do, not how the check works.
PERSON_REFUSAL = (
    "Only you can make this decision, and this window could not show that it is you. "
    "Choose “Open a confirmed window”, then decide in the window that opens. Nothing was "
    "changed."
)

#: What a pairing code that is not the current one is answered with.
PAIRING_REFUSAL = (
    "This window could not be confirmed. Go back to the story and choose “Open a confirmed "
    "window” again."
)

#: What is said when no browser window could be opened on this computer.
WINDOW_FAILURE = "No browser window could be opened on this computer."

#: What a request for a window is answered with when one was opened a moment ago.
WINDOW_TOO_SOON = (
    "A confirmed window was opened a moment ago. Look for it in your browser, or try again "
    "in a few seconds."
)

#: The least time between two windows this server opens.
WINDOW_INTERVAL_SECONDS = 10.0

#: `Sec-Fetch-Site` values a decision may carry: the dashboard's own page, a request the
#: person typed, or no header at all (an older browser, which still needs the cookie).
_OWN_SITES = frozenset({"same-origin", "none"})


class PersonGate:
    """The secret that marks a dashboard window this server opened, and its pairing code.

    One per app, made in `create_ui_app`; nothing reads the secret but `pair` and
    `confirms`, and `repr` does not show it.
    """

    def __init__(
        self,
        opener: Callable[[str], bool] = webbrowser.open,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._secret = secrets.token_urlsafe(32)
        self._pairing: str | None = None
        self._last_window: float | None = None
        self._lock = threading.Lock()
        #: Opens a URL in the person's browser. Replaced in tests.
        self.opener = opener
        #: Seconds on a clock that only moves forward. Replaced in tests.
        self.clock = clock

    def __repr__(self) -> str:
        return "PersonGate(secret=<hidden>)"

    @staticmethod
    def cookie_name(port: int) -> str:
        """Per port: a browser keeps one cookie jar for every port of `127.0.0.1`."""
        return f"uclone_person_{port}"

    def window_url(self, base_url: str) -> str:
        """A URL that confirms the window it is opened in; any earlier code stops working."""
        code = secrets.token_urlsafe(32)
        with self._lock:
            self._pairing = code
        return f"{base_url.rstrip('/')}/#pair={code}"

    def may_open_window(self) -> bool:
        """Whether a window may open now; if so, the next may not for the interval.

        Counted when asked, not when a window opened: a request whose browser failed is
        still a request, and the model could otherwise ask as often as the failures come.
        """
        now = self.clock()
        with self._lock:
            last = self._last_window
            if last is not None and now - last < WINDOW_INTERVAL_SECONDS:
                return False
            self._last_window = now
            return True

    def open_window(self, base_url: str) -> bool:
        """Open a confirming window in the person's browser; False when none opened."""
        try:
            return bool(self.opener(self.window_url(base_url)))
        except Exception:
            return False

    def pair(self, code: str) -> str | None:
        """The secret, in exchange for the current pairing code, which is spent; else None."""
        with self._lock:
            current = self._pairing
            if current is None or not hmac.compare_digest(code.encode(), current.encode()):
                return None
            self._pairing = None
            return self._secret

    def confirms(self, request: Request) -> bool:
        """Whether `request` came from a window this server confirmed."""
        if _from_elsewhere(request):
            return False
        sent = request.cookies.get(self.cookie_name(_server_port(request)))
        return sent is not None and hmac.compare_digest(sent.encode(), self._secret.encode())

    def require(self, request: Request) -> None:
        """Refuse, with `PERSON_REFUSAL`, a request `confirms` does not accept."""
        if not self.confirms(request):
            raise HTTPException(status_code=403, detail=PERSON_REFUSAL)


def _from_elsewhere(request: Request) -> bool:
    """Whether the browser says the request came from a page other than the dashboard's."""
    site = request.headers.get("sec-fetch-site")
    return site is not None and site not in _OWN_SITES


def _server_port(request: Request) -> int:
    """The port this server answered on, from the socket, never from a header."""
    server: Any = request.scope.get("server")
    if isinstance(server, tuple | list) and len(server) >= 2 and isinstance(server[1], int):  # pyright: ignore[reportUnknownArgumentType]
        return server[1]
    return 80


def _server_base_url(request: Request) -> str:
    """This server's own address, from the socket rather than the `Host` header.

    A program on this computer chooses the `Host` it sends. Were the window opened at that
    name, a program listening on another port could serve the page that reads the code.
    """
    server: Any = request.scope.get("server")
    host = "127.0.0.1"
    if isinstance(server, tuple | list) and server and isinstance(server[0], str):  # pyright: ignore[reportUnknownArgumentType]
        host = server[0]
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{_server_port(request)}"


class _PairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str


def register_person_routes(app: FastAPI, gate: PersonGate) -> None:
    """Mount `/api/person/pair` and `/api/person/window`."""

    @app.post("/api/person/pair")
    async def pair_window(request: Request, body: _PairRequest) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Confirm this window: the pairing code it was opened with, for the cookie."""
        secret = gate.pair(body.code)
        if secret is None:
            raise HTTPException(status_code=403, detail=PAIRING_REFUSAL)
        response = JSONResponse({"confirmed": True})
        response.set_cookie(
            gate.cookie_name(_server_port(request)),
            secret,
            httponly=True,
            samesite="strict",
            path="/api/",
        )
        return response

    @app.post("/api/person/window")
    async def open_confirmed_window(request: Request) -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
        """Open a confirmed dashboard window in the person's browser, on this computer."""
        if _from_elsewhere(request):
            raise HTTPException(status_code=403, detail=WINDOW_FAILURE)
        if not gate.may_open_window():
            raise HTTPException(status_code=429, detail=WINDOW_TOO_SOON)
        if not gate.open_window(_server_base_url(request)):
            raise HTTPException(status_code=500, detail=WINDOW_FAILURE)
        return {"opened": True}
