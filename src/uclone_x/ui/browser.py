"""Settings ▸ Browser and the extension's link (design `browser-agent.md` §3.6).

- `WS /api/browser/extension`: the UClone-X extension dials here from U0's Chrome. Only an
  extension origin is let in; the pairing token is checked by `ExtensionHub.serve`.
- `GET /api/browser/extension`: whether an extension is connected, the pairing code to
  paste into it, and the folder to load unpacked.
- `POST /api/browser/extension/pairing-code`: a new code; the paired extension is dropped
  and must be given the new one.
- `WS /api/browser/{conversation_id}`: the dock's Browser tab, live (§3.4, §6 step 3):
  state, screencast frames, overlay and step notices from `browser.live.LiveView`, and the
  take-over controls. Only a page of this dashboard is let in.
"""

from __future__ import annotations

import contextlib
import json
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websockets.exceptions import ConnectionClosed

from uclone_x.browser.extension import EXTENSION_FOLDER, ExtensionHub, pairing_code
from uclone_x.browser.live import LiveView
from uclone_x.browser.service import BrowserService
from uclone_x.browser.tool import default_browser_service
from uclone_x.llm.connectors.saved_choice import settings_data, update_settings_file

EXTENSION_ORIGIN_PREFIX = "chrome-extension://"
TOKEN_KEY = "browser_extension_token"

_SETTINGS_UNREADABLE = (
    "UClone-X could not make a pairing code because its settings could not be read or "
    "saved. Check that the UClone-X folder is writable and try again."
)


def saved_pairing_token(settings_file: Path) -> str | None:
    """The current token, or None when none was made yet (or the file cannot be read)."""
    token = settings_data(settings_file).get(TOKEN_KEY)
    return token if isinstance(token, str) and token else None


def pairing_token(settings_file: Path) -> str:
    """The current token, made and saved on first use.

    Raises `ValueError` when the settings file exists but cannot be read, and `OSError`
    when it cannot be written.
    """
    token = saved_pairing_token(settings_file)
    if token is not None:
        return token
    return regenerate_pairing_token(settings_file)


def regenerate_pairing_token(settings_file: Path) -> str:
    """Replace the token (128 random bits). An extension paired with the old one is refused."""
    token = secrets.token_hex(16)
    update_settings_file({TOKEN_KEY: token}, path=settings_file)
    return token


class StarletteSocket:
    """FastAPI's WebSocket with the websockets-style surface `CdpConnection` reads."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def recv(self) -> str:
        try:
            return await self._ws.receive_text()
        except (WebSocketDisconnect, RuntimeError) as exc:
            raise ConnectionClosed(None, None) from exc

    async def send(self, message: str) -> None:
        try:
            await self._ws.send_text(message)
        except (WebSocketDisconnect, RuntimeError) as exc:
            raise ConnectionClosed(None, None) from exc

    async def __aiter__(self) -> AsyncIterator[str]:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            async for message in self._ws.iter_text():
                yield message

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self._ws.application_state == WebSocketState.DISCONNECTED:
            return
        with contextlib.suppress(RuntimeError):
            await self._ws.close(code, reason)


class LiveSocketAdapter:
    """FastAPI's WebSocket with the surface `LiveView` reads."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def send_json(self, message: dict[str, Any]) -> None:
        await self._ws.send_text(json.dumps(message, ensure_ascii=False))

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def receive_json(self) -> dict[str, Any] | None:
        try:
            text = await self._ws.receive_text()
        except (WebSocketDisconnect, RuntimeError, KeyError):
            return None
        try:
            value: object = json.loads(text)
        except ValueError:
            return {}
        return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def register_browser_routes(
    app: FastAPI,
    *,
    hub: ExtensionHub,
    settings_file: Path,
    refuse_cross_origin: Callable[[Request], None],
    service: BrowserService | None = None,
    on_stop: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Mount the extension's link, the browser dock WebSocket, and Settings ▸ Browser routes."""

    def _service() -> BrowserService:
        nonlocal service
        if service is None:
            service = default_browser_service()
        return service

    def _view(request: Request, token: str) -> dict[str, Any]:
        server = request.scope.get("server")
        port = int(server[1]) if server and server[1] else 80
        return {
            **hub.status(),
            "pairing_code": pairing_code(port, token),
            "extension_folder": str(EXTENSION_FOLDER),
        }

    def _unreadable() -> JSONResponse:
        return JSONResponse(
            {"reason_code": "settings_unreadable", "message": _SETTINGS_UNREADABLE},
            status_code=503,
        )

    @app.websocket("/api/browser/extension")
    async def extension_link(websocket: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        """The extension's socket: pairing, then CDP to the tabs it opens for clones."""
        origin = websocket.headers.get("origin", "")
        if not origin.startswith(EXTENSION_ORIGIN_PREFIX):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await hub.serve(StarletteSocket(websocket))

    @app.get("/api/browser/extension", response_model=None)
    async def extension_status(request: Request) -> dict[str, Any] | JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Whether the extension is connected, and the pairing code to give it."""
        refuse_cross_origin(request)
        try:
            token = pairing_token(settings_file)
        except (ValueError, OSError):
            return _unreadable()
        return _view(request, token)

    @app.post("/api/browser/extension/pairing-code", response_model=None)
    async def new_pairing_code(request: Request) -> dict[str, Any] | JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Replace the pairing code; the extension paired with the old one is dropped."""
        refuse_cross_origin(request)
        try:
            token = regenerate_pairing_token(settings_file)
        except (ValueError, OSError):
            return _unreadable()
        await hub.disconnect()
        return _view(request, token)

    @app.websocket("/api/browser/{conversation_id}")
    async def browser_dock_ws(websocket: WebSocket, conversation_id: str) -> None:  # pyright: ignore[reportUnusedFunction]
        """The Dock ▸ Browser tab: what the conversation's clones see and do, live (§3.4).

        `browser.live.LiveView` serves it over whichever route the tab is on (R1 or R2):
        state pushed whenever a clone acts, frames only while the head is looking, sized to
        the dock, stills for a tab that is not painting, overlays and step notices.
        """
        try:
            refuse_cross_origin(cast(Request, websocket))
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        view = LiveView(
            _service(),
            conversation_id,
            LiveSocketAdapter(websocket),
            extension_connected=lambda: bool(hub.status().get("connected")),
            on_stop=on_stop,
        )
        with contextlib.suppress(WebSocketDisconnect, ConnectionClosed, RuntimeError):
            await view.run()
