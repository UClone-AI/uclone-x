"""An in-process uClone2 for the link tests, answering only as the contract allows.

`FakeUclone2.transport` is an `httpx.MockTransport`, so nothing leaves the process. Every
answer it can give is a (route, status) entry in `ANSWERS`, and a contract test checks
each entry against the vendored `openapi.yaml`: the status must be one the spec declares
for that route and the body must validate against the spec's schema for it. A test that
wants the client's handling of a server *outside* the contract builds its own transport
and says so.

The error envelopes carry deliberately internal-looking `message` text: the client must
never show it, and the tests assert that it does not.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import httpx

#: A token shaped like the real thing. Tests search every output for it.
TOKEN: Final = "ucl_Zt8qLw3NvXe5RkP0sYcJ2mHa9bFd"
#: The last four characters, which a masked view may show.
TOKEN_TAIL: Final = TOKEN[-4:]
CODE: Final = "Ab3dE6gH9j"
ORIGIN: Final = "https://staging.uclone.test"
CONNECT_URL: Final = f"{ORIGIN}/link/{CODE}"
BOT_ID: Final = "bot_7f3a"

_CLONE: Final[dict[str, object]] = {
    "id": BOT_ID,
    "username": "haru",
    "display_name": "Haru",
    "avatar_url": "https://cdn.uclone.test/avatars/haru.png",
    "persona": {"text": "A cheerful gardener.", "identity_matrix": {}},
}

#: A later profile: the owner edited the clone in uClone2.
EDITED_CLONE: Final[dict[str, object]] = {**_CLONE, "display_name": "Haru (봄)"}


def _ok(data: object) -> dict[str, object]:
    return {"status": "success", "request_id": "req_1", "data": data}


def _err(code: str, message: str) -> dict[str, object]:
    return {"status": "error", "request_id": "req_1", "error": {"code": code, "message": message}}


def self_body(
    *, clone: Mapping[str, object] = _CLONE, updated_at: str = "2026-09-26T22:40:00Z"
) -> dict[str, object]:
    return _ok(
        {
            "bot_id": BOT_ID,
            "clone": dict(clone),
            "clone_updated_at": updated_at,
            "link": {
                "created_at": "2026-09-27T09:00:00Z",
                "runtime": {"name": "uclone-x", "version": "0.3.0"},
            },
            "pending": 3,
            "online": False,
        }
    )


CONNECT: Final = ("POST", "/linked/connect")
GET_SELF: Final = ("GET", "/linked/self")
DELETE_SELF: Final = ("DELETE", "/linked/self")

#: Every answer the fake can give, by (method, contract path) and status. `None` is an
#: empty body (a 204).
ANSWERS: Final[dict[tuple[str, str], dict[int, dict[str, object] | None]]] = {
    CONNECT: {
        200: _ok(
            {
                "token": TOKEN,
                "bot_id": BOT_ID,
                "ws_url": "wss://staging.uclone.test/api/v4/linked/ws",
                "protocol": 1,
                "clone": _CLONE,
            }
        ),
        400: _err("LINKED_CODE_INVALID", "redis GET linked:code:Ab3dE6gH9j -> nil"),
        409: _err("LINKED_CAP_REACHED", "owner 42 has 5 linked_clone rows"),
        429: _err("RATE_LIMIT_EXCEEDED", "limiter ip=10.0.0.7 bucket=connect"),
        503: _err("LINKED_DISABLED", "LINKED_ENABLED=false in config.go"),
    },
    GET_SELF: {
        200: self_body(),
        401: _err("LINKED_TOKEN_INVALID", "token hash not found in linked_tokens"),
        503: _err("LINKED_DISABLED", "LINKED_ENABLED=false in config.go"),
    },
    DELETE_SELF: {
        204: None,
        401: _err("LINKED_TOKEN_INVALID", "token hash not found in linked_tokens"),
        503: _err("LINKED_DISABLED", "LINKED_ENABLED=false in config.go"),
    },
}

_DEFAULT_STATUS: Final[dict[tuple[str, str], int]] = {CONNECT: 200, GET_SELF: 200, DELETE_SELF: 204}
_API_PREFIX: Final = "/api/v4"


@dataclass
class FakeUclone2:
    """Answers the link routes from `ANSWERS`; records every request it receives."""

    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])
    status: dict[tuple[str, str], int] = field(default_factory=lambda: dict(_DEFAULT_STATUS))
    #: Routes that fail as a network error instead of answering.
    unreachable: set[tuple[str, str]] = field(default_factory=set[tuple[str, str]])
    #: Replaces the 200 body of `GET /linked/self`, for a profile that changed.
    self_answer: dict[str, object] | None = None

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def answer(self, route: tuple[str, str], status: int) -> None:
        if status not in ANSWERS[route]:
            raise AssertionError(f"the fake has no {status} for {route}; add it to ANSWERS")
        self.status[route] = status

    def calls(self, route: tuple[str, str]) -> list[httpx.Request]:
        method, path = route
        return [r for r in self.requests if r.method == method and r.url.path == _API_PREFIX + path]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        route = (request.method, path[len(_API_PREFIX) :]) if path.startswith(_API_PREFIX) else None
        if route is None or route not in ANSWERS:
            raise AssertionError(f"the client called a route the fake does not serve: {path}")
        if route in self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        status = self.status[route]
        body = ANSWERS[route][status]
        if route == GET_SELF and status == 200 and self.self_answer is not None:
            body = self.self_answer
        if body is None:
            return httpx.Response(status)
        return httpx.Response(status, content=json.dumps(body).encode(), headers=_JSON)


_JSON: Final = {"Content-Type": "application/json"}


#: What must never reach a user-facing sentence: a status code, a URL, the API path, a
#: contract error code, an exception class name, a traceback, the token or the code.
_INTERNALS: Final = (
    ("status code", re.compile(r"\b[1-5]\d\d\b")),
    ("url", re.compile(r"://")),
    ("api path", re.compile(r"/api/")),
    ("error code", re.compile(r"\b(?:LINKED|RATE_LIMIT)_[A-Z_]+\b")),
    ("exception name", re.compile(r"\b[A-Za-z]*(?:Error|Exception)\b")),
    ("traceback", re.compile(r"Traceback")),
    ("token", re.compile(re.escape(TOKEN))),
    ("code", re.compile(re.escape(CODE))),
    ("server prose", re.compile(r"redis|linked_tokens|config\.go|limiter")),
)


def internals_in(text: str) -> list[str]:
    """The kinds of internal detail `text` shows a user; empty when it is plain copy."""
    return [name for name, pattern in _INTERNALS if pattern.search(text)]
