"""`Uclone2LinkClient`: the REST half of the Linked Runtime contract, as a runtime calls it.

Three routes in this step: `POST /linked/connect` (a link code for a token), `GET
/linked/self` (the link and the clone's profile) and `DELETE /linked/self` (unlink from
this side). The contract is uClone2's `docs/contracts/linked-runtime/openapi.yaml`, owned
by both repositories; UClone-X keeps a pinned copy as test fixtures.

**The token stays in here.** It goes out in one place, the `Authorization` header this
module builds, and never into a URL, a log line or an exception. Every failure is a
`LinkError` whose message is a plain sentence a user can act on; the status and the
server's error code travel separately in `LinkError.diagnostic` for the Diagnostics tab,
and neither of them ever quotes a request or a response body.

The code is exchanged by `POST` only. uclone_mm's exchange was a `GET` URL, which put the
credential into access logs and into its agent's transcript.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeVar, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from uclone_x import __version__
from uclone_x.link.uclone2.models import ConnectResult, LinkSelf

__all__ = [
    "API_PREFIX",
    "CALLED_ROUTES",
    "PRODUCTION_ORIGIN",
    "LinkError",
    "LinkFailure",
    "ParsedConnect",
    "Uclone2LinkClient",
    "UnlinkResult",
    "parse_connect_input",
]

logger = logging.getLogger(__name__)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

#: A bare code means this origin (the contract's `servers[0]` default).
PRODUCTION_ORIGIN: Final = "https://uclone.ai"
API_PREFIX: Final = "/api/v4"

#: Every (method, path) this client calls, as the contract spells the path. A contract test
#: fails on one the vendored `openapi.yaml` does not declare.
CALLED_ROUTES: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("POST", "/linked/connect"),
        ("GET", "/linked/self"),
        ("DELETE", "/linked/self"),
    }
)

#: The contract gives a code as exactly 10 characters and says nothing of the alphabet;
#: this admits the URL-safe ones, which is all a code pasted from a URL path can hold.
_CODE_RE: Final = re.compile(r"\A[A-Za-z0-9_-]{10}\Z")
_LINK_PATH_RE: Final = re.compile(r"\A/link/([^/]+)/?\Z")

_RUNTIME_NAME: Final = "uclone-x"
_TIMEOUT_S: Final = 15.0


class LinkFailure(StrEnum):
    """Why a link call failed, in terms a head can act on."""

    BAD_INPUT = "bad_input"
    SERVER_WITH_URL = "server_with_url"
    CODE_INVALID = "code_invalid"
    CAP_REACHED = "cap_reached"
    RATE_LIMITED = "rate_limited"
    DISABLED = "disabled"
    TOKEN_INVALID = "token_invalid"
    UNREACHABLE = "unreachable"
    SERVER_ERROR = "server_error"
    NOT_FOUND = "not_found"
    #: This machine has no clone that could answer for the uClone2 clone.
    NO_LOCAL_CLONE = "no_local_clone"
    #: No local clone shares the uClone2 clone's name and there is no default clone; the
    #: exchange was undone.
    NO_MATCHING_CLONE = "no_matching_clone"
    #: Removing a record from this machine alone is for a *해제 대기* record only.
    NOT_PENDING = "not_pending"
    #: Another runtime on this machine (`ucx link run`) holds the links; this one dials none.
    ELSEWHERE = "elsewhere"


#: What the user reads for each failure. Plain sentences: no status code, no error code,
#: no URL, no exception name. `CODE_INVALID` and `TOKEN_INVALID` are the design's own copy.
MESSAGES: Final[dict[LinkFailure, str]] = {
    LinkFailure.BAD_INPUT: (
        "붙여 넣은 내용이 uClone2 연결 주소나 코드가 아닙니다. "
        "uClone2의 클론 페이지에서 받은 연결 주소를 그대로 붙여 넣으십시오"
    ),
    LinkFailure.SERVER_WITH_URL: (
        "연결 주소에는 서버가 이미 들어 있습니다. 서버 지정은 코드만 붙여 넣을 때 쓰십시오"
    ),
    LinkFailure.CODE_INVALID: "코드가 만료되었거나 이미 사용되었습니다. uClone2에서 새로 받으십시오",
    LinkFailure.CAP_REACHED: (
        "uClone2 계정 하나에 연결할 수 있는 클론은 5개까지입니다. "
        "다른 클론의 연결을 해제한 뒤 다시 시도하십시오"
    ),
    LinkFailure.RATE_LIMITED: "연결 시도가 너무 잦았습니다. 1분쯤 뒤에 다시 시도하십시오",
    LinkFailure.DISABLED: "uClone2가 연결을 잠시 멈췄습니다. 나중에 다시 시도하십시오",
    LinkFailure.TOKEN_INVALID: (
        "이 연결은 끝났습니다 — uClone2에서 해제했거나 다른 곳에서 다시 연결했습니다"
    ),
    LinkFailure.UNREACHABLE: (
        "uClone2에 닿을 수 없습니다. 인터넷 연결을 확인하고 다시 시도하십시오"
    ),
    LinkFailure.SERVER_ERROR: "uClone2가 요청을 처리하지 못했습니다. 잠시 후 다시 시도하십시오",
    LinkFailure.NOT_FOUND: "그런 연결이 없습니다. 연결 목록에서 번호를 확인하십시오",
    LinkFailure.NO_LOCAL_CLONE: (
        "이 컴퓨터에 답할 클론이 없습니다. 클론을 먼저 만들거나 있는 클론을 고른 뒤 다시 연결하십시오"
    ),
    LinkFailure.NO_MATCHING_CLONE: (
        "같은 이름의 로컬 클론도 기본 클론도 없어서 연결을 되돌렸습니다. "
        "답할 클론을 고른 뒤 uClone2에서 새 코드를 받아 다시 연결하십시오"
    ),
    LinkFailure.NOT_PENDING: (
        "이 연결은 해제 대기 중이 아닙니다. 이 컴퓨터에서만 지우는 것은 해제 대기 중인 연결에만 쓸 수 있습니다"
    ),
    LinkFailure.ELSEWHERE: (
        "이 컴퓨터에서 따로 실행 중인 UClone-X가 uClone2 연결을 맡고 있습니다. "
        "그쪽을 멈춘 뒤 다시 시도하십시오"
    ),
}


class LinkError(Exception):
    """A link call failed. `str(error)` is the sentence to show; nothing else is.

    `diagnostic` is for the Diagnostics tab only: the HTTP status and the contract's error
    code (for example `400 LINKED_CODE_INVALID`), never a URL, a header or a body.
    """

    def __init__(self, failure: LinkFailure, diagnostic: str = "") -> None:
        super().__init__(MESSAGES[failure])
        self.failure = failure
        self.diagnostic = diagnostic


class UnlinkResult(StrEnum):
    UNLINKED = "unlinked"
    #: 401 `LINKED_TOKEN_INVALID`: the link was already gone, which counts as success.
    ALREADY_GONE = "already_gone"


@dataclass(frozen=True)
class ParsedConnect:
    """A pasted connect URL or bare code, split into the server's origin and the code."""

    server_url: str
    code: SecretStr


def _is_loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _origin(url: str) -> str | None:
    """`scheme://host[:port]` of `url`, or `None` when it is not a usable server address.

    `https` everywhere; plain `http` only for a uClone2 on this machine, where there is no
    wire for the code or the token to cross.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        # `.port` parses lazily and raises on `host:abc`; ask now, not inside httpx.
        _ = parts.port
    except ValueError:
        # An unbalanced `[` or a non-numeric port. Not an address; the caller says so.
        return None
    if not host or parts.username or parts.password:
        return None
    if parts.scheme == "https" or (parts.scheme == "http" and _is_loopback(host)):
        return f"{parts.scheme}://{parts.netloc}"
    return None


def parse_connect_input(text: str, server_url: str | None = None) -> ParsedConnect:
    """Split what the user pasted into the origin to call and the one-time code.

    A connect URL is `https://<origin>/link/<code>` and carries its own origin, so a
    staging or local uClone2 needs no separate server field. A bare code means
    `server_url` when one is given, and the production origin otherwise. A server given
    together with a URL is refused: the two could disagree, and guessing which one the
    user meant would send the code to the other.
    """
    pasted = text.strip()
    if "://" in pasted:
        if server_url:
            raise LinkError(LinkFailure.SERVER_WITH_URL)
        origin = _origin(pasted)
        if origin is None:
            raise LinkError(LinkFailure.BAD_INPUT)
        parts = urlsplit(pasted)
        match = _LINK_PATH_RE.match(parts.path)
        if match is None or parts.query or parts.fragment:
            raise LinkError(LinkFailure.BAD_INPUT)
        code = match.group(1)
    else:
        code = pasted
        if server_url:
            chosen = _origin(server_url.strip())
            if chosen is None or urlsplit(server_url.strip()).path not in ("", "/"):
                raise LinkError(LinkFailure.BAD_INPUT)
            origin = chosen
        else:
            origin = PRODUCTION_ORIGIN
    if not _CODE_RE.match(code):
        raise LinkError(LinkFailure.BAD_INPUT)
    return ParsedConnect(server_url=origin, code=SecretStr(code))


def _error_code(response: httpx.Response) -> str:
    """The envelope's `error.code`, or `""`. Never the message: it is the server's prose."""
    try:
        body = cast(object, response.json())
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    error = cast(dict[str, object], body).get("error")
    if not isinstance(error, dict):
        return ""
    code = cast(dict[str, object], error).get("code")
    return code if isinstance(code, str) and re.fullmatch(r"[A-Z0-9_]{1,64}", code) else ""


def _diagnostic(response: httpx.Response) -> str:
    return f"{response.status_code} {_error_code(response)}".strip()


def _data(response: httpx.Response, model: type[_ModelT]) -> _ModelT:
    """The envelope's `data` as `model`, or a plain `SERVER_ERROR`.

    `from None`, and nothing of the failure is logged: a validation error quotes its
    input, and a `connect` answer's input is the token.
    """
    try:
        body = cast(object, response.json())
        data = cast(dict[str, object], body).get("data") if isinstance(body, dict) else None
        return model.model_validate(data)
    except (ValueError, ValidationError):
        pass
    logger.warning("uClone2 answered %s with a body the contract does not describe", model.__name__)
    raise LinkError(LinkFailure.SERVER_ERROR, f"{response.status_code} unexpected body")


class Uclone2LinkClient:
    """Calls one uClone2 server's link routes. Holds no state between calls.

    `transport` replaces the network, for tests and for a host with its own HTTP stack.
    """

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = _TIMEOUT_S,
        runtime_version: str = __version__,
    ) -> None:
        self._transport = transport
        self._timeout_s = timeout_s
        self._runtime_version = runtime_version

    async def connect(self, server_url: str, code: SecretStr) -> ConnectResult:
        """Exchange a one-time link code for this runtime's token (`POST /linked/connect`)."""
        response = await self._send(
            "POST",
            server_url,
            "/linked/connect",
            json={
                "code": code.get_secret_value(),
                "runtime": {"name": _RUNTIME_NAME, "version": self._runtime_version},
            },
        )
        status = response.status_code
        if status == 200:
            return _data(response, ConnectResult)
        failure = {
            400: LinkFailure.CODE_INVALID,
            409: LinkFailure.CAP_REACHED,
            429: LinkFailure.RATE_LIMITED,
            503: LinkFailure.DISABLED,
        }.get(status, LinkFailure.SERVER_ERROR)
        raise LinkError(failure, _diagnostic(response))

    async def get_self(self, server_url: str, token: SecretStr) -> LinkSelf:
        """The link as the server sees it, and the clone's current profile."""
        response = await self._send("GET", server_url, "/linked/self", token=token)
        if response.status_code == 200:
            return _data(response, LinkSelf)
        raise LinkError(self._link_failure(response.status_code), _diagnostic(response))

    async def unlink(self, server_url: str, token: SecretStr) -> UnlinkResult:
        """End the link from this side (`DELETE /linked/self`).

        A 401 means the token is already revoked -- the owner unlinked in uClone2, or a
        repeat of this call -- and is reported as `ALREADY_GONE`, not as a failure.
        """
        response = await self._send("DELETE", server_url, "/linked/self", token=token)
        if response.status_code == 204:
            return UnlinkResult.UNLINKED
        if response.status_code == 401:
            return UnlinkResult.ALREADY_GONE
        raise LinkError(self._link_failure(response.status_code), _diagnostic(response))

    @staticmethod
    def _link_failure(status: int) -> LinkFailure:
        if status == 401:
            return LinkFailure.TOKEN_INVALID
        if status == 503:
            return LinkFailure.DISABLED
        return LinkFailure.SERVER_ERROR

    async def _send(
        self,
        method: str,
        server_url: str,
        path: str,
        *,
        token: SecretStr | None = None,
        json: dict[str, object] | None = None,
    ) -> httpx.Response:
        headers = {"Accept": "application/json"}
        if token is not None:
            # The one place the token leaves this process: a header, never the URL.
            headers["Authorization"] = f"Bearer {token.get_secret_value()}"
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout_s, follow_redirects=False
            ) as http:
                return await http.request(
                    method, f"{server_url}{API_PREFIX}{path}", headers=headers, json=json
                )
        except httpx.InvalidURL:
            # Not an `HTTPError`: a server address `_origin` let through that httpx still
            # cannot build a request for. It is what the user typed that is wrong.
            logger.info("uClone2 %s %s had an unusable server address", method, path)
            raise LinkError(LinkFailure.BAD_INPUT) from None
        except httpx.HTTPError as err:
            # The class name only: an httpx message can quote the URL, and the Diagnostics
            # line should not depend on which exception happened to say what.
            logger.info("uClone2 %s %s could not be sent: %s", method, path, type(err).__name__)
            raise LinkError(LinkFailure.UNREACHABLE, type(err).__name__) from None
