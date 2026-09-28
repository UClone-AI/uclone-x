"""Settings → 연결 → uClone2: the `/api/links*` routes over the link supervisor.

See the uClone2 link design, §3.6. A head over `link/uclone2/`: the service decides
what is stored, kept and retried, and the supervisor runs the sessions; these routes only
read and call them.

**Nothing here carries the token**, not even its masked hint: a link is shown by its clone
pair and its state, and the user never needs the credential again. A refusal is
`{code, message}` -- `code` is a `LinkFailure` value (or a store case) for the dashboard to
word in the user's language, `message` the plain Korean sentence the CLI prints. Neither
carries a status, an error code from uClone2, a URL or an exception name; the session's
protocol detail stays in `LinkSession.diagnostic`, for the Diagnostics tab.

`GET /api/links/{link_id}/activity` is not mounted yet: no activity is recorded before
step 3's `LinkActivity`, and an empty list would say "nothing happened" where the truth is
"nothing is counted".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from uclone_x.link.uclone2.client import LinkError, LinkFailure
from uclone_x.link.uclone2.models import LinkRecord
from uclone_x.link.uclone2.service import UnlinkOutcome, check_local_choice, local_chooser
from uclone_x.link.uclone2.session import LinkSessionState
from uclone_x.link.uclone2.store import LinkStoreError
from uclone_x.link.uclone2.supervisor import LinkSupervisor

__all__ = ["ELSEWHERE_STATE", "UNLINK_PENDING_STATE", "link_state", "register_link_routes"]

#: The card's state for a record whose unlink uClone2 has not confirmed. Not a session
#: state: no session runs for it.
UNLINK_PENDING_STATE = "unlink_pending"
#: The card's state while another runtime on this machine (`ucx link run`) holds the links
#: and this dashboard dials none of them. Not a session state either: no session runs here.
ELSEWHERE_STATE = "elsewhere"

#: What each refusal answers with. The body's `code` is what the dashboard reads; the status
#: is only for HTTP's sake.
_STATUS: dict[LinkFailure, int] = {
    LinkFailure.BAD_INPUT: 400,
    LinkFailure.SERVER_WITH_URL: 400,
    LinkFailure.CODE_INVALID: 400,
    LinkFailure.NO_LOCAL_CLONE: 400,
    LinkFailure.NO_MATCHING_CLONE: 400,
    LinkFailure.CAP_REACHED: 409,
    LinkFailure.TOKEN_INVALID: 409,
    LinkFailure.NOT_PENDING: 409,
    LinkFailure.ELSEWHERE: 409,
    LinkFailure.NOT_FOUND: 404,
    LinkFailure.RATE_LIMITED: 429,
    LinkFailure.DISABLED: 503,
    LinkFailure.UNREACHABLE: 502,
    LinkFailure.SERVER_ERROR: 502,
}

_BAD_REQUEST_MESSAGE = "요청을 읽을 수 없습니다. 연결 주소를 다시 붙여 넣으십시오"
_NOT_WRITTEN_MESSAGE = (
    "이 컴퓨터에 연결 정보를 저장하지 못했습니다. 저장 공간과 권한을 확인하십시오"
)


class _ConnectBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    connect: str
    #: The local clone that answers; `None` is "the same name" (then the default clone).
    local_agent_id: str | None = None
    server_url: str | None = None


class _EnabledBody(BaseModel):
    enabled: bool


def _refusal(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"code": code, "message": message}, status_code=status)


def _link_refusal(err: LinkError) -> JSONResponse:
    return _refusal(err.failure.value, str(err), _STATUS.get(err.failure, 400))


def _store_refusal(err: LinkStoreError) -> JSONResponse:
    return _refusal(err.kind, str(err), 500)


def link_state(
    record: LinkRecord, session_states: dict[str, LinkSessionState], *, elsewhere: bool = False
) -> str:
    """The card's state: what the record says first, then the live session's.

    `elsewhere`: another runtime holds the links, so a link with no session here is not
    offline -- it is served (or being dialled) by that runtime.
    """
    if record.unlink_pending:
        return UNLINK_PENDING_STATE
    if not record.enabled:
        return LinkSessionState.ENDED.value
    if record.paused:
        return LinkSessionState.PAUSED.value
    if record.link_id not in session_states and elsewhere:
        return ELSEWHERE_STATE
    return session_states.get(record.link_id, LinkSessionState.OFFLINE).value


def _page_url(record: LinkRecord) -> str:
    """The uClone2 clone's minihompy. The contract names no page URL; this is uClone2's
    own web route, `/hompy/<username>`."""
    return f"{record.server_url}/hompy/{quote(record.remote_username, safe='')}"


def _card(
    record: LinkRecord, session_states: dict[str, LinkSessionState], *, elsewhere: bool
) -> dict[str, Any]:
    return {
        "link_id": record.link_id,
        "local_agent_id": record.local_agent_id,
        "remote_username": record.remote_username,
        "remote_display_name": record.remote_display_name,
        "page_url": _page_url(record),
        "state": link_state(record, session_states, elsewhere=elsewhere),
        "created_at": record.created_at.isoformat(),
        "last_connected_at": (
            record.last_connected_at.isoformat() if record.last_connected_at else None
        ),
    }


def register_link_routes(
    app: FastAPI,
    *,
    supervisor: LinkSupervisor,
    local_clone_names: Callable[[], list[str]],
    refuse_cross_origin: Callable[[Request], None],
) -> None:
    """Mount `/api/links*` over `supervisor` and the local clone list."""

    @app.get("/api/links")
    async def list_links(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Every link with its live state; `unreadable` when the links file is damaged."""
        refuse_cross_origin(request)
        try:
            records = supervisor.store.records()
        except LinkStoreError:
            return {"links": [], "unreadable": True}
        states = supervisor.states()
        return {
            "links": [_card(r, states, elsewhere=supervisor.held_elsewhere) for r in records],
            "unreadable": False,
        }

    @app.post("/api/links/uclone2", response_model=None)
    async def connect_uclone2(request: Request) -> dict[str, Any] | JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Link from a pasted connect URL or code; the session starts at once."""
        refuse_cross_origin(request)
        try:
            body = _ConnectBody.model_validate(await request.json())
        except (ValueError, ValidationError):
            return _refusal(LinkFailure.BAD_INPUT.value, _BAD_REQUEST_MESSAGE, 400)
        named = body.local_agent_id or None
        known = local_clone_names()
        try:
            check_local_choice(named, known)
            record = await supervisor.link(
                body.connect,
                choose_local=local_chooser(named, known),
                server_url=body.server_url or None,
            )
        except LinkError as err:
            return _link_refusal(err)
        except LinkStoreError as err:
            return _store_refusal(err)
        return _card(record, supervisor.states(), elsewhere=supervisor.held_elsewhere)

    @app.delete("/api/links/{link_id}", response_model=None)
    async def remove_link(  # pyright: ignore[reportUnusedFunction]
        link_id: str, request: Request, local: bool = False
    ) -> dict[str, Any] | JSONResponse:
        """Unlink in uClone2 (`removed`, or `pending` when it could not be reached).

        `?local=true` removes a *해제 대기* record from this machine only.
        """
        refuse_cross_origin(request)
        try:
            if local:
                await supervisor.forget_local(link_id)
                return {"outcome": "forgotten"}
            outcome = await supervisor.unlink(link_id)
        except LinkError as err:
            return _link_refusal(err)
        except LinkStoreError as err:
            return _store_refusal(err)
        except OSError:
            return _refusal("not_written", _NOT_WRITTEN_MESSAGE, 500)
        return {"outcome": "removed" if outcome is UnlinkOutcome.REMOVED else "pending"}

    @app.post("/api/links/{link_id}/enabled", response_model=None)
    async def set_link_enabled(  # pyright: ignore[reportUnusedFunction]
        link_id: str, request: Request
    ) -> dict[str, Any] | JSONResponse:
        """*온라인으로 전환* / *오프라인으로 전환*: kept across restarts, not an unlink."""
        refuse_cross_origin(request)
        try:
            body = _EnabledBody.model_validate(await request.json())
        except (ValueError, ValidationError):
            return _refusal(LinkFailure.BAD_INPUT.value, _BAD_REQUEST_MESSAGE, 400)
        try:
            record = await supervisor.set_online(link_id, body.enabled)
        except LinkError as err:
            return _link_refusal(err)
        except LinkStoreError as err:
            return _store_refusal(err)
        except OSError:
            return _refusal("not_written", _NOT_WRITTEN_MESSAGE, 500)
        return _card(record, supervisor.states(), elsewhere=supervisor.held_elsewhere)
